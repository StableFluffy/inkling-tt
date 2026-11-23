from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import torch
import ttnn

from inkling_tt import rowvm
from inkling_tt.model import HD, HKV, LOOP0, LOOP1, NB, NPASS, VOCAB, D, visit_slot
from inkling_tt.model2 import BF, DRAM, FP, L1, MTP_TAP_VISITS, RM, TILE, Step, _DecodeBank
from inkling_tt.mtp_experts import VerifyExperts
from inkling_tt.mtp_weights import PADDED_VOCAB
from inkling_tt.prefill import _alloc, _grid, _launch, _program, _tasks, _tile_cb

_KERNELS = Path(__file__).resolve().parent / "kernels" / "mtp"


class StagedVisit(NamedTuple):
    k: ttnn.Tensor
    v: ttnn.Tensor
    conv: tuple[ttnn.Tensor, ...]


@lru_cache(maxsize=16)
def _commit_program(tasks, rows, accessors):
    grid, rt = _grid(tasks)
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "target_commit.cpp"),
        core_ranges=grid,
        runtime_args=rt,
        compile_time_args=[rows, *sum(accessors, ())],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, BF, size=64 + 2048)])


class TargetCommit:
    """Commit the accepted prefix from staged BF16 KV and FP32 pre-convolution rows."""

    def __init__(self, device, visits, rows):
        if not 2 <= rows <= 8 or not visits:
            raise ValueError("Target commit requires visits and a verify capacity of 2-8 rows")
        shape = (1, HKV, 32, HD)
        self.staged = {
            slot: StagedVisit(
                _alloc(device, shape),
                _alloc(device, shape),
                tuple(_alloc(device, (1, 1, 32, width), FP) for width in (HKV * HD, HKV * HD, D, D)),
            )
            for slot in visits
        }
        self.tensors = []
        table = []
        for slot, live in visits.items():
            scratch = self.staged[slot]
            visit = (live.k, live.v, scratch.k, scratch.v, *live.conv, *scratch.conv)
            self.tensors.extend(visit)
            table.append([*[t.buffer_address() for t in visit], live.klen, 0, 0, 0])
        self.addresses = ttnn.from_torch(
            torch.tensor(table, dtype=torch.int64), dtype=ttnn.uint32, layout=RM, device=device, memory_config=DRAM
        )
        self.meta = _alloc(device, (1, 8), ttnn.uint32, RM)
        tiles_per_visit = (2 * HKV * HD + 2 * (HKV * HD + D)) // 32
        tasks = _tasks(device, len(visits) * tiles_per_visit)
        accessors = tuple(
            tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args())
            for t in (self.meta, self.addresses, self.tensors[0], self.tensors[4])
        )
        self.program, kernels = _commit_program(tasks, rows, accessors)
        self.kernel = kernels[0]

    def commit(self):
        self.kernel.common_runtime_args = [self.meta.buffer_address(), self.addresses.buffer_address()]
        self.program.kernels = [self.kernel]
        ttnn.generic_op([self.meta, self.addresses, *self.tensors], self.program)


@lru_cache(maxsize=64)
def _tile_row_program(tasks, dtype, src_row, dst_row, specs):
    grid, rt = _grid(tasks)
    tile = 4096 if dtype == FP else 2048
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "tile_row.cpp"),
        core_ranges=grid,
        runtime_args=rt,
        compile_time_args=[tile, src_row, dst_row, *sum(specs[0], ())],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, dtype)])


def _tile_row(src, dst, *, source=0, target=0):
    args = (_tasks(src.device(), src.shape[-1] // 32), src.dtype, source, target)
    _launch(_tile_row_program, args, (src, dst))


class TargetVerify:
    """Run a layer-major block with B1 row arithmetic and read-only committed state."""

    def __init__(self, model, proposals, *, streaming=False):
        c = model.c
        if (
            model.trace_id is not None
            or len(model.layers) != 24
            or not (c.visit_fusion and c.fused_attn_decode and c.fused_router and c.fused_head_tail)
        ):
            raise ValueError("Create verify before capture with the full fused B1 target")
        self.model, self.rows = model, proposals + 1
        device = c.dev
        self.commit = TargetCommit(device, model.visits, self.rows)
        self.experts = VerifyExperts(device, self.rows, streaming=streaming)
        self.steps = tuple(Step(c, model.lmax) for _ in range(self.rows))
        self.slots = tuple(c.up(torch.tensor([r], dtype=torch.int32), ttnn.int32, RM) for r in range(self.rows))
        bank_layout = RM if c.row_major_state else TILE
        self.banks = tuple(c.up(torch.zeros(1, 1, 32, D), FP, bank_layout, L1) for _ in range(self.rows))
        self.scratch = tuple(
            tuple(
                ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, device, t.memory_config())
                for t in c.visit_scratch
            )
            for _ in range(self.rows)
        )
        self.histories = tuple(_alloc(device, (1, 1, 4, width), FP) for width in (HKV * HD, HKV * HD, D, D))
        self.mlp = _alloc(device, (1, 1, 32, D))
        self.branch = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, D)), BF, TILE, device, L1)
        self.routes = tuple(
            (
                ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, 8)), ttnn.uint16, RM, device, L1),
                ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, 8)), FP, TILE, device, L1),
            )
            for _ in range(self.rows)
        )
        self.logits = tuple(_alloc(device, (1, 1, 1, PADDED_VOCAB), BF, RM) for _ in range(self.rows))
        self.ids = tuple(_alloc(device, (1, 1, 1, 1), ttnn.uint32, RM) for _ in range(self.rows))
        self.taps = tuple(_alloc(device, (1, 1, 32, D)) for _ in range(4))
        self.head_input = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, 32, D)), BF, TILE, device, model.head.input_shards
        )
        self.head_output = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, 32, PADDED_VOCAB)), BF, TILE, device, model.head.output_shards
        )
        self.outputs = {}
        for layer in model.layers:
            for p in range(NPASS if layer.is_loop else 1):
                next_mix = model._next_mix(layer, p)
                state_layout = bank_layout if next_mix is not None else TILE
                dtype = FP if next_mix is not None and next_mix[1] is None else BF
                self.outputs[visit_slot(layer.i, p)] = tuple(
                    (
                        _alloc(device, (1, 1, 1, D), BF, state_layout),
                        _alloc(device, (1, 1, 1, D), dtype) if next_mix is not None else None,
                    )
                    for _ in range(self.rows)
                )

    def stage(self, block, start):
        if len(block) != self.rows or not 0 <= start <= self.model.lmax - self.rows:
            raise ValueError("Verify block must fit within the allocated context")
        for step, tok, pos in zip(self.steps, block, range(start, start + self.rows)):
            step.write(self.model.embed_row(tok), pos)

    def verify(self):
        m = self.model
        c = m.c
        states = [(step.x, _DecodeBank(bank, 0, 1), None) for step, bank in zip(self.steps, self.banks)]
        scratch = c.visit_scratch
        try:
            states = self._range(0, LOOP0, 0, states)
            prefix = states
            states = self._range(LOOP0, LOOP1, 0, states)
            for p in range(1, NPASS):
                reenter = [
                    (x, bank._replace(tail_start=1 + p * ((LOOP1 - LOOP0) // NB)), None) for x, bank, _ in prefix
                ]
                states = self._range(LOOP0, LOOP1, p, reenter, states)
            states = self._range(LOOP1, len(m.layers), 0, states)
            for i, (_, _, hidden) in enumerate(states):
                _tile_row(hidden, self.taps[3], target=i)
            head = m.head
            ttnn.interleaved_to_sharded(self.taps[3], head.input_shards, preallocated_output=self.head_input)
            ttnn.linear(
                self.head_input,
                head.tensor,
                program_config=head.program,
                compute_kernel_config=head.compute,
                memory_config=head.output_shards,
                optional_output_tensor=self.head_output,
            )
            for i, (logits, index) in enumerate(zip(self.logits, self.ids)):
                rowvm.head_tail(self.head_output, logits, index, VOCAB, row=i)
        finally:
            c.visit_scratch = scratch
        return self.logits, self.ids

    def _range(self, first, stop, pass_id, states, previous=None):
        m = self.model
        for layer in m.layers[first:stop]:
            slot = visit_slot(layer.i, pass_id if layer.is_loop else 0)
            live = m.visits[slot]
            staged = self.commit.staged[slot]
            visit = SimpleNamespace(klen=live.klen, k=live.k, v=live.v, conv=self.histories)
            for source, dest in zip(live.conv, self.histories):
                ttnn.copy(source, dest)
            pending = []
            for i, ((x, bank, attention_input), step, row_slot) in enumerate(zip(states, self.steps, self.slots)):
                if (tap := MTP_TAP_VISITS.get((layer.i, pass_id))) is not None:
                    _tile_row(x, self.taps[tap], target=i)
                messages = None
                if previous is not None:
                    prev_x, prev_bank, _ = previous[i]
                    messages = rowvm.attnres_mix(
                        prev_bank.tensor,
                        prev_x,
                        m.dec_msg_q[layer.i - LOOP0][2],
                        prev_bank.count - 2,
                        bank_start=prev_bank.tail_start + 1,
                    )
                m.c.visit_scratch = self.scratch[i]
                mixed, bank, residual = layer._forward(
                    x,
                    bank,
                    visit,
                    step,
                    messages,
                    attention_input=attention_input,
                    staging=(staged.k, staged.v, self.steps[0].pos_g, row_slot),
                    defer_experts=True,
                )
                ids, gamma = self.routes[i]
                rowvm.fused_router(mixed, layer.router_w.tensor, layer.dec.router_b, ids_out=ids, gamma_out=gamma)
                _tile_row(mixed, self.mlp, target=i)
                for source, target in zip(self.histories[:3], staged.conv[:3]):
                    _tile_row(source, target, target=i)
                pending.append((bank, residual))
            grouped = self.experts.run(self.mlp, self.routes, layer.expert_stream)
            states = []
            for i, (bank, residual) in enumerate(pending):
                _tile_row(grouped, self.branch, source=i)
                out, mixed_out = self.outputs[slot][i]
                states.append(
                    layer._finish_visit(
                        self.branch,
                        bank,
                        residual,
                        self.histories[3],
                        m._next_mix(layer, pass_id),
                        out=out,
                        mixed_out=mixed_out,
                    )
                )
                _tile_row(self.histories[3], staged.conv[3], target=i)
        return states
