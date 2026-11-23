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
from inkling_tt.mtp_projection import SharedProjection, _pack_rows, _tile_row
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


@lru_cache(maxsize=8)
def _verify_stage_program(lmax, tasks, specs):
    rows = len(tasks)
    grid, rt = _grid(tasks)
    unique = (*specs[0][:4], *specs[0][3 + rows : 6 + rows])
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "verify_stage.cpp"),
        core_ranges=grid,
        compile_time_args=[rows, lmax, *sum(unique, ())],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, BF, size=128 + 2 * D)])


class TargetVerify:
    """Run a layer-major block with B1 row arithmetic and read-only committed state."""

    def __init__(self, model, proposals, *, streaming=False, compare_serial=False):
        c = model.c
        if (
            model.trace_id is not None
            or len(model.layers) != 24
            or model.lmax % 32
            or not 32 <= model.lmax <= 32768
            or not (
                c.fused_attn_decode
                and c.join_attention_prep
                and c.fused_router
                and c.fused_head_tail
                and c.fused_message_down
                and c.message_down_inputs
                and c.message_fusion
            )
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
        row = lambda size, dtype=BF, mem=DRAM: ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, 1, size)), dtype, TILE, device, mem
        )
        layer = model.layers[0]
        q_weight, o_weight = layer.dq.weight, layer.wo
        global_layer = next(layer for layer in model.layers if layer.is_global)
        self.shared = True
        self.q, self.k, self.v, self.k_dyn, self.v_dyn, self.o = (
            SharedProjection(device, w)
            for w in (q_weight, layer.wk, layer.wv, global_layer.k_wproj, global_layer.v_wproj, o_weight)
        )
        self.scratch = tuple(
            SimpleNamespace(
                visit=tuple(
                    ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, device, t.memory_config())
                    for t in c.visit_scratch
                ),
                inputs_fp=tuple(row(D, FP) for _ in range(3)),
                inputs_bf=tuple(row(D) for _ in range(3)),
                qkv_down=tuple(row(64, mem=L1) for _ in range(3)),
                qkv_blend=(row(D, mem=q_weight.input_shards), row(D), row(D)),
                mixed_fp=row(D, FP),
                mlp_down=(row(64, mem=L1),),
                messages=tuple(row(D, FP) for _ in range(4)),
            )
            for _ in range(self.rows)
        )
        if compare_serial:
            for scratch in self.scratch:
                scratch.sharded_q = row(D, mem=q_weight.input_shards)
                scratch.qrsg = row(q_weight.n, mem=q_weight.output_shards)
                scratch.k, scratch.v = (row(HKV * HD) for _ in range(2))
                scratch.k_dyn, scratch.v_dyn = (row(32) for _ in range(2))
                scratch.attended = row(o_weight.n, mem=o_weight.output_shards)
        self.histories = tuple(_alloc(device, (1, 1, 4, width), FP) for width in (HKV * HD, HKV * HD, D, D))
        self.mlp = _alloc(device, (1, 1, 32, D))
        self.mlp_tasks = _tasks(device, D // 32)
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

    def stage_device(self, anchor, drafts, start):
        m = self.model
        if len(drafts) != self.rows - 1 or not m.c.row_major_state or not hasattr(m, "embed_table"):
            raise ValueError("Device verification requires normalized feedback embeddings and a full draft block")
        inputs = (start, m.embed_table, m.tau_table, anchor, *drafts)
        outputs = tuple(tensor for step in self.steps for tensor in (step.x, step.pos_g, step.tau))
        _launch(_verify_stage_program, (m.lmax, _tasks(m.c.dev, self.rows)), (*inputs, *outputs))

    def verify(self):
        m = self.model
        states = [(step.x, _DecodeBank(bank, 0, 1), None) for step, bank in zip(self.steps, self.banks)]
        states = self._range(0, LOOP0, 0, states)
        prefix = states
        states = self._range(LOOP0, LOOP1, 0, states)
        for p in range(1, NPASS):
            reenter = [(x, bank._replace(tail_start=1 + p * ((LOOP1 - LOOP0) // NB)), None) for x, bank, _ in prefix]
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
            messages_by_row = []
            for i, (x, _, _) in enumerate(states):
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
                        out=self.scratch[i].messages,
                    )
                messages_by_row.append(messages)
            inputs = [
                layer._attention_inputs(x, bank, messages, attention_input, scratch=scratch)
                for (x, bank, attention_input), messages, scratch in zip(states, messages_by_row, self.scratch)
            ]
            projections = None
            if self.shared:
                xq, xk, xv = zip(*inputs)
                q = self.q.project(layer.dq.weight, xq)
                k = self.k.project(layer.wk, xk)
                v = self.v.project(layer.wv, xv)
                kd = self.k_dyn.project(layer.k_wproj, xk) if layer.is_global else None
                vd = self.v_dyn.project(layer.v_wproj, xv) if layer.is_global else None
                projections = (q, k, v, kd, vd)
            attention = []
            for i, (xqkv, step, row_slot, scratch) in enumerate(zip(inputs, self.steps, self.slots, self.scratch)):
                attention.append(
                    layer.attention(
                        *xqkv,
                        visit,
                        step,
                        staging=(staged.k, staged.v, self.steps[0].pos_g, row_slot),
                        scratch=scratch,
                        projections=projections,
                        output=self.o.input if self.shared else None,
                        row=i if self.shared else 0,
                    )
                )
                for h in range(2):
                    _tile_row(self.histories[h], staged.conv[h], target=i)
            if self.shared:
                projected = self.o.project(layer.wo)
                attention = [projected] * self.rows
            pending = []
            mixed_rows = []
            for i, ((x, bank, _), a, messages, scratch) in enumerate(
                zip(states, attention, messages_by_row, self.scratch)
            ):
                mixed, bank, residual = layer._finish_attention(
                    x,
                    bank,
                    self.histories[2],
                    a,
                    messages,
                    scratch=scratch,
                    row=i if self.shared else 0,
                )
                if self.shared:
                    mixed_rows.append(mixed)
                else:
                    ids, gamma = self.routes[i]
                    rowvm.fused_router(mixed, layer.router_w.tensor, layer.dec.router_b, ids_out=ids, gamma_out=gamma)
                    _tile_row(mixed, self.mlp, target=i)
                _tile_row(self.histories[2], staged.conv[2], target=i)
                pending.append((bank, residual))
            if self.shared:
                _pack_rows(self.mlp, mixed_rows, self.mlp_tasks)
                rowvm.fused_router_block(self.mlp, layer.router_w.tensor, layer.dec.router_b, self.routes)
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
