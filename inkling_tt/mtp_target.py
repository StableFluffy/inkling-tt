from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
import ttnn

from inkling_tt import attn_decode, rowvm
from inkling_tt.model import DR, HD, HKV, HQ, LOOP0, LOOP1, NB, NPASS, VOCAB, D, visit_slot
from inkling_tt.model2 import BF, DRAM, FP, L1, MTP_TAP_VISITS, RM, TILE, Step, _DecodeBank
from inkling_tt.mtp_experts import VerifyExperts
from inkling_tt.mtp_projection import SharedProjection, _pack_rows, _tile_row
from inkling_tt.mtp_weights import PADDED_VOCAB
from inkling_tt.prefill import _alloc, _grid, _launch, _program, _tasks, _tile_cb

_KERNELS = Path(__file__).resolve().parent / "kernels" / "mtp"


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
    """Run a layer-major block with B1 row arithmetic and position-indexed state."""

    def __init__(self, model, proposals, *, streaming=False, compare_serial=False):
        c = model.c
        if (
            model.trace_id is not None
            or len(model.layers) != 24
            or model.lmax % 32
            or not 32 <= model.lmax <= 32768
            or not (
                c.indexed_state
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
        self.experts = VerifyExperts(device, self.rows, streaming=streaming)
        self.steps = tuple(Step(c, model.lmax) for _ in range(self.rows))
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
                    ttnn.allocate_tensor_on_device(
                        t.shape, t.dtype, t.layout, device, DRAM if i < 2 else t.memory_config()
                    )
                    for i, t in enumerate(c.visit_scratch)
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
        self.prepared_heads = tuple(
            tuple(_alloc(device, t.shape) for t in c.attention_glue[:3]) for _ in range(self.rows)
        )
        self.mlp = _alloc(device, (1, 1, 32, D))
        self.mlp_tasks = _tasks(device, D // 32)
        self.branch = _alloc(device, (1, 1, 1, D))
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
        self.head_input, self.head_output = (
            model.head_workspace if model.head_workspace is not None else model._alloc_head_workspace()
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
        rows = len(drafts) + 1
        if not 2 <= rows <= self.rows or not m.c.row_major_state or not hasattr(m, "embed_table"):
            raise ValueError("Device verification requires feedback embeddings and a draft block within capacity")
        inputs = (start, m.embed_table, m.tau_table, anchor, *drafts)
        outputs = tuple(tensor for step in self.steps[:rows] for tensor in (step.x, step.pos_g, step.tau))
        _launch(_verify_stage_program, (m.lmax, _tasks(m.c.dev, rows)), (*inputs, *outputs))

    def verify(self, *, rows=None, grouped_glue=True, grouped_attention=True, grouped_front=True, grouped_inputs=True):
        m = self.model
        rows = self.rows if rows is None else rows
        if not 2 <= rows <= self.rows:
            raise ValueError("Verify width must fit the allocated row capacity")
        states = [(step.x, _DecodeBank(bank, 0, 1), None) for step, bank in zip(self.steps[:rows], self.banks)]
        batch = self.shared and grouped_glue
        block_attention = self.shared and grouped_attention
        front = batch and block_attention and grouped_front
        input_groups = front and grouped_inputs and m.c.shard_fusion and m.c.group_gates
        states = self._range(
            0, LOOP0, 0, states, batch=batch, block_attention=block_attention, front=front, input_groups=input_groups
        )
        prefix = states
        states = self._range(
            LOOP0,
            LOOP1,
            0,
            states,
            batch=batch,
            block_attention=block_attention,
            front=front,
            input_groups=input_groups,
        )
        for p in range(1, NPASS):
            reenter = [(x, bank._replace(tail_start=1 + p * ((LOOP1 - LOOP0) // NB)), None) for x, bank, _ in prefix]
            states = self._range(
                LOOP0,
                LOOP1,
                p,
                reenter,
                states,
                batch=batch,
                block_attention=block_attention,
                front=front,
                input_groups=input_groups,
            )
        states = self._range(
            LOOP1,
            len(m.layers),
            0,
            states,
            batch=batch,
            block_attention=block_attention,
            front=front,
            input_groups=input_groups,
        )
        if front:
            _pack_rows(self.taps[3], [hidden for _, _, hidden in states], self.mlp_tasks)
        else:
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
        for i, (logits, index) in enumerate(zip(self.logits[:rows], self.ids[:rows])):
            rowvm.head_tail(self.head_output, logits, index, VOCAB, row=i)
        return self.logits[:rows], self.ids[:rows]

    def _range(self, first, stop, pass_id, states, previous=None, *, batch, block_attention, front, input_groups):
        m = self.model
        for layer in m.layers[first:stop]:
            slot = visit_slot(layer.i, pass_id if layer.is_loop else 0)
            visit = m.visits[slot]
            if (tap := MTP_TAP_VISITS.get((layer.i, pass_id))) is not None:
                if front:
                    _pack_rows(self.taps[tap], [x for x, _, _ in states], self.mlp_tasks)
                else:
                    for i, (x, _, _) in enumerate(states):
                        _tile_row(x, self.taps[tap], target=i)
            messages_by_row = []
            for i in range(len(states)):
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
            mix_group = rowvm._Group() if input_groups else None
            inputs = [
                layer._attention_inputs(
                    x, bank, messages, attention_input, scratch=scratch, group=mix_group, blend=not input_groups
                )
                for (x, bank, attention_input), messages, scratch in zip(states, messages_by_row, self.scratch)
            ]
            if mix_group is not None:
                mix_group.launch()
                if previous is not None:
                    inputs = self._blend_attention_rows(layer, inputs, messages_by_row)
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
            prep_group = rowvm._Group() if front else None
            for i, (xqkv, step, scratch) in enumerate(zip(inputs, self.steps, self.scratch)):
                attention.append(
                    layer.attention(
                        *xqkv,
                        visit,
                        step,
                        scratch=scratch,
                        projections=projections,
                        output=self.o.input if self.shared else None,
                        row=i if self.shared else 0,
                        prepare_heads=self.prepared_heads[i] if block_attention else None,
                        prep_group=prep_group,
                    )
                )
            if prep_group is not None:
                prep_group.launch()
            if block_attention:
                attn_decode.decode_rows(
                    attention,
                    visit.k,
                    visit.v,
                    layer._decode_rel_table,
                    projections[0],
                    global_cache=layer.is_global,
                    gate_col=HQ * (HD + DR) + 32,
                    out=self.o.input,
                )
            if self.shared:
                projected = self.o.project(layer.wo)
                attention = [projected] * len(states)
            after_attention = rowvm._Group() if batch else None
            pending = []
            for i, ((x, bank, _), a, messages, scratch) in enumerate(
                zip(states, attention, messages_by_row, self.scratch)
            ):
                mixed, bank, residual = layer._mix_attention_output(
                    x,
                    bank,
                    visit.conv[2],
                    a,
                    messages is not None,
                    position=self.steps[i].pos_g,
                    scratch=scratch,
                    row=i if self.shared else 0,
                    group=after_attention,
                )
                pending.append((mixed, bank, residual))
            if after_attention is not None:
                after_attention.launch()
            if front and previous is not None:
                mixed_rows = self._blend_mlp_rows(layer, pending, messages_by_row)
            else:
                mixed_rows = [
                    layer._blend_mlp(mixed, messages, scratch=scratch)
                    for (mixed, _, _), messages, scratch in zip(pending, messages_by_row, self.scratch)
                ]
            if not self.shared:
                for i, (mixed, (ids, gamma)) in enumerate(zip(mixed_rows, self.routes)):
                    rowvm.fused_router(mixed, layer.router_w.tensor, layer.dec.router_b, ids_out=ids, gamma_out=gamma)
                    _tile_row(mixed, self.mlp, target=i)
            if self.shared:
                _pack_rows(self.mlp, mixed_rows, self.mlp_tasks)
                rowvm.fused_router_block(
                    self.mlp, layer.router_w.tensor, layer.dec.router_b, self.routes[: len(states)]
                )
            grouped = self.experts.run(self.mlp, self.routes[: len(states)], layer.expert_stream)
            after_mlp = rowvm._Group() if batch else None
            states = []
            for i, (_, bank, residual) in enumerate(pending):
                if not batch:
                    _tile_row(grouped, self.branch, source=i)
                out, mixed_out = self.outputs[slot][i]
                states.append(
                    layer._finish_visit(
                        grouped if batch else self.branch,
                        bank,
                        residual,
                        visit.conv[3],
                        m._next_mix(layer, pass_id),
                        position=self.steps[i].pos_g,
                        out=out,
                        mixed_out=mixed_out,
                        row=i if batch else 0,
                        group=after_mlp,
                    )
                )
            if after_mlp is not None:
                after_mlp.launch()
        return states

    def _blend_attention_rows(self, layer, inputs, messages):
        weights = tuple(layer.gate_down[s] for s in "qkv")
        gates = tuple(layer.gates[s] for s in "qkv")
        d = layer.dec
        norms = (d.norm_q, d.norm_k, d.norm_v) if layer.is_global else (d.norm_in,) * 3
        grid = self.model.c.dev.compute_with_storage_grid_size()
        batch = grid.x * grid.y // (3 * rowvm._MESSAGE_DOWN_STRIPES)
        result = []
        for start in range(0, len(inputs), batch):
            end = min(start + batch, len(inputs))
            operands = [
                (u, m, weight)
                for row, message in zip(inputs[start:end], messages[start:end])
                for u, m, weight in zip(row, message, weights)
            ]
            destinations = tuple(t for scratch in self.scratch[start:end] for t in scratch.qkv_down)
            rowvm.message_down(operands, out=destinations)
            for row, message, scratch in zip(inputs[start:end], messages[start:end], self.scratch[start:end]):
                result.append(
                    rowvm.qkv_message_projections(
                        (
                            ((u, m, h, wf, wg, norm), out.memory_config())
                            for u, m, h, (_, wf, wg), norm, out in zip(
                                row, message, scratch.qkv_down, gates, norms, scratch.qkv_blend
                            )
                        ),
                        out=scratch.qkv_blend,
                    )
                )
        return result

    def _blend_mlp_rows(self, layer, pending, messages):
        mixed = [row for row, _, _ in pending]
        mlp_messages = [row[3] for row in messages]
        weight = layer.gate_down["mlp"]
        _, forget, update = layer.gates["mlp"]
        down = rowvm.message_down(
            [(u, m, weight) for u, m in zip(mixed, mlp_messages)],
            out=tuple(s.mlp_down[0] for s in self.scratch[: len(mixed)]),
        )
        blends = rowvm._Group()
        result = [
            rowvm.message_project(u, m, h, forget, update, layer.dec.norm_post, out=s.visit[2], _group=blends)
            for u, m, h, s in zip(mixed, mlp_messages, down, self.scratch)
        ]
        blends.launch()
        return result
