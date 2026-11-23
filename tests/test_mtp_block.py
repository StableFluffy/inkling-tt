import argparse
import statistics
import time

import torch
import ttnn
from mtp_reference import DraftContext, DraftReference

from inkling_tt import dense, paths
from inkling_tt.model import EPS, VOCAB, D, Weights
from inkling_tt.model2 import BF, DRAM, RM
from inkling_tt.mtp_block import DraftBlock, DraftHeadProjection
from inkling_tt.mtp_context import DraftContextProjection, DraftPositions
from inkling_tt.mtp_markov import MarkovDraftHead
from inkling_tt.mtp_weights import PADDED_VOCAB, fold_head, load_drafter
from inkling_tt.prefill import _alloc
from serve.runtime import DEVICE_PARAMS


def stage(data, dest):
    src = ttnn.from_torch(data.contiguous(), layout=dest.layout, dtype=dest.dtype)
    ttnn.copy_host_to_device_tensor(src, dest)


def make_cases(reference, embeddings):
    generator = torch.Generator().manual_seed(724)
    for length, token in ((29, 42), (1055, 198)):
        features = torch.randn(length, 4, D, generator=generator, dtype=torch.bfloat16)
        ctx = DraftContext()
        reference.extend(ctx, features)
        hidden = reference.hidden(ctx, embeddings[token])
        expected = reference.score(hidden[1:], ctx.last, token)
        yield ctx, token, features, hidden, expected


def stage_case(context, features, extend, case, anchor, position):
    ctx, token, taps, _, _ = case
    stage(torch.tensor([[token]], dtype=torch.int32), anchor)
    stage(torch.tensor([ctx.length], dtype=torch.int32), position)
    for start in range(0, ctx.length, context.chunk):
        valid = min(context.chunk, ctx.length - start)
        stage(torch.tensor([[start, valid, 0, 0, 0, 0, 0, 0]], dtype=torch.int32), context.meta)
        for i, dest in enumerate(features):
            padded = torch.zeros(tuple(dest.shape), dtype=torch.bfloat16)
            padded[0, 0, :valid] = taps[start : start + valid, i]
            stage(padded, dest)
        extend()


def check_context(ctx, block, last):
    for (keys, vals), caches in zip(ctx.kv, block.cache):
        cap = caches[0].shape[-2] - 32
        slots = torch.arange(ctx.length - len(keys), ctx.length) % cap
        for expected, cache in zip((keys, vals), caches):
            actual = ttnn.to_torch(cache)[0, :, slots, :].transpose(0, 1).float()
            error = (actual - expected).norm() / expected.float().norm()
            assert error < 0.02, f"prefix {ctx.length}, KV relative L2={error:.5f}"
    actual = ttnn.to_torch(last).float().reshape(-1, D)
    error = (actual - ctx.last).norm() / ctx.last.float().norm()
    assert error < 0.01, f"Markov tap mixture relative L2={error:.5f}"


def main(compare_queries=False, compare_scores=False, compare_projections=False, compare_rope=False):
    count = 3
    weights = load_drafter(paths.snapshot())
    target = Weights(paths.hf_sd())
    head = target.get("lm_head.weight")
    fold, bias = fold_head(head, weights)
    reference = DraftReference(weights, head, fold, bias, count + 1)
    embeddings = target.get("embed_tokens.weight")[:256].float()
    embeddings *= torch.rsqrt(embeddings.square().mean(-1, keepdim=True) + EPS)
    embeddings *= target.get("embed_tokens.embed_norm.weight")
    embeddings = embeddings.bfloat16()
    cases = list(make_cases(reference, embeddings))
    device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        device.enable_program_cache()
        labels = ("ordinary queries", "packed queries") if compare_queries else ("ordinary MLP", "sharded MLP")
        if compare_scores:
            labels = ("ordinary scores", "fused scores")
        if compare_projections:
            labels = ("ordinary projections", "sharded projections")
        if compare_rope:
            labels = ("ordinary RoPE", "fused RoPE")
        blocks = [
            DraftBlock(
                device,
                weights,
                count + 1,
                1088,
                sharded_mlp=True if compare_queries or compare_scores or compare_projections or compare_rope else mode,
                pack_queries=mode if compare_queries else True,
                sharded_attention=mode if compare_projections else True,
                fused_rope=mode if compare_rope else True,
            )
            for mode in (False, True)
        ]
        head = dense.prepare_weight(device, torch.nn.functional.pad(head.T, (0, PADDED_VOCAB - VOCAB)))
        projection = DraftHeadProjection(device, head, weights, count)
        markov = MarkovDraftHead(device, weights, fold, bias, count)
        anchor = _alloc(device, (1, 1), ttnn.uint32, RM)
        position = _alloc(device, (1,), ttnn.uint32, RM)
        embedding = ttnn.from_torch(embeddings, dtype=BF, layout=RM, device=device, memory_config=DRAM)
        last = _alloc(device, (1, 1, 1, D))
        contexts = [DraftContextProjection(block, weights, 128, last) for block in blocks]
        context = contexts[0]
        positions = DraftPositions(device, 1088)
        features = tuple(_alloc(device, (1, 1, context.chunk, D)) for _ in range(4))

        def extend(context):
            positions.rotate(context.meta, context.cos, context.sin)
            context.extend(features)

        def forward(block):
            positions.prepare(block, anchor, position, embedding)
            base, rnn_context = projection.project(block.forward(), last)
            markov.fused_scores = block is blocks[1] if compare_scores else True
            markov.score(base, rnn_context, anchor)

        def read(block):
            hidden = ttnn.to_torch(block.hidden)[0, 0, : count + 1]
            ids = [ttnn.to_torch(index).item() for index in markov.ids]
            logits = torch.stack([ttnn.to_torch(row).flatten()[:VOCAB] for row in markov.logits])
            return ids, hidden, logits

        eager = []
        for case in cases:
            for block, context in zip(blocks, contexts):
                stage_case(context, features, lambda context=context: extend(context), case, anchor, position)
                check_context(case[0], block, last)
            block = blocks[0]
            positions.prepare(block, anchor, position, embedding)
            expected, token = case[:2]
            assert torch.equal(ttnn.to_torch(block.embedded)[0, 0, 0], embeddings[token]), "anchor"
            for angles, actual in zip(reference._angles(expected.length, 32), (block.cos, block.sin)):
                assert torch.equal(ttnn.to_torch(actual).reshape(angles.shape), angles), "RoPE positions"
            for mask in block.mask:
                cap = mask.shape[-1] - 32
                bits = torch.isfinite(ttnn.to_torch(mask).flatten())
                per_position = torch.arange(mask.shape[-1])
                expected_mask = (per_position < min(expected.length, cap)) | (
                    (per_position >= cap) & (per_position < cap + block.rows)
                )
                assert torch.equal(bits, expected_mask), "draft attention mask"
            forward(blocks[0])
            ids, hidden, logits = read(blocks[0])
            forward(blocks[1])
            actual = read(blocks[1])
            delta = (actual[2].float() - logits.float()).norm() / logits.float().norm()
            print(f"{labels[1]}: {actual[0]} vs {ids}, relative L2 {delta:.5f}", flush=True)
            assert actual[0] == ids and delta < 0.01
            if compare_rope:
                assert torch.equal(actual[1], hidden) and torch.equal(actual[2], logits)
            hidden_error = (hidden.float() - case[3]).norm() / case[3].float().norm()
            expected_ids, expected_logits = case[4]
            logit_error = (logits.float() - expected_logits).norm() / expected_logits.float().norm()
            print(
                f"prefix {case[0].length}: hidden relative L2={hidden_error:.5f}, "
                f"draft IDs {ids} vs CPU {expected_ids}, logits relative L2={logit_error:.5f}",
                flush=True,
            )
            assert ids == expected_ids
            assert hidden_error < 0.03
            assert logit_error < 0.03
            eager.append(((ids, hidden, logits), actual))

        ttnn.synchronize_device(device)
        context_traces, traces = [], []
        for context, block in zip(contexts, blocks):
            trace = ttnn.begin_trace_capture(device, cq_id=0)
            extend(context)
            ttnn.end_trace_capture(device, trace, cq_id=0)
            context_traces.append(trace)
            trace = ttnn.begin_trace_capture(device, cq_id=0)
            forward(block)
            ttnn.end_trace_capture(device, trace, cq_id=0)
            traces.append(trace)
        try:
            ms = [[], []]
            for i in range(16):
                j = i % len(cases)
                for context, trace in zip(contexts, context_traces):
                    stage_case(
                        context,
                        features,
                        lambda trace=trace: ttnn.execute_trace(device, trace, cq_id=0, blocking=False),
                        cases[j],
                        anchor,
                        position,
                    )
                ttnn.synchronize_device(device)
                for k in (0, 1) if i % 2 == 0 else (1, 0):
                    start = time.perf_counter()
                    ttnn.execute_trace(device, traces[k], cq_id=0, blocking=True)
                    ms[k].append(1000 * (time.perf_counter() - start))
                    ids, hidden, logits = read(blocks[k])
                    assert ids == eager[j][k][0]
                    assert torch.equal(hidden, eager[j][k][1])
                    assert torch.equal(logits, eager[j][k][2])
            for label, timings in zip(labels, ms):
                p95 = statistics.quantiles(timings[4:], n=20, method="inclusive")[-1]
                print(
                    f"DSpark {count} drafts ({label}): median {statistics.median(timings[4:]):.2f} ms, "
                    f"p95 {p95:.2f} ms",
                    flush=True,
                )
        finally:
            for trace in traces:
                ttnn.release_trace(device, trace)
            for trace in context_traces:
                ttnn.release_trace(device, trace)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    comparisons = parser.add_mutually_exclusive_group()
    comparisons.add_argument("--compare-queries", action="store_true")
    comparisons.add_argument("--compare-scores", action="store_true")
    comparisons.add_argument("--compare-projections", action="store_true")
    comparisons.add_argument("--compare-rope", action="store_true")
    options = parser.parse_args()
    main(
        options.compare_queries,
        options.compare_scores,
        options.compare_projections,
        options.compare_rope,
    )
