import statistics
import time

import torch
import ttnn
from mtp_reference import DraftContext, DraftReference

from inkling_tt import paths
from inkling_tt.model import VOCAB, D, Weights
from inkling_tt.model2 import BF, FP, RM, TILE
from inkling_tt.mtp_markov import MarkovDraftHead, _cast
from inkling_tt.mtp_weights import PADDED_VOCAB, RANK, fold_head, load_drafter
from inkling_tt.prefill import _alloc
from serve.runtime import DEVICE_PARAMS


def main():
    count = 3
    weights = load_drafter(paths.snapshot())
    target = Weights(paths.hf_sd())
    head = target.get("lm_head.weight")
    fold, bias = fold_head(head, weights)
    reference = DraftReference(weights, head, fold, bias, count + 1)
    generator = torch.Generator().manual_seed(869)
    cases = []
    for length, token in ((29, 42), (128, 198)):
        features = torch.randn(length, 4, D, generator=generator, dtype=torch.bfloat16)
        anchor = target.get("embed_tokens.weight")[token].float()
        anchor = (
            anchor * torch.rsqrt(anchor.square().mean() + 1e-6) * target.get("embed_tokens.embed_norm.weight")
        ).bfloat16()
        ctx = DraftContext()
        reference.extend(ctx, features)
        hidden = reference.hidden(ctx, anchor)[1:]
        expected_ids, expected_logits = reference.score(hidden, ctx.last, token)
        base = torch.nn.functional.linear(hidden, head)
        base = torch.nn.functional.pad(base, (0, PADDED_VOCAB - VOCAB))
        fixed = torch.nn.functional.linear(
            torch.cat((hidden, ctx.last.expand_as(hidden)), -1).float(),
            reference.rnn_weight[:, 2 * RANK :],
            reference.rnn_bias,
        )
        cases.append((token, base, fixed, expected_ids, expected_logits))

    device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        device.enable_program_cache()
        values = torch.linspace(-1, 1, 1024).reshape(1, 1, 32, 32)
        source = ttnn.from_torch(values, dtype=FP, layout=TILE, device=device)
        rounded = _alloc(device, values.shape)
        _cast(source, rounded)
        assert torch.equal(ttnn.to_torch(rounded), values.bfloat16())
        draft = MarkovDraftHead(device, weights, fold, bias, count)
        anchor = _alloc(device, (1, 1), ttnn.uint32, RM)
        base = [_alloc(device, (1, 1, 1, PADDED_VOCAB), BF) for _ in range(count)]
        context = [_alloc(device, (1, 1, 1, 3 * RANK), FP) for _ in range(count)]

        def stage(case):
            token, rows, fixed, _, _ = case
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([[token]]), dtype=ttnn.uint32), anchor)
            for src, dst in zip([*rows, *fixed], [*base, *context]):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(src.reshape(1, 1, 1, -1), dtype=dst.dtype, layout=TILE), dst
                )

        def read():
            ids = [ttnn.to_torch(index).item() for index in draft.ids]
            logits = torch.stack([ttnn.to_torch(row).flatten() for row in draft.logits])
            assert torch.isneginf(logits[:, VOCAB:]).all()
            return ids, logits[:, :VOCAB]

        eager = []
        for i, case in enumerate(cases):
            stage(case)
            draft.score(base, context, anchor)
            ids, logits = read()
            expected_ids, expected_logits = case[-2:]
            relative_l2 = (logits.float() - expected_logits).norm() / expected_logits.float().norm()
            print(f"case {i}: IDs {ids} versus CPU {expected_ids}; relative L2={relative_l2:.5f}", flush=True)
            assert ids == expected_ids
            assert relative_l2 < 0.007
            eager.append((ids, logits))

        ttnn.synchronize_device(device)
        trace = ttnn.begin_trace_capture(device, cq_id=0)
        draft.score(base, context, anchor)
        ttnn.end_trace_capture(device, trace, cq_id=0)
        try:
            times = []
            for i in range(32):
                j = i % len(cases)
                stage(cases[j])
                ttnn.synchronize_device(device)
                start = time.perf_counter()
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                times.append((time.perf_counter() - start) * 1e3)
                ids, logits = read()
                assert ids == eager[j][0]
                assert torch.equal(logits, eager[j][1])
            warm = times[4:]
            print(
                f"Markov {count} drafts: median {statistics.median(warm):.2f} ms, "
                f"p95 {statistics.quantiles(warm, n=20)[-1]:.2f} ms",
                flush=True,
            )
        finally:
            ttnn.release_trace(device, trace)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
