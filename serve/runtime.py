import time

import torch
import ttnn

from inkling_tt import paths
from inkling_tt.device import L1_SMALL_SIZE, TRACE_REGION_SIZE
from inkling_tt.model import VOCAB, Precision
from inkling_tt.model2 import InklingTT2

DEVICE_PARAMS = {"l1_small_size": L1_SMALL_SIZE, "trace_region_size": TRACE_REGION_SIZE}


def default_precision():
    prec = Precision(
        expert_dtype=ttnn.bfloat8_b,
        dense_dtype=ttnn.bfloat16,
        expert_fidelity=ttnn.MathFidelity.HiFi4,
        dense_fidelity=ttnn.MathFidelity.HiFi4,
    )
    prec.experts = paths.experts("stream")
    return prec


class InklingRuntime:
    def __init__(self, device, weights, lmax=32768, chunk=2048, log=print, *, device_sampling=False, mtp=0):
        if lmax % 32 or chunk % 32:
            raise ValueError(f"lmax and chunk must be multiples of 32, got {lmax}, {chunk}")
        self.dev = device
        self.lmax = lmax
        self.vocab = VOCAB
        self.fast_prefill = True
        self.chunk = chunk
        self.pf = None
        self.mtp = mtp
        self.speculative = None
        self.buckets = []
        t0 = time.time()
        self.m = InklingTT2(device, weights, default_precision(), lmax=lmax, log=lambda s: None, capture_taps=bool(mtp))
        self.sampler = None
        if device_sampling and hasattr(self.m, "head_logits"):
            from inkling_tt.sampling import DecodeSampler

            self.sampler = DecodeSampler(self.m.head_logits)
        log(f"inkling: model loaded in {time.time() - t0:.1f} s")
        self._zeros = {}
        self.pos = None

    @property
    def traced(self):
        return self.m.trace_id is not None

    def capture(self, log=print):
        if self.traced:
            return
        self._prepare_prefill(log)
        t0 = time.time()
        if self.mtp:
            from inkling_tt.mtp_round import SpeculativeRound

            self.speculative = SpeculativeRound(
                self.m, self.buckets, paths.snapshot(), proposals=self.mtp, adaptive=self.mtp <= 3
            )
            for bucket in self.buckets:
                bucket._warm()
            self.speculative.warm()
        if self.sampler is not None:
            self.sampler.warm()
        self.m.capture()
        if self.speculative is not None:
            self.speculative.capture()
        if self.sampler is not None:
            self.sampler.capture()
        self.pos = None
        log(f"inkling: trace captured in {time.time() - t0:.1f} s")

    def _prepare_prefill(self, log):
        if not self.chunk:
            return
        from inkling_tt.prefill import Prefill

        t0 = time.time()
        # With MTP, warm the buckets after allocating the draft and verifier workspaces.
        self.buckets = [
            Prefill(
                self.m,
                c,
                warmup=not self.mtp,
                global_score_tiles=56 if self.mtp else None,
                weight_buffers=1 if self.mtp and c >= 512 else 2,
            )
            for c in sorted({128, self.chunk})
        ]
        self.pf = self.buckets[0]
        self.reset()
        log(f"inkling: chunked prefill tables and programs prepared in {time.time() - t0:.1f} s")

    def release(self):
        if self.speculative is not None:
            self.speculative.close()
            self.speculative = None
        if self.sampler is not None:
            self.sampler.release()
        if self.traced:
            ttnn.release_trace(self.dev, self.m.trace_id)
            self.m.trace_id = None

    def reset(self):
        # KV slots beyond the sequence are masked out of attention, so only the conv histories need clearing.
        for v in self.m.visits.values():
            for t in v.conv:
                key = (tuple(t.shape), t.dtype)
                if key not in self._zeros:
                    self._zeros[key] = ttnn.from_torch(torch.zeros(key[0]), dtype=t.dtype, layout=ttnn.TILE_LAYOUT)
                ttnn.copy_host_to_device_tensor(self._zeros[key], t)
        self.pos = 0

    def _advance(self, tok, pos):
        m = self.m
        if not self.traced or not self.fast_prefill:
            m.step(tok, pos)
            return
        m.st.write(m.embed_row(tok), pos)
        ttnn.execute_trace(m.c.dev, m.trace_id, cq_id=0, blocking=False)

    def feed(self, ids, start, cancelled=None, *, draft_context=False):
        end = start + len(ids)
        if end > self.lmax:
            raise ValueError(f"sequence length {end} exceeds lmax {self.lmax}")
        if start == 0 and self.pos != 0:
            self.reset()
        elif self.pos != start:
            raise RuntimeError(f"sequence state is at position {self.pos}, asked to continue from {start}")
        self.pos = None
        if self.pf is not None and self.chunk and start % 32 == 0:
            logits = self._chunked(ids, start, cancelled, draft_context)
        else:
            logits = self._tokenwise(ids, start, cancelled)
        if logits is not None:
            self.pos = end
        return logits

    def _chunked(self, ids, start, cancelled, draft_context):
        on_chunk = self.speculative.append_prompt if draft_context else None
        if start == 0 and len(self.buckets) > 1:
            from inkling_tt.prefill import prefill_buckets

            return prefill_buckets(self.buckets, ids, on_chunk=on_chunk)[0]
        logits = None
        for i in range(0, len(ids), self.chunk):
            if cancelled is not None and cancelled.is_set():
                return None
            tokens = ids[i : i + self.chunk]
            bucket = next(p for p in self.buckets if p.chunk >= len(tokens))
            logits = bucket.run_chunk(tokens, start + i)
            if on_chunk is not None:
                on_chunk(bucket)
        return logits

    def _tokenwise(self, ids, start, cancelled):
        end = start + len(ids)
        for p in range(start, end - 1):
            if cancelled is not None and cancelled.is_set():
                return None
            self._advance(ids[p - start], p)
        return self.m.step(ids[-1], end - 1)

    def speculate(self, tok, pos, rows):
        if self.pos != pos:
            raise RuntimeError(f"sequence state is at position {self.pos}, asked to decode position {pos}")
        self.pos = None
        emitted, self.pos = self.speculative.run(tok, pos, rows)
        return emitted

    def _begin_step(self, pos):
        if self.pos != pos:
            raise RuntimeError(f"sequence state is at position {self.pos}, asked to decode position {pos}")
        if pos >= self.lmax:
            raise ValueError(f"position {pos} exceeds lmax {self.lmax}")
        self.pos = None

    def step_greedy(self, tok, pos):
        """Decode one token with the on-device argmax; returns the next token id."""
        self._begin_step(pos)
        nxt = self.m.step_greedy(tok, pos)
        self.pos = pos + 1
        return nxt

    def step(self, tok, pos):
        self._begin_step(pos)
        logits = self.m.step(tok, pos)
        self.pos = pos + 1
        return logits

    def step_sampled(self, tok, pos, temperature, top_k, top_p, presence, frequency, draw):
        self._begin_step(pos)
        sampler = self.sampler
        sampler.stage(tok, max(0.0, temperature), top_k, top_p, presence, frequency, draw)
        self.m.run(tok, pos)
        next_token = sampler.sample()
        self.pos = pos + 1
        return next_token
