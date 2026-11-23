import argparse
import asyncio
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from inkling_tt import paths

log = logging.getLogger("inkling.serve")

EOS = 200006
MESSAGE_MODEL = 200001
CONTENT_TEXT = 200004
CONTENT_THINKING = 200008
END_MESSAGE = 200010


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None
    name: str | None = None
    reasoning_content: str | None = None


class SamplingFields(BaseModel):
    model: str | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    seed: int | None = None
    stream: bool = False
    n: int = 1
    ignore_eos: bool = False
    mtp: bool = True


class ChatRequest(SamplingFields):
    messages: list[ChatMessage]
    reasoning_effort: str | float | None = None
    include_reasoning: bool = True


class CompletionRequest(SamplingFields):
    prompt: str | list[int]
    echo: bool = False


@dataclass
class Sampling:
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    frequency_penalty: float
    stop: list[str]
    stop_ids: set[int]
    seed: int | None
    mtp: bool = True


@dataclass
class Timings:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    reset_s: float = 0.0
    finish_reason: str = "length"

    def as_dict(self):
        ttft = self.reset_s + self.prefill_s
        dec = self.completion_tokens - 1
        return {
            "reset_ms": round(self.reset_s * 1000, 1),
            "prefill_ms": round(self.prefill_s * 1000, 1),
            "prefill_tok_s": round(self.prompt_tokens / self.prefill_s, 2) if self.prefill_s else None,
            "ttft_ms": round(ttft * 1000, 1),
            "decode_ms": round(self.decode_s * 1000, 1),
            "decode_tok_s": round(dec / self.decode_s, 2) if dec > 0 and self.decode_s else None,
            "ms_per_token": round(self.decode_s * 1000 / dec, 1) if dec > 0 else None,
        }


@dataclass
class Job:
    ids: list[int]
    sampling: Sampling
    out: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    cancelled: threading.Event = field(default_factory=threading.Event)
    timings: Timings = field(default_factory=Timings)

    def emit(self, item):
        self.loop.call_soon_threadsafe(self.out.put_nowait, item)


class Engine:
    def __init__(self, args):
        import ttnn

        from inkling_tt.device import open_device
        from serve.runtime import InklingRuntime

        mtp = getattr(args, "mtp", 0)
        if mtp and (args.no_trace or not args.prefill_chunk):
            raise ValueError("--mtp requires trace capture and prefill")
        self.ttnn = ttnn
        self.lmax = args.lmax
        self.dev = open_device(args.device_id, large_dispatch_ring=not mtp)
        self.device_sampling = not args.no_trace and os.getenv("INK_DEVICE_SAMPLING") != "0"
        self.rt = InklingRuntime(
            self.dev,
            args.weights,
            lmax=args.lmax,
            chunk=args.prefill_chunk,
            log=log.info,
            device_sampling=self.device_sampling,
            mtp=mtp,
        )
        self.rt.fast_prefill = not args.no_fast_prefill
        if not args.no_trace:
            self.rt.capture(log=log.info)
        self.queue: asyncio.Queue[Job] | None = None
        self.busy = False
        self.served = 0

    def close(self):
        self.rt.release()
        self.ttnn.close_device(self.dev)

    def run(self, job: Job):
        s, t = job.sampling, job.timings
        ids = job.ids
        t.prompt_tokens = len(ids)
        gen = torch.Generator()
        if s.seed is None:
            gen.seed()
        else:
            gen.manual_seed(s.seed)
        greedy = s.temperature <= 0.0 and not (s.presence_penalty or s.frequency_penalty)
        score_bound = 30 + abs(s.presence_penalty) + s.max_tokens * abs(s.frequency_penalty)
        device_sampling = (
            self.device_sampling
            and self.rt.sampler is not None
            and self.rt.traced
            and not greedy
            and math.isfinite(s.temperature)
            and math.isfinite(s.top_p)
            and score_bound < 3e38 * min(s.temperature if s.temperature > 0.0 else 1.0, 1.0)
        )
        counts: dict[int, int] = {}
        t0 = time.perf_counter()
        self.rt.reset()
        if device_sampling and (s.presence_penalty or s.frequency_penalty):
            self.rt.sampler.reset()
        t1 = time.perf_counter()
        t.reset_s = t1 - t0
        spec = self.rt.speculative if greedy and s.mtp else None
        if spec is not None:
            spec.start_request()
            if spec.pick(len(ids), s.max_tokens - 1) == 1:
                spec = None
        logits = self.rt.feed(ids, 0, job.cancelled, draft_context=spec is not None)
        t0 = time.perf_counter()
        t.prefill_s = t0 - t1
        if logits is None:
            t.finish_reason = "cancelled"
            return
        pos = len(ids)
        pending = deque()
        for k in range(s.max_tokens):
            tok = pending.popleft() if pending else sample(logits, s, counts, gen)
            counts[tok] = counts.get(tok, 0) + 1
            t.completion_tokens += 1
            if tok in s.stop_ids:
                t.finish_reason = "stop"
                break
            job.emit(("token", tok))
            if job.cancelled.is_set():
                t.finish_reason = "cancelled"
                break
            if k == s.max_tokens - 1 or pos >= self.lmax:
                t.finish_reason = "length"
                break
            if greedy:
                if not pending:
                    rows = spec.pick(pos, s.max_tokens - k - 1) if spec is not None else 1
                    if rows > 1:
                        pending.extend(self.rt.speculate(tok, pos, rows))
                    else:
                        pending.append(self.rt.step_greedy(tok, pos))
            elif device_sampling:
                draw = torch.rand(1, generator=gen).item() if s.temperature > 0.0 else 0.0
                pending.append(
                    self.rt.step_sampled(
                        tok, pos, s.temperature, s.top_k, s.top_p, s.presence_penalty, s.frequency_penalty, draw
                    )
                )
            else:
                logits = self.rt.step(tok, pos)
            pos += 1
        t.decode_s = time.perf_counter() - t0

    async def worker(self):
        while True:
            job = await self.queue.get()
            self.busy = True
            try:
                await asyncio.to_thread(self.run, job)
                job.emit(("done", None))
            except Exception as e:
                log.exception("generation failed")
                job.emit(("error", repr(e)))
            finally:
                self.busy = False
                self.served += 1
                d = job.timings
                log.info(
                    "req prompt=%d gen=%d finish=%s %s",
                    d.prompt_tokens,
                    d.completion_tokens,
                    d.finish_reason,
                    json.dumps(d.as_dict()),
                )


def sample(logits, s: Sampling, counts, gen):
    x = logits
    if counts and (s.presence_penalty or s.frequency_penalty):
        idx = torch.tensor(list(counts.keys()))
        cnt = torch.tensor(list(counts.values()), dtype=x.dtype)
        x = x.clone()
        x[idx] -= s.presence_penalty + s.frequency_penalty * cnt
    if s.temperature <= 0.0:
        return int(x.argmax())
    x = x / s.temperature
    idx = None
    if s.top_k > 0:
        x, idx = torch.topk(x, min(s.top_k, x.numel()))
    probs = torch.softmax(x, -1)
    if s.top_p < 1.0:
        if idx is None:
            head, idx = torch.topk(probs, min(1024, probs.numel()))
            if head.sum() < s.top_p:
                head, idx = torch.sort(probs, descending=True)
            probs = head
        keep = probs.cumsum(-1) - probs < s.top_p
        probs, idx = probs[keep], idx[keep]
    cdf = probs.cumsum(-1)
    j = min(int(torch.searchsorted(cdf, torch.rand(1, generator=gen) * cdf[-1])), cdf.numel() - 1)
    return j if idx is None else int(idx[j])


class Detok:
    def __init__(self, tok, split_reasoning):
        self.tok = tok
        self.split = split_reasoning
        self.mode = "content"
        self.seg: list[int] = []
        self.sent = 0

    def _flush(self, final):
        s = self.tok.decode(self.seg, skip_special_tokens=False)
        if not final and s.endswith("\ufffd"):
            return []
        d = s[self.sent :]
        self.sent = len(s)
        return [(self.mode, d)] if d else []

    def push(self, t):
        if self.split and t in (MESSAGE_MODEL, CONTENT_TEXT, CONTENT_THINKING, END_MESSAGE):
            out = self._flush(True)
            self.seg, self.sent = [], 0
            if t == CONTENT_THINKING:
                self.mode = "reasoning"
            elif t in (CONTENT_TEXT, END_MESSAGE):
                self.mode = "content"
            return out
        self.seg.append(t)
        return self._flush(False)

    def finish(self):
        return self._flush(True)


def find_stop(text, stops):
    hit = [text.find(x) for x in stops if x and x in text]
    return min(hit) if hit else -1


def make_sampling(req: SamplingFields, n_prompt, lmax):
    if req.n != 1:
        raise ValueError("only n=1 is supported")
    room = lmax - n_prompt
    if room <= 0:
        raise ValueError(f"prompt has {n_prompt} tokens; this build supports at most {lmax - 1}")
    mt = req.max_completion_tokens or req.max_tokens or room
    stop = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
    stop_ids = set(req.stop_token_ids or [])
    if not req.ignore_eos:
        stop_ids.add(EOS)
    return Sampling(
        max_tokens=max(1, min(mt, room)),
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        stop=stop,
        stop_ids=stop_ids,
        seed=req.seed,
        mtp=req.mtp,
    )


def build_app(engine: Engine | None, tok, model_name, lmax):
    @asynccontextmanager
    async def lifespan(app):
        if engine is not None:
            engine.queue = asyncio.Queue()
            task = asyncio.create_task(engine.worker())
        yield
        if engine is not None:
            task.cancel()
            engine.close()

    app = FastAPI(lifespan=lifespan)

    def err(code, msg):
        return JSONResponse(
            {"error": {"message": msg, "type": "invalid_request_error", "code": code}}, status_code=code
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "busy": engine.busy, "queued": engine.queue.qsize(), "served": engine.served}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "tt-inkling", "max_model_len": lmax}],
        }

    async def submit(ids, sampling):
        job = Job(ids=ids, sampling=sampling, out=asyncio.Queue(), loop=asyncio.get_running_loop())
        await engine.queue.put(job)
        return job

    async def events(job: Job, detok: Detok, request: Request):
        stops = job.sampling.stop
        hold = max((len(x) for x in stops), default=1) - 1
        buf, sent = "", 0
        try:
            while True:
                if await request.is_disconnected():
                    job.cancelled.set()
                kind, val = await job.out.get()
                if kind == "error":
                    raise RuntimeError(val)
                deltas = detok.finish() if kind == "done" else detok.push(val)
                for ch, d in deltas:
                    if ch != "content" or not stops:
                        yield ch, d
                        continue
                    buf += d
                    cut = find_stop(buf, stops)
                    if cut >= 0:
                        if cut > sent:
                            yield "content", buf[sent:cut]
                        job.cancelled.set()
                        while (await job.out.get())[0] not in ("done", "error"):
                            pass
                        job.timings.finish_reason = "stop"
                        yield "finish", "stop"
                        return
                    safe = len(buf) if kind == "done" else max(sent, len(buf) - hold)
                    if safe > sent:
                        yield "content", buf[sent:safe]
                        sent = safe
                if kind == "done":
                    break
            yield "finish", job.timings.finish_reason
        finally:
            job.cancelled.set()

    def usage(job):
        t = job.timings
        return {
            "prompt_tokens": t.prompt_tokens,
            "completion_tokens": t.completion_tokens,
            "total_tokens": t.prompt_tokens + t.completion_tokens,
        }

    def sse(obj):
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def envelope(rid, obj, created, choices, job=None):
        body = {"id": rid, "object": obj, "created": created, "model": model_name, "choices": choices}
        if job is not None:
            body["usage"] = usage(job)
            body["timings"] = job.timings.as_dict()
        return body

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, request: Request):
        msgs = [m.model_dump(exclude_none=True) for m in req.messages]
        try:
            ids = list(
                tok.apply_chat_template(
                    msgs,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    reasoning_effort=req.reasoning_effort,
                )["input_ids"]
            )
            sampling = make_sampling(req, len(ids), lmax)
        except Exception as e:
            return err(400, str(e))
        job = await submit(ids, sampling)
        rid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        detok = Detok(tok, split_reasoning=True)
        obj = "chat.completion.chunk"

        if req.stream:

            async def gen():
                first = {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
                yield sse(envelope(rid, obj, created, [first]))
                try:
                    async for ch, d in events(job, detok, request):
                        if ch == "finish":
                            yield sse(envelope(rid, obj, created, [{"index": 0, "delta": {}, "finish_reason": d}]))
                            break
                        if ch == "reasoning" and not req.include_reasoning:
                            continue
                        key = "reasoning_content" if ch == "reasoning" else "content"
                        yield sse(envelope(rid, obj, created, [{"index": 0, "delta": {key: d}, "finish_reason": None}]))
                    yield sse(envelope(rid, obj, created, [], job))
                except RuntimeError as e:
                    yield sse({"error": {"message": str(e)}})
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        parts = {"content": [], "reasoning": []}
        finish = "length"
        try:
            async for ch, d in events(job, detok, request):
                if ch == "finish":
                    finish = d
                else:
                    parts[ch].append(d)
        except RuntimeError as e:
            return err(500, str(e))
        msg = {"role": "assistant", "content": "".join(parts["content"]), "reasoning_content": None}
        if req.include_reasoning:
            msg["reasoning_content"] = "".join(parts["reasoning"]) or None
        choice = {"index": 0, "message": msg, "finish_reason": finish, "logprobs": None}
        return envelope(rid, "chat.completion", created, [choice], job)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        if isinstance(req.prompt, str):
            ids = tok.encode(req.prompt, add_special_tokens=False)
            prompt_text = req.prompt
        else:
            ids, prompt_text = list(req.prompt), tok.decode(req.prompt, skip_special_tokens=False)
        if not ids:
            return err(400, "empty prompt")
        try:
            sampling = make_sampling(req, len(ids), lmax)
        except Exception as e:
            return err(400, str(e))
        job = await submit(ids, sampling)
        rid, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())
        detok = Detok(tok, split_reasoning=False)
        obj = "text_completion"

        def choice(text, finish):
            return [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}]

        if req.stream:

            async def gen():
                try:
                    if req.echo:
                        yield sse(envelope(rid, obj, created, choice(prompt_text, None)))
                    async for ch, d in events(job, detok, request):
                        if ch == "finish":
                            yield sse(envelope(rid, obj, created, choice("", d)))
                            break
                        yield sse(envelope(rid, obj, created, choice(d, None)))
                    yield sse(envelope(rid, obj, created, [], job))
                except RuntimeError as e:
                    yield sse({"error": {"message": str(e)}})
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        text, finish = [prompt_text] if req.echo else [], "length"
        try:
            async for ch, d in events(job, detok, request):
                if ch == "finish":
                    finish = d
                else:
                    text.append(d)
        except RuntimeError as e:
            return err(500, str(e))
        return envelope(rid, obj, created, choice("".join(text), finish), job)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=paths.hf_sd())
    ap.add_argument("--tokenizer", default=paths.snapshot())
    ap.add_argument("--served-model-name", default="Inkling-2-Mini-TT")
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address. The API has no authentication; bind beyond loopback "
        "(e.g. 0.0.0.0) only on a trusted network or behind an authenticating proxy",
    )
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--lmax", type=int, default=32768)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument(
        "--mtp",
        nargs="?",
        const=3,
        default=0,
        type=int,
        choices=range(1, 8),
        help="enable greedy DSpark with up to this many proposals (3 when omitted)",
    )
    ap.add_argument("--no-fast-prefill", action="store_true")
    ap.add_argument("--prefill-chunk", type=int)
    args = ap.parse_args()
    if args.prefill_chunk is None:
        args.prefill_chunk = 2048
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    engine = Engine(args)
    app = build_app(engine, tok, args.served_model_name, args.lmax)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
