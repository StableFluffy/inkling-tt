from unittest.mock import patch

import torch
import ttnn

from inkling_tt import paths
from inkling_tt.model import VOCAB
from inkling_tt.model2 import InklingTT2
from inkling_tt.prefill import Prefill
from serve.runtime import DEVICE_PARAMS, default_precision


def read_taps(taps, valid):
    return tuple(ttnn.to_torch(t)[0, 0, :valid] for t in taps)


def reference(m, pf, tokens):
    m.reset()
    prefill_inputs, decode_inputs = [], []
    prefill_layer, readout = pf.layer, m._readout
    layers = [m.layers[i] for i in (5, 11, 17)]
    decode_layers = [layer._forward for layer in layers]
    counts = [0] * len(layers)

    def prefill_visit(layer, x, bank, visit, messages):
        if layer in layers:
            i = layers.index(layer)
            if counts[i] == i:
                prefill_inputs.append(ttnn.to_torch(x)[0, 0, : len(tokens)])
            counts[i] += 1
        return prefill_layer(layer, x, bank, visit, messages)

    def decode_visit(i, x, *args, **kwargs):
        if counts[i] == i:
            decode_inputs.append(ttnn.to_torch(x)[0, 0])
        counts[i] += 1
        return decode_layers[i](x, *args, **kwargs)

    def decode_readout(h):
        decode_inputs.append(ttnn.to_torch(h)[0, 0])
        return readout(h)

    with patch.object(pf, "taps", None), patch.object(pf, "layer", prefill_visit):
        plain_pf = pf.run_chunk(tokens, 0)
        prefill_inputs.append(ttnn.to_torch(pf.last)[0, 0])

    counts[:] = [0] * len(layers)
    tok, pos = int(plain_pf.argmax()), len(tokens)
    with (
        patch.object(m, "taps", None),
        patch.object(m, "_readout", decode_readout),
        patch.object(layers[0], "_forward", lambda x, *a, **kw: decode_visit(0, x, *a, **kw)),
        patch.object(layers[1], "_forward", lambda x, *a, **kw: decode_visit(1, x, *a, **kw)),
        patch.object(layers[2], "_forward", lambda x, *a, **kw: decode_visit(2, x, *a, **kw)),
    ):
        m.run(tok, pos)
        plain_decode = ttnn.to_torch(m.logits).flatten()[:VOCAB]
        plain_id = ttnn.to_torch(m.next_id).item()

    return plain_pf, plain_decode, plain_id, prefill_inputs, decode_inputs


def check(m, pf, tokens, expected):
    plain_pf, plain_decode, plain_id, prefill_inputs, decode_inputs = expected
    m.reset()
    assert torch.equal(pf.run_chunk(tokens, 0), plain_pf), "prefill logits"
    captured = read_taps(pf.taps, len(tokens))
    for tap, expected in zip(captured[:3], prefill_inputs[:3], strict=True):
        assert torch.equal(tap, expected), "prefill incoming partial"
    assert torch.equal(captured[3][-1:], prefill_inputs[-1]), "prefill final norm"
    assert captured[3].dtype == torch.bfloat16 and not torch.equal(captured[3][0], captured[3][-1])

    tok, pos = int(plain_pf.argmax()), len(tokens)
    assert m.step_greedy(tok, pos) == plain_id, "decode greedy ID"
    assert torch.equal(ttnn.to_torch(m.logits).flatten()[:VOCAB], plain_decode), "decode raw logits"
    for tap, expected in zip(read_taps(m.taps, 1), decode_inputs, strict=True):
        assert torch.equal(tap, expected), "decode tap"
    print(f"{len(tokens)} prompt tokens: prefill and decode taps and logits match", flush=True)


def main():
    device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        m = InklingTT2(device, paths.hf_sd(), default_precision(), capture_taps=True)
        pf = Prefill(m, 128)
        try:
            cap = torch.load(paths.golden("capture_haiku.pt"), weights_only=False)
            ids = cap["ids"][0].tolist()
            references = [(tokens, reference(m, pf, tokens)) for tokens in (ids, ids[:-1])]
            m.capture()
            for tokens, expected in references:
                check(m, pf, tokens, expected)
        finally:
            pf.close()
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
