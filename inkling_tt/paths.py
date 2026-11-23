"""Filesystem locations, configured through environment variables.

INKLING_WORK      work root; tools that take --root expect hf_sd/, golden/ and Inkling-2-Mini/ below it
INKLING_HF_SD     converted per-layer BF16 shards (tools/convert/convert_checkpoint.py output)
INKLING_SNAPSHOT  Hugging Face snapshot of thinkingmachines/Inkling-2-Mini (tokenizer, chat template, config)
INKLING_GOLDEN    HF CPU reference captures (tools/golden/*.py output)
INKLING_EXPERTS   expert decode path: "stream" (inkling_tt.experts_stream) or "stock" (indexed sparse_matmul)
"""

import os


def _env(name, default):
    return os.environ.get(name, default)


def work():
    return _env("INKLING_WORK", ".")


def hf_sd():
    return _env("INKLING_HF_SD", os.path.join(work(), "hf_sd"))


def snapshot():
    return _env("INKLING_SNAPSHOT", os.path.join(work(), "Inkling-2-Mini"))


def golden(name=""):
    return os.path.join(_env("INKLING_GOLDEN", os.path.join(work(), "golden")), name)


def experts(default="stream"):
    return _env("INKLING_EXPERTS", os.environ.get("INK_EXPERTS", default))
