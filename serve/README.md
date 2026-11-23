# serve

- `openai_server.py`: OpenAI-compatible HTTP server.
- `runtime.py`: the model plus its sequence state, prefill buckets and decode loop.
- `vllm_inkling/`: experimental vLLM TT-plugin adapter.
- `Dockerfile`, `build_image.sh`: the release image.

Usage is in [docs/serving.md](../docs/serving.md); building the image is in [docs/development.md](../docs/development.md#release-image).
