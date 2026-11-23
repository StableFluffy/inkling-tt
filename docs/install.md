# Install from source

A source checkout runs the tests, the benchmarks and the server. If you only want to serve the model, the container image is simpler: convert the checkpoint (step 3) and follow [serving.md](serving.md#container-image).

You need an x86-64 Linux host with one p150a, Docker, 1 GiB hugepages mounted at `/dev/hugepages-1G`, and about 100 GB of free disk: 20 GB for the official checkpoint, 53 GB for the converted shards, 15 GB for tt-metal and 12 GB for the dev image. Device and TTNN commands run inside the pinned tt-metal dev image through `scripts/dev_run.sh`; the converter and the reference captures run in a separate CPU Python environment.

## 0. Clone and set the paths

```bash
git clone https://github.com/StableFluffy/inkling-tt.git ~/inkling/inkling-tt
cd ~/inkling/inkling-tt

export TT_METAL_HOME=~/inkling/tt-metal
export TT_METAL_VENV=~/inkling/ttnn-venv
export INKLING_WORK=~/inkling/work
export INKLING_SNAPSHOT=$INKLING_WORK/Inkling-2-Mini
export INKLING_HF_SD=$INKLING_WORK/hf_sd
export INKLING_GOLDEN=$INKLING_WORK/golden
export INKLING_DEVICE=0
mkdir -p "$INKLING_WORK"
```

Keep these exported in the shell you work in. The wrappers in `scripts/` read them, and so do the converter and golden scripts, which run outside the wrappers. Paths must be absolute. [development.md](development.md#environment) lists every variable.

## 1. Build tt-metal

```bash
git clone https://github.com/tenstorrent/tt-metal.git "$TT_METAL_HOME"
git -C "$TT_METAL_HOME" checkout 79951bd44b7484a136b3619e0d81f26baa8a1936
git -C "$TT_METAL_HOME" submodule update --init --recursive
scripts/build_tt_metal.sh
```

The build takes about 6 minutes on 64 cores. Use a full clone with tags; without tags CMake stops with "No VERSION specified". The script mounts the tree at `/opt/tt-metal` inside the dev image and runs `./build_metal.sh --release --use-system-sfpi`.

## 2. Create the TTNN environment

```bash
scripts/setup_venv.sh
scripts/run_tests.sh host
```

`setup_venv.sh` builds `$TT_METAL_VENV` on top of the dev image's Python 3.10 environment, installs tt-metal as an editable `ttnn` package and adds the server dependencies. The `host` tier needs no device: it byte-compiles the sources and imports ttnn, the model and the server.

## 3. Download and convert the checkpoint

To skip the conversion, download the result instead and only fetch the tokenizer files from the official repository:

```bash
hf download maywell/Inkling-2-Mini-TT --include 'hf_sd/*' --local-dir "$INKLING_WORK"
hf download thinkingmachines/Inkling-2-Mini --revision b85c3cfa615fc413cc9af3379f491b963f3b586a \
  --exclude '*.safetensors' --local-dir "$INKLING_SNAPSHOT"
```

The golden scripts in step 4 still need the full official checkpoint, so download it without `--exclude` if you want to run the parity tests.

The converter needs a Transformers release that knows `inkling2_loop`, so it gets its own CPU environment:

```bash
REV=b85c3cfa615fc413cc9af3379f491b963f3b586a
uv venv --python 3.12 "$INKLING_WORK/cpu-venv"
CPU="$INKLING_WORK/cpu-venv/bin"
uv pip install --python "$CPU/python" --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
uv pip install --python "$CPU/python" transformers==5.24.0 safetensors==0.8.0 huggingface_hub

"$CPU/hf" download thinkingmachines/Inkling-2-Mini --revision $REV --local-dir "$INKLING_SNAPSHOT"
"$CPU/python" tools/convert/convert_checkpoint.py --src "$INKLING_SNAPSHOT" --out "$INKLING_HF_SD" --revision $REV
```

The converter loads the model in BF16 on the CPU, which dequantizes the checkpoint's NVFP4 and MXFP8 tensors, and writes one safetensors file per layer plus a manifest. It needs about 60 GB of RAM and takes a few minutes. Packing into device formats (BFP8_B experts, tiles) happens later, when the model loads.

Check the inputs and outputs against the reference hashes:

```bash
(cd "$INKLING_SNAPSHOT" && grep -v '^#' ~/inkling/inkling-tt/tools/convert/SHA256SUMS.source | sha256sum -c -)
(cd "$INKLING_HF_SD"    && grep -v '^#' ~/inkling/inkling-tt/tools/convert/SHA256SUMS.converted | sha256sum -c -)
```

The converted hashes hold for the versions above. Other torch or Transformers versions may produce different bytes that are still correct; run the parity tests in that case.

## 4. Generate the reference captures

The parity and evaluation tests compare against HF eager outputs that you generate locally:

```bash
"$CPU/python" tools/golden/capture.py   # capture_haiku.pt: per-layer captures and 8 greedy tokens
"$CPU/python" tools/golden/hf_eval.py   # eval3x128.pt: 3 prompts x 128 tokens with logits
"$CPU/python" tools/golden/hf_long.py   # long300.pt: a 300-token prompt
```

These are pickle files read with `torch.load`, so only use captures you made yourself. `hf_long.py` builds its prompt from the snapshot's README, which is why the revision is pinned.

## 5. Check, benchmark and serve

```bash
scripts/run_tests.sh parity
scripts/run_tests.sh eval
scripts/bench_decode.sh
PORT=8000 scripts/serve.sh
```

The first run compiles every kernel into `$INKLING_WORK/kcache`, which takes several minutes; later runs reuse it. [development.md](development.md#tests) describes all test tiers, and [serving.md](serving.md) the server.
