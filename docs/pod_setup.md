# GPU pod setup

## Model choice

There is **no Qwen 3.6 at 4B or 9B**. As of 2026-09-03 the only Qwen-authored 3.6 checkpoints are
`Qwen/Qwen3.6-27B` (+FP8) and `Qwen/Qwen3.6-35B-A3B` (+FP8). Use:

| Where | `MODEL_ID` (src/config.py) | Notes |
|---|---|---|
| Laptop (MPS, 16GB) | `Qwen/Qwen3-4B-Instruct-2507` | dense, standard attention; pipeline testing only |
| Pod, default | `Qwen/Qwen3.5-9B` | hybrid, 32 layers, hidden 4096, ~18GB bf16 |
| Pod, if GPU allows | `Qwen/Qwen3.6-27B` | hybrid, 64 layers, hidden 5120, ~54GB bf16 (80GB card) |

Both `Qwen/Qwen3.5-*` and `Qwen/Qwen3.6-*` are `Qwen3_5ForConditionalGeneration` hybrids:
3 of every 4 layers are Gated DeltaNet linear attention. Decoder blocks live at
`model.model.language_model.layers`; `src/model.py:get_decoder_layers` finds them automatically.

## Kernels the hybrid models need

Without these, transformers silently falls back to a pure-PyTorch chunked/recurrent implementation
that is ~3 s/token on MPS and still slow on CUDA. Install on the pod **before** loading the model:

```bash
pip install flash-linear-attention   # provides the `fla` package (chunk_gated_delta_rule etc.)
pip install causal-conv1d             # fused causal conv used by the DeltaNet layers
python -c "import fla, causal_conv1d; print('kernels ok')"
```

`flash-linear-attention` needs a recent Triton (bundled with the CUDA torch wheel). If the
`causal-conv1d` wheel fails to build, `pip install causal-conv1d --no-build-isolation` after
`pip install ninja packaging`. A quick timing check after install: a 60-token greedy reply from
Qwen3.5-9B should take a few seconds on an A100, not minutes.

## Everything else

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch here is the CPU/MPS wheel; on the pod install the CUDA wheel first
export HF_TOKEN=...                    # optional; faster, rate-limit-free downloads
```

Set `MATS_MODEL_ID` (env or `.env`), then `python -m src.model` to load and sanity-check a reply.
Weights cache in `~/.cache/huggingface/hub`; nothing under `models/` or `activations/` is committed.

## Thinking mode (applies to Qwen3.5-9B and Qwen3.6-27B on the pod too)

Qwen3, Qwen3.5 and Qwen3.6 chat templates all carry an `enable_thinking` switch. With it on, the
reply starts with a `<think>...</think>` block of chain-of-thought, which would (a) change what the
"reply" is for every behavioural metric and (b) put hundreds of extra tokens before the answer.
`src/config.py:ENABLE_THINKING = False` is passed to `apply_chat_template` by `chat()`,
`steer_generate()` and `get_residual_activations()`; the template then emits an empty
`<think>\n\n</think>\n\n` at the end of the *prompt* so the model answers directly.

After loading a new model on the pod, re-run the check (prints raw decoded output, must contain no
`<think>` and must not start with a think token):

```bash
python -m src.model --check-thinking
```

## Connecting from the laptop (what worked on 2026-09-05)

- Pod created via the Runpod MCP: A40 48GB, secure cloud on-demand ($0.49/hr), template
  `runpod-torch-v280`, 30GB container disk, 60GB volume at `/workspace`, ports `22/tcp,8888/http`,
  env `PUBLIC_KEY=<ssh public key>` and `JUPYTER_PASSWORD=<token>` (Jupyter only starts if that is set).
- **`PUBLIC_KEY` env overrides account keys.** With `PUBLIC_KEY` set on the pod, the template's
  start script installs only that key and ignores keys registered on the Runpod account (verified
  2026-09-05: an account key added while the pod existed was not injected on restart). So either put
  every key you need in `PUBLIC_KEY` (newline-separated) at creation, or create the pod without
  `PUBLIC_KEY` and rely on account keys.
- **Use a passphrase-free key for the pod.** A passphrase-protected key fails non-interactively
  (server accepts the key, then "Permission denied") unless it is loaded into ssh-agent. A dedicated
  `~/.ssh/runpod_mats_ed25519` (no passphrase) is registered on the pod; the `~/.ssh/config` alias is
  `runpod-mats`. If the pod is ever recreated, pass that key's `.pub` as `PUBLIC_KEY`. To add a key to a
  running pod without SSH, run Python through JupyterLab's kernel API and append to
  `/root/.ssh/authorized_keys`.
- Runpod re-maps the external SSH port on every stop/start: re-read it from `get-pod` and update
  `~/.ssh/config`.
- **Copy the repo with rsync, not git** (the pod has no GitHub credentials). macOS ships openrsync, so
  use `--stats`, not `--info=...`. Always pull `results/` back **first**, then push with `--delete` so
  files removed from the repo (e.g. the untracked third-party docs formerly in `docs/`) are removed
  on the pod too. Excluded paths (`.venv`, `models/`, `activations/`, ...) are protected from
  deletion by rsync, and `results/` is protected explicitly so pod-generated outputs are never lost:

  ```bash
  # 1. pull results back (never overwrite the local README)
  rsync -az --stats --exclude README.md runpod-mats:/workspace/mats-user-models/results/ results/
  # 2. push the repo, deleting anything on the pod that no longer exists locally
  rsync -az --stats --delete --filter='P results/' \
    --exclude .venv --exclude models/ --exclude activations/ --exclude .git/ --exclude __pycache__ \
    --exclude .ipynb_checkpoints --exclude .DS_Store --exclude default_600k.md --exclude docs/private/ \
    ./ runpod-mats:/workspace/mats-user-models/
  ```
  The `chown ... Operation not permitted` warnings on `/workspace` are harmless (rsync exits 23; the
  files still transfer).
- On the pod the venv is `/workspace/venv` (activate it first). Its `bin/activate` exports
  `MATS_MODEL_ID=Qwen/Qwen3.5-9B` (read by `src/config.py`, laptop default is Qwen3-1.7B) and
  `HF_HOME=/workspace/hf_cache` so weights survive a container reset. `src/config.py` is identical on
  both machines; never edit it per machine. Pull results back with
  `rsync -az runpod-mats:/workspace/mats-user-models/results/ results/`.
- JupyterLab: `https://<pod-id>-8888.proxy.runpod.net/?token=<JUPYTER_PASSWORD>`.
- **If the pod's host is fully allocated on restart**, Runpod cannot resume it; use the console's
  pod migration to move the volume to a new host. After migrating (2026-09-06) the HF cache and repo
  copy were intact but `/workspace/venv` came back truncated (320 MB, torch importing as an empty
  module). Check the imports first and rebuild only the venv if so (`pip` steps 2-3 of the checklist,
  then re-append the `MATS_MODEL_ID` / `HF_HOME` exports to `bin/activate`).
- **Stop the pod when not in use** (Runpod console or MCP `stop-pod`); a stopped pod bills only the
  volume. Terminate it at the end of the project.
