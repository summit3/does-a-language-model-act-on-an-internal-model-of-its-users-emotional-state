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

Set `MODEL_ID` in `src/config.py`, then `python -m src.model` to load and sanity-check a reply.
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
- **Use a passphrase-free key for the pod.** A passphrase-protected key fails non-interactively
  (server accepts the key, then "Permission denied") unless it is loaded into ssh-agent. A dedicated
  `~/.ssh/runpod_mats_ed25519` (no passphrase) is registered on the pod; the `~/.ssh/config` alias is
  `runpod-mats`. If the pod is ever recreated, pass that key's `.pub` as `PUBLIC_KEY`. To add a key to a
  running pod without SSH, run Python through JupyterLab's kernel API and append to
  `/root/.ssh/authorized_keys`.
- Runpod re-maps the external SSH port on every stop/start: re-read it from `get-pod` and update
  `~/.ssh/config`.
- **Copy the repo with rsync, not git** (the pod has no GitHub credentials). macOS ships openrsync, so
  use `--stats`, not `--info=...`:

  ```bash
  rsync -az --stats --exclude .venv --exclude models/ --exclude activations/ --exclude .git/ \
    --exclude __pycache__ --exclude .ipynb_checkpoints --exclude .DS_Store --exclude default_600k.md \
    ./ runpod-mats:/workspace/mats-user-models/
  ```
  The `chown ... Operation not permitted` warnings on `/workspace` are harmless.
- On the pod the venv is `/workspace/venv` (activate it first); the HF cache is `/workspace/hf_cache`
  (`export HF_HOME=/workspace/hf_cache`) so weights survive a container reset. Pull results back with
  `rsync -az runpod-mats:/workspace/mats-user-models/results/ results/`.
- JupyterLab: `https://<pod-id>-8888.proxy.runpod.net/?token=<JUPYTER_PASSWORD>`.
- **Stop the pod when not in use** (Runpod console or MCP `stop-pod`); a stopped pod bills only the
  volume. Terminate it at the end of the project.
