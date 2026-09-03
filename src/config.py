"""Single place to choose the model. Change MODEL_ID and everything in src/ follows."""

# Local (Apple Silicon, 16GB) pipeline-testing model: dense, ~3.4GB bf16, no swapping.
# Qwen3-1.7B is the post-trained chat model (there is no separate "-Instruct" id at 1.7B);
# thinking is switched off via the chat template's enable_thinking flag in model.py.
MODEL_ID = "Qwen/Qwen3-1.7B"

# GPU options. Both are Qwen3_5ForConditionalGeneration hybrids (Gated DeltaNet + full
# attention) and need the `fla` and `causal_conv1d` CUDA kernels; see docs/pod_setup.md.
# MODEL_ID = "Qwen/Qwen3.5-9B"      # primary GPU model: 32 layers, hidden 4096, ~18GB bf16
# MODEL_ID = "Qwen/Qwen3.6-27B"     # replication on an 80GB card: 64 layers, hidden 5120, ~54GB bf16
