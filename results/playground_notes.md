# Playground notes (untimed prep, 2026-09-04)

Local exploration on the laptop model, **not** part of the timed project. Signal only; see caveats.

## Setup
- Model: `Qwen/Qwen3-1.7B`, 28 layers, hidden 2048, bf16 on MPS. Thinking mode confirmed off
  (`python -m src.model --check-thinking`).
- Speed: first call after load ~190 s (cold start, MPS kernel warm-up), then ~8-9 tok/s.

## Residual stream
- Norms ramp ~15 -> ~2600 across layers with a dip at the final layer. Layer 18 norm ~250.
- Cosine similarity by layer, last prompt token:
  - same question, neutral vs stressed framing: 0.997 -> 0.80
  - different questions: 0.997 -> 0.49
  - the split opens around layers 10-15.
- **Anisotropy**: early-layer cosines near 1.0 for unrelated prompts mean the residual stream has a
  large shared component. Mean-centre (subtract the dataset mean per layer) before any cosine or PCA
  in the real project.

## Mean-difference "stress" direction
- Layer 18, mean(stressed) - mean(neutral) over 5 prompt pairs, unit-normalised.
- Cosine with a random unit vector: 0.018 (expected ~1/sqrt(2048) = 0.022). Sanity check only.

## Steering sweep
Prompt: "What's the capital of Australia?". Unit direction added at layers 12-23.

| N | Behaviour |
|---|---|
| 8 | Fact intact (Canberra), tone softer, emoji. |
| 15 | **Task abandoned.** "I can't help with that. I'm here to support you... You're not alone." No answer given. |
| 25-35 | Incoherent but still in the comfort register ("I'm here for you", "deep breath"). |
| 50+ | Degenerate repetition ("you're you're", "chill chill"). |
| random dir, matched N | Degenerate but unrelated ("except except"). |

## Lessons
- **Calibration**: Chen et al.'s N=8 and Empathic Machines' N=7 do not transfer; the right strength
  depends on the residual norm of the model and layer. Going forward, record N as a fraction of the
  mean residual norm at the target layers (`src/hooks.py:steer_generate_relative`), so the setting
  transfers from 1.7B to 9B. Measured mean last-token norm over layers 12-23 on the capital-of-Australia
  prompt is ~530 (the ~250 figure above is layer 18 alone), so the sweep in fractions of the band mean
  is: N=8 ~ 0.015, N=15 ~ 0.03, N=25-35 ~ 0.05-0.07, N=50 ~ 0.1. Note also that the layer-18 figure
  and `src/hooks.py` layer indexing (index i = output of block i, 0-based) should be reconciled by hand
  before the real runs: on the compound-interest prompt, block-16 output is ~258 and block-18 is ~461.
- **Design**: 80-token replies on maths questions were cut off before the answer, so correctness was
  not measurable. Phase 1 needs a concise-answer system prompt (applied identically in every
  condition) and/or a larger token budget.
- **Metric**: at moderate strength the model dropped the factual task for emotional support.
  Task-abandonment / refusal rate must be a primary Phase 1 metric, not an afterthought.

## Caveats
5 in-sample pairs; explicit stress words (no implied-emotion or third-party control); one steering
prompt; one random seed; tiny model; no unrelated-learned-direction control (Fomin 2026). None of
this is a result.
