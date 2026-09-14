# Does the user model change behaviour on unrelated tasks?

**Write-up:** the final report is [docs/writeup.pdf](docs/writeup.pdf) ("How Would You Rate This Human?", 21 pages).

Interpretability project. Probes a Qwen chat model's residual stream for the *user's* emotional state, then tests whether that representation changes task behaviour (accuracy, hedging, refusals, sycophancy) on emotion-unrelated prompts, with steering and matched-norm control directions. See `docs/project_brief.md`.

Run: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`, then `jupyter lab` and open `notebooks/`. Reusable code lives in `src/` (`config.py` holds the single `MODEL_ID`; `model.py` loads the model and chats; `hooks.py` extracts residual activations and steers). GPU notes in `docs/pod_setup.md`. Generated prompts/labels go in `data/`, outputs in `results/`.
