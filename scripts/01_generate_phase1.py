"""Phase 1: greedy replies for all neutral/stressed prompt pairs.

Run on the pod from the repo root with the venv active (MATS_MODEL_ID set there):
    python scripts/01_generate_phase1.py
Writes results/phase1/phase1_replies.csv and results/phase1/phase1_meta.json.
"""
import csv, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.config import MODEL_ID, CONCISE_SYSTEM_PROMPT, ENABLE_THINKING
from src.model import load_model, chat

MAX_NEW_TOKENS = 150
pairs = list(csv.DictReader(open("data/phase1_pairs.csv", newline="", encoding="utf-8")))
assert len(pairs) == 30, len(pairs)

t0 = time.time(); model, tok = load_model(); load_s = time.time() - t0
chat(model, tok, "Say hi.", system=CONCISE_SYSTEM_PROMPT, max_new_tokens=5)   # warm-up, not timed

out = []; t_gen = time.time()
for r in pairs:
    for cond in ("neutral", "stressed"):
        prompt = r[cond]
        t = time.time()
        reply = chat(model, tok, prompt, system=CONCISE_SYSTEM_PROMPT, max_new_tokens=MAX_NEW_TOKENS)
        n = len(tok(reply)["input_ids"])
        out.append({"pair_id": r["pair_id"], "task_type": r["task_type"], "condition": cond,
                    "prompt": prompt, "reply": reply, "n_tokens": n})
        print(f"pair {r['pair_id']:>2} {cond:<8} {n:>3} tok {time.time()-t:4.1f}s", flush=True)
gen_s = time.time() - t_gen

Path("results/phase1").mkdir(parents=True, exist_ok=True)
with open("results/phase1/phase1_replies.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["pair_id", "task_type", "condition", "prompt", "reply", "n_tokens"], lineterminator="\n")
    w.writeheader(); w.writerows(out)
meta = {"model_id": MODEL_ID, "system_prompt": CONCISE_SYSTEM_PROMPT, "enable_thinking": ENABLE_THINKING,
        "decoding": "greedy", "max_new_tokens": MAX_NEW_TOKENS, "n_prompts": len(out),
        "load_seconds": round(load_s, 1), "generation_seconds": round(gen_s, 1),
        "device": str(model.device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pairs_file": "data/phase1_pairs.csv"}
json.dump(meta, open("results/phase1/phase1_meta.json", "w"), indent=2)
print(f"\ndone: {len(out)} replies in {gen_s:.0f}s (load {load_s:.0f}s) -> results/phase1/phase1_replies.csv, results/phase1/phase1_meta.json")
