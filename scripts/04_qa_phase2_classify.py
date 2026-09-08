"""Blind label-consistency check for data/phase2_prompts.csv: the pod model classifies each row's
text as neutral / distressed / frustrated WITHOUT seeing the label. NOTE: the judge here is the
subject model (Qwen3.5-9B); this is a consistency check, not an independent ground truth.
Run on the pod: python scripts/04_qa_phase2_classify.py -> data/phase2_qa_labels.csv
"""
import csv, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import load_model, chat
from src.config import MODEL_ID

SYSTEM = ("You classify the emotional state of the person writing a message to an assistant. Reply with exactly one "
          "word from this list and nothing else: neutral, distressed, frustrated. 'distressed' = anxious, sad, "
          "overwhelmed or frightened. 'frustrated' = angry, irritated or annoyed. 'neutral' = no emotional state expressed. "
          "Judge only the writer's own state, not anyone else's.")
rows = list(csv.DictReader(open("data/phase2_prompts.csv", newline="", encoding="utf-8")))
model, tok = load_model(); print(f"judge = {MODEL_ID} (the subject model)"); chat(model, tok, "hi", system=SYSTEM, max_new_tokens=3)
out = []; t0 = time.time()
for i, r in enumerate(rows):
    reply = chat(model, tok, f"Message: {r['text']}\n\nOne word:", system=SYSTEM, max_new_tokens=4)
    m = re.search(r"neutral|distressed|frustrated", reply.lower())
    out.append({"id": r["id"], "condition": r["condition"], "split": r["split"], "judge_label": m.group(0) if m else "other", "judge_raw": reply.strip()[:40]})
    if (i + 1) % 50 == 0: print(f"  {i+1}/{len(rows)} in {time.time()-t0:.0f}s", flush=True)
with open("data/phase2_qa_labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(out)
print(f"done {len(out)} rows in {time.time()-t0:.0f}s -> data/phase2_qa_labels.csv")
