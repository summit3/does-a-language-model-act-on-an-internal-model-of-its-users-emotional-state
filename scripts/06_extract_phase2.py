"""Extract last-token residual activations at every layer for all rows of data/phase2_prompts.csv.
Run on the pod. Saves activations/phase2_acts_<model>.pt (float16 tensor [n_rows, n_layers, hidden])
and results/phase2/phase2_activation_index.csv (row order = tensor order). Layer i = output of block i.
"""
import csv, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.config import MODEL_ID
from src.model import load_model
from src.hooks import get_residual_activations

rows = list(csv.DictReader(open("data/phase2_prompts.csv", newline="", encoding="utf-8")))
Path("activations").mkdir(exist_ok=True); Path("results/phase2").mkdir(parents=True, exist_ok=True)
tag = MODEL_ID.split("/")[-1].lower().replace(".", "_")
model, tok = load_model(); n_layers = model.config.get_text_config().num_hidden_layers; hid = model.config.get_text_config().hidden_size
acts = torch.empty(len(rows), n_layers, hid, dtype=torch.float16)
t0 = time.time()
for i, r in enumerate(rows):
    acts[i] = get_residual_activations(model, tok, r["text"]).to(torch.float16)
    if (i + 1) % 100 == 0: print(f"  {i+1}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
dt = time.time() - t0
assert torch.isfinite(acts.float()).all()
torch.save({"acts": acts, "model": MODEL_ID, "layer_convention": "index i = output of decoder block i, 0-based", "position": "last prompt token (after assistant header + empty think block)", "seconds": dt}, f"activations/phase2_acts_{tag}.pt")
with open("results/phase2/phase2_activation_index.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["row", "id", "base_task", "condition", "split", "author", "judge_agrees"], lineterminator="\n"); w.writeheader()
    for i, r in enumerate(rows): w.writerow({"row": i, **{k: r[k] for k in ["id", "base_task", "condition", "split", "author", "judge_agrees"]}})
print(f"done: shape {tuple(acts.shape)} float16, {acts.numel()*2/1e6:.0f} MB, {dt:.0f}s for {len(rows)} rows -> activations/phase2_acts_{tag}.pt, results/phase2/phase2_activation_index.csv")
