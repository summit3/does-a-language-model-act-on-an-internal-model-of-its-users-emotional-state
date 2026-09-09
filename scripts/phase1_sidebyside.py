"""Side-by-side view of Phase 1 replies (no model needed). Used by the notebook and to write
results/phase1/phase1_replies.md:    python scripts/phase1_sidebyside.py
"""
import csv, sys
from collections import OrderedDict
from pathlib import Path

TASK_ORDER = ["arithmetic", "factual", "false_premise", "coding", "advice", "instruction_following"]

def load_pairs(path="results/phase1/phase1_replies.csv"):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    pairs = OrderedDict()
    for r in rows:
        pairs.setdefault(int(r["pair_id"]), {"task_type": r["task_type"]})[r["condition"]] = r
    by_task = OrderedDict((t, []) for t in TASK_ORDER)
    for pid, p in sorted(pairs.items()):
        by_task.setdefault(p["task_type"], []).append((pid, p))
    return by_task

def pair_markdown(pid, p):
    n, s = p["neutral"], p["stressed"]
    return "\n".join([
        f"### Pair {pid}  ({p['task_type']})", "",
        f"**Neutral prompt:** {n['prompt']}", "",
        f"**Stressed prompt:** {s['prompt']}", "",
        f"**Neutral reply** ({n['n_tokens']} tokens):", "", "```", n["reply"], "```", "",
        f"**Stressed reply** ({s['n_tokens']} tokens):", "", "```", s["reply"], "```", ""])

def write_markdown(out="results/phase1/phase1_replies.md", src="results/phase1/phase1_replies.csv"):
    by_task = load_pairs(src)
    parts = ["# Phase 1 replies, side by side", "", f"Source: `{src}`. Grouped by task type. Unscored.", ""]
    for task, items in by_task.items():
        if not items: continue
        parts += [f"## {task}  ({len(items)} pairs)", ""]
        parts += [pair_markdown(pid, p) for pid, p in items]
    Path(out).write_text("\n".join(parts), encoding="utf-8")
    return out

if __name__ == "__main__":
    print("wrote", write_markdown(*(sys.argv[1:2] or [])))
