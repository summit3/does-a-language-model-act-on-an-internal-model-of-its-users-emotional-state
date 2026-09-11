"""T7: run J rates by task type (steering at 0.04 of mean band norm on all 150 bare base tasks, greedy, rules v3), one block per direction.
Reads results/phase4/phase4_J_by_task_type.csv (written by scripts/04b_score_phase4.py) and writes results/phase4/figures/T7_J_by_task_type.md.
Columns: task_type, n, acknowledges, infers, abandons, correct; rates to two decimals; n = 25 per task type (150 tasks / 6 types).
Usage: python scripts/12_render_T7.py [out.md]. Laptop-only; no model needed."""
import csv, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "results/phase4/phase4_J_by_task_type.csv"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/phase4/figures/T7_J_by_task_type.md"
scores = [r for r in csv.DictReader(open(ROOT / "results/phase4/phase4_scores.csv", newline="", encoding="utf-8")) if r["run"] == "J"]
n_by = collections.Counter((r["direction"], r["task_type"]) for r in scores)
rows = list(csv.DictReader(open(SRC, newline="", encoding="utf-8")))
f2 = lambda v: f"{float(v):.2f}" if v != "" else ""
L = ["# T7. Run J by task type: distressed_md, random and unrelated_coding_probe at 0.04 of mean band norm on all 150 bare tasks (greedy, rules v3)", "",
     "Rates from results/phase4/phase4_J_by_task_type.csv; n per task type from phase4_scores.csv. Correct is over closed tasks only, so it is blank for advice.", ""]
for d in ["distressed_md", "random", "unrelated_coding_probe"]:
    L += [f"## {d} @ 0.04", "", "| task_type | n | acknowledges | infers | abandons | correct |", "|---|---|---|---|---|---|"]
    L += [f"| {r['task_type']} | {n_by[(d, r['task_type'])]} | {f2(r['acknowledges_emotion'])} | {f2(r['user_state_inference'])} | {f2(r['task_abandoned'])} | {f2(r['correct'])} |" for r in rows if r["direction"] == d]
    L.append("")
OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(L), encoding="utf-8"); print(f"wrote {OUT}")
