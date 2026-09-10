"""T1 as an image: headline run J (steering at 0.04 of mean band norm on all 150 bare base tasks, greedy), rules v1 vs v3.
Reads results/phase4/phase4_J_power_ci_v1_v2_v3.csv (written by scripts/04c_handcheck_agreement.py) and writes
results/phase4/figures/T1_headline_J_v1_v3.png. Usage: python scripts/08_render_T1.py [out.png]. Laptop-only; no model needed."""
import csv, re, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "results/phase4/phase4_J_power_ci_v1_v2_v3.csv"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/phase4/figures/T1_headline_J_v1_v3.png"
rows = {(r["direction"], r["metric"]): r for r in csv.DictReader(open(SRC, newline="", encoding="utf-8"))}
def fmt(s): m = re.match(r"([\d.]+) \[([\d.]+),([\d.]+)\] n=(\d+)", s); return f"{m[1]} [{m[2]}, {m[3]}]", int(m[4])
metrics = [("acknowledges_emotion", "acknowledges"), ("user_state_inference", "infers user state"), ("task_abandoned", "abandons task"), ("correct", "correct"), ("incoherent", "incoherent")]
dirs = [("distressed_md", "distressed mean-diff", "#3f85d9"), ("random", "random", "#ec7648"), ("unrelated_coding_probe", "unrelated coding probe", "#31b687")]
cells, ns = [], set()
for m, lab in metrics:
    row = [lab]
    for d, _, _ in dirs:
        for v in ("v1", "v3"): t, n = fmt(rows[(d, m)][v]); row.append(t); ns.add((lab, v, n))
    cells.append(row)
fig, ax = plt.subplots(figsize=(12.5, 2.9), dpi=200); fig.patch.set_facecolor("#fcfcfb"); ax.axis("off")
col_labels = ["metric"] + [f"{lab}\n{v}" for _, lab, _ in dirs for v in ("v1", "v3 (final)")]
tb = ax.table(cellText=cells, colLabels=col_labels, loc="center", cellLoc="center", colWidths=[0.13] + [0.145] * 6)
tb.auto_set_font_size(False); tb.set_fontsize(8.6); tb.scale(1, 1.9)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#d9d9d6"); cell.set_linewidth(0.6)
    if r == 0:
        cell.set_height(cell.get_height() * 1.35); cell.get_text().set_fontweight("bold")
        cell.set_facecolor("#f0f0ee" if c == 0 else dirs[(c - 1) // 2][2]); cell.get_text().set_color("#222222" if c == 0 else "white")
        if c > 0 and c % 2 == 1: cell.set_alpha(0.75)
    elif c == 0: cell.get_text().set_ha("left"); cell.get_text().set_fontweight("bold"); cell.set_facecolor("#f7f7f5")
    elif c % 2 == 0: cell.set_facecolor("#ffffff")
    else: cell.set_facecolor("#f7f7f5")
fig.suptitle("T1. Headline run J: steering at 0.04 of mean band norm, all 150 bare tasks (bare form never entered the direction; greedy), rules v1 vs v3 (final)", x=0.03, y=0.97, ha="left", fontsize=9.2)
fig.text(0.03, 0.04, "Wilson 95% CIs; n = 150 per cell except correct, which is over closed tasks only (v1 n = 121, four instruction-following keys did not resolve; v3 n = 125).\n"
         "v1 = original keyword rules; v3 = after hand-check reconciliation (referent check, life-event inference, refusal-phrase abandonment, first-answer-sentence false-premise scoring).",
         ha="left", va="bottom", fontsize=7.4, color="#444444", linespacing=1.5)
fig.tight_layout(rect=(0, 0.16, 1, 0.97)); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=200, facecolor=fig.get_facecolor()); print(f"wrote {OUT}")
