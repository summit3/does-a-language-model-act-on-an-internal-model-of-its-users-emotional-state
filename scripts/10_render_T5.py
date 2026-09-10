"""T5 as an image: run M, sampled subtraction on the 30 distressed val prompts (T=0.7, 5 samples each, 150 per fraction), rules v3.
Reads results/phase4/phase4_M_sampled_subtraction_ci.csv (written by scripts/04b_score_phase4.py) and writes
results/phase4/figures/T5_run_M_sampled_subtraction.png. Usage: python scripts/10_render_T5.py [out.png]. Laptop-only."""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "results/phase4/phase4_M_sampled_subtraction_ci.csv"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/phase4/figures/T5_run_M_sampled_subtraction.png"
rows = {(r["fraction"], r["metric"]): r for r in csv.DictReader(open(SRC, newline="", encoding="utf-8"))}
metrics = [("acknowledges_emotion", "acknowledges"), ("user_state_inference", "infers user state"), ("task_abandoned", "abandons task"), ("correct", "correct")]
fracs = [("0.0", "fraction 0 (unsteered)", "#f0f0ee", "#222222"), ("0.04", "-0.04 along distressed mean-diff", "#3f85d9", "white")]
fmt = lambda r: (f"{float(r['rate']):.3f} [{float(r['ci_lo']):.3f}, {float(r['ci_hi']):.3f}]", int(r["n"]))
cells, ns = [], {}
for m, lab in metrics:
    row = [lab]
    for f, *_ in fracs: t, n = fmt(rows[(f, m)]); row.append(t); ns[lab] = n
    cells.append(row)
fig, ax = plt.subplots(figsize=(8.0, 2.7), dpi=200); fig.patch.set_facecolor("#fcfcfb"); ax.axis("off")
tb = ax.table(cellText=cells, colLabels=["metric"] + [lab for _, lab, _, _ in fracs], loc="center", cellLoc="center", colWidths=[0.24, 0.36, 0.40])
tb.auto_set_font_size(False); tb.set_fontsize(8.8); tb.scale(1, 2.4)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#d9d9d6"); cell.set_linewidth(0.6)
    if r == 0:
        cell.set_height(cell.get_height() * 1.2); cell.get_text().set_fontweight("bold")
        cell.set_facecolor(fracs[c - 1][2] if c else "#f0f0ee"); cell.get_text().set_color(fracs[c - 1][3] if c else "#222222")
    elif c == 0: cell.get_text().set_ha("left"); cell.get_text().set_fontweight("bold"); cell.set_facecolor("#f7f7f5")
    else: cell.set_facecolor("#ffffff")
fig.suptitle("T5. Run M: subtracting the distressed direction from distressed prompts\n30 distressed val prompts, T = 0.7, 5 samples each (150 per column), rules v3", x=0.03, y=0.99, ha="left", fontsize=9, linespacing=1.4)
fig.text(0.03, 0.05, f"Wilson 95% CIs; n = 150 per cell except correct, over the {ns['correct']} closed-task samples.\nBand 12-23; strength as a fraction of the prompt's mean band norm.", ha="left", va="bottom", fontsize=7.4, color="#444444", linespacing=1.5)
fig.tight_layout(rect=(0, 0.15, 1, 0.9)); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=200, facecolor=fig.get_facecolor()); print(f"wrote {OUT}")
