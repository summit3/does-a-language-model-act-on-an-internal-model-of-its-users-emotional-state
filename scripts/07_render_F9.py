"""F9: headline bar chart for run J (steering at 0.04 of mean band norm on all 150 bare base tasks, greedy, rules v3).
Reads the v3 column of results/phase4/phase4_J_power_ci_v1_v2_v3.csv (written by scripts/04c_handcheck_agreement.py) and writes
results/phase4/figures/F9_headline_J_bar.png. Usage: python scripts/07_render_F9.py [out.png]. Laptop-only; no model needed."""
import csv, re, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "results/phase4/phase4_J_power_ci_v1_v2_v3.csv"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/phase4/figures/F9_headline_J_bar.png"
rows = {(r["direction"], r["metric"]): r["v3"] for r in csv.DictReader(open(SRC, newline="", encoding="utf-8"))}
def parse(s): m = re.match(r"([\d.]+) \[([\d.]+),([\d.]+)\] n=(\d+)", s); return float(m[1]), float(m[2]), float(m[3])
metrics = [("acknowledges_emotion", "acknowledges"), ("user_state_inference", "infers user state"), ("task_abandoned", "abandons task"), ("correct", "correct")]
dirs = [("distressed_md", "distressed mean-diff", "#3f85d9"), ("random", "random", "#ec7648"), ("unrelated_coding_probe", "unrelated coding probe", "#31b687")]
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150); fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
x = np.arange(len(metrics)); w = 0.26
for i, (d, lab, c) in enumerate(dirs):
    vals = [parse(rows[(d, m)]) for m, _ in metrics]; p = [v[0] for v in vals]; lo = [v[0] - v[1] for v in vals]; hi = [v[2] - v[0] for v in vals]
    xs = x + (i - 1) * w; ax.bar(xs, p, w, color=c, label=lab, yerr=[lo, hi], capsize=3, error_kw=dict(lw=1.2, ecolor="black"))
    for xi, v in zip(xs, vals): ax.text(xi, v[2] + 0.02, f"{v[0]:.2f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in metrics], fontsize=9.5); ax.set_ylabel("rate (Wilson 95% CI)", fontsize=9.5); ax.set_ylim(0, 1.1)
ax.yaxis.grid(True, color="#e8e8e6", lw=0.8); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ("top", "right")]; [ax.spines[s].set_color("#cccccc") for s in ("left", "bottom")]
ax.tick_params(colors="#444444", labelsize=9); ax.legend(loc="upper left", frameon=False, fontsize=9)
fig.suptitle("F9. Steering at 0.04 of mean band norm, all 150 bare tasks (bare form never entered the direction; greedy, rules v3)\nn = 150 per bar; correct over the 125 closed tasks", x=0.03, y=0.99, ha="left", fontsize=8.8)
fig.tight_layout(rect=(0, 0, 1, 0.95)); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=150, facecolor=fig.get_facecolor()); print(f"wrote {OUT}")
