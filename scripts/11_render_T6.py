"""T6 as an image: rule-score rates (v3) for runs D, E, F, I, K, N (the specificity / robustness runs of section 4.3), one row per
run x form x direction x fraction. Same computation as rate_table() in scripts/09_assemble_writeup.py (which writes the markdown twin,
results/writeup/04_3_specificity_robustness/rates_E_F_I_D_K_N.md). Reads results/phase4/phase4_scores.csv and writes
results/phase4/figures/T6_rates_D_E_F_I_K_N.png. Usage: python scripts/11_render_T6.py [out.png]. Laptop-only; no model needed."""
import csv, sys, collections
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "results/phase4/phase4_scores.csv"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/phase4/figures/T6_rates_D_E_F_I_K_N.png"
RUNS = "DEFIKN"
g = collections.defaultdict(list)
for s in csv.DictReader(open(SRC, newline="", encoding="utf-8")):
    if s["run"] in RUNS: g[(s["run"], s["form"], s["direction"], s["fraction"])].append(s)
DIRS = {"distressed_md": ("distressed mean-diff", "#3f85d9"), "random": ("random", "#ec7648"), "unrelated_coding_probe": ("unrelated coding probe", "#31b687"),
        "np_minus_bare_md": ("preamble-presence (np - bare)", "#222222"), "distressed_md_band8-19": ("distressed mean-diff, band 8-19", "#3f85d9"),
        "distressed_md_band16-27": ("distressed mean-diff, band 16-27", "#3f85d9"), "shared_pc1": ("shared PC1", "#222222"), "valence_resid": ("valence residual", "#222222")}
cells, run_of, colors = [], [], []
for k in sorted(g, key=lambda k: (k[0], k[1], k[2], float(k[3]))):
    run, form, d, frac = k; rs = g[k]; closed = [r for r in rs if r["correct"] != ""]
    m = lambda col, sub=rs: f"{sum(float(r[col]) for r in sub) / len(sub):.3f}" if sub else "n/a"
    lab, col = DIRS.get(d, (d, "#222222"))
    cells.append([run, form, lab, frac, str(len(rs)), f"{m('correct', closed)} ({len(closed)})", m("task_abandoned"), m("acknowledges_emotion"), m("user_state_inference"), m("incoherent"), m("affect_leak_lexical"), f"{sum(float(r['n_tokens']) for r in rs) / len(rs):.1f}"])
    run_of.append(run); colors.append(col)
# show the run letter only on the first row of each block
for i in range(len(cells) - 1, 0, -1):
    if run_of[i] == run_of[i - 1]: cells[i][0] = ""
head = ["run", "form", "direction", "fraction", "n", "correct (n closed)", "abandons", "acknowledges", "infers user state", "incoherent", "leak (lexical)", "mean tokens"]
widths = [0.045, 0.06, 0.20, 0.06, 0.04, 0.10, 0.07, 0.085, 0.09, 0.075, 0.08, 0.075]
fig, ax = plt.subplots(figsize=(13.0, 9.8), dpi=200); fig.patch.set_facecolor("#fcfcfb"); ax.axis("off")
tb = ax.table(cellText=cells, colLabels=head, loc="center", cellLoc="center", colWidths=widths)
tb.auto_set_font_size(False); tb.set_fontsize(8.0); tb.scale(1, 1.55)
block = {r: (sum(1 for j in range(r) if run_of[j] != run_of[j - 1] and j > 0)) for r in range(len(cells))}  # block index per body row
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#d9d9d6"); cell.set_linewidth(0.5); t = cell.get_text()
    if r == 0:
        cell.set_height(cell.get_height() * 1.3); t.set_fontweight("bold"); cell.set_facecolor("#f0f0ee"); t.set_color("#222222"); continue
    i = r - 1; shade = "#ffffff" if block[i] % 2 == 0 else "#f5f5f3"; cell.set_facecolor(shade)
    if i > 0 and run_of[i] != run_of[i - 1]: cell.set_edgecolor("#9a9a97")  # heavier top rule at each run boundary (drawn as the cell's full edge)
    if c == 0: t.set_fontweight("bold"); t.set_ha("left")
    elif c == 2: t.set_ha("left"); t.set_color(colors[i])
    elif c == 5: t.set_fontweight("bold")
fig.suptitle("T6. Specificity and robustness runs D, E, F, I, K, N: rule-score rates (v3), one row per run x form x direction x fraction",
             x=0.03, y=0.985, ha="left", fontsize=9.6)
fig.text(0.03, 0.02,
         "Rates are fractions of the n generations in the row; correct is over closed tasks only (n in parentheses). Rules v3, from phase4_scores.csv.\n"
         "All rows use the 30 bare val prompts except N (30 neutral_preamble val prompts). Greedy decoding except D and N (T = 0.7, 5 samples per prompt, so n = 150).\n"
         "Band 12-23 except run F (bands 8-19 and 16-27); strength as a fraction of the prompt's mean band norm. D fraction 0.0 is the unsteered baseline.",
         ha="left", va="bottom", fontsize=7.4, color="#444444", linespacing=1.5)
fig.tight_layout(rect=(0, 0.07, 1, 1.0)); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=200, facecolor=fig.get_facecolor()); print(f"wrote {OUT}")
