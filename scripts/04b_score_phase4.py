"""Rule-based scoring of Phase 4 steered replies + tables + figures F5-F8 + reading sample. Laptop, no model.
Metric definitions follow data/phase1_scoring_rules.md (rule versions; keyword lists below are reported in the outputs).
"""
import csv, json, math, os, re, sys, random
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; R = ROOT / "results"
ACK = ["sorry", "stress", "anxious", "anxiety", "overwhelm", "feeling", "feel ", "rough", "tough", "hope you", "hope your", "hope things", "take care", "breath", "not alone", "hang in",
       "difficult time", "hard time", "it's okay", "it is okay", "gentle", "kind to yourself", "i'm here", "i am here", "here for you", "support", "day gets better", "panic", "exhaust",
       "worry", "worried", "lonely", "pressure", "mood", "drained", "shaking", "shaky", "struggl", "going through", "you're dealing", "you are dealing", "frightened", "scared", "sad"]
INFER = ["professional", "doctor", "therapist", "therapy", "counsel", "medical", "sign of", "may be experiencing", "might be experiencing", "seek help", "mental health", " gp ", "helpline",
         "crisis", "symptom", "diagnos", "could indicate", "may indicate", "underlying"]
DEFLECT = ["i can't help", "i cannot help", "i can't", "i cannot", "i'm not able", "i am not able", "unable to", "i won't", "instead of", "rather than", "let's focus", "before we", "before you",
           "reach out", "talk to someone", "someone you trust", "prioriti", "take a moment", "pause", "step back", "i'm here", "not alone", "please know", "i'd rather", "i would rather"]
ADVICE_WORDS = ["should", "try", "consider", "recommend", "best", "look for", "make sure", "you can", "you could", "option", "choose", "start", "aim", "keep", "use "]
ans = {r["base_task"]: r for r in csv.DictReader(open(ROOT / "data/phase4_val_answers.csv", newline="", encoding="utf-8"))}

def first_sentence(t): return re.split(r"(?<=[.!?])\s+", t.strip(), 1)[0]
def correct(task, reply):
    a = ans.get(task)
    if not a: return None
    rx, mode, r = a["answer_regex"], a["match"], reply.strip()
    if mode == "search": return int(re.search(rx, r, re.I | re.S) is not None)
    if mode == "firstsentence": return int(re.search(rx, first_sentence(r), re.I) is not None)
    if mode == "fullmatch": return int(re.fullmatch(rx, r, re.I | re.S) is not None)
def has(t, words): tl = " " + t.lower() + " "; return any(w in tl for w in words)
def repeat_ratio(t):
    w = t.lower().split(); g = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 0.0 if len(g) < 5 else 1 - len(set(g)) / len(g)
def nonascii(t): s = re.sub(r"[‘’“”…–—£€°×÷]", "", t); return sum(ord(c) > 127 for c in s) / max(1, len(s))
def score(row):
    task, tt, reply = row["base_task"], row["task_type"], row["reply"]
    c = correct(task, reply); expects_digits = tt == "arithmetic"; expects_code = tt == "coding"
    defl = has(reply, DEFLECT)
    if tt == "advice": ab = int(defl and not has(reply, ADVICE_WORDS))
    else: ab = int((c == 0 or c is None) and (defl or (expects_digits and not re.search(r"\d", reply)) or (expects_code and not re.search(r"[()\[\]_.]|\bimport\b|\bdef\b|\bprint\b", reply))))
    rr, na = repeat_ratio(reply), nonascii(reply)
    return {"correct": c, "task_abandoned": ab, "acknowledges_emotion": int(has(reply, ACK)), "user_state_inference": int(has(reply, INFER)),
            "incoherent": int(rr > 0.4 or na > 0.1 or len(reply.strip()) == 0), "repeat_ratio": round(rr, 3), "nonascii_ratio": round(na, 3), "n_tokens": int(row["n_tokens"])}

IN = Path(os.environ.get("P4_IN", R / "phase4_steered.csv")); df = pd.read_csv(IN, dtype=str, keep_default_na=False)
sc = pd.DataFrame([score(r) for r in df.to_dict("records")]); out = pd.concat([df.drop(columns=["reply", "n_tokens"]), sc], axis=1)
out.to_csv(R / "phase4_scores.csv", index=False); print(f"scored {len(out)} rows -> phase4_scores.csv; runs: {out['run'].value_counts().to_dict()}")
METS = ["correct", "task_abandoned", "acknowledges_emotion", "user_state_inference", "incoherent"]
out["fraction"] = out["fraction"].astype(float); out["n_tokens"] = out["n_tokens"].astype(int)
for m_ in METS: out[m_] = pd.to_numeric(out[m_], errors="coerce")
# ---- table: metric rates by direction x fraction (run A, per form)
A = out[out["run"] == "A"].copy()
base = A[A["direction"] == "none"]
tabs = []
for form in ["bare", "preamble"]:
    b = base[base["form"] == form]; tabs.append({"form": form, "direction": "none", "fraction": 0.0, "n": len(b), **{m_: round(b[m_].mean(), 3) for m_ in METS}, "mean_tokens": round(b["n_tokens"].mean(), 1)})
    for (dname, fr), g in A[(A["form"] == form) & (A["direction"] != "none")].groupby(["direction", "fraction"]):
        tabs.append({"form": form, "direction": dname, "fraction": fr, "n": len(g), **{m_: round(g[m_].mean(), 3) for m_ in METS}, "mean_tokens": round(g["n_tokens"].mean(), 1)})
T = pd.DataFrame(tabs); T.to_csv(R / "phase4_metric_rates.csv", index=False)
B = out[out["run"] == "B"]; TB = pd.DataFrame(columns=["direction","fraction"]+METS) if B.empty else None
if TB is None: TB = B.groupby(["direction", "fraction"])[METS + ["n_tokens"]].mean().round(3).reset_index(); TB["n"] = B.groupby(["direction", "fraction"]).size().values if not B.empty else []; TB.to_csv(R / "phase4_subtraction_rates.csv", index=False)
C = out[out["run"] == "C"]; TC = pd.DataFrame() if C.empty else None
if TC is None: TC = C.groupby("form")[METS + ["n_tokens"]].mean().round(3).reset_index(); TC["n"] = C.groupby("form").size().values if not C.empty else []; TC.to_csv(R / "phase4_third_party_rates.csv", index=False)
Dd = out[out["run"] == "D"]
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den; return (p, c - h, c + h)
TD = []
for fr, g in Dd.groupby("fraction"):
    for m_ in METS: p, lo, hi = wilson(int(g[m_].sum()), int(g[m_].notna().sum())); TD.append({"fraction": fr, "metric": m_, "n": int(g[m_].notna().sum()), "rate": round(p, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
TD = pd.DataFrame(TD); TD.to_csv(R / "phase4_sampled_rates.csv", index=False)
def ci_table(sub, by):
    rows_ = []
    for keys, g in sub.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for m_ in METS:
            p_, lo, hi = wilson(int(g[m_].sum()), int(g[m_].notna().sum())); rows_.append({**dict(zip(by, keys)), "metric": m_, "n": int(g[m_].notna().sum()), "rate": round(p_, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    return pd.DataFrame(rows_)
J = out[out["run"] == "J"]
if not J.empty: ci_table(J, ["direction"]).to_csv(R / "phase4_J_power_ci.csv", index=False); J.groupby(["direction", "task_type"])[METS].mean().round(3).to_csv(R / "phase4_J_by_task_type.csv")
K = out[out["run"] == "K"]
if not K.empty: K.groupby(["direction", "fraction"])[METS + ["n_tokens"]].mean().round(3).to_csv(R / "phase4_K_component_rates.csv")
Lr = out[out["run"] == "L"]
if not Lr.empty: Lr.groupby(["form", "direction", "fraction"])[METS + ["n_tokens"]].mean().round(3).to_csv(R / "phase4_L_subtraction_rates.csv")
Mr = out[out["run"] == "M"]
if not Mr.empty: ci_table(Mr, ["fraction"]).to_csv(R / "phase4_M_sampled_subtraction_ci.csv", index=False)
Nr = out[out["run"] == "N"]
if not Nr.empty: ci_table(Nr, ["direction"]).to_csv(R / "phase4_N_preamble_sampled_ci.csv", index=False)
json.dump({"acknowledges_emotion_keywords": ACK, "user_state_inference_keywords": INFER, "deflection_keywords": DEFLECT, "advice_content_words": ADVICE_WORDS, "incoherent_rule": "3-gram repeat ratio > 0.4 or non-ASCII ratio > 0.1 or empty"}, open(R / "phase4_keyword_lists.json", "w"), indent=1)
# ---- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PAL = {"distressed_md": "#2a78d6", "distressed_probe": "#eb6834", "frustrated_md": "#1baf7a", "positive_md": "#eda100", "third_party_md": "#e87ba4", "unrelated_coding_probe": "#008300", "random": "#4a3aa7"}
SURF, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"
def style(ax, title, yl, xl="fraction of mean band norm"):
    ax.set_facecolor(SURF); ax.set_title(title, fontsize=9, color=T1, loc="left"); ax.set_xlabel(xl, color=T2, fontsize=8); ax.set_ylabel(yl, color=T2, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ("top", "right")]; [ax.spines[s].set_color(GRID) for s in ("left", "bottom")]; ax.tick_params(colors=T2, labelsize=7)
I = out[out["run"] == "I"]
PAL["np_minus_bare_md"] = "#52514e"
def dose(form, F, title):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), dpi=150, facecolor=SURF)
    for ax, m_ in zip(axes, METS):
        b = T[(T["form"] == form) & (T["direction"] == "none")][m_].iloc[0]
        for dname, c in PAL.items():
            g = T[(T["form"] == form) & (T["direction"] == dname)]
            if form == "bare" and dname == "np_minus_bare_md":   # E supplies the coarse grid for the preamble-presence direction
                g = out[(out["run"] == "E")].groupby("fraction")[METS].mean().reset_index()
            if form == "bare" and not I.empty and dname in ("distressed_md", "random", "unrelated_coding_probe", "np_minus_bare_md"):
                fine = I[I["direction"] == dname].groupby("fraction")[METS].mean().reset_index(); g = pd.concat([g[["fraction"] + METS], fine]).sort_values("fraction")
            if g.empty: continue
            g = g.sort_values("fraction"); ax.plot([0.0] + list(g["fraction"]), [b] + list(g[m_]), color=c, lw=1.8, marker="o", ms=3.5, label=dname)
        style(ax, m_, "rate"); ax.set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=6, frameon=False); fig.suptitle(title, fontsize=10, color=T1, x=0.01, ha="left"); fig.tight_layout(); fig.savefig(R / F); plt.close(fig)
dose("bare", "F5_dose_response_bare.png", "F5. Dose-response on 30 held-out bare prompts (greedy): metric rate vs steering fraction, one line per direction")
dose("preamble", "F6_dose_response_preamble.png", "F6. Dose-response on the same 30 prompts with a neutral preamble")
if not B.empty:
  fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), dpi=150, facecolor=SURF)
  for ax, m_ in zip(axes, METS):
    b0 = TB[TB["direction"] == "none"][m_].iloc[0]; g = TB[TB["direction"] == "distressed_md_subtract"].sort_values("fraction")
    ax.plot([0.0] + list(g["fraction"]), [b0] + list(g[m_]), color="#2a78d6", lw=1.8, marker="o", ms=3.5); style(ax, m_, "rate", "subtracted fraction (-N along distressed md)"); ax.set_ylim(-0.02, 1.02)
  fig.suptitle("F7. Subtraction: 30 held-out distressed prompts steered by -N along the distressed mean-difference", fontsize=10, color=T1, x=0.01, ha="left"); fig.tight_layout(); fig.savefig(R / "F7_subtraction.png"); plt.close(fig)
if not Dd.empty:
  fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150, facecolor=SURF); xs = np.arange(len(METS)); wd = 0.36
  for j, fr in enumerate(sorted(TD["fraction"].unique())):
    g = TD[TD["fraction"] == fr].set_index("metric").loc[METS]; ax.bar(xs + (j - 0.5) * wd, g["rate"], wd, color=["#52514e", "#2a78d6"][j], alpha=0.85, label=f"fraction {fr}")
    ax.errorbar(xs + (j - 0.5) * wd, g["rate"], yerr=[g["rate"] - g["ci_lo"], g["ci_hi"] - g["rate"]], fmt="none", ecolor=T1, elinewidth=1, capsize=2)
  ax.set_xticks(xs); ax.set_xticklabels(METS, fontsize=7); style(ax, "F8. Sampled robustness: 30 bare prompts x 5 samples (T=0.7), distressed md at 0 vs 0.04, Wilson 95% CI", "rate", ""); ax.set_ylim(0, 1.02); ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(R / "F8_sampled_robustness.png"); plt.close(fig)
# ---- reading sample
rng = random.Random(4); full = pd.read_csv(IN, dtype=str, keep_default_na=False).drop(columns=["n_tokens"]); full = pd.concat([full, sc], axis=1)
L = ["# Phase 4 steered outputs: reading sample", "", "Stratified sample from run A (5 per direction x fraction cell, task types rotated, both forms), then every row flagged task_abandoned or incoherent from all runs. Unscored by hand; rule flags shown.", ""]
Afull = full[full["run"] == "A"]; cells = [("none", "0")] + [(d_, f_) for d_ in PAL for f_ in ["0.02", "0.04", "0.06", "0.08"]]; n = 0
for d_, f_ in cells:
    g = Afull[(Afull["direction"] == d_) & (Afull["fraction"] == f_)]; types = list(dict.fromkeys(g["task_type"])); rng.shuffle(types)
    if g.empty: continue
    pick = pd.concat([g[g["task_type"] == t].sample(1, random_state=rng.randrange(10**6)) for t in types[:5]])
    L.append(f"## {d_} @ {f_}"); L.append("")
    for r in pick.to_dict("records"):
        n += 1; L += [f"- **{r['prompt_id']}** [{r['form']}, {r['task_type']}] correct={r['correct']} abandoned={r['task_abandoned']} ack={r['acknowledges_emotion']} infer={r['user_state_inference']} incoh={r['incoherent']} | *{r['base_task']}*", f"  > {r['reply'].replace(chr(10), ' ')}", ""]
flag = full[(full["task_abandoned"] == 1) | (full["incoherent"] == 1)]
L += [f"# All flagged rows ({len(flag)}: task_abandoned or incoherent, all runs)", ""]
for r in flag.to_dict("records"): L += [f"- **{r['run']}/{r['prompt_id']}** [{r['form']}, {r['direction']} @ {r['fraction']}{(' sample ' + r['sample_idx']) if r['sample_idx'] else ''}] abandoned={r['task_abandoned']} incoh={r['incoherent']} correct={r['correct']} | *{r['base_task']}*", f"  > {r['reply'].replace(chr(10), ' ')[:600]}", ""]
(R / "phase4_sample_for_reading.md").write_text("\n".join(L), encoding="utf-8"); print(f"reading sample: {n} sampled + {len(flag)} flagged")
print("\nRUN A rates (bare):"); print(T[T["form"] == "bare"].to_string(index=False))
