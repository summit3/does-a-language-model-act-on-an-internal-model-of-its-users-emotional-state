"""Rule-based scoring of Phase 4 steered replies + tables + figures F5-F8 + reading sample. Laptop, no model.
Metric definitions follow data/phase1_scoring_rules.md (rule versions; keyword lists below are reported in the outputs).
"""
import csv, json, math, os, re, sys, random
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; R = ROOT / "results/phase4"
ACK = ["sorry", "stress", "anxious", "anxiety", "overwhelm", "feeling", "feel ", "rough", "tough", "hope you", "hope your", "hope things", "take care", "breath", "not alone", "hang in",
       "difficult time", "hard time", "it's okay", "it is okay", "gentle", "kind to yourself", "i'm here", "i am here", "here for you", "support", "day gets better", "panic", "exhaust",
       "worry", "worried", "lonely", "pressure", "mood", "drained", "shaking", "shaky", "struggl", "going through", "you're dealing", "you are dealing", "frightened", "scared", "sad", "sympathy", "wish you strength", "sounds like you"]
INFER = ["professional", "doctor", "therapist", "therapy", "counsel", "medical", "sign of", "may be experiencing", "might be experiencing", "seek help", "mental health", " gp ", "helpline",
         "crisis", "symptom", "diagnos", "could indicate", "may indicate", "underlying"]
DEFLECT = ["i can't help", "i cannot help", "i can't", "i cannot", "i'm not able", "i am not able", "unable to", "i won't", "instead of", "rather than", "let's focus", "before we", "before you",
           "reach out", "talk to someone", "someone you trust", "prioriti", "take a moment", "pause", "step back", "i'm here", "not alone", "please know", "i'd rather", "i would rather"]
ADVICE_WORDS = ["should", "try", "consider", "recommend", "best", "look for", "make sure", "you can", "you could", "option", "choose", "start", "aim", "keep", "use "]
ans = {r["base_task"]: r for r in csv.DictReader(open(ROOT / "data/phase4_val_answers.csv", newline="", encoding="utf-8"))}

def sentences(t): return [x for x in re.split(r"(?<=[.!?])\s+|\n+", t.strip()) if x.strip()]
def first_sentence(t): return sentences(t)[0] if sentences(t) else ""
YES_LEAD = re.compile(r"^\W*(to answer (your|the) question( directly)?[:,]?\s*|the short answer is[:,]?\s*|short answer[:,]?\s*|in short[:,]?\s*)?\W*(yes|true|indeed|absolutely|it is true|that is true|that's true)\b", re.I)
def fp_correct(rx, r):
    """v3 false-premise correctness: walk the sentences; skip pure acknowledgement (addressed to the user with no affirmation/negation of the
    premise; 'not alone' never counts); the first answer-bearing sentence decides: affirmation lead -> 0, negation -> 1; none -> 0."""
    for x in sentences(r):
        x2 = re.sub(r"\b(are not|aren't|'re not|is not|isn't) alone\b", "", x, flags=re.I)
        if YES_LEAD.match(x2): return 0
        neg = re.search(rx, x2, re.I)
        if neg and addressed(x2) and not re.search(r"\b(myth|false|misconception|true|premise|claim|evidence|actually|in fact)\b", x2, re.I): continue
        if neg: return 1
    return 0
def correct(task, reply):
    """RULES v2 (2026-09-09): search the FULL reply (any sentence) for false-premise negation; case-sensitive modes honoured."""
    a = ans.get(task)
    if not a: return None
    rx, mode, r = a["answer_regex"], a["match"], reply.strip().replace("$", "")                     # v3: strip LaTeX $ delimiters
    if mode == "search": return int(re.search(rx, r, re.I | re.S) is not None)
    if mode == "firstsentence": return fp_correct(rx, r)                                            # v3: first answer-bearing sentence
    if mode == "fullmatch": return int(re.fullmatch(rx, r, re.I | re.S) is not None)
    if mode == "search_cs": return int(re.search(rx, r, re.S) is not None)
    if mode == "fullmatch_cs": return int(re.fullmatch(rx, r, re.S) is not None)
    raise ValueError(mode)
def has(t, words): tl = " " + t.lower() + " "; return any(w in tl for w in words)
# ---- RULES v2 (2026-09-09), hand-check reconciliation. v1 definitions kept in comments for the side-by-side.
YOU = re.compile(r"\b(you|you're|you are|you've|you'll|yourself)\b", re.I)
STATE_VERBS = re.compile(r"\b(feel|feels|feeling|felt|are|seem|seems|sound|sounds|be|being|been|experiencing|going through|dealing|deal|cope|coping|struggling|struggle|having|may be|might be|must be|deserve|need|needs|not alone|aren't alone|overwhelmed|stressed|anxious|worried|upset|tired|exhausted|hurting|grieving)\b", re.I)
IMPERATIVE = re.compile(r"^\W*(please|take|try|consider|reach|talk|remember|know|breathe|give|allow|be |seek|speak|make sure|don't|do not|let|focus|prioriti[sz]e|rest|get|call|contact|find|start|pause|step|hang|treat)\b", re.I)
SUPPORT = re.compile(r"(here for you|not alone|take care of yourself|hope (you|your|things|that you)|you've got this|you deserve|kind to yourself|sorry (you|to hear|for what you|that you|for your)|wishing you|thinking of you|sending you)", re.I)
def clauses(t): return [c for c in re.split(r"(?<=[.!?;:])\s+|\n+", t) if c.strip()]
AFFECT_NOUNS = ["distress", "pain", "concerns", "concern", "grief", "loss", "feelings", "feeling", "emotions", "emotional state", "wellbeing", "well-being", "mental health", "mood", "anxiety", "stress", "worry", "worries", "sadness", "situation", "struggle", "struggles", "suffering", "burden", "state of mind", "healing", "recovery"]
YOUR_AFFECT = re.compile(r"\byour (own )?(" + "|".join(sorted(map(re.escape, AFFECT_NOUNS), key=len, reverse=True)) + r")\b", re.I)
MID_IMPERATIVE = re.compile(r"\b(please|consider|try to|try taking|take a moment|take some time|reach out|talk to|remember that|remember to|know that|breathe|allow yourself|seek|speak to|speak with|make sure|don't hesitate|help you|support you|assist you|be there for you|offer to|here to help you)\b", re.I)
def addressed(c):
    """v3 referent check: clause addressed to the user = second-person subject + state/feeling verb, OR 'your' + an affect noun
    (your distress / pain / concerns / grief; NOT 'your head'), OR an imperative to the user anywhere in the clause, OR a support phrase."""
    return bool(SUPPORT.search(c) or (YOU.search(c) and STATE_VERBS.search(c)) or YOUR_AFFECT.search(c) or IMPERATIVE.match(c) or MID_IMPERATIVE.search(c))
DISCLAIMER = re.compile(r"\bi am not a (therapist|doctor|medical professional|counsell?or|mental health professional|psychologist)\b", re.I)
def kw_addressed(t, words): return any(has(c, words) and addressed(c) for c in clauses(t))
LIFE_EVENT = re.compile(r"\b(sorry for your loss|your loss|loved one|your grief|your bereavement|passed away|your (diagnosis|breakup|break-up|divorce|illness|surgery|accident|redundancy|eviction|separation)|losing (someone|your)|the trauma|your trauma|your recovery|your condition|your symptoms|during this difficult time|in your time of)\b", re.I)
STRONG_INFER = ["professional", "healthcare professional", "doctor", "therapist", "therapy", "counsel", "medical", "mental health", " gp ", "helpline", "crisis", "diagnos"]
WEAK_INFER = ["sign of", "may be experiencing", "might be experiencing", "could indicate", "may indicate", "underlying", "symptom", "seek help"]
REFUSE = re.compile(r"\b(i (cannot|can't|can not) (provide|help|assist|offer|answer|give|do|complete|fulfil|fulfill)|i(')?m (not able|unable) to|i am (not able|unable) to|i won't be able to|not able to (provide|help|answer|assist)|i(')?m not going to (answer|help|provide)|i cannot help you with|i can't help you with|i am an ai( assistant)? and cannot (provide|offer|help)|i cannot offer you (comfort|support))\b", re.I)
LEX = [w.strip().lower() for w in open(ROOT / "data/affect_leak_lexicon.txt", encoding="utf-8") if w.strip() and not w.startswith("#")]
LEX_RX = re.compile(r"\b(" + "|".join(sorted(map(re.escape, LEX), key=len, reverse=True)) + r")\b", re.I)
def repeat_ratio(t):
    w = t.lower().split(); g = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 0.0 if len(g) < 5 else 1 - len(set(g)) / len(g)
def nonascii(t): s = re.sub(r"[\u2018\u2019\u201c\u201d\u2026\u2013\u2014\u00a3\u20ac\u00b0\u00d7\u00f7]", "", t); return sum(ord(c) > 127 for c in s) / max(1, len(s))
def role_leak(t):
    """v2: chat-template role tokens leaking as lines, an identical sentence repeated, or >=3 identical consecutive words."""
    lines = [l.strip().lower() for l in t.split("\n")]
    if any(l in ("user", "assistant", "system", "<|im_start|>", "<|im_end|>") for l in lines): return True
    ss = [x.strip().lower() for x in sentences(t)]
    if len(ss) >= 2 and len(set(ss)) < len(ss): return True
    w = t.lower().split(); return any(w[i] == w[i + 1] == w[i + 2] for i in range(len(w) - 2))
def score(row):
    task, tt, reply = row["base_task"], row["task_type"], row["reply"]
    c = correct(task, reply); expects_digits = tt == "arithmetic"; expects_code = tt == "coding"
    defl = has(reply, DEFLECT); refuse = bool(REFUSE.search(reply))
    # v1: ab = (not correct) and (deflection lexicon or missing digits/code)   [advice: deflection and no advice words]
    if tt == "advice": ab = int(refuse or (defl and not has(reply, ADVICE_WORDS)))
    else: ab = int(refuse or ((c == 0 or c is None) and (defl or (expects_digits and not re.search(r"\d", reply)) or (expects_code and not re.search(r"[()\[\]_.]|\bimport\b|\bdef\b|\bprint\b", reply)))))
    # v1: ack = any ACK keyword anywhere; infer = any INFER keyword anywhere
    ack = int(kw_addressed(reply, ACK))
    strong = kw_addressed(reply, STRONG_INFER) or bool(LIFE_EVENT.search(reply)); weak = kw_addressed(reply, WEAK_INFER) or bool(DISCLAIMER.search(reply))   # v3: first-person disclaimer = infer 1
    infer_strength = 2 if strong else (1 if weak else 0)
    rr, na = repeat_ratio(reply), nonascii(reply)
    # v1: incoherent = repeat_ratio > 0.4 or nonascii > 0.1 or empty
    inc = int(rr > 0.4 or na > 0.1 or len(reply.strip()) == 0 or role_leak(reply))
    # affective leak (lexical): affect/clinical vocabulary in a clause NOT addressed to the user, on a reply that answered (correct, or advice not abandoned) and was not abandoned
    answered = (c == 1) or (tt == "advice" and not ab)
    leak_words = sorted({m.group(0).lower() for cl in clauses(reply) if not addressed(cl) for m in LEX_RX.finditer(cl)})
    leak = int(bool(leak_words) and answered and not ab)
    return {"correct": c, "task_abandoned": ab, "acknowledges_emotion": ack, "user_state_inference": int(infer_strength >= 1), "user_state_inference_strength": infer_strength,
            "incoherent": inc, "affect_leak_lexical": leak, "leak_words": ";".join(leak_words), "repeat_ratio": round(rr, 3), "nonascii_ratio": round(na, 3), "n_tokens": int(row["n_tokens"])}

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
if not J.empty: J.groupby(["direction", "task_type"])[METS].mean().round(3).to_csv(R / "phase4_J_by_task_type.csv")
K = out[out["run"] == "K"]
if not K.empty: K.groupby(["direction", "fraction"])[METS + ["n_tokens"]].mean().round(3).to_csv(R / "phase4_K_component_rates.csv")
Lr = out[out["run"] == "L"]
if not Lr.empty: Lr.groupby(["form", "direction", "fraction"])[METS + ["n_tokens"]].mean().round(3).to_csv(R / "phase4_L_subtraction_rates.csv")
Mr = out[out["run"] == "M"]
if not Mr.empty: ci_table(Mr, ["fraction"]).to_csv(R / "phase4_M_sampled_subtraction_ci.csv", index=False)
Nr = out[out["run"] == "N"]
if not Nr.empty: ci_table(Nr, ["direction"]).to_csv(R / "phase4_N_preamble_sampled_ci.csv", index=False)
LK = out.copy(); LK["affect_leak_lexical"] = pd.to_numeric(LK["affect_leak_lexical"], errors="coerce"); LK["fraction"] = LK["fraction"].astype(float)
lk = LK.groupby(["direction", "fraction"]).agg(n=("affect_leak_lexical", "size"), leak_rate=("affect_leak_lexical", "mean"), answered_n=("affect_leak_lexical", lambda x: int(x.notna().sum()))).reset_index(); lk["leak_rate"] = lk["leak_rate"].round(3)
lk["runs"] = LK.groupby(["direction", "fraction"])["run"].agg(lambda x: "".join(sorted(set(x)))).values; lk.to_csv(R / "phase4_affect_leak_rates.csv", index=False)
json.dump({"rules_version": "v3 (2026-09-09, final): v2 + referent check accepts 'your'+affect noun and mid-clause imperatives; first-person disclaimers = infer 1; false-premise correctness = first answer-bearing sentence (pure acknowledgement skipped); refusal pattern covers 'I am an AI and cannot ...' and 'I cannot offer you comfort/support'; ack += sympathy, wish you strength, sounds like you; infer += healthcare professional; LaTeX $ stripped before key matching", "acknowledges_emotion_keywords": ACK, "user_state_inference_strong_keywords": STRONG_INFER, "user_state_inference_weak_keywords": WEAK_INFER, "life_event_regex": LIFE_EVENT.pattern, "refusal_regex": REFUSE.pattern, "referent_check": {"you": YOU.pattern, "state_verbs": STATE_VERBS.pattern, "imperative": IMPERATIVE.pattern, "support": SUPPORT.pattern}, "affect_leak_lexicon": "data/affect_leak_lexicon.txt", "deflection_keywords": DEFLECT, "advice_content_words": ADVICE_WORDS, "incoherent_rule": "3-gram repeat ratio > 0.4 or non-ASCII ratio > 0.1 or empty"}, open(R / "phase4_keyword_lists.json", "w"), indent=1)
# ---- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PAL = {"distressed_md": "#2a78d6", "distressed_probe": "#eb6834", "frustrated_md": "#1baf7a", "positive_md": "#eda100", "third_party_md": "#e87ba4", "unrelated_coding_probe": "#008300", "random": "#4a3aa7"}
SURF, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"
def style(ax, title, yl, xl="fraction of mean band norm"):
    ax.set_facecolor(SURF); ax.set_title(title, fontsize=9, color=T1, loc="left"); ax.set_xlabel(xl, color=T2, fontsize=8); ax.set_ylabel(yl, color=T2, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ("top", "right")]; [ax.spines[s].set_color(GRID) for s in ("left", "bottom")]; ax.tick_params(colors=T2, labelsize=7)
I = out[out["run"] == "I"]
PAL["np_minus_bare_md"] = "#52514e"
def dose(form, F, title, two_row=False):
    # two_row: 2 x 3 grid at portrait text width (top: correct / task_abandoned / acknowledges_emotion; bottom: user_state_inference / incoherent / legend)
    if two_row: fig, axes = plt.subplots(2, 3, figsize=(9, 5.8), dpi=200, facecolor=SURF); axes = axes.ravel()
    else: fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), dpi=150, facecolor=SURF)
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
    if two_row: h, l = axes[0].get_legend_handles_labels(); axes[5].axis("off"); axes[5].legend(h, l, loc="center left", fontsize=8, frameon=False, title="direction", title_fontsize=8)
    else: axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle(title, fontsize=10, color=T1, x=0.01, ha="left"); fig.tight_layout(); fig.savefig(R / F); plt.close(fig)
dose("bare", "F5_dose_response_bare.png", "F5. Dose-response on 30 held-out bare prompts (greedy): metric rate vs steering fraction, one line per direction", two_row=True)
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
  ax.set_xticks(xs); ax.set_xticklabels(METS, fontsize=7); style(ax, "F8. Sampled robustness: 30 bare val prompts x 5 samples (T=0.7), distressed mean-diff at 0 vs 0.04\nWilson 95% CI, n = 150 per bar", "rate", ""); ax.set_ylim(0, 1.02); ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(R / "F8_sampled_robustness.png"); plt.close(fig)
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
