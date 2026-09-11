"""Phase 3b: probes with neutral_preamble as the neutral class (bare neutral reported separately, never trained on).
Tasks: (a') neutral_preamble vs distressed, (b') neutral_preamble vs frustrated, (c') 3-way, (d') distressed vs positive.
Outputs results/phase3/phase3b_*.csv, F1-F4 (v1 copies archived in results/archive/phase3_v1/ as *_v1_confounded_*), results/phase3/directions_layer<L>_v2.pt.
"""
import csv, json, sys, time, warnings
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent; R = ROOT / "results/phase3"; R2 = ROOT / "results/phase2"; rng = np.random.default_rng(0)
C_GRID = [0.001, 0.01, 0.1]; CV = StratifiedKFold(3, shuffle=True, random_state=0); NEU = "neutral_preamble"
STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"

d = torch.load(ROOT / "activations/phase2_acts_qwen3_5-9b.pt"); A = d["acts"].float().numpy(); N, NL, H = A.shape
idx = list(csv.DictReader(open(R2 / "phase2_activation_index.csv", newline="", encoding="utf-8")))
prompts = {r["id"]: r for r in csv.DictReader(open(ROOT / "data/phase2_prompts.csv", newline="", encoding="utf-8"))}
judge = {r["text"]: r["judge_label"] for r in csv.DictReader(open(ROOT / "data/phase2_qa_labels.csv", newline="", encoding="utf-8"))}
cond = np.array([r["condition"] for r in idx]); split = np.array([r["split"] for r in idx]); author = np.array([r["author"] for r in idx])
jagree = np.array([r["judge_agrees"] == "1" for r in idx]); text = np.array([prompts[r["id"]]["text"] for r in idx]); base = np.array([r["base_task"] for r in idx])
jlab = np.array([judge.get(t, "?") for t in text]); assert len(idx) == N
def m(**kw):
    mask = np.ones(N, bool)
    for k, v in kw.items(): mask &= np.isin({"cond": cond, "split": split, "author": author}[k], v if isinstance(v, (list, tuple)) else [v])
    return mask
TR = m(split="train"); VA = m(split="val"); CL = m(author="claude")
G = {"bare_neutral_val": VA & m(cond="neutral"), "neutral_preamble_val": VA & m(cond=NEU), "distressed_val": VA & m(cond="distressed"), "frustrated_val": VA & m(cond="frustrated"),
     "positive_all": m(cond="positive"), "positive_val": VA & m(cond="positive"), "implied": CL & m(cond="implied"), "implied_bare_neutral": m(split="implied", cond="neutral"),
     "third_party": CL & m(cond="third_party"), "third_party_neutral": CL & m(cond="third_party_neutral"),
     "human_neutral": m(split="human_neutral"), "human_distressed": m(split="human_distressed"), "human_frustrated": m(split="human_frustrated"), "human_implied": m(split="human_implied"),
     "human_third_party": m(split="human_third_party"), "human_third_party_neutral": m(split="human_third_party_neutral"),
     "jm_distressed": (~jagree) & m(cond="distressed") & ~TR, "jm_frustrated": (~jagree) & m(cond="frustrated") & ~TR}
print("rows per group:", {k: int(v.sum()) for k, v in G.items()}); print("train rows by condition:", {c: int((TR & m(cond=c)).sum()) for c in [NEU, "neutral", "distressed", "frustrated", "positive"]})
PD_GROUPS = ["bare_neutral_val", "neutral_preamble_val", "distressed_val", "frustrated_val", "positive_all", "implied", "third_party", "third_party_neutral",
             "human_neutral", "human_distressed", "human_frustrated", "human_implied", "human_third_party", "human_third_party_neutral"]
Y3 = np.array([{NEU: 0, "distressed": 1, "frustrated": 2}.get(c, -1) for c in cond])
def prep(L): X = A[:, L, :]; mu = X[TR].mean(0); sd = X[TR].std(0) + 1e-6; return (X - mu) / sd, mu, sd
def fit(X, y, cv=True):
    if cv: g = GridSearchCV(LogisticRegression(max_iter=3000), {"C": C_GRID}, cv=CV, scoring="balanced_accuracy", n_jobs=-1).fit(X, y); return g.best_estimator_, g.best_params_["C"]
    return LogisticRegression(C=0.01, max_iter=3000).fit(X, y), 0.01
def ba(clf, X, y): return balanced_accuracy_score(y, clf.predict(X)) if len(set(y)) > 1 else np.nan
def rec(clf, X, pos=1): return float((clf.predict(X) == pos).mean()) if X.shape[0] else np.nan
def pdis(clf, X): return clf.predict_proba(X)[:, list(clf.classes_).index(1)]
def two(c1, c2, mask): return mask & m(cond=[c1, c2]), (cond[mask & m(cond=[c1, c2])] == c2).astype(int)

def eval_a(clf, X, t):
    o = {}; y = (cond == "distressed").astype(int)
    mt, yt = two(NEU, "distressed", TR); o[f"{t}_train"] = ba(clf, X[mt], yt); mv, yv = two(NEU, "distressed", VA); o[f"{t}_val"] = ba(clf, X[mv], yv)
    bv, yb = two("neutral", "distressed", VA); o[f"{t}_val_bare"] = ba(clf, X[bv], yb)                       # deployment-realistic, never trained on
    o[f"{t}_implied_recall"] = rec(clf, X[G["implied"]]); ib = G["implied"] | G["implied_bare_neutral"]; o[f"{t}_implied_ba_bare"] = ba(clf, X[ib], (cond[ib] == "implied").astype(int))
    inp = G["implied"] | G["neutral_preamble_val"]; o[f"{t}_implied_ba_np"] = ba(clf, X[inp], (cond[inp] == "implied").astype(int))
    hu = G["human_neutral"] | G["human_distressed"]; o[f"{t}_human"] = ba(clf, X[hu], y[hu]); o[f"{t}_human_recall"] = rec(clf, X[G["human_distressed"]])
    o[f"{t}_jm_recall"] = rec(clf, X[G["jm_distressed"]]); jp = G["jm_distressed"] | G["neutral_preamble_val"]; o[f"{t}_jm_pooled"] = ba(clf, X[jp], y[jp])
    o[f"{t}_positive_as_distressed"] = rec(clf, X[G["positive_all"]])
    for g in PD_GROUPS: o[f"{t}_pdist_{g}"] = float(pdis(clf, X[G[g]]).mean())
    return o
def eval_b(clf, X, t):
    o = {}; y = (cond == "frustrated").astype(int)
    mt, yt = two(NEU, "frustrated", TR); o[f"{t}_train"] = ba(clf, X[mt], yt); mv, yv = two(NEU, "frustrated", VA); o[f"{t}_val"] = ba(clf, X[mv], yv)
    bv, yb = two("neutral", "frustrated", VA); o[f"{t}_val_bare"] = ba(clf, X[bv], yb)
    hu = G["human_neutral"] | G["human_frustrated"]; o[f"{t}_human"] = ba(clf, X[hu], y[hu]); o[f"{t}_human_recall"] = rec(clf, X[G["human_frustrated"]])
    o[f"{t}_jm_recall"] = rec(clf, X[G["jm_frustrated"]]); o[f"{t}_positive_as_frustrated"] = rec(clf, X[G["positive_all"]]); return o
def eval_c(clf, X, t):
    o = {}; o[f"{t}_train"] = ba(clf, X[TR & (Y3 >= 0)], Y3[TR & (Y3 >= 0)]); o[f"{t}_val"] = ba(clf, X[VA & (Y3 >= 0)], Y3[VA & (Y3 >= 0)])
    hu = G["human_neutral"] | G["human_distressed"] | G["human_frustrated"]; yh = np.array([{"neutral": 0, "distressed": 1, "frustrated": 2}[c] for c in cond[hu]]); o[f"{t}_human"] = ba(clf, X[hu], yh); return o
def eval_d(clf, X, t):
    o = {}; mt, yt = two("positive", "distressed", TR); o[f"{t}_train"] = ba(clf, X[mt], yt); mv, yv = two("positive", "distressed", VA); o[f"{t}_val"] = ba(clf, X[mv], yv)
    o[f"{t}_val_n"] = int(mv.sum()); o[f"{t}_implied_as_distressed"] = rec(clf, X[G["implied"]]); o[f"{t}_np_as_distressed"] = rec(clf, X[G["neutral_preamble_val"]]); return o
TASKS = {"a": (NEU, "distressed", eval_a), "b": (NEU, "frustrated", eval_b), "d": ("positive", "distressed", eval_d)}

if STAGE in ("probes", "all"):
    rows, models, t0 = [], {}, time.time()
    for L in range(NL):
        X, mu, sd = prep(L); row = {"layer": L}; ms = {}
        for t, (c0, c1, ev) in TASKS.items():
            mt, yt = two(c0, c1, TR); clf, C = fit(X[mt], yt); row.update(ev(clf, X, t)); row[f"{t}_C"] = C; ms[t] = clf
            ys = rng.permutation(yt); sh, _ = fit(X[mt], ys, cv=False); mv, yv = two(c0, c1, VA); row[f"shuffled_{t}_val"] = ba(sh, X[mv], yv)
        clf_c, Cc = fit(X[TR & (Y3 >= 0)], Y3[TR & (Y3 >= 0)]); row.update(eval_c(clf_c, X, "c")); row["c_C"] = Cc; ms["c"] = clf_c
        cs = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)); rnd = rng.standard_normal(H)
        row["cos_wa_wb"] = cs(ms["a"].coef_[0], ms["b"].coef_[0]); row["cos_wa_rand"] = cs(ms["a"].coef_[0], rnd); row["cos_wb_rand"] = cs(ms["b"].coef_[0], rnd); row["cos_wa_wd"] = cs(ms["a"].coef_[0], ms["d"].coef_[0])
        models[L] = (ms, mu, sd); rows.append(row)
        print(f"L{L:02d} a val={row['a_val']:.2f} bare={row['a_val_bare']:.2f} impl_rec={row['a_implied_recall']:.2f} hum={row['a_human']:.2f} jm={row['a_jm_recall']:.2f} tpn_P={row['a_pdist_third_party_neutral']:.2f} | b val={row['b_val']:.2f} | c val={row['c_val']:.2f} | d val={row['d_val']:.2f} | shuf a={row['shuffled_a_val']:.2f} | {time.time()-t0:.0f}s", flush=True)
    with open(R / "phase3b_probe_by_layer.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    def key(t):
        def k(r):
            held = (r["a_implied_recall"] + r["a_human"]) / 2 if t == "a" else (r[f"{t}_human"] if t in "bc" else r["d_train"])
            return (round(r[f"{t}_val"], 3), round(held, 3), -abs(r["layer"] - NL // 2))
        return k
    best = {t: max(rows, key=key(t))["layer"] for t in "abcd"}; json.dump({**best, "rule": "argmax val BA; ties by held-out (a: implied recall+human; b,c: human), then mid-depth"}, open(R / "phase3b_best_layers.json", "w")); print("best layers:", best)
    # baselines
    bl = []; tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(text[TR]); XT = tf.transform(text)
    try:
        from transformers import AutoTokenizer; tk = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B"); ln = np.array([len(tk(t)["input_ids"]) for t in text], float); lname = "length_tokens"
    except Exception: ln = np.array([len(t.split()) for t in text], float); lname = "length_words"
    XL = ((ln - ln[TR].mean()) / ln[TR].std())[:, None]
    for name, Xb in [("bow_tfidf", XT), (lname, XL)]:
        o = {"baseline": name}
        for t, (c0, c1, ev) in TASKS.items(): mt, yt = two(c0, c1, TR); clf, _ = fit(Xb[mt], yt); o.update(ev(clf, Xb, t))
        clf, _ = fit(Xb[TR & (Y3 >= 0)], Y3[TR & (Y3 >= 0)]); o.update(eval_c(clf, Xb, "c")); bl.append(o)
    jd = (jlab == "distressed").astype(int); jf = (jlab == "frustrated").astype(int); o = {"baseline": "ask_the_model_judge"}
    mv, yv = two(NEU, "distressed", VA); o["a_val"] = balanced_accuracy_score(yv, jd[mv]); bv, yb = two("neutral", "distressed", VA); o["a_val_bare"] = balanced_accuracy_score(yb, jd[bv])
    o["a_implied_recall"] = float(jd[G["implied"]].mean()); hu = G["human_neutral"] | G["human_distressed"]; o["a_human"] = balanced_accuracy_score((cond[hu] == "distressed").astype(int), jd[hu]); o["a_jm_recall"] = 0.0
    for g in PD_GROUPS: o[f"a_pdist_{g}"] = float(jd[G[g]].mean())
    mv, yv = two(NEU, "frustrated", VA); o["b_val"] = balanced_accuracy_score(yv, jf[mv]); j3 = np.array([{"neutral": 0, "distressed": 1, "frustrated": 2}.get(x, 0) for x in jlab]); o["c_val"] = balanced_accuracy_score(Y3[VA & (Y3 >= 0)], j3[VA & (Y3 >= 0)])
    mv, yv = two("positive", "distressed", VA); o["d_val"] = balanced_accuracy_score(yv, jd[mv]); bl.append(o)
    bkeys = sorted({k for b in bl for k in b}, key=lambda k: (k != "baseline", k))
    with open(R / "phase3b_baselines.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=bkeys, lineterminator="\n"); w.writeheader(); w.writerows(bl)
    # per-row P(distressed) at best a layer; headline
    La = best["a"]; ms, mu, sd = models[La]; X = (A[:, La, :] - mu) / sd; p = pdis(ms["a"], X); br = rows[La]; bb = rows[best["b"]]; bc = rows[best["c"]]; bd = rows[best["d"]]; bow, lenb, ask = bl
    with open(R / "phase3b_pdist_bestlayer.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["row", "id", "condition", "split", "author", "judge_agrees", "p_distressed"]); [w.writerow([i, r["id"], r["condition"], r["split"], r["author"], r["judge_agrees"], f"{p[i]:.4f}"]) for i, r in enumerate(idx)]
    pdl = lambda o: " / ".join(f"{o[f'a_pdist_{g}']:.2f}" for g in PD_GROUPS)
    head = [["neutral class", NEU], ["task (a') neutral_preamble vs distressed, layer", La], ["a' val / train", f"{br['a_val']:.3f} / {br['a_train']:.3f}"], ["a' bare-neutral val vs distressed val (never trained on)", f"{br['a_val_bare']:.3f}"],
            ["a' implied recall / BA vs bare matched neutrals / BA vs neutral_preamble val", f"{br['a_implied_recall']:.3f} / {br['a_implied_ba_bare']:.3f} / {br['a_implied_ba_np']:.3f}"],
            ["a' human BA / recall (n=5+6)", f"{br['a_human']:.3f} / {br['a_human_recall']:.3f}"], ["a' judge-missed non-train recall / pooled BA (n=%d)" % G["jm_distressed"].sum(), f"{br['a_jm_recall']:.3f} / {br['a_jm_pooled']:.3f}"],
            ["a' positive rows classified distressed (n=50)", f"{br['a_positive_as_distressed']:.3f}"], ["a' shuffled-label val BA at this layer", f"{br['shuffled_a_val']:.3f}"],
            ["P(distressed) groups: " + " / ".join(PD_GROUPS), pdl(br)],
            ["LENGTH-ONLY baseline: a' val / a' bare val / b' val / c' val / d' val", f"{lenb['a_val']:.3f} / {lenb['a_val_bare']:.3f} / {lenb['b_val']:.3f} / {lenb['c_val']:.3f} / {lenb['d_val']:.3f}"],
            ["LENGTH-ONLY P(distressed) groups", pdl(lenb)],
            ["bag-of-words: a' val / a' bare val / implied recall / human / jm recall / b' val / c' val / d' val", f"{bow['a_val']:.3f} / {bow['a_val_bare']:.3f} / {bow['a_implied_recall']:.3f} / {bow['a_human']:.3f} / {bow['a_jm_recall']:.3f} / {bow['b_val']:.3f} / {bow['c_val']:.3f} / {bow['d_val']:.3f}"],
            ["ask-the-model: a' val / a' bare val / implied recall / human / b' val / c' val / d' val", f"{ask['a_val']:.3f} / {ask['a_val_bare']:.3f} / {ask['a_implied_recall']:.3f} / {ask['a_human']:.3f} / {ask['b_val']:.3f} / {ask['c_val']:.3f} / {ask['d_val']:.3f}"],
            ["task (b') neutral_preamble vs frustrated, layer / val / bare val / human / jm recall (n=1)", f"{best['b']} / {bb['b_val']:.3f} / {bb['b_val_bare']:.3f} / {bb['b_human']:.3f} / {bb['b_jm_recall']:.3f}"],
            ["task (c') 3-way, layer / val / train / human", f"{best['c']} / {bc['c_val']:.3f} / {bc['c_train']:.3f} / {bc['c_human']:.3f}"],
            ["task (d') distressed vs positive, layer / val (n=%d) / train / implied as distressed / neutral_preamble as distressed" % bd["d_val_n"], f"{best['d']} / {bd['d_val']:.3f} / {bd['d_train']:.3f} / {bd['d_implied_as_distressed']:.3f} / {bd['d_np_as_distressed']:.3f}"],
            ["a' val BA at layer 0 / 1 / 2", f"{rows[0]['a_val']:.3f} / {rows[1]['a_val']:.3f} / {rows[2]['a_val']:.3f}"]]
    with open(R / "phase3b_headline.csv", "w", newline="") as f: w = csv.writer(f, lineterminator="\n"); w.writerow(["item", "value"]); w.writerows(head)
    print("\nHEADLINE"); [print(f"  {k}: {v}") for k, v in head]
    print(f"sanity: shuffled a' val mean = {np.mean([r['shuffled_a_val'] for r in rows]):.3f}; train-val gap L{La} = {br['a_train']-br['a_val']:.3f}")
    # geometry (relative to neutral_preamble) + directions v2
    Xr = A[:, La, :]; nm = lambda mask: Xr[mask].mean(0); unit = lambda v: (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32); cs = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    np_tr = nm(TR & m(cond=NEU)); md_d = nm(TR & m(cond="distressed")) - np_tr; md_f = nm(TR & m(cond="frustrated")) - np_tr
    pos_tasks = set(base[m(cond="positive")]); md_pos = nm(TR & m(cond="positive")) - nm(TR & m(cond=NEU) & np.isin(base, list(pos_tasks)))
    md_npb = np_tr - nm(TR & m(cond="neutral"))
    tp_t, tpn_t = set(base[G["third_party"]]), set(base[G["third_party_neutral"]]); nptv = m(cond=NEU) & np.isin(split, ["train", "val"])
    md_tp = nm(G["third_party"]) - nm(nptv & np.isin(base, list(tp_t))); md_tpn = nm(G["third_party_neutral"]) - nm(nptv & np.isin(base, list(tpn_t)))
    w_a, w_b, w_d = ms["a"].coef_[0] / sd, ms["b"].coef_[0] / sd, ms["d"].coef_[0] / sd
    yc = np.array([("Python" in t or "pip" in t) for t in text]).astype(int); Xs = (Xr - mu) / sd; clf_u, _ = fit(Xs[TR], yc[TR]); w_u = clf_u.coef_[0] / sd
    geo = {"layer": La, "neutral_class": NEU, "cos_md_d_md_f": cs(md_d, md_f), "cos_md_d_md_pos": cs(md_d, md_pos), "cos_md_d_md_np_minus_bare": cs(md_d, md_npb), "cos_md_tp_md_tpn": cs(md_tp, md_tpn),
           "cos_md_tp_md_d": cs(md_tp, md_d), "cos_md_tpn_md_d": cs(md_tpn, md_d), "cos_unrelated_probe_md_d": cs(w_u, md_d), "cos_probe_a_md_d": cs(w_a, md_d), "cos_probe_a_probe_b": cs(w_a, w_b), "cos_probe_a_probe_d": cs(w_a, w_d),
           "cos_md_f_md_pos": cs(md_f, md_pos), "norm_md_d": float(np.linalg.norm(md_d)), "norm_md_f": float(np.linalg.norm(md_f)), "norm_md_pos": float(np.linalg.norm(md_pos)), "norm_md_np_minus_bare": float(np.linalg.norm(md_npb)),
           "unrelated_probe_val_BA": ba(clf_u, Xs[VA], yc[VA]), "mean_resid_norm_train": float(np.linalg.norm(Xr[TR], axis=1).mean())}
    with open(R / f"phase3b_geometry_layer{La}.csv", "w", newline="") as f: w = csv.writer(f, lineterminator="\n"); w.writerow(["item", "value"]); [w.writerow([k, f"{v:.4f}" if isinstance(v, float) else v]) for k, v in geo.items()]
    torch.save({"layer": La, "model": d["model"], "neutral_class": NEU, "space": "raw residual stream (output of block L, last prompt token); mean-differences relative to neutral_preamble",
                "distressed_meandiff": torch.tensor(unit(md_d)), "frustrated_meandiff": torch.tensor(unit(md_f)), "positive_meandiff": torch.tensor(unit(md_pos)), "neutral_preamble_minus_bare_meandiff": torch.tensor(unit(md_npb)),
                "third_party_meandiff": torch.tensor(unit(md_tp)), "third_party_neutral_meandiff": torch.tensor(unit(md_tpn)), "distressed_probe": torch.tensor(unit(w_a)), "frustrated_probe": torch.tensor(unit(w_b)),
                "distressed_vs_positive_probe": torch.tensor(unit(w_d)), "unrelated_coding_probe": torch.tensor(unit(w_u)), "train_mean": torch.tensor(mu.astype(np.float32)), "train_std": torch.tensor(sd.astype(np.float32)), "geometry": geo}, R / f"directions_layer{La}_v2.pt")
    print("geometry:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in geo.items()}); print(f"saved results/phase3/directions_layer{La}_v2.pt")

if STAGE in ("figures", "all"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(R / "phase3b_probe_by_layer.csv"))); bl = {b["baseline"]: b for b in csv.DictReader(open(R / "phase3b_baselines.csv"))}; best = json.load(open(R / "phase3b_best_layers.json"))
    g = lambda k: np.array([float(r[k]) for r in rows]); Ls = g("layer"); bow = bl["bow_tfidf"]; lenb = [v for k, v in bl.items() if k.startswith("length")][0]
    S1, S2, S3, S4, S5, SURF, T1, T2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"
    def style(ax, title, yl):
        ax.set_facecolor(SURF); ax.set_title(title, fontsize=10, color=T1, loc="left"); ax.set_xlabel("decoder block i (activation = output of block i)", color=T2, fontsize=9); ax.set_ylabel(yl, color=T2, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ("top", "right")]; [ax.spines[s].set_color(GRID) for s in ("left", "bottom")]; ax.tick_params(colors=T2, labelsize=8)
    for t, F, title, series in [("a", "F1", "Task (a') neutral_preamble vs distressed: balanced accuracy by layer", [("a_val", "val", S1), ("a_val_bare", "bare-neutral val (not trained on)", S5), ("a_implied_recall", "implied recall", S2), ("a_human", "human", S3), ("a_jm_recall", "judge-missed recall (non-train)", S4)]),
                               ("b", "F2", "Task (b') neutral_preamble vs frustrated: balanced accuracy by layer", [("b_val", "val", S1), ("b_val_bare", "bare-neutral val (not trained on)", S5), ("b_human", "human", S3), ("b_jm_recall", "judge-missed recall (non-train)", S4)])]:
        fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150, facecolor=SURF)
        for k, lab, c in series: ax.plot(Ls, g(k), color=c, lw=2, marker="o", ms=4, label=lab)
        ax.axhline(float(bow[f"{t}_val"]), color=T2, lw=1.5, ls="--", label=f"bag-of-words val ({float(bow[f'{t}_val']):.2f})"); ax.axhline(float(lenb[f"{t}_val"]), color=T2, lw=1.5, ls=":", label=f"length-only val ({float(lenb[f'{t}_val']):.2f})"); ax.axhline(0.5, color=GRID, lw=1)
        ax.set_ylim(0.3, 1.02); style(ax, title, "balanced accuracy / recall"); ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=2); fig.tight_layout(); fig.savefig(R / f"{F}_probe_acc_by_layer_task_{t}.png"); plt.close(fig)
    pr = list(csv.DictReader(open(R / "phase3b_pdist_bestlayer.csv")))
    groups = [("bare neutral", lambda r: r["split"] == "val" and r["condition"] == "neutral"), ("neutral_preamble", lambda r: r["split"] == "val" and r["condition"] == NEU), ("distressed", lambda r: r["split"] == "val" and r["condition"] == "distressed"),
              ("frustrated", lambda r: r["split"] == "val" and r["condition"] == "frustrated"), ("positive", lambda r: r["condition"] == "positive"), ("implied", lambda r: r["condition"] == "implied" and r["author"] == "claude"),
              ("third_party", lambda r: r["condition"] == "third_party" and r["author"] == "claude"), ("third_party_neutral", lambda r: r["condition"] == "third_party_neutral" and r["author"] == "claude"),
              ("human dist.", lambda r: r["split"] == "human_distressed"), ("human neut.", lambda r: r["split"] == "human_neutral")]
    data = [[float(r["p_distressed"]) for r in pr if f(r)] for _, f in groups]
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150, facecolor=SURF); bp = ax.boxplot(data, tick_labels=[n for n, _ in groups], patch_artist=True, widths=0.5, medianprops={"color": T1})
    for b in bp["boxes"]: b.set(facecolor=S1, alpha=0.35, edgecolor=S1)
    jrng = np.random.default_rng(1)  # jitter has its own fixed seed so F3 renders identically whether or not the fitting stage ran first
    for i, dd in enumerate(data): ax.scatter(np.full(len(dd), i + 1) + jrng.uniform(-0.12, 0.12, len(dd)), dd, s=9, color=S1, alpha=0.6, zorder=3)
    style(ax, f"P(distressed) from the task (a') probe (neutral_preamble vs distressed) at layer {best['a']}", "P(distressed)"); ax.set_xlabel(""); ax.tick_params(axis="x", labelsize=7); ax.set_ylim(-0.02, 1.02); fig.tight_layout(); fig.savefig(R / "F3_pdist_boxplots_bestlayer.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150, facecolor=SURF)
    for k, lab, c in [("cos_wa_wb", "distressed probe vs frustrated probe", S1), ("cos_wa_wd", "distressed probe vs distressed-vs-positive probe", S5), ("cos_wa_rand", "distressed probe vs random", S2), ("cos_wb_rand", "frustrated probe vs random", S3)]: ax.plot(Ls, g(k), color=c, lw=2, marker="o", ms=4, label=lab)
    ax.axhline(0, color=GRID, lw=1); style(ax, "Cosine similarity between probe directions by layer (standardised space, neutral = neutral_preamble)", "cosine"); ax.legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(R / "F4_probe_direction_cosines.png"); plt.close(fig)
    print("figures saved: F1-F4 (v2)")
