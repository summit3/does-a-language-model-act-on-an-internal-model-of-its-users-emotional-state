"""ARCHIVED (v1 probes, bare-neutral class; confounded, superseded by scripts/03b_probes_v2.py). Reads/writes results/archive/phase3_v1/.
Phase 3: per-layer linear probes on cached activations + baselines + geometry + directions.
Runs on the laptop, no model. Inputs: activations/phase2_acts_qwen3_5-9b.pt, results/phase2/phase2_activation_index.csv,
data/phase2_prompts.csv, data/phase2_qa_labels.csv. Outputs: results/phase3/phase3_*.csv, F1..F4 PNGs,
results/phase3/directions_layer<L>.pt. Preprocessing statistics come from the TRAIN split only.
"""
import csv, json, sys, time, warnings
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "results/archive/phase3_v1"  # archived: v1 outputs, superseded by scripts/03b_probes_v2.py; R2 = ROOT / "results/phase2"
rng = np.random.default_rng(0)
C_GRID = [0.001, 0.01, 0.1]; CV = StratifiedKFold(3, shuffle=True, random_state=0)
STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"   # "probes" | "figures" | "all"

# ----------------------------------------------------------------------------- data
d = torch.load(ROOT / "activations/phase2_acts_qwen3_5-9b.pt")
A = d["acts"].float().numpy(); N, NL, H = A.shape
idx = list(csv.DictReader(open(R2 / "phase2_activation_index.csv", newline="", encoding="utf-8")))
prompts = {r["id"]: r for r in csv.DictReader(open(ROOT / "data/phase2_prompts.csv", newline="", encoding="utf-8"))}
judge = {r["text"]: r["judge_label"] for r in csv.DictReader(open(ROOT / "data/phase2_qa_labels.csv", newline="", encoding="utf-8"))}
cond = np.array([r["condition"] for r in idx]); split = np.array([r["split"] for r in idx]); author = np.array([r["author"] for r in idx])
jagree = np.array([r["judge_agrees"] == "1" for r in idx]); text = np.array([prompts[r["id"]]["text"] for r in idx])
base = np.array([r["base_task"] for r in idx]); jlab = np.array([judge.get(t, "?") for t in text])
assert len(idx) == N
def m(**kw):
    mask = np.ones(N, bool)
    for k, v in kw.items():
        arr = {"cond": cond, "split": split, "author": author}[k]; mask &= np.isin(arr, v if isinstance(v, (list, tuple)) else [v])
    return mask
TR = m(split="train"); VA = m(split="val")
S = {"train": TR, "val": VA,
     "val_neutral": VA & m(cond="neutral"), "val_distressed": VA & m(cond="distressed"), "val_frustrated": VA & m(cond="frustrated"),
     "implied": m(cond="implied", author="claude"), "implied_neutral": m(split="implied", cond="neutral"),
     "third_party": m(cond="third_party", author="claude"), "third_party_neutral": m(cond="third_party_neutral", author="claude"),
     "human_neutral": m(split="human_neutral"), "human_distressed": m(split="human_distressed"), "human_frustrated": m(split="human_frustrated"),
     "human_implied": m(split="human_implied"), "human_third_party": m(split="human_third_party"), "human_third_party_neutral": m(split="human_third_party_neutral"),
     # judge-missed explicit rows, excluding train (so they are held out from the probe)
     "jm_distressed": (~jagree) & m(cond="distressed") & ~TR, "jm_frustrated": (~jagree) & m(cond="frustrated") & ~TR,
     "jm_distressed_train": (~jagree) & m(cond="distressed") & TR, "jm_frustrated_train": (~jagree) & m(cond="frustrated") & TR}
print("rows per split used:", {k: int(v.sum()) for k, v in S.items()})
Y3 = np.array([{"neutral": 0, "distressed": 1, "frustrated": 2}.get(c, -1) for c in cond])

def prep(layer):
    X = A[:, layer, :]; mu = X[TR].mean(0); sd = X[TR].std(0) + 1e-6
    return (X - mu) / sd, mu, sd

def fit(X, y, cv=True):
    if cv:
        g = GridSearchCV(LogisticRegression(max_iter=3000), {"C": C_GRID}, cv=CV, scoring="balanced_accuracy", n_jobs=-1).fit(X, y)
        return g.best_estimator_, g.best_params_["C"]
    return LogisticRegression(C=0.01, max_iter=3000).fit(X, y), 0.01

def ba(clf, X, y): return balanced_accuracy_score(y, clf.predict(X)) if len(set(y)) > 1 else np.nan
def rec(clf, X, pos=1): return float((clf.predict(X) == pos).mean()) if X.shape[0] else np.nan
def pdis(clf, X): return clf.predict_proba(X)[:, list(clf.classes_).index(1)]

def eval_a(clf, X, tag):
    """task (a) neutral(0) vs distressed(1)."""
    y = (cond == "distressed").astype(int); o = {}
    o[f"{tag}_train"] = ba(clf, X[TR & m(cond=["neutral", "distressed"])], y[TR & m(cond=["neutral", "distressed"])])
    v = VA & m(cond=["neutral", "distressed"]); o[f"{tag}_val"] = ba(clf, X[v], y[v])
    im = S["implied"] | S["implied_neutral"]; o[f"{tag}_implied"] = ba(clf, X[im], (cond[im] == "implied").astype(int)); o[f"{tag}_implied_recall"] = rec(clf, X[S["implied"]])
    hu = S["human_neutral"] | S["human_distressed"]; o[f"{tag}_human"] = ba(clf, X[hu], y[hu]); o[f"{tag}_human_recall"] = rec(clf, X[S["human_distressed"]])
    o[f"{tag}_jm_recall"] = rec(clf, X[S["jm_distressed"]]); jp = S["jm_distressed"] | S["val_neutral"]; o[f"{tag}_jm_pooled"] = ba(clf, X[jp], y[jp])
    o[f"{tag}_jm_train_recall"] = rec(clf, X[S["jm_distressed_train"]])
    for g in ["val_neutral", "val_distressed", "implied", "third_party", "third_party_neutral", "human_third_party", "human_third_party_neutral", "human_implied"]:
        o[f"{tag}_pdist_{g}"] = float(pdis(clf, X[S[g]]).mean())
    return o

def eval_b(clf, X, tag):
    y = (cond == "frustrated").astype(int); o = {}
    o[f"{tag}_train"] = ba(clf, X[TR & m(cond=["neutral", "frustrated"])], y[TR & m(cond=["neutral", "frustrated"])])
    v = VA & m(cond=["neutral", "frustrated"]); o[f"{tag}_val"] = ba(clf, X[v], y[v])
    hu = S["human_neutral"] | S["human_frustrated"]; o[f"{tag}_human"] = ba(clf, X[hu], y[hu]); o[f"{tag}_human_recall"] = rec(clf, X[S["human_frustrated"]])
    o[f"{tag}_jm_recall"] = rec(clf, X[S["jm_frustrated"]]); jp = S["jm_frustrated"] | S["val_neutral"]; o[f"{tag}_jm_pooled"] = ba(clf, X[jp], y[jp])
    return o

def eval_c(clf, X, tag):
    o = {}; tr = TR; o[f"{tag}_train"] = ba(clf, X[tr], Y3[tr]); o[f"{tag}_val"] = ba(clf, X[VA], Y3[VA])
    hu = S["human_neutral"] | S["human_distressed"] | S["human_frustrated"]; o[f"{tag}_human"] = ba(clf, X[hu], Y3[hu])
    return o

# ----------------------------------------------------------------------------- probes per layer
if STAGE in ("probes", "all"):
    rows, t0 = [], time.time(); models = {}
    for L in range(NL):
        X, mu, sd = prep(L); row = {"layer": L}
        ma = TR & m(cond=["neutral", "distressed"]); clf_a, Ca = fit(X[ma], (cond[ma] == "distressed").astype(int)); row.update(eval_a(clf_a, X, "a")); row["a_C"] = Ca
        mb = TR & m(cond=["neutral", "frustrated"]); clf_b, Cb = fit(X[mb], (cond[mb] == "frustrated").astype(int)); row.update(eval_b(clf_b, X, "b")); row["b_C"] = Cb
        clf_c, Cc = fit(X[TR], Y3[TR]); row.update(eval_c(clf_c, X, "c")); row["c_C"] = Cc
        # shuffled-label baseline (fixed C, no CV)
        ys = rng.permutation((cond[ma] == "distressed").astype(int)); sh, _ = fit(X[ma], ys, cv=False)
        v = VA & m(cond=["neutral", "distressed"]); row["shuffled_a_val"] = ba(sh, X[v], (cond[v] == "distressed").astype(int))
        ysb = rng.permutation((cond[mb] == "frustrated").astype(int)); shb, _ = fit(X[mb], ysb, cv=False)
        vb = VA & m(cond=["neutral", "frustrated"]); row["shuffled_b_val"] = ba(shb, X[vb], (cond[vb] == "frustrated").astype(int))
        # direction cosines (probe weights, standardised space)
        wa, wb = clf_a.coef_[0], clf_b.coef_[0]; rnd = rng.standard_normal(H)
        cs = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        row["cos_wa_wb"] = cs(wa, wb); row["cos_wa_rand"] = cs(wa, rnd); row["cos_wb_rand"] = cs(wb, rnd)
        models[L] = (clf_a, clf_b, clf_c, mu, sd); rows.append(row)
        print(f"L{L:02d} a_val={row['a_val']:.2f} a_impl={row['a_implied']:.2f} a_hum={row['a_human']:.2f} a_jm={row['a_jm_recall']:.2f} | b_val={row['b_val']:.2f} | c_val={row['c_val']:.2f} | shuf={row['shuffled_a_val']:.2f} | {time.time()-t0:.0f}s", flush=True)
    keys = list(rows[0].keys())
    with open(R / "phase3_probe_by_layer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    # best layer: max val BA; ties (val is at ceiling on many layers) broken by mean held-out BA (implied + human for a; human for b, c)
    def key(t):
        def k(r):
            held = (r["a_implied"] + r["a_human"]) / 2 if t == "a" else r[f"{t}_human"]
            return (round(r[f"{t}_val"], 3), round(held, 3), -abs(r["layer"] - NL // 2))
        return k
    best = {t: max(rows, key=key(t))["layer"] for t in "abc"}
    json.dump({**best, "rule": "argmax val BA; ties broken by mean held-out BA (a: implied+human; b,c: human), then closeness to mid-depth"}, open(R / "phase3_best_layers.json", "w"))
    print("best layers (val, tie-break held-out):", best)

    # ------------------------------------------------------------------------- baselines: BoW, length, ask-the-model
    bl = []
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(text[TR]); XT = tf.transform(text)
    ba_bow, _ = fit(XT[ma], (cond[ma] == "distressed").astype(int)); bl.append({"baseline": "bow_tfidf", **eval_a(ba_bow, XT, "a")})
    bb_bow, _ = fit(XT[mb], (cond[mb] == "frustrated").astype(int)); bl[-1].update(eval_b(bb_bow, XT, "b"))
    bc_bow, _ = fit(XT[TR], Y3[TR]); bl[-1].update(eval_c(bc_bow, XT, "c"))
    # length in tokens (Qwen tokenizer if available, else words)
    try:
        from transformers import AutoTokenizer; tk = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B"); ln = np.array([len(tk(t)["input_ids"]) for t in text], float); lunit = "tokens"
    except Exception as e:
        ln = np.array([len(t.split()) for t in text], float); lunit = f"words (tokenizer unavailable: {type(e).__name__})"
    XL = ((ln - ln[TR].mean()) / ln[TR].std())[:, None]
    la, _ = fit(XL[ma], (cond[ma] == "distressed").astype(int)); bl.append({"baseline": f"length_{lunit.split()[0]}", **eval_a(la, XL, "a")})
    lb, _ = fit(XL[mb], (cond[mb] == "frustrated").astype(int)); bl[-1].update(eval_b(lb, XL, "b")); lc, _ = fit(XL[TR], Y3[TR]); bl[-1].update(eval_c(lc, XL, "c"))
    # ask-the-model: judge labels
    jd = (jlab == "distressed").astype(int); jf = (jlab == "frustrated").astype(int); o = {"baseline": "ask_the_model_judge"}
    v = VA & m(cond=["neutral", "distressed"]); o["a_val"] = balanced_accuracy_score((cond[v] == "distressed").astype(int), jd[v])
    im = S["implied"] | S["implied_neutral"]; o["a_implied"] = balanced_accuracy_score((cond[im] == "implied").astype(int), jd[im]); o["a_implied_recall"] = float(jd[S["implied"]].mean())
    hu = S["human_neutral"] | S["human_distressed"]; o["a_human"] = balanced_accuracy_score((cond[hu] == "distressed").astype(int), jd[hu]); o["a_human_recall"] = float(jd[S["human_distressed"]].mean())
    o["a_jm_recall"] = 0.0
    for g in ["val_neutral", "val_distressed", "implied", "third_party", "third_party_neutral", "human_third_party", "human_third_party_neutral", "human_implied"]: o[f"a_pdist_{g}"] = float(jd[S[g]].mean())
    vb = VA & m(cond=["neutral", "frustrated"]); o["b_val"] = balanced_accuracy_score((cond[vb] == "frustrated").astype(int), jf[vb])
    hub = S["human_neutral"] | S["human_frustrated"]; o["b_human"] = balanced_accuracy_score((cond[hub] == "frustrated").astype(int), jf[hub]); o["b_jm_recall"] = 0.0
    j3 = np.array([{"neutral": 0, "distressed": 1, "frustrated": 2}.get(x, 0) for x in jlab]); o["c_val"] = balanced_accuracy_score(Y3[VA], j3[VA])
    bl.append(o)
    bkeys = sorted({k for b in bl for k in b}, key=lambda k: (k != "baseline", k))
    with open(R / "phase3_baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bkeys, lineterminator="\n"); w.writeheader(); w.writerows(bl)
    print("baselines:"); [print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items() if k in ("baseline", "a_val", "a_implied", "a_human", "a_jm_recall", "b_val", "c_val", "a_pdist_third_party", "a_pdist_third_party_neutral")}) for b in bl]

    # ------------------------------------------------------------------------- per-row P(distressed) at best layer; headline
    La = best["a"]; clf_a, clf_b, clf_c, mu, sd = models[La]; X = (A[:, La, :] - mu) / sd; p = pdis(clf_a, X)
    with open(R / "phase3_pdist_bestlayer.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["row", "id", "condition", "split", "author", "judge_agrees", "p_distressed"])
        for i, r in enumerate(idx): w.writerow([i, r["id"], r["condition"], r["split"], r["author"], r["judge_agrees"], f"{p[i]:.4f}"])
    br = rows[La]; bb = rows[best["b"]]; bc = rows[best["c"]]; bow = bl[0]
    head = [["task (a) neutral vs distressed, layer", La], ["a val BA", br["a_val"]], ["a train BA", br["a_train"]], ["a implied BA (with matched neutrals)", br["a_implied"]],
            ["a implied recall", br["a_implied_recall"]], ["a human BA / recall", f"{br['a_human']:.3f} / {br['a_human_recall']:.3f}"],
            ["a judge-missed (non-train) recall / pooled BA (n=%d)" % S["jm_distressed"].sum(), f"{br['a_jm_recall']:.3f} / {br['a_jm_pooled']:.3f}"],
            ["a judge-missed (train, in-sample) recall (n=%d)" % S["jm_distressed_train"].sum(), br["a_jm_train_recall"]],
            ["a mean P(distressed): val_neutral / val_distressed / implied / third_party / third_party_neutral", " / ".join(f"{br[f'a_pdist_{g}']:.2f}" for g in ["val_neutral", "val_distressed", "implied", "third_party", "third_party_neutral"])],
            ["a shuffled-label val BA at this layer", br["shuffled_a_val"]],
            ["task (b) neutral vs frustrated, layer", best["b"]], ["b val BA / train BA", f"{bb['b_val']:.3f} / {bb['b_train']:.3f}"], ["b human BA / judge-missed recall", f"{bb['b_human']:.3f} / {bb['b_jm_recall']:.3f}"],
            ["task (c) 3-way, layer", best["c"]], ["c val BA / train BA / human BA", f"{bc['c_val']:.3f} / {bc['c_train']:.3f} / {bc['c_human']:.3f}"],
            ["BoW baseline a val / implied / human / jm recall", f"{bow['a_val']:.3f} / {bow['a_implied']:.3f} / {bow['a_human']:.3f} / {bow['a_jm_recall']:.3f}"],
            ["BoW baseline b val / c val", f"{bow['b_val']:.3f} / {bow['c_val']:.3f}"],
            ["length baseline a val / b val / c val", f"{bl[1]['a_val']:.3f} / {bl[1]['b_val']:.3f} / {bl[1]['c_val']:.3f}"],
            ["ask-the-model a val / implied / human / b val / c val", f"{bl[2]['a_val']:.3f} / {bl[2]['a_implied']:.3f} / {bl[2]['a_human']:.3f} / {bl[2]['b_val']:.3f} / {bl[2]['c_val']:.3f}"]]
    with open(R / "phase3_headline.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["item", "value"]); [w.writerow([k, (f"{v:.3f}" if isinstance(v, float) else v)]) for k, v in head]
    print("\nHEADLINE"); [print(f"  {k}: {(f'{v:.3f}' if isinstance(v, float) else v)}") for k, v in head]
    print(f"sanity: shuffled a_val mean over layers = {np.mean([r['shuffled_a_val'] for r in rows]):.3f}; train-val gap at L{La} = {br['a_train']-br['a_val']:.3f}")

    # ------------------------------------------------------------------------- directions + geometry at best layer (raw residual space)
    Xr = A[:, La, :]; nmean = lambda mask: Xr[mask].mean(0)
    md_d = nmean(TR & m(cond="distressed")) - nmean(TR & m(cond="neutral")); md_f = nmean(TR & m(cond="frustrated")) - nmean(TR & m(cond="neutral"))
    tp_tasks = set(base[S["third_party"]]); tpn_tasks = set(base[S["third_party_neutral"]])
    neu_tp = m(cond="neutral") & np.isin(base, list(tp_tasks)) & np.isin(split, ["train", "val"]); neu_tpn = m(cond="neutral") & np.isin(base, list(tpn_tasks)) & np.isin(split, ["train", "val"])
    md_tp = nmean(S["third_party"]) - nmean(neu_tp); md_tpn = nmean(S["third_party_neutral"]) - nmean(neu_tpn)
    w_a_raw = clf_a.coef_[0] / sd; w_b_raw = clf_b.coef_[0] / sd          # standardised-space weights -> raw-space direction
    yc = np.array([("Python" in t or "pip" in t) for t in text]).astype(int)   # coding vs not (task_type proxy from text)
    Xs = (Xr - mu) / sd; clf_u, _ = fit(Xs[TR], yc[TR]); w_u_raw = clf_u.coef_[0] / sd
    unit = lambda v: (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
    cs = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    anyemo = md_d / np.linalg.norm(md_d) + md_f / np.linalg.norm(md_f)
    M = np.stack([md_d / np.linalg.norm(md_d), md_f / np.linalg.norm(md_f)]); U, Sv, Vt = np.linalg.svd(M, full_matrices=False)
    geo = {"layer": La, "cos_md_d_md_f": cs(md_d, md_f), "cos_probe_a_md_d": cs(w_a_raw, md_d), "cos_probe_b_md_f": cs(w_b_raw, md_f), "cos_probe_a_probe_b": cs(w_a_raw, w_b_raw),
           "cos_md_d_anyemo": cs(md_d, anyemo), "cos_md_f_anyemo": cs(md_f, anyemo), "pca_var_explained_pc1": float(Sv[0]**2 / (Sv**2).sum()),
           "cos_md_tp_md_d": cs(md_tp, md_d), "cos_md_tpn_md_d": cs(md_tpn, md_d), "cos_md_tp_md_tpn": cs(md_tp, md_tpn), "cos_unrelated_md_d": cs(w_u_raw, md_d), "cos_unrelated_probe_a": cs(w_u_raw, w_a_raw),
           "norm_md_d": float(np.linalg.norm(md_d)), "norm_md_f": float(np.linalg.norm(md_f)), "norm_md_tp": float(np.linalg.norm(md_tp)), "norm_md_tpn": float(np.linalg.norm(md_tpn)),
           "unrelated_probe_val_BA": ba(clf_u, Xs[VA], yc[VA]), "mean_resid_norm_train": float(np.linalg.norm(Xr[TR], axis=1).mean())}
    with open(R / f"phase3_geometry_layer{La}.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["item", "value"]); [w.writerow([k, f"{v:.4f}" if isinstance(v, float) else v]) for k, v in geo.items()]
    torch.save({"layer": La, "model": d["model"], "space": "raw residual stream (output of block L, last prompt token)",
                "distressed_meandiff": torch.tensor(unit(md_d)), "frustrated_meandiff": torch.tensor(unit(md_f)),
                "distressed_probe": torch.tensor(unit(w_a_raw)), "frustrated_probe": torch.tensor(unit(w_b_raw)),
                "third_party_meandiff": torch.tensor(unit(md_tp)), "third_party_neutral_meandiff": torch.tensor(unit(md_tpn)),
                "unrelated_coding_probe": torch.tensor(unit(w_u_raw)), "anyemotion": torch.tensor(unit(anyemo)),
                "train_mean": torch.tensor(mu.astype(np.float32)), "train_std": torch.tensor(sd.astype(np.float32)), "geometry": geo}, R / f"directions_layer{La}.pt")
    print("geometry:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in geo.items()}); print(f"saved results/phase3/directions_layer{La}.pt")

# ----------------------------------------------------------------------------- figures
if STAGE in ("figures", "all"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(R / "phase3_probe_by_layer.csv"))); bl = list(csv.DictReader(open(R / "phase3_baselines.csv"))); best = json.load(open(R / "phase3_best_layers.json"))
    best = {k: v for k, v in best.items() if k in "abc"}
    g = lambda k: np.array([float(r[k]) for r in rows]); Ls = g("layer"); bow = bl[0]
    S1, S2, S3, S4, SURF, T1, T2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"
    def style(ax, title, yl):
        ax.set_facecolor(SURF); ax.set_title(title, fontsize=10, color=T1, loc="left"); ax.set_xlabel("decoder block i (activation = output of block i)", color=T2, fontsize=9); ax.set_ylabel(yl, color=T2, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ("top", "right")]; [ax.spines[s].set_color(GRID) for s in ("left", "bottom")]; ax.tick_params(colors=T2, labelsize=8)
    for tag, F, title, series in [("a", "F1", "Task (a) neutral vs distressed: balanced accuracy by layer", [("a_val", "val", S1), ("a_implied", "implied (+matched neutrals)", S2), ("a_human", "human", S3), ("a_jm_recall", "judge-missed recall (non-train)", S4)]),
                                  ("b", "F2", "Task (b) neutral vs frustrated: balanced accuracy by layer", [("b_val", "val", S1), ("b_human", "human", S3), ("b_jm_recall", "judge-missed recall (non-train)", S4)])]:
        fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150, facecolor=SURF)
        for k, lab, c in series: ax.plot(Ls, g(k), color=c, lw=2, marker="o", ms=4, label=lab)
        ax.axhline(float(bow[f"{tag}_val"]), color=T2, lw=1.5, ls="--", label=f"bag-of-words val ({float(bow[f'{tag}_val']):.2f})"); ax.axhline(0.5, color=GRID, lw=1)
        ax.set_ylim(0.3, 1.02); style(ax, title, "balanced accuracy / recall"); ax.legend(fontsize=8, frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(R / f"{F}_v1_confounded_probe_acc_by_layer_task_{tag}.png"); plt.close(fig)
    pd_rows = list(csv.DictReader(open(R / "phase3_pdist_bestlayer.csv"))); groups = [("val neutral", lambda r: r["split"] == "val" and r["condition"] == "neutral"), ("val distressed", lambda r: r["split"] == "val" and r["condition"] == "distressed"),
              ("third_party", lambda r: r["condition"] == "third_party" and r["author"] == "claude"), ("third_party_neutral", lambda r: r["condition"] == "third_party_neutral" and r["author"] == "claude"), ("implied", lambda r: r["condition"] == "implied" and r["author"] == "claude")]
    data = [[float(r["p_distressed"]) for r in pd_rows if f(r)] for _, f in groups]
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150, facecolor=SURF); bp = ax.boxplot(data, tick_labels=[n for n, _ in groups], patch_artist=True, widths=0.5, medianprops={"color": T1})
    for b in bp["boxes"]: b.set(facecolor=S1, alpha=0.35, edgecolor=S1)
    for i, dd in enumerate(data): ax.scatter(np.full(len(dd), i + 1) + rng.uniform(-0.12, 0.12, len(dd)), dd, s=10, color=S1, alpha=0.6, zorder=3)
    style(ax, f"P(distressed) from the task (a) probe at layer {best['a']}", "P(distressed)"); ax.set_xlabel(""); ax.set_ylim(-0.02, 1.02); fig.tight_layout(); fig.savefig(R / "F3_v1_confounded_pdist_boxplots_bestlayer.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150, facecolor=SURF)
    ax.plot(Ls, g("cos_wa_wb"), color=S1, lw=2, marker="o", ms=4, label="distressed probe vs frustrated probe"); ax.plot(Ls, g("cos_wa_rand"), color=S2, lw=2, marker="o", ms=4, label="distressed probe vs random"); ax.plot(Ls, g("cos_wb_rand"), color=S3, lw=2, marker="o", ms=4, label="frustrated probe vs random")
    ax.axhline(0, color=GRID, lw=1); style(ax, "Cosine similarity between probe directions by layer (standardised space)", "cosine"); ax.legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(R / "F4_v1_confounded_probe_direction_cosines.png"); plt.close(fig)
    print("figures saved: F1-F4")
