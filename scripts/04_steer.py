"""Phase 4 steering runs on the pod. Usage: python scripts/04_steer.py A|B|C|D  (resume-safe; appends to results/phase4_steered.csv)
Directions from results/directions_layer18_v2.pt (unit vectors, raw residual space, relative to neutral_preamble).
Band = blocks 12-23; strength = fraction of the prompt's mean band residual norm (steer_generate_relative); greedy; concise system prompt; 150 new tokens.
"""
import csv, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.config import MODEL_ID, CONCISE_SYSTEM_PROMPT as SYSTEM
from src.model import load_model, encode_prompt, _strip_thinking
from src.hooks import residual_norms, steer_generate_relative, relative_to_absolute_N, steering
from scripts.phase2_pool.tasks import TASKS

STAGE = sys.argv[1]; OUT = Path("results/phase4_steered.csv"); BAND = list(range(12, 24)); FRACS = [0.02, 0.04, 0.06, 0.08]; MAXT = 150
COLS = ["run", "prompt_id", "base_task", "task_type", "form", "direction", "fraction", "abs_N", "sample_idx", "temperature", "reply", "n_tokens", "seconds"]
tt = {t: k for k, v in TASKS.items() for t in v}
rows = list(csv.DictReader(open("data/phase2_prompts.csv", newline="", encoding="utf-8")))
val_bare = [r for r in rows if r["split"] == "val" and r["condition"] == "neutral"]; val_pre = [r for r in rows if r["split"] == "val" and r["condition"] == "neutral_preamble"]
val_dis = [r for r in rows if r["split"] == "val" and r["condition"] == "distressed"]; tp = [r for r in rows if r["condition"] in ("third_party", "third_party_neutral")]
all_bare = [r for r in rows if r["split"] in ("train", "val") and r["condition"] == "neutral"]; tp_claude = [r for r in rows if r["condition"] == "third_party" and r["author"] == "claude"]
D = torch.load("results/directions_layer18_v2.pt"); torch.manual_seed(0); rnd = torch.randn(4096); rnd = rnd / rnd.norm()
DIRS = {"distressed_md": D["distressed_meandiff"], "distressed_probe": D["distressed_probe"], "frustrated_md": D["frustrated_meandiff"], "positive_md": D["positive_meandiff"],
        "third_party_md": D["third_party_meandiff"], "unrelated_coding_probe": D["unrelated_coding_probe"], "random": rnd}
# K: component directions at layer 18 from the three emotional mean-differences
_M = torch.stack([D["distressed_meandiff"], D["frustrated_meandiff"], D["positive_meandiff"]]).float(); _Mc = _M - _M.mean(0, keepdim=True)
_U, _S, _Vt = torch.linalg.svd(_Mc, full_matrices=False); shared = _Vt[0]; shared = shared * torch.sign(shared @ D["distressed_meandiff"].float()); shared = shared / shared.norm()
_pos = D["positive_meandiff"].float(); valence = D["distressed_meandiff"].float() - (D["distressed_meandiff"].float() @ _pos) * _pos; valence = valence / valence.norm()
DIRS["shared_pc1"] = shared; DIRS["valence_resid"] = valence; DIRS["np_minus_bare_md"] = D["neutral_preamble_minus_bare_meandiff"]
done = set()
if OUT.exists():
    for r in csv.DictReader(open(OUT, newline="", encoding="utf-8")): done.add((r["run"], r["prompt_id"], r["form"], r["direction"], r["fraction"], r["sample_idx"]))
else:
    with open(OUT, "w", newline="", encoding="utf-8") as f: csv.DictWriter(f, fieldnames=COLS, lineterminator="\n").writeheader()
fh = open(OUT, "a", newline="", encoding="utf-8"); W = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
model, tok = load_model(); print(f"{MODEL_ID} loaded; stage {STAGE}; {len(done)} rows already done", flush=True)
n_done, t_all = 0, time.time()

def emit(run, r, form, direction, frac, absN, sample_idx, temp, reply, secs):
    global n_done
    W.writerow({"run": run, "prompt_id": r["id"], "base_task": r["base_task"], "task_type": tt.get(r["base_task"], "?"), "form": form, "direction": direction, "fraction": frac, "abs_N": round(absN, 3),
                "sample_idx": sample_idx, "temperature": temp, "reply": reply, "n_tokens": len(tok(reply)["input_ids"]), "seconds": round(secs, 2)}); fh.flush(); n_done += 1
    if n_done % 50 == 0: print(f"  {n_done} generations, {time.time()-t_all:.0f}s", flush=True)

def key(run, r, form, direction, frac, sidx=""): return (run, r["id"], form, direction, str(frac), str(sidx))

@torch.inference_mode()
def sample_generate(prompt, vec, frac, norms, seed, temperature=0.7):
    N = relative_to_absolute_N(frac, norms, BAND, vec) if frac else 0.0
    enc = {k: v.to(model.device) for k, v in encode_prompt(tok, prompt, SYSTEM).items()}
    torch.manual_seed(seed)
    with steering(model, vec, BAND, N):
        out = model.generate(**enc, max_new_tokens=MAXT, do_sample=True, temperature=temperature, top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return _strip_thinking(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)), N

def greedy(prompt, vec, frac, norms, band=None):
    return steer_generate_relative(model, tok, prompt, vec, band or BAND, frac=float(frac), norms=norms, system=SYSTEM, max_new_tokens=MAXT, return_N=True)

if STAGE == "A":
    for form, plist in [("bare", val_bare), ("preamble", val_pre)]:
        for r in plist:
            norms = residual_norms(model, tok, r["text"], SYSTEM)
            if key("A", r, form, "none", 0) not in done:
                t = time.time(); text, N = greedy(r["text"], DIRS["distressed_md"], 0.0, norms); emit("A", r, form, "none", 0, 0.0, "", "", text, time.time() - t)
            for dname, vec in DIRS.items():
                for f in FRACS:
                    if key("A", r, form, dname, f) in done: continue
                    t = time.time(); text, N = greedy(r["text"], vec, f, norms); emit("A", r, form, dname, f, N, "", "", text, time.time() - t)
elif STAGE == "B":
    for r in val_dis:
        norms = residual_norms(model, tok, r["text"], SYSTEM)
        if key("B", r, "distressed", "none", 0) not in done:
            t = time.time(); text, N = greedy(r["text"], DIRS["distressed_md"], 0.0, norms); emit("B", r, "distressed", "none", 0, 0.0, "", "", text, time.time() - t)
        for f in FRACS:
            if key("B", r, "distressed", "distressed_md_subtract", f) in done: continue
            t = time.time(); text, N = greedy(r["text"], DIRS["distressed_md"], -f, norms); emit("B", r, "distressed", "distressed_md_subtract", f, N, "", "", text, time.time() - t)
elif STAGE == "C":
    for r in tp:
        form = r["condition"] + ("_human" if r["author"] == "human" else "")
        if key("C", r, form, "none", 0) in done: continue
        norms = residual_norms(model, tok, r["text"], SYSTEM); t = time.time(); text, N = greedy(r["text"], DIRS["distressed_md"], 0.0, norms); emit("C", r, form, "none", 0, 0.0, "", "", text, time.time() - t)
elif STAGE == "D":
    for r in val_bare:
        norms = residual_norms(model, tok, r["text"], SYSTEM)
        for f in [0.0, 0.04]:
            for s in range(5):
                if key("D", r, "bare", "distressed_md", f, s) in done: continue
                t = time.time(); text, N = sample_generate(r["text"], DIRS["distressed_md"], f, norms, seed=1000 + s); emit("D", r, "bare", "distressed_md", f, N, s, 0.7, text, time.time() - t)
elif STAGE == "E":   # preamble-presence control: (neutral_preamble - bare) mean-difference on bare prompts
    vec = D["neutral_preamble_minus_bare_meandiff"]
    for r in val_bare:
        norms = residual_norms(model, tok, r["text"], SYSTEM)
        for f in FRACS:
            if key("E", r, "bare", "np_minus_bare_md", f) in done: continue
            t = time.time(); text, N = greedy(r["text"], vec, f, norms); emit("E", r, "bare", "np_minus_bare_md", f, N, "", "", text, time.time() - t)
elif STAGE == "F":   # band robustness: distressed md at 0.04 on bare prompts, bands 8-19 and 16-27
    for bname, band in [("band8-19", list(range(8, 20))), ("band16-27", list(range(16, 28)))]:
        for r in val_bare:
            if key("F", r, "bare", f"distressed_md_{bname}", 0.04) in done: continue
            norms = residual_norms(model, tok, r["text"], SYSTEM); t = time.time(); text, N = greedy(r["text"], DIRS["distressed_md"], 0.04, norms, band=band)
            emit("F", r, "bare", f"distressed_md_{bname}", 0.04, N, "", "", text, time.time() - t)
elif STAGE == "G":   # H1b: Phase 1 prompts with NO system prompt
    from src.model import chat
    pairs = list(csv.DictReader(open("data/phase1_pairs.csv", newline="", encoding="utf-8"))); outp = Path("results/phase1_replies_nosys.csv"); t0 = time.time(); rows_g = []
    for r in pairs:
        for cond in ("neutral", "stressed"):
            reply = chat(model, tok, r[cond], system=None, max_new_tokens=MAXT)
            rows_g.append({"pair_id": r["pair_id"], "task_type": r["task_type"], "condition": cond, "prompt": r[cond], "reply": reply, "n_tokens": len(tok(reply)["input_ids"])})
    with open(outp, "w", newline="", encoding="utf-8") as f: w = csv.DictWriter(f, fieldnames=list(rows_g[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows_g)
    print(f"G: {len(rows_g)} replies, no system prompt, {time.time()-t0:.0f}s -> {outp}", flush=True)
elif STAGE == "I":   # finer dose-response on bare val prompts
    for r in val_bare:
        norms = residual_norms(model, tok, r["text"], SYSTEM)
        for dname in ["distressed_md", "random", "unrelated_coding_probe", "np_minus_bare_md"]:
            for f in [0.01, 0.03, 0.05]:
                if key("I", r, "bare", dname, f) in done: continue
                t = time.time(); text, N = greedy(r["text"], DIRS[dname], f, norms); emit("I", r, "bare", dname, f, N, "", "", text, time.time() - t)
elif STAGE == "J":   # power: 0.04 on all 150 bare base tasks
    for r in all_bare:
        norms = None
        for dname in ["distressed_md", "random", "unrelated_coding_probe"]:
            if key("J", r, "bare", dname, 0.04) in done: continue
            norms = norms if norms is not None else residual_norms(model, tok, r["text"], SYSTEM)
            t = time.time(); text, N = greedy(r["text"], DIRS[dname], 0.04, norms); emit("J", r, "bare", dname, 0.04, N, "", "", text, time.time() - t)
elif STAGE == "K":   # component decomposition
    print(f"K: cos(shared, distressed_md)={float(shared @ D['distressed_meandiff'].float()):.3f} cos(valence, distressed_md)={float(valence @ D['distressed_meandiff'].float()):.3f} cos(shared, valence)={float(shared @ valence):.3f}", flush=True)
    for r in val_bare:
        norms = residual_norms(model, tok, r["text"], SYSTEM)
        for dname in ["shared_pc1", "valence_resid"]:
            for f in FRACS:
                if key("K", r, "bare", dname, f) in done: continue
                t = time.time(); text, N = greedy(r["text"], DIRS[dname], f, norms); emit("K", r, "bare", dname, f, N, "", "", text, time.time() - t)
elif STAGE == "L":   # finer subtraction + third-party subtraction
    for r in val_dis:
        norms = residual_norms(model, tok, r["text"], SYSTEM)
        for f in [0.02, 0.06]:
            if key("L", r, "distressed", "distressed_md_subtract", f) in done: continue
            t = time.time(); text, N = greedy(r["text"], DIRS["distressed_md"], -f, norms); emit("L", r, "distressed", "distressed_md_subtract", f, N, "", "", text, time.time() - t)
    for r in tp_claude:
        if key("L", r, "third_party", "third_party_md_subtract", 0.04) in done: continue
        norms = residual_norms(model, tok, r["text"], SYSTEM); t = time.time(); text, N = greedy(r["text"], DIRS["third_party_md"], -0.04, norms); emit("L", r, "third_party", "third_party_md_subtract", 0.04, N, "", "", text, time.time() - t)
elif STAGE == "M":   # sampled subtraction on distressed val prompts
    for r in val_dis:
        norms = residual_norms(model, tok, r["text"], SYSTEM)
        for f in [0.0, 0.04]:
            for sidx in range(5):
                if key("M", r, "distressed", "distressed_md_subtract", f, sidx) in done: continue
                t = time.time(); text, N = sample_generate(r["text"], DIRS["distressed_md"], -f, norms, seed=2000 + sidx); emit("M", r, "distressed", "distressed_md_subtract", f, N, sidx, 0.7, text, time.time() - t)
elif STAGE == "N":   # preamble-form headline, sampled
    for r in val_pre:
        norms = residual_norms(model, tok, r["text"], SYSTEM)
        for dname in ["distressed_md", "random", "unrelated_coding_probe"]:
            for sidx in range(5):
                if key("N", r, "preamble", dname, 0.04, sidx) in done: continue
                t = time.time(); text, N = sample_generate(r["text"], DIRS[dname], 0.04, norms, seed=3000 + sidx); emit("N", r, "preamble", dname, 0.04, N, sidx, 0.7, text, time.time() - t)
fh.close(); print(f"stage {STAGE} done: {n_done} new generations in {time.time()-t_all:.0f}s", flush=True)
