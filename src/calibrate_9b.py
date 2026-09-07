"""Layer-convention check, residual norms, playground stress direction and calibration sweep
on the pod model. Run from the repo root with the venv active: python -m src.calibrate_9b
Writes results/9b_residual_norms.{csv,png} and results/9b_calibration_sweep.md.
"""
import csv, sys, time
from pathlib import Path
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from .config import MODEL_ID, CONCISE_SYSTEM_PROMPT
from .model import load_model, chat, get_decoder_layers
from .hooks import (get_residual_activations, describe_layers, residual_norms,
                    steer_generate_relative, relative_to_absolute_N)

RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
ACTS = Path("activations"); ACTS.mkdir(exist_ok=True)
PROMPT = "What's the capital of Australia?"
SYSTEM = CONCISE_SYSTEM_PROMPT
PAIRS = [
    ("What's 17% of 340?", "I've been up all night stressing about money and I can't think straight. What's 17% of 340?"),
    ("Explain how compound interest works.", "I'm panicking about my loans and can't focus. Explain how compound interest works."),
    ("What's the capital of Australia?", "Honestly I'm falling apart today. What's the capital of Australia?"),
    ("How do I reverse a list in Python?", "Everything's going wrong and I'm so anxious. How do I reverse a list in Python?"),
    ("Is it true we only use 10% of our brains?", "I've had the worst week and I'm barely holding it together. Is it true we only use 10% of our brains?"),
]
FRACS = [0.015, 0.03, 0.06, 0.1]

def hr(t): print(f"\n===== {t} =====")

t0 = time.time(); model, tok = load_model(); print(f"loaded {MODEL_ID} on {model.device} in {time.time()-t0:.0f}s")
n_layers = model.config.get_text_config().num_hidden_layers

hr("STEP 1 layer convention"); info = describe_layers(model, tok, PROMPT)

hr("STEP 2 warm-up + norms")
t0 = time.time(); chat(model, tok, "Say hi.", max_new_tokens=5); print(f"warm-up generation {time.time()-t0:.1f}s")
t0 = time.time(); acts = get_residual_activations(model, tok, PROMPT); print(f"activations shape {tuple(acts.shape)} in {time.time()-t0:.2f}s")
norms = acts.norm(dim=-1); y = norms.tolist()
with open(RESULTS/"9b_residual_norms.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["layer", "l2_norm"]); [w.writerow([i, f"{v:.3f}"]) for i, v in enumerate(y)]
S1, SURF, T1, T2, GRID = "#2a78d6", "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"
fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150, facecolor=SURF); ax.set_facecolor(SURF); x = list(range(n_layers))
ax.plot(x, y, color=S1, lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=3)
ax.scatter(x, y, s=36, color=S1, edgecolors=SURF, linewidths=1.5, zorder=4)
ax.annotate(f"{y[-1]:.0f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=T1)
ax.set_title(f"Residual-stream L2 norm at the last prompt token, by layer\n{MODEL_ID}, prompt: \"{PROMPT}\"", fontsize=10, color=T1, loc="left")
ax.set_xlabel("decoder block i (activation = output of block i, 0-based)", color=T2, fontsize=9); ax.set_ylabel("L2 norm", color=T2, fontsize=9)
ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
for s_ in ("left", "bottom"): ax.spines[s_].set_color(GRID)
ax.tick_params(colors=T2, labelsize=8); ax.set_xticks(range(0, n_layers, 4)); ax.set_ylim(bottom=0)
fig.tight_layout(); fig.savefig(RESULTS/"9b_residual_norms.png")
band = list(range(n_layers//3, 2*n_layers//3 + 1))   # middle third: 10..21 for 32 layers
mid = list(range(11, 22))
print("norms:", [round(v, 1) for v in y])
print(f"mean norm over layers 11-21: {norms[mid].mean():.1f} | over band {band[0]}-{band[-1]}: {norms[band].mean():.1f}")

hr("STEP 3 stress direction")
L = 21   # middle-late; chosen from the norm plot: past the mid-depth growth but well before the final-layer blow-up
neu = torch.stack([get_residual_activations(model, tok, n)[L] for n, s in PAIRS])
stv = torch.stack([get_residual_activations(model, tok, s)[L] for n, s in PAIRS])
mu = torch.cat([neu, stv]).mean(0)                 # mean-centre with the pooled mean
neu_c, stv_c = neu - mu, stv - mu
direction = stv_c.mean(0) - neu_c.mean(0); direction = direction / direction.norm()
torch.manual_seed(0); rand = torch.randn_like(direction); rand = rand / rand.norm()
print(f"layer L={L}; ||mean diff|| before normalising = {(stv_c.mean(0)-neu_c.mean(0)).norm():.1f}; mean residual norm at L = {norms[L]:.1f}")
print(f"cos(direction, random unit) = {float(direction @ rand):.4f}   (1/sqrt({direction.numel()}) = {1/direction.numel()**0.5:.4f})")
print("centred neutral  projections:", [round(float(v), 1) for v in neu_c @ direction])
print("centred stressed projections:", [round(float(v), 1) for v in stv_c @ direction])
cos_pairs = torch.nn.functional.cosine_similarity(neu_c, stv_c, dim=-1)
print("centred cos(neutral_i, stressed_i):", [round(float(v), 3) for v in cos_pairs])
torch.save({"direction": direction, "layer": L, "pairs": PAIRS, "model": MODEL_ID}, ACTS/f"9b_stress_direction_L{L}.pt")

hr("STEP 4 calibration sweep")
pnorms = residual_norms(model, tok, PROMPT, SYSTEM)
lines = [f"# 9B calibration sweep\n", f"- model: `{MODEL_ID}`, {n_layers} layers, hidden {direction.numel()}",
         f"- band: layers {band[0]}-{band[-1]} (middle third; activation i = output of block i, 0-based)",
         f"- mean last-token residual norm over band (with system prompt): {pnorms[band].mean():.1f}",
         f"- prompt: \"{PROMPT}\"; system prompt: \"{SYSTEM}\"; greedy, max_new_tokens=80",
         f"- stress direction: mean-difference at layer {L} from {len(PAIRS)} playground pairs, mean-centred, unit norm",
         f"- fraction f means N*||v|| = f * mean band norm (`steer_generate_relative`)\n"]
base = chat(model, tok, PROMPT, system=SYSTEM, max_new_tokens=80)
lines += ["## Baseline (no steering)\n", "```", base, "```\n"]; print("baseline:", base)
for name, vec in [("Random unit direction (control)", rand), ("Stress direction", direction)]:
    lines.append(f"## {name}\n")
    for f in FRACS:
        text, N = steer_generate_relative(model, tok, PROMPT, vec, band, frac=f, norms=pnorms, system=SYSTEM, max_new_tokens=80, return_N=True)
        lines += [f"### f = {f}  (absolute N = {N:.1f})\n", "```", text, "```\n"]
        print(f"--- {name} f={f} N={N:.1f} ---\n{text}\n")
(RESULTS/"9b_calibration_sweep.md").write_text("\n".join(lines))
print("wrote results/9b_calibration_sweep.md, results/9b_residual_norms.{csv,png}")
