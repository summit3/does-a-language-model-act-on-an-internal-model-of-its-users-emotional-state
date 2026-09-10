"""Assemble results/writeup/: one folder per write-up section, holding COPIES of the figures/tables that section needs plus a
few derived files (per-run rate table, quoted replies, prompt counts). Originals stay where they are; rerun after any change.
Usage: python scripts/09_assemble_writeup.py. Laptop-only; no model needed."""
import csv, json, re, shutil, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results/writeup"; R = ROOT / "results"
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir()

# ---------- shared data ----------
prompts = {r["id"]: r for r in csv.DictReader(open(ROOT / "data/phase2_prompts.csv", newline="", encoding="utf-8"))}
steered = list(csv.DictReader(open(R / "phase4/phase4_steered.csv", newline="", encoding="utf-8")))
scores = list(csv.DictReader(open(R / "phase4/phase4_scores.csv", newline="", encoding="utf-8")))
SK = lambda r: (r["run"], r["prompt_id"], r["form"], r["direction"], r["fraction"], r["sample_idx"])
score_by = {SK(r): r for r in scores}
def flags(r):
    s = score_by.get(SK(r)); return "" if s is None else f"correct={s['correct'] or 'n/a'} abandoned={s['task_abandoned']} ack={s['acknowledges_emotion']} infer={s['user_state_inference']}({s['user_state_inference_strength']}) incoh={s['incoherent']} leak={s['affect_leak_lexical']}"
def dedupe(rs):
    seen = set(); return [r for r in rs if not (SK(r) in seen or seen.add(SK(r)))]
def quote(r):
    p = prompts.get(r["prompt_id"], {}); hdr = f"**{r['run']}/{r['prompt_id']}** [{r['form']}, {r['direction']} @ {r['fraction']}" + (f", sample {r['sample_idx']}" if r["sample_idx"] else "") + f"] {flags(r)}"
    return f"- {hdr}\n  - prompt: *{p.get('text', r['base_task'])}*\n  - reply: > {r['reply'].replace(chr(10), ' / ')}\n"
def rows(**kw):
    out = []
    for r in steered:
        ok = True
        for k, v in kw.items():
            vals = v if isinstance(v, (list, tuple, set)) else [v]
            if k == "task_contains": ok &= any(t.lower() in r["base_task"].lower() for t in vals)
            elif k == "prompt_contains": ok &= any(t.lower() in prompts.get(r["prompt_id"], {}).get("text", "").lower() for t in vals)
            else: ok &= r[k] in [str(x) for x in vals]
        if ok: out.append(r)
    return out
def md_table(header, body): return "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in body) + "\n"

# per-run rate table from phase4_scores.csv (rule scores v3)
def rate_table(runs=None):
    g = collections.defaultdict(list)
    for s in scores:
        if runs is None or s["run"] in runs: g[(s["run"], s["form"], s["direction"], s["fraction"])].append(s)
    body = []
    for k in sorted(g, key=lambda k: (k[0], k[1], k[2], float(k[3]))):
        rs = g[k]; closed = [r for r in rs if r["correct"] != ""]; m = lambda col, sub=rs: f"{sum(float(r[col]) for r in sub) / len(sub):.3f}" if sub else "n/a"
        body.append([*k, len(rs), f"{m('correct', closed)} (n={len(closed)})", m("task_abandoned"), m("acknowledges_emotion"), m("user_state_inference"), m("incoherent"), m("affect_leak_lexical"), f"{sum(float(r['n_tokens']) for r in rs) / len(rs):.1f}"])
    return md_table(["run", "form", "direction", "fraction", "n", "correct", "abandons", "acknowledges", "infers", "incoherent", "leak (lexical)", "mean tokens"], body)

def extract_section(path, heading, until_level):
    L = open(path, encoding="utf-8").read().split("\n"); out, on = [], False
    for ln in L:
        if ln.startswith(heading): on = True
        elif on and re.match(rf"^#{{1,{until_level}}} ", ln): break
        if on: out.append(ln)
    return "\n".join(out).strip() + "\n"

sysprompt = re.search(r'CONCISE_SYSTEM_PROMPT = "(.*)"', open(ROOT / "src/config.py", encoding="utf-8").read())[1]
counts = collections.Counter((p["condition"], p["split"], "generated" if p["author"] == "claude" else "human") for p in prompts.values())
prompt_counts = "# Appendix B. Prompt counts: condition x split x source (data/phase2_prompts.csv)\n\n" + md_table(["condition", "split", "source", "n"], [[*k, v] for k, v in sorted(counts.items())]) + f"\nTotal {len(prompts)} rows; generated {sum(v for k, v in counts.items() if k[2] == 'generated')}, human {sum(v for k, v in counts.items() if k[2] == 'human')}.\n"
run_table = extract_section(R / "README.md", "## phase4/ runs", 2)

# ---------- sections ----------
S = []  # (folder, title, plan, copies[(src, note)], generated{name: text})
S.append(("03_setup", "3. Setup (3.1 model and compute, 3.3 directions and controls, 3.4 steering and generation, 3.5 metrics)", [
    "3.3 distressed_md = mean(train distressed) - mean(train neutral_preamble), layer 18, last prompt token; no bare row entered. Controls: random, coding probe, np_minus_bare (E), frustrated/positive, third_party, shared_pc1/valence_resid (K). Raw cosine 0.85 is not the F4 quantity.",
    "3.4 band 12-23, fraction of per-prompt mean band norm, 150 tokens, system prompt verbatim (system_prompt.txt), greedy; sampled T=0.7 top_p=1 x 5 seeds; run G no system prompt; run table -> Appendix A.",
    "3.5 five flags + leak (lexicon frozen at 083ba00); v1 -> v3 one line -> section 6."], [
    ("results/phase3/directions_summary.csv", "per-direction summary of directions_layer18_v2.pt"),
    ("results/phase3/phase3b_geometry_layer18.csv", "raw-space cosines at layer 18 (cos(distressed md, frustrated md) 0.85 lives here)"),
    ("results/phase3/phase3b_best_layers.json", "layer choice rule"),
    ("results/phase4/phase4_keyword_lists.json", "rule lexicons (3.5)"),
    ("data/affect_leak_lexicon.txt", "frozen leak lexicon, 95 words"),
    ("data/phase1_scoring_rules.md", "metric definitions"),
    ("results/playground/9b_residual_norms.png", "3.1/3.4: residual norm by block, basis of the band and relative-strength convention"),
    ("results/playground/9b_residual_norms.csv", ""),
    ("results/playground/9b_calibration_sweep.md", "calibration sweep (in-sample, n=1; cite only as calibration)")],
    {"system_prompt.txt": sysprompt + "\n", "run_table.md": "# Appendix A. Phase 4 runs\n\n" + run_table, "prompt_counts.md": prompt_counts}))

S.append(("04_1_headline", "4.1 Headline (T1, F9)", ["Run J: 0.04 on all 150 bare base tasks, greedy, rules v3; three directions."], [
    ("results/phase4/figures/F9_headline_J_bar.png", "headline bar chart (v3)"),
    ("results/phase4/figures/T1_headline_J_v1_v3.png", "T1 as image"),
    ("results/phase4/figures/T1_headline_J_v1_v3.md", "T1 markdown"),
    ("results/phase4/phase4_J_power_ci_v1_v2_v3.csv", "source of T1/F9, all three rule versions"),
    ("results/phase4/phase4_J_by_task_type.csv", "J rates by task type (v3): the task-type reversal")], {}))

S.append(("04_2_sufficiency_necessity", "4.2 Sufficiency and necessity (F5, F7/T5; F6, F8 one line each)", [
    "F5 (runs A + I, bare val n=30): verify the four-fraction numbers against rates_all_runs.md rows run=A form=bare.",
    "Random/coding correctness drift at 0.08 = non-specific cost; distressed cliff far below it.",
    "M (T5): 0 vs -0.04; quote one -0.04 reply (run_M_quotes.md); caveats: 5 samples x 30 prompts, flips 4 up / 4 down, 2/150 identical, incoherence 1/150."], [
    ("results/phase4/F5_dose_response_bare.png", "dose-response, bare val (A + I)"),
    ("results/phase4/F6_dose_response_preamble.png", "dose-response, neutral_preamble val (A)"),
    ("results/phase4/F7_subtraction.png", "subtraction on distressed val (B + L)"),
    ("results/phase4/F8_sampled_robustness.png", "sampled addition, bare val (D)"),
    ("results/phase4/figures/T5_run_M_sampled_subtraction.md", "T5"),
    ("results/phase4/phase4_metric_rates.csv", "run A rates by form x direction x fraction (v3)"),
    ("results/phase4/phase4_subtraction_rates.csv", "run B rates"),
    ("results/phase4/phase4_L_subtraction_rates.csv", "run L rates"),
    ("results/phase4/phase4_M_sampled_subtraction_ci.csv", "run M Wilson CIs"),
    ("results/phase4/phase4_sampled_rates.csv", "run D Wilson CIs")],
    {"rates_all_runs.md": "# Rule-score rates (v3) for every run x form x direction x fraction, from phase4_scores.csv\n\nSubtraction runs (B, L, M) list the magnitude; direction name ends in _subtract. Correct is over closed tasks only.\n\n" + rate_table(),
     "run_M_quotes.md": "# Run M replies for the quotable prompts (T=0.7, 5 samples; fraction 0.04 here means -0.04 along distressed_md)\n\n" + "".join(quote(r) for r in sorted(dedupe(rows(run="M", task_contains=["Great Wall", "spider", "ocean"]) + rows(run="M", prompt_contains=["rejection"])), key=lambda r: (r["base_task"], float(r["fraction"]), int(r["sample_idx"]))))}))

S.append(("04_3_specificity_robustness", "4.3 Specificity and robustness (E, K, F, N, D)", [
    "E: preamble-presence direction rates; K: shared_pc1 vs valence_resid; F: bands 8-19 / 16-27; N: neutral_preamble prompts sampled; D: bare val sampled matches greedy.",
    "All five runs are in rates_E_F_I_D_K_N.md (computed from phase4_scores.csv, rules v3); K and N also have their own CSVs."], [
    ("results/phase4/phase4_K_component_rates.csv", "run K"),
    ("results/phase4/phase4_N_preamble_sampled_ci.csv", "run N Wilson CIs"),
    ("results/phase4/phase4_sampled_rates.csv", "run D Wilson CIs"),
    ("results/phase4/phase4_metric_rates.csv", "run A (for the greedy comparison to D and N)")],
    {"rates_E_F_I_D_K_N.md": "# Rule-score rates (v3) for runs D, E, F, I, K, N, from phase4_scores.csv\n\n" + rate_table(runs="DEFIKN")}))

q44 = rows(run="J", prompt_id=["p2_0373", "p2_0325", "p2_0361", "p2_0355", "p2_0163", "p2_0319", "p2_0067", "p2_0196"])
q44 += rows(run="K", prompt_id="p2_0196")
q44 += rows(run="A", form="bare", prompt_id=["p2_0064", "p2_0040", "p2_0415"], direction=["none", "random", "distressed_md", "third_party_md"], fraction=["0", "0.04", "0.06"])
S.append(("04_4_what_replies_look_like", "4.4 What the replies look like (T3 + quotes)", [
    "Displacement by affect p2_0373; substitution p2_0325, p2_0361; leak lexical p2_0355, p2_0163, K/p2_0196; leak framing p2_0319, p2_0067 (flag: my reading); idiom G pair 30 (in 04_6); T3 contrasts p2_0064/0040/0415; leak-rate table with none@0 and random."], [
    ("results/phase4/figures/T3_within_prompt_contrasts.md", "T3"),
    ("results/phase4/phase4_affect_leak_rates.csv", "leak rate by direction x fraction, pooled over runs"),
    ("results/phase4/phase4_handcheck_set.md", "the 71 hand-read rows with rule flags")],
    {"quotes.md": "# Quoted replies for 4.4 (prompt text from data/phase2_prompts.csv; flags = rules v3; infer(strength))\n\n" + "".join(quote(r) for r in dedupe(q44))}))

# bereavement: hand-check rows whose reply asserts a loss, plus the whole A/0.06 abandoned group
hc = [(m[1], m[2], m[3], m[4]) for m in re.finditer(r"\*\*([A-Z])/(p2_\d+)\*\* \[(?:bare|preamble|[a-z_]+), [a-z_]+, ([a-z_0-9\-]+) @ ([\d.]+)\]", open(R / "phase4/phase4_handcheck_set.md", encoding="utf-8").read())]
form_of = lambda pid: "preamble" if prompts[pid]["condition"] == "neutral_preamble" else ("distressed" if prompts[pid]["condition"] == "distressed" else "bare")
hc_rows = [r for r in steered for (run, pid, d, f) in hc if r["run"] == run and r["prompt_id"] == pid and r["direction"] == d and float(r["fraction"]) == float(f) and r["sample_idx"] == ""]
BER = re.compile(r"your loss|loved one|grief|griev|bereave|condolence|passed away|sorry for your loss|the loss of", re.I)
ber_rows = [r for r in hc_rows if BER.search(r["reply"])]
ber_rows = dedupe(ber_rows)
a06 = [r for r in hc_rows if r["run"] == "A" and r["direction"] == "distressed_md" and r["fraction"] == "0.06"]
S.append(("04_5_bereavement", "4.5 Bereavement", [
    f"{len(ber_rows)} hand-check rows whose reply asserts a loss (bereavement_rows.md, regex: your loss / loved one / grief / bereave / condolence / passed away / the loss of); the A @ 0.06 abandoned group is listed in full below it.",
    "4f: grief/funeral/bereavement once each in the 150 distressed training preambles; loss/loved one/died/death/passed 0 (docs/hypotheses.md).",
    "Refuse-to-comfort variant: p2_0373, p2_0758 (A, preamble form), p2_0040 (A @ 0.06)."], [
    ("results/phase4/phase4_handcheck_set.md", "source of the 11 rows")],
    {"bereavement_rows.md": "# Hand-check rows asserting a loss\n\n" + "".join(quote(r) for r in ber_rows) + "\n# Run A, distressed_md @ 0.06, rule-abandoned group (10 rows)\n\n" + "".join(quote(r) for r in a06)}))

p1 = list(csv.DictReader(open(R / "phase1/phase1_replies.csv", newline="", encoding="utf-8"))); g = list(csv.DictReader(open(R / "phase1/phase1_replies_nosys.csv", newline="", encoding="utf-8")))
def p1q(rs, pairs):
    out = []
    for r in rs:
        if str(r.get("pair_id", "")) in pairs:
            meta = {k: v for k, v in r.items() if k not in ("reply", "prompt", "neutral", "stressed", "reply_neutral", "reply_stressed")}
            out.append(f"- pair {r.get('pair_id')} " + " ".join(f"{k}={v}" for k, v in meta.items() if k != "pair_id") + "\n" + "".join(f"  - {k}: {str(v).replace(chr(10), ' / ')}\n" for k, v in r.items() if k in ("prompt", "neutral", "stressed", "reply", "reply_neutral", "reply_stressed")))
    return "".join(out)
S.append(("04_6_prompted_baseline", "4.6 Prompted baseline (Phase 1, run G)", [
    "30 pairs, concise prompt: correct 20/20 closed; ack 18/30 by task type; infer 8/30; displacement 7/30; advice changed 3/5; pair 27 refusal.",
    "G no system prompt: ack 24/30, coding 5/5, factual 5/5, length 130 vs 43; 22/30 capped; pair 13 'Yes' on a false premise; pair 30 knot-in-stomach."], [
    ("results/phase1/phase1_replies.md", "all 60 Phase 1 replies, side by side"),
    ("results/phase1/phase1_scores.csv", "rule + manual scores (manual columns are the hand verdicts)"),
    ("results/phase1/phase1_replies_nosys.csv", "run G replies"),
    ("results/phase1/phase1_scores_nosys.csv", "run G rule scores (v3)"),
    ("results/phase1/phase1_meta.json", "generation settings")],
    {"quotes_phase1_and_G.md": "# Phase 1 pairs 9, 11, 22, 27 (concise prompt)\n\n" + p1q(p1, {"9", "11", "22", "27"}) + "\n# Run G pairs 13 and 30 (no system prompt)\n\n" + p1q(g, {"13", "30"})}))

S.append(("04_7_is_it_the_words", "4.7 Is it the words? (F1-F4)", [
    "a' layer 18 val 0.950, shuffled ~0.50, layer-0 0.767; bare-neutral 0.983 untrained; implied recall 0.850 / BA 0.912; human 0.917; bag-of-words 0.850 / 0.575 / 0.350; length-only 0.583; ask-the-model implied 0.150. Verify all against phase3b_headline.csv and phase3b_baselines.csv.",
    "F4 plots probe-weight cosines in standardised space (0.53-0.77 distressed vs frustrated); the 0.85 in the text is the raw-space mean-difference cosine (phase3b_geometry_layer18.csv). Verify the 0.24-0.50 distressed-vs-positive range against phase3b_probe_by_layer.csv.",
    "v1 confounded renders are included only for a before/after panel."], [
    ("results/phase3/F1_probe_acc_by_layer_task_a.png", "v2"), ("results/phase3/F2_probe_acc_by_layer_task_b.png", "v2"),
    ("results/phase3/F3_pdist_boxplots_bestlayer.png", "v2"), ("results/phase3/F4_probe_direction_cosines.png", "v2"),
    ("results/phase3/F1_v1_confounded_probe_acc_by_layer_task_a.png", "v1, before panel only"), ("results/phase3/F4_v1_confounded_probe_direction_cosines.png", "v1, before panel only"),
    ("results/phase3/phase3b_headline.csv", "headline numbers"), ("results/phase3/phase3b_baselines.csv", "bag-of-words, length-only, ask-the-model"),
    ("results/phase3/phase3b_probe_by_layer.csv", "per-layer accuracies and cosines (F1, F2, F4 source)"), ("results/phase3/phase3b_geometry_layer18.csv", "raw-space cosines"),
    ("results/phase3/phase3b_best_layers.json", "")], {}))

S.append(("04_8_self_vs_other", "4.8 Self vs other (T4, F3, run C)", [
    "Probe: third_party 0.90, third_party_neutral ~0.01 (check F3/phase3b_pdist_bestlayer.csv; the plan's 0.11 needs verifying); T4 (run A, 0.06); C unsteered third-party replies (quotes_C_and_p2_0732.md); p2_0732 escalation; unresolved, n=30."], [
    ("results/phase4/figures/T4_distressed_vs_third_party.md", "T4"),
    ("results/phase3/F3_pdist_boxplots_bestlayer.png", "P(distressed) by group at layer 18"),
    ("results/phase3/phase3b_pdist_bestlayer.csv", "per-row P(distressed)"),
    ("results/phase4/phase4_third_party_rates.csv", "run C rates by form"),
    ("results/phase4/phase4_metric_rates.csv", "run A rates incl. third_party_md rows")],
    {"quotes_C_and_p2_0732.md": "# p2_0732 (neutral_preamble form: 'My daughter's teacher set this as an open question for parents'), run A\n\n" + "".join(quote(r) for r in rows(run="A", prompt_id="p2_0732", direction=["none", "distressed_md"], fraction=["0", "0.04", "0.06"])) + "\n# Run C: unsteered replies to the 70 third-party and third-party-neutral prompts\n\n" + "".join(quote(r) for r in rows(run="C"))}))

S.append(("06_verification", "6. Verification (T2, hand-check, rule versions)", [
    "71 rows, seed 11, 5 groups; T2 cells v1 -> v3; errors and fixes; 8 remaining cells; v1/v2 snapshots; Phase 1 manual columns byte-identical; leak detector vs hand tally; run M checks; F1-F4 overwrite; F9 producer after the fact."], [
    ("results/phase4/figures/T2_handcheck_agreement_v1_v3.md", "T2"),
    ("results/phase4/phase4_handcheck_set.md", "the 71 rows"),
    ("results/phase4/phase4_handcheck_disagreements.csv", "hand verdicts and disagreements"),
    ("results/phase4/phase4_handcheck_agreement.csv", "per-cell agreement"),
    ("results/phase4/phase4_handcheck_agreement_v1.csv", ""), ("results/phase4/phase4_handcheck_agreement_v2.csv", ""), ("results/phase4/phase4_handcheck_agreement_v3.csv", ""),
    ("results/phase4/phase4_J_power_ci_v1_v2_v3.csv", "headline under all three rule versions"),
    ("results/phase4/rules_v1/F5_dose_response_bare_rules_v1.png", "F5 under v1"), ("results/phase4/rules_v2/F5_dose_response_bare_rules_v2.png", "F5 under v2"),
    ("results/phase4/F5_dose_response_bare.png", "F5 under v3 (final)")], {}))

S.append(("09_appendix", "Appendices A-E and 9. time log", ["A run table; B prompt counts; C rules v3 + v1->v3 diffs (from docs/hypotheses.md); D hypothesis ledger; E file index (results/README.md)."], [
    ("data/phase1_scoring_rules.md", "C: rule text"), ("results/phase4/phase4_keyword_lists.json", "C: lexicons"), ("results/README.md", "E: file index")],
    {"A_run_table.md": "# Appendix A. Phase 4 runs\n\n" + run_table, "B_prompt_counts.md": prompt_counts,
     "C_rule_diffs.md": "# Appendix C. Rule changes v1 -> v2 -> v3 (from docs/hypotheses.md)\n\n" + extract_section(ROOT / "docs/hypotheses.md", "### Hand-check reconciliation", 3) + "\n" + extract_section(ROOT / "docs/hypotheses.md", "### Rules v3", 3),
     "D_hypothesis_ledger.md": "# Appendix D. Hypothesis ledger (docs/hypotheses.md section 2)\n\n" + extract_section(ROOT / "docs/hypotheses.md", "## 2. Hypotheses", 2),
     "time_log.md": extract_section(ROOT / "docs/hypotheses.md", "## 8. Time log", 2)}))

# ---------- write ----------
top = ["# Write-up assembly", "", "Generated by `scripts/09_assemble_writeup.py`; every file here is a COPY of, or derived from, files elsewhere in results/, data/ or docs/. Edit the originals and rerun. Sections 1, 2, 5, 7 and 8 need no result files (2 and 7 draw on docs/hypotheses.md).", ""]
for folder, title, plan, copies, gen in S:
    d = OUT / folder; d.mkdir(); idx = [f"# {title}", "", "Plan bullets:", *[f"- {p}" for p in plan], "", "Files:"]
    for src, note in copies:
        dst = d / Path(src).name; shutil.copy2(ROOT / src, dst); idx.append(f"- `{dst.name}` <- `{src}`" + (f": {note}" if note else ""))
    for name, text in gen.items(): (d / name).write_text(text, encoding="utf-8"); idx.append(f"- `{name}` (derived, generated by this script)")
    (d / "INDEX.md").write_text("\n".join(idx) + "\n", encoding="utf-8"); top.append(f"- `{folder}/` {title} ({len(copies)} copied, {len(gen)} derived)")
(OUT / "README.md").write_text("\n".join(top) + "\n", encoding="utf-8"); print("\n".join(top))
