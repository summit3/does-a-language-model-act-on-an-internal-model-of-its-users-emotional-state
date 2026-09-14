"""Write data/phase2_qa.md: blind-judge agreement, preamble word frequencies (+ >10% flags), length
stats, duplicate check, and 10 random rows per condition. No model needed."""
import csv, random, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.phase2_pool.checks import words, norm, STOP

rows = list(csv.DictReader(open("data/phase2_prompts.csv", newline="", encoding="utf-8")))
_labs = list(csv.DictReader(open("data/phase2_qa_labels.csv", newline="", encoding="utf-8")))
_by_text = {l["text"]: l for l in _labs}
labels = {}
def _bind():
    for r in rows: labels[r["id"]] = _by_text.get(r["text"], {"judge_label": "?"})
def preamble(r): return r["text"][:-len(r["base_task"])].strip() if r["condition"] != "neutral" else ""
_bind()
conds = ["neutral", "distressed", "frustrated", "implied", "third_party", "third_party_neutral", "neutral_preamble", "positive"]
L = ["# Phase 2 dataset QA", "", f"`data/phase2_prompts.csv`: {len(rows)} rows. Built by `scripts/03_build_phase2.py` from hand-written pools in "
     "`scripts/phase2_pool/` (preambles and base tasks written by the assistant, not by any model). Seed 20260908.", ""]
# composition
L += ["## Composition", "", "| split | condition | rows |", "|---|---|---|"]
for (sp, c), n in sorted(Counter((r["split"], r["condition"]) for r in rows).items()): L.append(f"| {sp} | {c} | {n} |")
L += ["", "Main set: 150 base tasks (25 per Phase 1 task type) x neutral/distressed/frustrated; train/val split by base task, "
      "80/20 stratified by type (no base task in both). `implied` split: 40 new base tasks x neutral/implied. `third_party` split: "
      "40 in-distribution tasks (all 30 val + 10 train) with a third-party-emotion preamble; their matched neutral rows are the main-set neutral rows with the same `base_task`.", ""]
# judge agreement
L += ["## Blind label-consistency check", "",
      "**Judge: Qwen3.5-9B on the pod, i.e. the subject model itself.** It saw each row's text with no label and answered one word "
      "(neutral / distressed / frustrated), instructed to judge only the writer's own state. Treat as a consistency check, not ground truth. "
      "Expected label: neutral -> neutral, distressed -> distressed, frustrated -> frustrated, implied -> distressed (no emotion words), "
      "third_party -> neutral (the writer's own state is not stated), third_party_neutral -> neutral. `judge_agrees` in the prompts CSV records this per row so Phase 3 can report probe accuracy on judge-missed rows separately.", "",
      "| condition | n | judge = expected | % | judge labels (counts) |", "|---|---|---|---|---|"]
expected = {"neutral": "neutral", "distressed": "distressed", "frustrated": "frustrated", "implied": "distressed", "third_party": "neutral", "third_party_neutral": "neutral", "neutral_preamble": "neutral", "positive": "neutral"}
disagree = defaultdict(list)
for c in conds:
    rs = [r for r in rows if r["condition"] == c]; js = [labels[r["id"]]["judge_label"] for r in rs]
    agree = sum(j == expected[c] for j in js)
    L.append(f"| {c} | {len(rs)} | {agree} | {100*agree/len(rs):.0f}% | {dict(Counter(js))} |")
    for r, j in zip(rs, js):
        if j != expected[c]: disagree[c].append((r["id"], j, r["text"]))
L += ["", "### Disagreements (all listed for hand review)", ""]
for c in conds:
    if disagree[c]:
        L.append(f"**{c}** ({len(disagree[c])}):"); L += [f"- `{i}` judge={j}: {t}" for i, j, t in disagree[c]]; L.append("")
# word frequencies
L += ["## Preamble word frequencies (top 20 per condition; content words only, stopwords excluded)", "",
      "Flag = word appears in >10% of that condition's preambles.", ""]
for c in ["distressed", "frustrated", "implied", "third_party", "third_party_neutral", "neutral_preamble", "positive"]:
    pres = [preamble(r) for r in rows if r["condition"] == c and r["author"] == "claude"]
    cnt = Counter(w for p in pres for w in set(words(p)) if w not in STOP)
    L.append(f"**{c}** (n={len(pres)}): " + ", ".join(f"{w} {k}" + (" **FLAG**" if k > 0.10*len(pres) else "") for w, k in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:20]))  # ties sorted alphabetically so the report is deterministic
    flags = [w for w, k in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])) if k > 0.10*len(pres)]  # same order as the list above; cnt insertion order is hash-seed dependent
    L.append(f"  Flags: {flags or 'none'}"); L.append("")
# specific frequencies requested
import re as _re
L += ["## Specific word frequencies (preambles containing the word; generated rows only)", "", "| condition | n | irritat* | letter |", "|---|---|---|---|"]
for c in ["distressed", "frustrated", "implied", "third_party", "third_party_neutral"]:
    pres = [preamble(r) for r in rows if r["condition"] == c and r["author"] == "claude"]
    L.append(f"| {c} | {len(pres)} | {sum(bool(_re.search(r'irritat', p, _re.I)) for p in pres)} | {sum(bool(_re.search(r'\\bletters?\\b', p, _re.I)) for p in pres)} |")
L += ["", "## Notes on the generated set (author review, 2026-09-08)", "",
      "- Generated preambles skew literate (full sentences, varied but formal-leaning register).",
      "- Frustrated preambles typically name an external cause (a printer, a colleague, a company); distressed preambles describe an internal state.",
      "- Implied distress is carried mainly by situation severity (hospital, eviction, results) rather than by wording.",
      "- Third-party rows carry pronoun markers (his/her/since); the `third_party_neutral` set mentions someone else with no emotional content to cancel that marker.",
      "- The `human_*` splits (author=human, the user's own preambles, verbatim) test whether probe accuracy depends on the generated register.", ""]
# lengths
L += ["## Preamble length (words)", "", "| condition | min | mean | median | max |", "|---|---|---|---|---|"]
for c in ["distressed", "frustrated", "implied", "third_party", "third_party_neutral", "neutral_preamble", "positive"]:
    ls = sorted(len(words(preamble(r))) for r in rows if r["condition"] == c)
    L.append(f"| {c} | {ls[0]} | {sum(ls)/len(ls):.1f} | {ls[len(ls)//2]} | {ls[-1]} |")
# duplicates
pres_all = [preamble(r) for r in rows if r["condition"] != "neutral"]
p1 = [r for r in csv.DictReader(open("data/phase1_pairs.csv", newline="", encoding="utf-8"))]
p1pre = {norm(r["stressed"][:-len(r["neutral"])]) for r in p1}
dups = [p for p, k in Counter(norm(p) for p in pres_all).items() if k > 1]
L += ["", "## Duplicates", "", f"- preambles: {len(pres_all)}; distinct (normalised): {len(set(norm(p) for p in pres_all))}; duplicates: {dups or 'none'}",
      f"- preambles reused from Phase 1: {[p for p in pres_all if norm(p) in p1pre] or 'none'}",
      f"- duplicate text rows: {len(rows) - len(set(r['text'] for r in rows))}", ""]
# human set
L += ["", "## Human-written set (author=human, verbatim)", ""]
for r in rows:
    if r["author"] == "human": L.append(f"- `{r['id']}` [{r['split']}] judge={labels.get(r['id'], {}).get('judge_label', '?')}: {r['text']}")
# samples
rng = random.Random(7)
L += ["## Random samples (10 per condition, for hand reading)", ""]
for c in conds:
    L.append(f"### {c}"); L.append("")
    for r in rng.sample([r for r in rows if r["condition"] == c and r["author"] == "claude"], 10):
        L.append(f"- `{r['id']}` [{r['split']}] judge={labels[r['id']]['judge_label']}: {r['text']}")
    L.append("")
L += ["", "## Phase 3b additions (2026-09-08): neutral_preamble and positive controls", "",
      "Added after Phase 3 v1 showed the neutral class (bare tasks, no preamble) was separable by length alone. `neutral_preamble` puts a mundane, "
      "non-emotional personal preamble on every base task (150 rows, same train/val split by task; length-matched to the emotional preambles); "
      "`positive` puts a clearly happy/excited/relieved preamble on 50 tasks spread across types. The blind judge has only three labels, so the expected "
      "judge label for both is `neutral`. Their agreement, word frequencies and lengths are included in the tables above; samples below.", ""]
for c in ["neutral_preamble", "positive"]:
    L.append(f"### {c}"); L.append("")
    for r in rng.sample([r for r in rows if r["condition"] == c], 10): L.append(f"- `{r['id']}` [{r['split']}] judge={labels[r['id']]['judge_label']}: {r['text']}")
    L.append("")
Path("data/phase2_qa.md").write_text("\n".join(L), encoding="utf-8"); print("wrote data/phase2_qa.md")
