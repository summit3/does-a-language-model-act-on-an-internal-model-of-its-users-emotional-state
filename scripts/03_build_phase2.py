"""Assemble data/phase2_prompts.csv from the hand-written pools.

Columns: id, base_task, condition, text, split, author, judge_agrees.
- third_party_neutral: 20 of the third_party tasks (10 train + 10 val) with a non-emotional third-party preamble.
- human_<condition>: the user's own preambles (verbatim), attached to val tasks not used by third_party_neutral.
- judge_agrees: 1/0 from data/phase2_qa_labels.csv (blank if the row has not been judged yet).
- main set: 150 base tasks x {neutral, distressed, frustrated}; split train/val by base task (80/20,
  stratified by task type). Preambles assigned at random with a cap of 15% topically related to their task.
- implied: 40 held-out tasks x {neutral, implied} (split="implied").
- third_party: 40 in-distribution tasks (all 30 val + 10 train) x {third_party} (split="third_party");
  their matched neutral rows are the main-set neutral rows with the same base_task.
"""
import csv, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.phase2_pool.tasks import TASKS, HELDOUT_TASKS
from scripts.phase2_pool.distressed import DISTRESSED
from scripts.phase2_pool.frustrated import FRUSTRATED
from scripts.phase2_pool.heldout import IMPLIED, THIRD_PARTY, THIRD_PARTY_NEUTRAL, HUMAN
from scripts.phase2_pool.checks import check_batch, norm, STOP, words

OUT = Path("data/phase2_prompts.csv"); SEED = 20260908; CAP = 0.15
rng = random.Random(SEED)

def content(t): return {w for w in words(t) if w not in STOP and len(w) > 3}
def related(pre, task): return len(content(pre) & content(task)) > 0

def assign(preambles, tasks):
    """Random 1:1 assignment with at most CAP of pairs topically related (shared content word)."""
    pre = list(preambles); rng.shuffle(pre)
    limit = int(CAP * len(tasks))
    for _ in range(2000):
        rel = [i for i in range(len(tasks)) if related(pre[i], tasks[i])]
        if len(rel) <= limit: break
        i = rng.choice(rel); j = rng.randrange(len(tasks))
        if not related(pre[j], tasks[i]) and not related(pre[i], tasks[j]): pre[i], pre[j] = pre[j], pre[i]
    return pre, len([i for i in range(len(tasks)) if related(pre[i], tasks[i])])

# validate pools once more (must all pass)
for name, pool, kw in [("distressed", DISTRESSED, {}), ("frustrated", FRUSTRATED, {}), ("implied", IMPLIED, {"no_emotion_words": True}), ("third_party", THIRD_PARTY, {})]:
    ok, problems, _ = check_batch(pool, name, quiet=True, **kw)
    assert not problems, (name, problems)
assert len(DISTRESSED) == len(FRUSTRATED) == 150 and len(IMPLIED) == len(THIRD_PARTY) == 40
allpre = DISTRESSED + FRUSTRATED + IMPLIED + THIRD_PARTY + THIRD_PARTY_NEUTRAL + [h for _, h in HUMAN]
ok, problems, _ = check_batch(THIRD_PARTY_NEUTRAL, "third_party_neutral", quiet=True, no_emotion_words=True); assert not problems, problems
assert len({norm(p) for p in allpre}) == len(allpre), "duplicate preamble across pools"

# main tasks + stratified split
main = [(tt, t) for tt in TASKS for t in TASKS[tt]]
split = {}
for tt in TASKS:
    ts = list(TASKS[tt]); rng.shuffle(ts)
    for k, t in enumerate(ts): split[t] = "val" if k < 5 else "train"
tasks_only = [t for _, t in main]
dis, n_rel_d = assign(DISTRESSED, tasks_only)
fru, n_rel_f = assign(FRUSTRATED, tasks_only)

rows = []; nid = [0]
def add(base_task, condition, text, sp, author="claude"):
    nid[0] += 1; rows.append({"id": f"p2_{nid[0]:04d}", "base_task": base_task, "condition": condition, "text": text, "split": sp, "author": author})
for i, (tt, t) in enumerate(main):
    add(t, "neutral", t, split[t]); add(t, "distressed", f"{dis[i]} {t}", split[t]); add(t, "frustrated", f"{fru[i]} {t}", split[t])
# implied held-out (new tasks)
ho_tasks = [t for _, t in HELDOUT_TASKS]
imp, n_rel_i = assign(IMPLIED, ho_tasks)
for i, t in enumerate(ho_tasks):
    add(t, "neutral", t, "implied"); add(t, "implied", f"{imp[i]} {t}", "implied")
# third-party: 30 val tasks + 10 random train tasks
val_tasks = [t for t in tasks_only if split[t] == "val"]; train_tasks = [t for t in tasks_only if split[t] == "train"]
tp_tasks = val_tasks + rng.sample(train_tasks, 10)
tp, n_rel_t = assign(THIRD_PARTY, tp_tasks)
for i, t in enumerate(tp_tasks): add(t, "third_party", f"{tp[i]} {t}", "third_party")
# third_party_neutral: 10 of the train third-party tasks + 10 val tasks (leaves 20 val tasks free for the human set)
tpn_tasks = tp_tasks[30:40] + val_tasks[:10]
tpn, n_rel_n = assign(THIRD_PARTY_NEUTRAL, tpn_tasks)
for i, t in enumerate(tpn_tasks): add(t, "third_party_neutral", f"{tpn[i]} {t}", "third_party_neutral")
# human-written preambles -> the 20 val tasks not used by third_party_neutral, cycling
free_val = val_tasks[10:]; assert len(free_val) == 20
for k, (cond, pre) in enumerate(HUMAN):
    t = free_val[k % len(free_val)]
    add(t, cond, f"{pre} {t}", f"human_{cond}", author="human")

texts = [r["text"] for r in rows]
assert len(set(texts)) == len(texts), "duplicate text rows"
# judge_agrees from the blind check (by text, so ids may be renumbered safely)
EXPECTED = {"neutral": "neutral", "distressed": "distressed", "frustrated": "frustrated", "implied": "distressed",
            "third_party": "neutral", "third_party_neutral": "neutral"}
LAB = Path("data/phase2_qa_labels.csv")
judged = {}
if LAB.exists():
    for r in csv.DictReader(open(LAB, newline="", encoding="utf-8")):
        judged[r["text"]] = r["judge_label"]
for r in rows:
    j = judged.get(r["text"]); r["judge_agrees"] = "" if j is None else int(j == EXPECTED[r["condition"]])
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "base_task", "condition", "text", "split", "author", "judge_agrees"], lineterminator="\n"); w.writeheader(); w.writerows(rows)
print("judge_agrees filled:", sum(1 for r in rows if r["judge_agrees"] != ""), "/", len(rows))
from collections import Counter
print(f"wrote {OUT}: {len(rows)} rows"); print("by split/condition:", dict(Counter((r['split'], r['condition']) for r in rows)))
print(f"topically related: distressed {n_rel_d}/150, frustrated {n_rel_f}/150, implied {n_rel_i}/40, third_party {n_rel_t}/40 (cap {CAP:.0%})")
print("train/val base tasks:", Counter(split.values()), "| third_party tasks from val/train:", len(val_tasks), "/", 10)
