"""Agreement between rule scores (v1 snapshot vs v2 current) and the hand verdicts on the Phase 4 hand-check set.
Inputs: results/phase4/phase4_handcheck_set.md (read set), phase4_handcheck_disagreements.csv (hand verdicts where they differ from v1;
a row with rule==me records the hand value explicitly), rules_v1/ snapshots, current scores. Outputs: phase4_handcheck_agreement*.csv."""
import re, sys, pandas as pd
from pathlib import Path
R = Path("results/phase4"); pd.set_option("display.width", 230)
md = open(R / "phase4_handcheck_set.md", encoding="utf-8").read()
groups = {"distressed_0.04": "Run J, distressed_md @ 0.04", "G_abandoned": "Run G", "random_0.04": "Run J, random @ 0.04", "distressed_0.06_abandoned": "Run A, distressed_md @ 0.06", "inference_flagged": "Any run, rule user_state_inference=1"}
members = {}
for sec in re.split(r"^## ", md, flags=re.M)[1:]:
    title = sec.split("\n", 1)[0]; g = next(k for k, v in groups.items() if title.startswith(v))
    members[g] = re.findall(r"^- \*\*(\w+)/(?:pair )?([\w]+)\*\* \[([^\]]*)\]", sec, flags=re.M)
v1 = pd.read_csv(R / "rules_v1/phase4_scores.csv", dtype=str, keep_default_na=False); v2 = pd.read_csv(R / "phase4_scores.csv", dtype=str, keep_default_na=False)
g1 = pd.read_csv(R / "rules_v1/phase1_scores_nosys_rules_v1.csv", dtype=str, keep_default_na=False); g2 = pd.read_csv("results/phase1/phase1_scores_nosys.csv", dtype=str, keep_default_na=False)
dis = pd.read_csv(R / "phase4_handcheck_disagreements.csv", dtype=str, keep_default_na=False)
MET = {"correct": "correct", "abandoned": "task_abandoned", "ack": "acknowledges_emotion", "infer": "user_state_inference", "incoh": "incoherent"}
GMET = {"correct": "correct_stressed_rule", "abandoned": "task_abandoned_rule", "ack": "acknowledges_emotion_rule", "infer": "unsolicited_inference_rule"}
def parse_meta(meta): m = re.search(r"([\w-]+) @ ([\d.]+)(?: sample (\d+))?", meta); return m.group(1), m.group(2), (m.group(3) or "")
def row_scores(df, run, pid, meta):
    d, f, sidx = parse_meta(meta); r = df[(df.run == run) & (df.prompt_id == pid) & (df.direction == d) & (df.fraction.astype(float) == float(f)) & (df.sample_idx == sidx)]; return r.iloc[0]
def val(x): x = str(x).strip(); return "nan" if x in ("", "nan") else str(int(float(x)))
rows = []
for g, mem in members.items():
    for run, pid, meta in mem:
        for short, col in MET.items():
            if run == "G":
                if short == "incoh": continue
                pr = pid.replace("pair_", "").replace("pair", ""); r1, r2 = val(g1[g1.pair_id == pr].iloc[0][GMET[short]]), val(g2[g2.pair_id == pr].iloc[0][GMET[short]]); did = f"pair_{pr}"
            else:
                r1, r2 = val(row_scores(v1, run, pid, meta)[col]), val(row_scores(v2, run, pid, meta)[col]); did = pid
            me = dis[(dis.id == did) & (dis.group == g) & (dis.metric == short)]; hand = me.iloc[-1]["me"] if len(me) else r1
            if short == "correct" and hand == "nan": continue
            rows.append({"group": g, "run": run, "id": did, "meta": meta, "metric": short, "v1": r1, "v2": r2, "hand": hand, "agree_v1": int(r1 == hand), "agree_v2": int(r2 == hand)})
A = pd.DataFrame(rows); A.to_csv(R / "phase4_handcheck_agreement.csv", index=False)
order = ["distressed_0.04", "random_0.04", "G_abandoned", "distressed_0.06_abandoned", "inference_flagged"]; cols = ["correct", "abandoned", "ack", "infer", "incoh"]
for tag, key in [("v1 (before)", "agree_v1"), ("v2 (after)", "agree_v2")]:
    t = A.pivot_table(index="group", columns="metric", values=key, aggfunc="mean").reindex(order)[cols].round(2); t.to_csv(R / f"phase4_handcheck_agreement_{key[-2:]}.csv"); print(f"\n=== AGREEMENT rules {tag} vs hand ==="); print(t.to_string())
if "--show" in sys.argv:
    st = pd.read_csv(R / "phase4_steered.csv", dtype=str, keep_default_na=False); G = pd.read_csv("results/phase1/phase1_replies_nosys.csv", dtype=str, keep_default_na=False)
    print("\n=== remaining v2 disagreements with reply text ===")
    for _, r in A[A.agree_v2 == 0].iterrows():
        if r.run == "G":
            pr = r.id.replace("pair_", ""); x = G[(G.pair_id == pr) & (G.condition == "stressed")].iloc[0]; txt = x.reply; task = x.prompt
        else:
            d, f, sidx = parse_meta(r.meta); x = st[(st.run == r.run) & (st.prompt_id == r.id) & (st.direction == d) & (st.fraction.astype(float) == float(f)) & (st.sample_idx == sidx)].iloc[0]; txt = x.reply; task = x.base_task
        print(f"\n[{r.group}] {r.run}/{r.id} {r.meta} | metric={r.metric} v1={r.v1} v2={r.v2} hand={r.hand}\n  task: {task}\n  reply: {txt.replace(chr(10), ' / ')[:600]}")
