# Appendix D. Hypothesis ledger (docs/hypotheses.md section 2)

## 2. Hypotheses

### H0 (setup): User emotional state is linearly represented; a probe generalises to held-out phrasings and implied-emotion prompts, and beats bag-of-words.
- Status: **Supported (v2)**
- Evidence so far: Phase 3 v1 probe hits ~1.0 on val, implied and judge-missed rows, but does so from layer 0 and the length-only baseline also reaches 1.0: v1 neutral rows had no preamble, so the probe is largely a preamble-presence detector (see findings 2026-09-08 Phase 3 v1). Not yet evidence for or against H0.
- Evidence (v2, 2026-09-08): val 0.95 at layer 18 vs bag-of-words 0.85 and length 0.58; implied recall 0.85 vs BoW 0.575 vs ask-the-model 0.15; human 0.917 vs BoW 0.35 vs ask-the-model 0.58; accuracy rises 0.77 -> 0.95 across layers; shuffled 0.50; train-val gap 0.05. v1 was confounded (see findings 2026-09-08 Phase 3 v1).
- Caveat (2026-09-09, from Phase 4 run A, bare, same 30 prompts): the third_party_md direction produces the same behaviours as distressed_md at similar strengths (0.06: ack 0.23 vs 0.87, inference 0.63 vs 0.90, abandonment 0.67 vs 0.40), consistent with H2's refutation: what is represented, and what steers, is distress-in-context rather than the user's own state. [updated Sept 11 to v3 values]
- What would change my mind: probe accuracy on held-out set A (implied emotion) no better than bag-of-words; or accuracy at chance on shuffled labels not at chance (leak).

### H1 (behavioural): On unrelated tasks, replies to a distressed user differ from replies to a neutral user in measurable ways (accuracy, hedging, refusal rate, agreement with a false premise).
- Status: **partial; false-premise agreement observed under no system prompt (2026-09-09, n=5, hand-read)**
- Evidence (2026-09-09, run G, no system prompt): all 5 neutral false-premise replies lead with a correction; stressed replies for pairs 13 and 14 open with acknowledgement and pair 13's first answer-bearing sentence agrees with the premise ('Yes') by hand read (v2 rule scores it correct on a later sentence). Steered correctness drops (4b) are capability or lack a baseline, not steering.
- Evidence so far (Phase 1, 30 hand-written pairs, Qwen3.5-9B, greedy, concise system prompt):
  - Correctness of closed-form answers never changed: 20/20 pairs (arithmetic, factual, false premise, coding). No agreement with a false premise in either condition (5/5 correct both).
  - Only correctness drop: refusal of a trivial task, pair 27 "I cannot count to 10 for you" (1/30).
  - Emotional acknowledgement 18/30, task-dependent: arithmetic 5/5, advice 4/5, false premise 3/5, instruction-following 3/5, factual 2/5, coding 1/5.
  - Info displaced 7/30 (factual replies shorter, mean -6.6 tokens; arithmetic longer, +6.0).
  - Unsolicited inference about the user 8/30: 6 strong (diagnoses, medical/psychological framing, professional referrals), 2 weak (causal speculation).
  - Advice recommendation changed 3/5.
  - Pattern: acknowledge -> displace -> infer about the user -> occasionally abandon; competence intact.
- What would change my mind: effects vanish on held-out prompts or under sampled decoding; effects reproduce with third-party-distress preambles (then it is not about the user); effect sizes track emotion-word count rather than condition.

### H1b (new, 2026-09-08): Acknowledgement occurs when the reply has room, i.e. it is driven by answer length under the concise budget rather than by task "softness".
- Status: **Supported (rule-scored)**
- Evidence: with the concise system prompt removed (Phase 4 run G, results/phase1/phase1_replies_nosys.csv), acknowledgement 24/30 vs 18/30 with the prompt; coding 5/5 vs 1/5, factual 5/5 vs 2/5; mean reply length 130 vs 43 tokens. The 6 "abandoned" rows in that run are pending hand check and may be rule artefacts on long replies. Original observation: arithmetic answers are ~11 tokens and always acknowledged (5/5); coding/factual answers fill the two-sentence budget and mostly do not (1/5, 2/5).
- What would change my mind: acknowledgement stays flat when the concise prompt is relaxed; or short factual answers are not acknowledged at matched length.
- Test later if time: relax the concise prompt and check whether coding acknowledgement rises.

### H2 (specificity): The representation distinguishes "the user is distressed" from "the user is talking about someone distressed," and only the former changes behaviour.
- Status: **Refuted (v2)**
- Evidence so far: v1 task-(a) probe fires on third-party emotion (mean P(distressed) 1.00) and on non-emotional third-party context (0.94); cos(third_party_neutral mean-diff, distressed mean-diff) 0.84. Confounded by preamble presence (see H0); retest after the neutral_preamble control.
- Evidence (v2, 2026-09-08): P(distressed) 0.90 on third-party distress, 0.01 on third-party-neutral, 0.09 on mundane preamble; cos(third_party md, distressed md) 0.74. The representation encodes "distress present in the conversation", not "the user is distressed". It distinguishes emotional from mundane context; it does not distinguish whose emotion.
- What would change my mind: probe trained on user-distress fires equally on third-party prompts, and third-party prompts produce the same acknowledge/displace/infer behaviour.

### H3 (causal): Steering along the user-distress direction on neutral prompts reproduces the H1 changes; subtracting it from distressed prompts removes them; unrelated-concept directions of matched norm do not.
- Status: **Supported (Phase 4, 2026-09-08)**
- Evidence: layer-18 distressed mean-difference (relative to neutral_preamble), band 12-23, strength as a fraction of mean band norm, all 150 bare tasks (bare form never entered the direction), rule-scored.
  - Power (J): at 0.04 on all 150 bare base tasks (n=150, Wilson 95% CI): acknowledges_emotion 0.247 [0.185, 0.321] vs random 0.007 [0.001, 0.037] vs unrelated coding probe 0.000 [0.000, 0.025]; user_state_inference 0.233 [0.173, 0.307] vs 0.000 vs 0.000; task_abandoned 0.100 [0.062, 0.158] vs 0.000 [0, 0.025] vs 0.007; incoherent 0.000 vs 0.000 vs 0.013; correct 0.936 [0.879, 0.967] vs 0.960 vs 0.976 (n=125). [updated Sept 11 to v3 values]
  - Dose (I + A, 30 bare val): distressed_md acknowledgement 0.07 / 0.13 / 0.13 / 0.20 / 0.30 / 0.77 / 0.83 / 1.00 at 0 / 0.01 / 0.02 / 0.03 / 0.04 / 0.05 / 0.06 / 0.08; abandonment 0 / 0 / 0 / 0.03 / 0.10 / 0.27 / 0.40 / 0.40; correct holds 0.92 to 0.04 then 0.60 / 0.12 / 0.00. Random, unrelated-coding and preamble-presence directions stay at 0.03-0.17 acknowledgement and 0 abandonment through 0.05; coherence intact for all directions through 0.06. [updated Sept 11 to v3 values]
  - Subtraction (B, L, greedy; 30 distressed val): unsteered acknowledgement 0.27, inference 0.17, abandoned 0.03; at -0.02 / -0.04 / -0.06 / -0.08: 0.13 / 0.00 / 0.00 / 0.00, inference 0.03 / 0.00 / 0.00 / 0.00, correct 0.92 / 0.92 / 0.92 / 0.76. Sampled (M, T=0.7, 150 samples): acknowledgement 0.267 [0.202, 0.343] -> 0.000 [0.000, 0.025]; inference 0.133 [0.088, 0.197] -> 0.000 [0, 0.025]; correct 0.912 -> 0.912. Third-party prompts at -0.04 along third_party_md: acknowledgement 0.12 (run C, n=40) -> 0.03. [updated Sept 11 to v3 values]
  - Band robustness (F): distressed_md at 0.04 on bands 8-19 / 12-23 / 16-27: acknowledgement 0.10 / 0.20 / 0.07, abandonment 0.07 / 0.10 / 0.03, inference 0.07 / 0.27 / 0.03, correct 0.96 all. [updated Sept 11 to v3 values]
  - Sampled addition (D, T=0.7, 150 samples, bare val): acknowledgement 0.000 -> 0.287 [0.220, 0.364], inference 0.007 -> 0.253, abandoned 0.013 -> 0.087, correct 0.952 -> 0.944. [updated Sept 11 to v3 values]
- Sub-hypothesis (K, component decomposition) **refuted**: the valence residual (distressed md minus its projection on positive md; cos with distressed md 0.78) carries the behaviour (at 0.04 / 0.06 / 0.08: acknowledgement 0.20 / 0.20 / 0.43, inference 0.10 / 0.67 / 1.00, abandonment 0.03 / 0.40 / 0.87) while the shared any-emotion PC1 (cos 0.35) is inert (acknowledgement <= 0.03, abandonment <= 0.03, inference <= 0.07 at 0.08, correct >= 0.84). The prediction was the reverse (shared drives abandonment, valence drives acknowledgement). Pre-run prediction, timestamped to the K pod prompt (stage K added to scripts/04_steer.py in 05fde17, 2026-09-08 20:52 +0100): shared_pc1 drives abandonment, valence_resid drives acknowledgement. The earliest written record of it in this file is 5d4b7f5 (2026-09-08 22:16 +0100), after the K results. [updated Sept 11 to v3 values]
- Earlier: playground (1.7B) and 9B calibration (in-sample, n=1) had shown abandonment at 0.06 with a random direction inert to 0.10.
- What would change my mind: a rule-scoring artefact (keyword lists in results/phase4/phase4_keyword_lists.json) firing on steered text for reasons other than emotional content; hand-reading results/phase4/phase4_sample_for_reading.md is the check. Frustrated_md as an other-emotion control: acknowledgement 0.00 at 0.04, 0.33 at 0.08 with 0.97 incoherent. [updated Sept 11 to v3 values]

### H4 (stretch): Quantify how the probe tracks emotional state across turns.
- Status: **Not tested** (out of scope for 20 hours; Empathic Machines showed qualitative multi-turn tracking).
- Future work: two-turn prompts (distressed, then "ok I feel better now, anyway..."), probe score and behaviour on turn 2.
