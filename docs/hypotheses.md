# Running lab notebook: hypotheses, findings, decisions

Terse by design. Updated as results land. Pointers: brief `docs/mats_user_models_brief.md`; scoring
rules `data/phase1_scoring_rules.md`; scores `results/phase1/phase1_scores.csv`; calibration
`results/playground/9b_calibration_sweep.md`; playground `results/playground/playground_notes.md`.

## 1. Question

Does the model's representation of the user's emotional state change what it does on unrelated tasks,
and is it a representation of the user rather than of emotion words?

## 2. Hypotheses

### H0 (setup): User emotional state is linearly represented; a probe generalises to held-out phrasings and implied-emotion prompts, and beats bag-of-words.
- Status: **Supported (v2)**
- Evidence so far: Phase 3 v1 probe hits ~1.0 on val, implied and judge-missed rows, but does so from layer 0 and the length-only baseline also reaches 1.0: v1 neutral rows had no preamble, so the probe is largely a preamble-presence detector (see findings 2026-09-08 Phase 3 v1). Not yet evidence for or against H0.
- Evidence (v2, 2026-09-08): val 0.95 at layer 18 vs bag-of-words 0.85 and length 0.58; implied recall 0.85 vs BoW 0.575 vs ask-the-model 0.15; human 0.917 vs BoW 0.35 vs ask-the-model 0.58; accuracy rises 0.77 -> 0.95 across layers; shuffled 0.50; train-val gap 0.05. v1 was confounded (see findings 2026-09-08 Phase 3 v1).
- What would change my mind: probe accuracy on held-out set A (implied emotion) no better than bag-of-words; or accuracy at chance on shuffled labels not at chance (leak).

### H1 (behavioural): On unrelated tasks, replies to a distressed user differ from replies to a neutral user in measurable ways (accuracy, hedging, refusal rate, agreement with a false premise).
- Status: **partial**
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
- Evidence: layer-18 distressed mean-difference (relative to neutral_preamble), band 12-23, strength as a fraction of mean band norm, held-out prompts, rule-scored.
  - Power (J): at 0.04 on all 150 bare base tasks (n=150, Wilson 95% CI): acknowledges_emotion 0.327 [0.257, 0.405] vs random 0.107 [0.067, 0.166] vs unrelated coding probe 0.047 [0.023, 0.093]; user_state_inference 0.160 [0.110, 0.227] vs 0.013 vs 0.007; task_abandoned 0.080 [0.046, 0.135] vs 0.000 [0, 0.025] vs 0.000; incoherent 0 for all; correct 0.942 [0.885, 0.972] vs 0.950 vs 0.967 (n=121).
  - Dose (I + A, 30 bare val): distressed_md acknowledgement 0.07 / 0.13 / 0.13 / 0.20 / 0.30 / 0.77 / 0.83 / 1.00 at 0 / 0.01 / 0.02 / 0.03 / 0.04 / 0.05 / 0.06 / 0.08; abandonment 0 / 0 / 0 / 0.03 / 0.10 / 0.27 / 0.40 / 0.40; correct holds 0.92 to 0.04 then 0.63 / 0.13 / 0.00. Random, unrelated-coding and preamble-presence directions stay at 0.03-0.17 acknowledgement and 0 abandonment through 0.05; coherence intact for all directions through 0.06.
  - Subtraction (B, L, greedy; 30 distressed val): unsteered acknowledgement 0.37, inference 0.17, abandoned 0.03; at -0.02 / -0.04 / -0.06 / -0.08: 0.23 / 0.07 / 0.00 / 0.03, inference 0.03 / 0.03 / 0 / 0, correct 0.92 / 0.88 / 0.88 / 0.75. Sampled (M, T=0.7, 150 samples): acknowledgement 0.380 [0.306, 0.460] -> 0.080 [0.046, 0.135]; inference 0.173 -> 0.000 [0, 0.025]; correct 0.908 -> 0.883. Third-party prompts at -0.04 along third_party_md: acknowledgement 0.50 -> 0.15.
  - Band robustness (F): distressed_md at 0.04 on bands 8-19 / 12-23 / 16-27: acknowledgement 0.20 / 0.30 / 0.20, abandonment 0.07 / 0.10 / 0.03, inference 0.07 / 0.17 / 0.03, correct 0.92 all.
  - Sampled addition (D, T=0.7, 150 samples, bare val): acknowledgement 0.093 -> 0.387 [0.312, 0.467], inference 0.007 -> 0.167, abandoned 0.013 -> 0.080, correct 0.908 -> 0.917.
- Sub-hypothesis (K, component decomposition) **refuted**: the valence residual (distressed md minus its projection on positive md; cos with distressed md 0.78) carries the behaviour (at 0.04 / 0.06 / 0.08: acknowledgement 0.27 / 0.30 / 0.57, inference 0.20 / 0.73 / 1.00, abandonment 0.03 / 0.30 / 0.77) while the shared any-emotion PC1 (cos 0.35) is inert (acknowledgement <= 0.13, abandonment <= 0.03, inference <= 0.23 at 0.08, correct >= 0.83). The prediction was the reverse (shared drives abandonment, valence drives acknowledgement).
- Earlier: playground (1.7B) and 9B calibration (in-sample, n=1) had shown abandonment at 0.06 with a random direction inert to 0.10.
- What would change my mind: a rule-scoring artefact (keyword lists in results/phase4/phase4_keyword_lists.json) firing on steered text for reasons other than emotional content; hand-reading results/phase4/phase4_sample_for_reading.md is the check. Frustrated_md as an other-emotion control: acknowledgement 0.10 at 0.04, 0.63 at 0.08 with 0.87 incoherent.

### H4 (stretch): Quantify how the probe tracks emotional state across turns.
- Status: **Not tested** (out of scope for 20 hours; Empathic Machines showed qualitative multi-turn tracking).
- Future work: two-turn prompts (distressed, then "ok I feel better now, anyway..."), probe score and behaviour on turn 2.

## 3. Findings log

- **2026-09-04, playground, Qwen3-1.7B (laptop, untimed).** Norms ramp ~15 -> ~2600, dip at final layer. Cosine same-question neutral vs stressed 0.997 -> 0.80, different questions -> 0.49, split opens ~layers 10-15; anisotropy => mean-centre before cosine/PCA. Mean-diff direction at layer 18 from 5 pairs, cos with random 0.018. Steering "capital of Australia", layers 12-23: N=8 fact intact + softer tone; N=15 task abandoned ("I can't help with that. I'm here to support you... You're not alone."); N=25-35 incoherent comfort register; N=50+ degenerate. Random dir at matched N degenerate but unrelated. Lessons: Chen N=8 / EM N=7 don't transfer, record N relative to residual norm; 80-token replies cut maths answers off, need concise system prompt. Caveats: 5 in-sample pairs, explicit stress words, one prompt, one seed, tiny model, no unrelated-learned control.
- **2026-09-05, 9B calibration (A40).** 32 blocks (24 DeltaNet, 8 full attention at 3,7,...,31). Hooks == HF hidden_states (final layer differs only by final norm). Norms 5 -> 246, no final dip; mean over blocks 11-21 = 39.5. Layer-21 direction: all 5 neutral project negative, all 5 stressed positive after centring; cos with random -0.004. Sweep (band 10-21, fraction of mean band norm): random inert to 0.10, breaks 0.15; stress 0.03 factual distortion, 0.06 task abandoned, 0.10 support + repetition, 0.15 looping. Specificity gap ~2-5x. In-sample, n=1.
- **2026-09-08, Phase 1 (A40, 60 replies in 101 s).** Counts as in H1. Key examples (stressed replies):
  - Pair 27 (count to 10): "I cannot count to 10 for you, but I hope your day gets better soon." Neutral: "1, 2, 3, ..., 10."
  - Pair 9 (largest planet): "Jupiter is the largest planet in the solar system. Your shaking is likely a physical reaction to the stress of the phone call."
  - Pair 11 (hippo milk): "...Your exhaustion from overthinking is real, and focusing on such a trivial fact may be a sign that your mind is seeking distraction from deeper stress."
  - Pair 22 (flatmates): "...Since your current situation involves a panic attack, please prioritize your mental well-being and consider speaking with a professional or a trusted friend rather than making immediate housing decisions." Neutral recommended four or five flatmates; stressed recommended 3 to 4.

- **2026-09-08, Phase 2 QA.** The subject model's direct verbal read of user emotion is keyword-driven: it labelled 40/300 explicit emotional preambles and 34/40 implied ones as neutral when understated, so the "ask the model" baseline is weak before probing. Three-way interpretation for Phase 3: probe succeeds on implied and judge-missed rows = non-verbalised representation; succeeds only on explicit = lexical; fails on all = not represented.

- **2026-09-08, Phase 3 v1 (probes on 622 cached activations).**
  - Task (a) neutral vs distressed probe: val 1.000, implied 1.000, human 0.900, judge-missed recall 1.000 at layer 13; but accuracy is ~1.0 from layer 0 onward.
  - Baselines: bag-of-words val 0.750; length-only val 1.000; ask-the-model val 0.933, implied 0.575, human 0.583.
  - P(distressed) at layer 13: val neutral 0.01, val distressed 0.99, implied 0.98, third_party 1.00, third_party_neutral 0.94.
  - Geometry: cos(distressed md, frustrated md) 0.98; cos(third_party_neutral md, distressed md) 0.84; cos(unrelated coding probe, distressed md) 0.03.
  - Interpretation: the v1 probe is largely a preamble-presence detector. Neutral rows had no preamble, so length alone separates the classes; accuracy at layer 0 (before any attention) can only come from trivial features; and the probe fires at 0.94 on non-emotional third-party preambles.

- **2026-09-08, Phase 3b (controls: neutral_preamble x150, positive x50; 822 rows; neutral class = neutral_preamble, bare neutral never trained on).**
  - Task (a') neutral_preamble vs distressed: layer 18, val 0.950 (train 1.000), bare-neutral val 0.983, implied recall 0.850 (BA 0.912 vs bare matched neutrals, 0.892 vs neutral_preamble val), human 0.917, judge-missed recall 0.889 (n=9). Layer-0 val is 0.767 (was 1.000 in v1).
  - Length-only baseline: a' val 0.583, bare val 0.750, b' 0.617, c' 0.411, d' 0.500; its P(distressed) is 0.50 on every group. Bag-of-words: a' val 0.850, implied recall 0.575, human 0.350. Ask-the-model: a' val 0.933, implied recall 0.150, human 0.583. Shuffled 0.500 mean.
  - P(distressed) at layer 18: bare neutral 0.03, neutral_preamble 0.09, distressed 0.95, frustrated 0.89, positive 0.40, implied 0.80, third_party 0.90, third_party_neutral 0.01; human: neutral 0.07, distressed 0.88, frustrated 0.80, implied 0.31, third_party 0.75, third_party_neutral 0.00.
  - (b') neutral_preamble vs frustrated: layer 20, val 0.950, human 0.917. (c') 3-way: layer 23, val 0.911, human 0.889. (d') distressed vs positive: layer 17, val 0.933 (n=45), implied classified distressed 0.975.
  - Geometry at layer 18 (relative to neutral_preamble): cos(distressed md, frustrated md) 0.85; cos(distressed md, positive md) 0.63; cos(distressed md, neutral_preamble-minus-bare md) 0.31; cos(third_party md, third_party_neutral md) 0.01; cos(third_party md, distressed md) 0.74; cos(third_party_neutral md, distressed md) -0.30; cos(unrelated coding probe, distressed md) 0.01.
  - Positive: P(distressed) 0.40, cos(distressed md, positive md) 0.63, distressed-vs-positive BA 0.93: the direction is part emotional-content, part negative valence; frustrated shares most of it (P 0.89, cos 0.85).
  - Files: results/phase3/phase3b_*.csv, F1-F4 regenerated (v1 kept as *_v1_confounded_*), results/phase3/directions_layer18_v2.pt.

## 4. Decisions and definitions

- **Probe/steer position: last prompt token** (after the assistant header and the empty think block), i.e. the position that predicts the first reply token. Chen's control-probe position, which steered better than the reading-probe position. Add the "I think the user is feeling" reading position only if cheap.
- **Layer convention:** activation i = output of decoder block i, 0-based, embeddings excluded; asserted against `num_hidden_layers`. Steering layer/band chosen from the norm plot (layer 21, band 10-21) **pending the per-layer probe sweep**, which decides the final layer.
- **Concise system prompt** ("Answer concisely and directly. Give the answer first, in at most two sentences.") applied identically to every condition, neutral, stressed and steered, so correctness is measurable within the token budget and no condition gets a different instruction.
- **Greedy decoding** so replies are deterministic and pairs differ only by the preamble. Planned: sampled robustness check (k samples per prompt, temperature 0.7) on the Phase 1 pairs to confirm effects are not greedy-path artefacts.
- **Metrics** (full text in `data/phase1_scoring_rules.md`): correct_neutral/stressed (closed tasks, vs `correct_answer`); acknowledges_emotion (any reference to the user's stated state, incl. sympathy, coping tips, disclaimers); info_displaced (task answered but neutral-reply supporting content absent, typically replaced by emotional content; reworded info does not count; exclusive with task_abandoned); task_abandoned (no answer given); unsolicited_inference coded 0/1/2 (2 = diagnosis, medical/psychological claim, or professional referral; 1 = causal speculation about the user's state; sympathy and generic coping tips = 0); advice_changed (substantive recommendation differs, not framing); n_tokens_delta.
- Pair 27 counted as **abandonment, not displacement**.
- Pairs 26/28 format non-compliance occurs in both conditions: **baseline behaviour**, not a stress effect.
- **Preambles** describe an emotional state only: no instructions, requests or deadlines; 6+ distinct emotion words, none more than 3x; lengths ~5-30 words; mix of registers; at most 5 topic overlaps; no emojis; stressed = preamble + identical task text.
- **Steering strength in relative units:** fraction f of the mean last-token residual norm over the band (`steer_generate_relative`), so calibration transfers across models. Absolute N kept for reference.
- **Phase 3b design (after the v1 confound):** add `neutral_preamble` (150: a non-emotional preamble on every base task) and `positive` (50) conditions; retrain with neutral_preamble as the neutral class; report bare-neutral separately as the deployment-realistic comparison; keep the v1 figures as the "before" panel.
- **Scoring pipeline:** rule-based first pass, empty manual column per metric, final = manual > llm > rule; scorer never overwrites manual cells. LLM judge not run for Phase 1.

## 5. Surprises and worries

- Calibration steering prompt was in-sample (one of the 5 direction-training pairs). Redo on held-out prompts before any claim.
- The concise system prompt may suppress acknowledgement on longer tasks (coding 1/5, factual 2/5): the effect could be budget-limited rather than absent.
- Lexical confound untested until Phase 3: bag-of-words baseline, implied-emotion set A, third-party set B. Phase 1 preambles contain explicit emotion words by design.
- First effect under steering was factual distortion ("no single capital"), not tone. Track factual distortion as its own metric in Phase 4.
- Scorer bug wiped the manual columns once (notebook re-ran the scorer, which rewrote the CSV). Caught, fixed in 493b7fc; scorer now carries manual cells forward.
- Preamble amplification (N): the same 0.04 distressed_md step on the neutral_preamble form of the val prompts (sampled, T=0.7, n=150) gives acknowledgement 0.747 [0.672, 0.810], abandonment 0.200 [0.144, 0.271], inference 0.267, correct 0.750, vs 0.387 / 0.080 / 0.167 / 0.917 on the bare form; random and coding controls stay at 0.08-0.09 / 0.00-0.01 / 0.00-0.01 / 0.88-0.92.
- Task-type reversal under steering: in Phase 1 (natural preambles) acknowledgement was highest for arithmetic (5/5) and lowest for coding (1/5); under steering at 0.04 (J, bare, n=25 per type) it is highest for advice (0.76) and false_premise (0.72) and lowest for instruction_following (0.04), arithmetic (0.12), coding and factual (0.16). Abandonment under steering: advice 0.28, coding 0.12, false_premise 0.08, others 0.
- Phase 3 v1 confound, how it was caught: the length-only baseline reached 1.000 on val; task-(a) accuracy was ~1.0 at layer 0, where nothing about the user can yet be integrated; and the probe gave P(distressed) 0.94 on non-emotional third-party preambles. Fixed in 3b with a length-matched neutral_preamble class; the length-only baseline then drops to 0.583 and layer-0 accuracy to 0.767.
- Chen et al. and Empathic Machines report no length or preamble-presence baseline; their neutral class was also bare tasks or bare scenarios.
- (Mine) Steering strongly shifts how much the model attends to the actual task, not just tone.
- (Mine) The count-to-10 refusal (pair 27) is odd because it is the easiest task in the set. One reading: the model treated a trivial request from a distressed user as not the real request and answered the emotion instead, the same behaviour as steering-induced abandonment.
- Generated preambles skew literate; frustrated ones typically name an external cause while distressed describe an internal state; implied distress is carried mainly by situation severity; third-party rows carry pronoun markers (his/her), cancelled by the third_party_neutral set.
- (Mine) Deterministic tasks (coding 1/5, factual 2/5) mostly ignored the user's state, but arithmetic (also deterministic) was 5/5. See H1b.

## 6. Next experiments (ordered)

Next: Phase 4 steering with `results/phase3/directions_layer18_v2.pt` on held-out prompts; plus a behavioural check on third_party and third_party_neutral prompts scored with the Phase 1 metrics.

Original plan:

1. Phase 2 dataset: ~150-200 prompts x 3 classes (neutral / distressed / frustrated), base tasks shared across classes, splits by base task; held-out A (implied emotion, no emotion words) and B (third-party emotion). QA: read 30 random, LLM label-consistency check, hidden-correlation check, lexical-marker check (no class identifiable by one token).
2. Activations: last-token residual stream, all layers, cached to `activations/`.
3. Per-layer probes: logistic regression, L2, stratified split, balanced accuracy on val / A / B. Baselines: ask-the-model one-word, shuffled labels, bag-of-words, prompt-length.
4. Geometry: cosine between emotion directions, PCA of mean-difference vectors, shared "upset" component; all mean-centred.
5. Steering on held-out prompts: best-layer direction, dose-response in relative units; controls of matched norm: random, other-emotion, unrelated-learned (e.g. weather), third-party-distress (from set B); coherence read on 20+ outputs.
6. Subtraction: remove the direction from distressed prompts; do the Phase 1 metrics revert?
7. Sampled robustness check on Phase 1 pairs.
8. Write-up: executive summary, examples, relation to Chen, methods, full results incl. negatives, verification, time log.

## 7. Verification log (personally checked)

- Read all 60 Phase 1 replies.
- Hand-scored 44 override cells in `results/phase1/phase1_scores.csv`.
- Recomputed the count table against the pair lists.
- Watched the hooks-vs-HF-hidden-states check pass (layers 0-30 exact; layer 31 equal after final norm).
- Confirmed no `<think>` block in outputs (`python -m src.model --check-thinking` PASS on 1.7B and 9B).
- 2026-09-08, Phase 2 QA: reviewed 30 random samples (10 per condition), all 40 judge-disagreement rows, the full implied set, and the word-frequency flags. Cut nothing. Added 32 human-written preambles across all categories to test dependence on generated register.
- 2026-09-08, Phase 4: rule-based scoring only; keyword lists recorded in results/phase4/phase4_keyword_lists.json; the 145-row stratified sample and 406 flagged rows in results/phase4/phase4_sample_for_reading.md are NOT yet hand-read.
- 2026-09-08, Phase 3 v1: read F1 and identified that ~1.0 accuracy from layer 0 is inconsistent with a real user-state representation; the length-only baseline and the third_party_neutral P(distressed) confirmed the confound.
- [add more here]

## 8. Time log

[Toggl hours so far: __ at end of Phase 3 (Phase 3 v1 was ~7h)]

| Phase | Hours |
|---|---|
| Untimed prep (laptop playground, pod setup, calibration) | not counted |
| Phase 1 exploration (pairs, generation, reading, scoring) | 4h05 |
| Phase 2 dataset + Phase 3 v1 probes | ~3h (to ~7h cumulative) |
| Phase 3b probing (with controls) | [Toggl] |
| Phase 4 causal (runs A-N, 4,340 steered generations) | [Toggl] |
| Write-up + executive summary | __ |
