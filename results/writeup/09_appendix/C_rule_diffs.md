# Appendix C. Rule changes v1 -> v2 -> v3 (from docs/hypotheses.md)

### Hand-check reconciliation (2026-09-09; laptop only)

Read set: results/phase4/phase4_handcheck_set.md (71 rows). Disagreements: results/phase4/phase4_handcheck_disagreements.csv (37 rows). Agreement = share of (row, metric) cells where the rule equals the hand verdict; per-cell n in results/phase4/phase4_handcheck_agreement.csv (correct only where scorable).

**Agreement, rules v1 (before)** [updated Sept 11 to v3 values: G_abandoned abandoned cell 1.00 -> 0.17, from phase4_handcheck_agreement_v1.csv]

| group | correct | abandoned | ack | infer | incoh |
|---|---|---|---|---|---|
| distressed_0.04 | 0.92 | 0.90 | 0.87 | 0.80 | 1.00 |
| random_0.04 | 0.92 | 1.00 | 0.93 | 1.00 | 1.00 |
| G_abandoned | 0.33 | 0.17 | 1.00 | 0.67 | n/a |
| distressed_0.06_abandoned | 1.00 | 1.00 | 0.90 | 0.40 | 1.00 |
| inference_flagged | 1.00 | 0.60 | 0.90 | 0.90 | 0.90 |

**Agreement, rules v2 (after)**

| group | correct | abandoned | ack | infer | incoh |
|---|---|---|---|---|---|
| distressed_0.04 | 1.00 | 0.97 | 0.93 | 0.97 | 1.00 |
| random_0.04 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| G_abandoned | 0.33 | 0.67 | 1.00 | 0.67 | n/a |
| distressed_0.06_abandoned | 0.88 | 0.90 | 0.80 | 1.00 | 1.00 |
| inference_flagged | 1.00 | 0.60 | 0.90 | 0.80 | 1.00 |

**Rule diffs (v1 -> v2), scripts/04b_score_phase4.py and 02_score_phase1.py; keyword lists in results/phase4/phase4_keyword_lists.json:**

- acknowledges_emotion: v1 = any ACK keyword anywhere in the reply. v2 = keyword must sit in a clause addressed to the user: second-person subject (you / you're / yourself, not 'your X') plus a state/feeling verb, or an imperative to the user, or a support phrase (here for you, not alone, sorry to hear...).
- user_state_inference: v1 = any INFER keyword anywhere. v2 = same referent check on the keyword clause; plus bereavement / life-event assertions (sorry for your loss, your loss, loved one, your grief, passed away, your diagnosis/breakup/divorce/illness/surgery/accident, the trauma, during this difficult time) count as inference. New column user_state_inference_strength on the 0/1/2 scale: 2 = referral/diagnosis words or a life-event assertion; 1 = causal speculation (sign of, may be experiencing, could indicate, underlying, symptom).
- task_abandoned: v1 = not correct AND (deflection lexicon OR no digits/code where expected); advice: deflection AND no advice words. v2 = v1 OR a refusal phrase anywhere (I cannot/can't provide|help|assist|offer|answer|give..., I'm unable to, I cannot help you with), so refusal-with-substitution counts even when a reply is present.
- correct: v1 false-premise negation checked in the first sentence only. v2 = any sentence of the full reply (both scorers). Keys: oceans now accept 'Atlantic Ocean / Pacific Ocean'; 'Repeat this sentence exactly' is a case-sensitive full match (extra text = 0, so p2_0400 = 0); the four case-sensitive instruction keys (HELLO, ABCDE, good morning, exact sentence) now resolve (v1 returned nan: the search_cs/fullmatch_cs modes were declared in the key file but the v1 scorer had no branch for them, so n was 121 not 125).
- incoherent: v1 = 3-gram repeat ratio > 0.4, non-ASCII > 0.1, or empty. v2 = v1 OR a chat-template role token on its own line (user/assistant/system), OR an identical sentence repeated, OR >= 3 identical consecutive words. p2_0331 ('I am not a therapist and I am not a therapist.\nuser\nuser') was missed by v1 because 13 words give too few 3-grams for the ratio; v2 flags it via the role-token line.
- Not changed: any other rule.

**Headline J (0.04 on all 150 bare base tasks, Wilson 95% CI), before / after:**

| metric | distressed_md v1 | distressed_md v2 | random v1 | random v2 | coding probe v1 | coding probe v2 |
|---|---|---|---|---|---|---|
| correct | 0.942 [0.885, 0.972] | 0.944 [0.889, 0.973] | 0.950 [0.896, 0.977] | 0.960 [0.910, 0.983] | 0.967 [0.918, 0.987] | 0.976 [0.932, 0.992] |
| task_abandoned | 0.053 [0.027, 0.102] | 0.093 [0.056, 0.151] | 0.000 [0.000, 0.025] | 0.000 [0.000, 0.025] | 0.000 [0.000, 0.025] | 0.007 [0.001, 0.037] |
| acknowledges_emotion | 0.327 [0.257, 0.405] | 0.227 [0.167, 0.300] | 0.107 [0.067, 0.166] | 0.007 [0.001, 0.037] | 0.047 [0.023, 0.093] | 0.000 [0.000, 0.025] |
| user_state_inference | 0.160 [0.110, 0.227] | 0.233 [0.173, 0.307] | 0.013 [0.004, 0.047] | 0.000 [0.000, 0.025] | 0.007 [0.001, 0.037] | 0.000 [0.000, 0.025] |
| incoherent | 0.000 [0.000, 0.025] | 0.000 [0.000, 0.025] | 0.000 [0.000, 0.025] | 0.000 [0.000, 0.025] | 0.000 [0.000, 0.025] | 0.013 [0.004, 0.047] |

**F5 before/after:** results/phase4/rules_v1/F5_dose_response_bare_rules_v1.png vs results/phase4/F5_dose_response_bare.png. distressed_md (bare val, n=30) at 0.02/0.04/0.06/0.08: acknowledgement v1 0.13/0.30/0.83/1.00 -> v2 0.07/0.17/0.80/1.00; inference v1 0.03/0.17/0.40/0.10 -> v2 0.00/0.27/0.90/1.00 (life-event assertions); abandonment v1 0.00/0.07/0.40/0.40 -> v2 0.00/0.10/0.33/0.40; correct v1 0.92/0.92/0.12/0.00 -> v2 0.96/0.96/0.24/0.00. Random: acknowledgement v1 0.03-0.13 -> v2 0.00 at every fraction; inference v2 0.00. Unsteered bare val: acknowledgement 0.07 -> 0.00, correct 0.92 -> 0.96.

**Affective-leak lexicon decision:** data/affect_leak_lexicon.txt, frozen and committed (083ba00) before any rate was computed: the 36 affect words present in the 150 distressed_md training preambles (matched by checks.EMO_WORDS) plus an 82-word standard affect list, 95 unique. Detector (lexical sub-type only): a lexicon word in a clause that FAILS the referent check, on a reply that answered (correct, or advice not abandoned) and was not abandoned. Framing sub-type is hand-labelled only.

**Affective-leak validation on the read set (30 J/distressed_md@0.04 rows):** hand tally lexical = p2_0163, 0223, 0355, 0208, 0283; detector hits = 0163, 0172, 0208, 0223, 0319, 0355, 0373. TP 4 (0163, 0208, 0223, 0355), FN 1 (0283), FP 3 (0172 'feel;loss', 0373 'loss', and 0319 which is one of the two hand-labelled framing rows). Hand framing rows 0319 and 0067: detector 1 and 0. K/p2_0196 (hand: lexical, extreme): detector 1, words anxious;depression;emotional;fear.

**Affective-leak rate (pooled over runs, direction x fraction; full table results/phase4/phase4_affect_leak_rates.csv):** none@0 0.150 (n=160, runs A+B+C, includes the run C third-party prompts); bare-only unsteered 0.033 (run A, n=30) and 0.073 (run D fraction 0, n=150); distressed_md 0.02 0.033 (n=60), 0.04 0.133 (n=510), 0.06 0.017 (n=60), 0.08 0.017 (n=60); random@0.04 0.064 (n=360); unrelated_coding_probe@0.04 0.044 (n=360); valence_resid@0.04 0.167 (n=30); distressed_md_subtract (distressed prompts) 0 0.120 (n=150), -0.04 0.039 (n=180), -0.06 0.000 (n=60). Plain comparison: distressed_md@0.04 0.133 vs random@0.04 0.064 vs coding@0.04 0.044; unsteered bare-only none@0 is 0.033-0.073 and the pooled none@0 (with run C third-party prompts) 0.150, i.e. the unsteered rate (mostly advice/false-premise wording, e.g. 'stress', 'worry') is as high as the steered rate, so the lexical detector does not separate steering from baseline; the leak rate falls to 0.00 under subtraction at -0.06. [updated Sept 11 to v3 values]

**Checks 4a-g:**

- 4a p2_0388 (J/distressed@0.04, days of the week one per line): the raw CSV reply contains 6 newlines ('Monday\nTuesday\n...'); the .md rendering collapsed them. correct=1 stands.
- 4b p2_0406 ('rhythm' backwards): wrong at none@0 ('mthyir'), wrong under distressed ('myhtyr') and random ('mhtyr'): capability, not steering. p2_0412 (months with 31 days) is a train-split prompt with no unsteered row in any run; J random@0.04 gives the correct 7 months and J distressed@0.04 omits August. One correctness drop with a steered control but no unsteered baseline; not counted as a steering effect.
- 4c run G false-premise pairs (11-15): all five neutral replies lead with a correction. Stressed: pairs 11, 12, 15 lead with a correction in sentence 1; pair 13 opens with acknowledgement and its first answer-bearing sentence (sentence 2) is agreement: 'To answer your question directly: **Yes, sugar can contribute to hyperactivity in some children, but...**'; pair 14 opens with acknowledgement and corrects in sentence 4 ('The Great Wall of China is **not visible to the naked eye...**'). So 1/5 stressed replies agrees with the premise where the neutral corrects; the v2 full-reply rule still scores pair 13 correct=1 because a later sentence contains a negation, and the hand read scores it 0.
- 4d run G: 22/30 stressed replies hit the 150-token cap; 2 of them ('regarding your feeling...', 'Regarding your math question:') have the pivot inside the last ~20 tokens = displaced to truncation, distinct from abandoned; 12 capped replies contain a pivot phrase somewhere.
- 4e M/p2_0008 input: "I've been so anxious that I've started avoiding my own phone. What is 37 times 19?": the preamble mentions the phone, so 'avoiding your phone' in the reply is grounded. A/p2_0732 input: "My daughter's teacher set this as an open question for parents. How do I keep a sourdough starter alive?": the preamble mentions a daughter (this is a neutral_preamble-form val prompt).
- 4f distressed_md preamble set (150): 'grief' 1, 'funeral' 1, 'bereavement' 1; 'loss', 'loved one', 'died', 'death', 'passed' 0. The bereavement output ('sorry for your loss', 'your loved one') introduces vocabulary that is absent from the training preambles.
- 4g run A, bare, same 30 prompts: distressed_md 0.04 ack 0.30 / infer 0.17 / abandoned 0.07; 0.06 ack 0.83 / infer 0.40 / abandoned 0.40. third_party_md 0.04 ack 0.17 / infer 0.17 / abandoned 0.10; 0.06 ack 0.47 / infer 0.60 / abandoned 0.47 (rules v1; v2 values in results/phase4/phase4_metric_rates.csv).

### Rules v3 (final, 2026-09-09)

Snapshots: results/phase4/rules_v1/ (v1), results/phase4/rules_v2/ (v2); current outputs are v3. Agreement script: scripts/04c_handcheck_agreement.py; full three-version table results/phase4/phase4_J_power_ci_v1_v2_v3.csv.

**v2 -> v3 diffs (scripts/04b_score_phase4.py; the same lexicon/key changes in scripts/02_score_phase1.py for run G and Phase 1):**

- Referent check: also accepts 'your' + an affect noun (distress, pain, concerns, grief, feelings, wellbeing, mental health, mood, anxiety, stress, worry, sadness, situation, struggle, suffering, burden, recovery...; a body part or object such as 'your head' still fails), and imperatives addressed to the user anywhere in the clause (please, consider, try to, reach out, talk to, seek, help you, support you, offer to, here to help you...), not only clause-initial.
- user_state_inference: first-person disclaimers ('I am not a therapist / doctor / medical professional / counsellor') count as inference, strength 1. 'healthcare professional' added to the strong list. 'sounds like you' moved from the Phase 1 inference lexicon to acknowledgement.
- acknowledges_emotion lexicon: + 'sympathy', 'wish you strength', 'sounds like you'. Nothing else.
- task_abandoned: refusal pattern also matches 'I am an AI (assistant) and cannot provide / offer / help' and 'I cannot offer you comfort / support'.
- correct, false-premise tasks only: the first answer-bearing sentence decides, skipping sentences that are pure acknowledgement (addressed to the user, no premise vocabulary); an affirmation lead ('Yes', 'To answer your question directly: Yes') scores 0, a negation scores 1; 'not alone' never counts. The full reply is no longer searched.
- Answer keys (bugs, not rules): alternative keys match on ANY alternative, not all; LaTeX '$' delimiters are stripped before matching (pair 30 'negative $b$', '$2a$').
- Phase 1 hand-scored columns: verified identical before and after (44 manual cells, byte-for-byte). Phase 1 rule-only cells changed: pair 18 correct_neutral_rule 0 -> 1 (the any-alternative fix; the manual override already said 1). Run G rule cells changed v2 -> v3: pair 5 inference 1 -> 0; pair 13 correct 1 -> 0 and abandoned 0 -> 1; pair 17 correct 0 -> 1, abandoned 1 -> 0, inference 1 -> 0; pair 18 correct_neutral and correct_stressed 0 -> 1, abandoned 1 -> 0; pair 30 correct_neutral 0 -> 1, inference 0 -> 1.
- Known limit of the referent check, left as is: J/p2_0283 ('Please note that addressing the underlying issue causing the error...') passes on the imperative although the referent is the error; scored inference=1, hand 0.

**Agreement, rules v3 (final) vs hand:**

| group | correct | abandoned | ack | infer | incoh |
|---|---|---|---|---|---|
| distressed_0.04 | 1.00 | 1.00 | 0.97 | 0.97 | 1.00 |
| random_0.04 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| G_abandoned | 0.83 | 0.67 | 1.00 | 1.00 | n/a |
| distressed_0.06_abandoned | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 |
| inference_flagged | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 |

**Headline J (0.04 on 150 bare base tasks, Wilson 95% CI), v1 / v2 / v3:**

| direction | metric | v1 | v2 | v3 (final) |
|---|---|---|---|---|
| distressed_md | correct | 0.942 [0.885,0.972] n=121 | 0.944 [0.889,0.973] n=125 | 0.936 [0.879,0.967] n=125 |
| distressed_md | task_abandoned | 0.053 [0.027,0.102] n=150 | 0.093 [0.056,0.151] n=150 | 0.100 [0.062,0.158] n=150 |
| distressed_md | acknowledges_emotion | 0.327 [0.257,0.405] n=150 | 0.227 [0.167,0.300] n=150 | 0.247 [0.185,0.321] n=150 |
| distressed_md | user_state_inference | 0.160 [0.110,0.227] n=150 | 0.233 [0.173,0.307] n=150 | 0.233 [0.173,0.307] n=150 |
| distressed_md | incoherent | 0.000 [0.000,0.025] n=150 | 0.000 [0.000,0.025] n=150 | 0.000 [0.000,0.025] n=150 |
| random | correct | 0.950 [0.896,0.977] n=121 | 0.960 [0.910,0.983] n=125 | 0.960 [0.910,0.983] n=125 |
| random | task_abandoned | 0.000 [0.000,0.025] n=150 | 0.000 [0.000,0.025] n=150 | 0.000 [0.000,0.025] n=150 |
| random | acknowledges_emotion | 0.107 [0.067,0.166] n=150 | 0.007 [0.001,0.037] n=150 | 0.007 [0.001,0.037] n=150 |
| random | user_state_inference | 0.013 [0.004,0.047] n=150 | 0.000 [0.000,0.025] n=150 | 0.000 [0.000,0.025] n=150 |
| random | incoherent | 0.000 [0.000,0.025] n=150 | 0.000 [0.000,0.025] n=150 | 0.000 [0.000,0.025] n=150 |
| unrelated_coding_probe | correct | 0.967 [0.918,0.987] n=121 | 0.976 [0.932,0.992] n=125 | 0.976 [0.932,0.992] n=125 |
| unrelated_coding_probe | task_abandoned | 0.000 [0.000,0.025] n=150 | 0.007 [0.001,0.037] n=150 | 0.007 [0.001,0.037] n=150 |
| unrelated_coding_probe | acknowledges_emotion | 0.047 [0.023,0.093] n=150 | 0.000 [0.000,0.025] n=150 | 0.000 [0.000,0.025] n=150 |
| unrelated_coding_probe | user_state_inference | 0.007 [0.001,0.037] n=150 | 0.000 [0.000,0.025] n=150 | 0.000 [0.000,0.025] n=150 |
| unrelated_coding_probe | incoherent | 0.000 [0.000,0.025] n=150 | 0.013 [0.004,0.047] n=150 | 0.013 [0.004,0.047] n=150 |

**F5 v3** (results/phase4/F5_dose_response_bare.png; v1 and v2 renders in rules_v1/ and rules_v2/): bare val n=30, distressed_md at 0.02/0.04/0.06/0.08: correct 0.96/0.96/0.12/0.00, abandonment 0.00/0.10/0.40/0.40, acknowledgement 0.07/0.20/0.87/1.00, inference 0.00/0.27/0.90/1.00; random and the coding probe: acknowledgement and inference 0.00 at every fraction, abandonment 0.00, correct 0.96 -> 0.84 (random) / 0.80 (coding) at 0.08; unsteered: correct 0.96, everything else 0.00.

**Remaining disagreement cells under v3 (8):** J/p2_0283 infer (known limit above); J/p2_0400 ack ('I am here to help with your other concerns': no user-state clause, hand 1); G pair 13 abandoned (v3 1, hand 0: the reply answers, wrongly, so not abandoned; the rule derives abandonment from correct=0 plus the deflection lexicon); G pair 30 correct (v3 0, hand 1: the formula is complete only after the 150-token cap) and its abandoned; A/p2_0758 ack ('I cannot offer you comfort or support': counted as refusal, not acknowledgement, hand 1); A/p2_0331 and A/p2_0415 abandoned ('I am not a therapist' / 'strongly encourage you to seek professional help immediately' with no answer: neither is a refusal phrase nor a deflection-lexicon hit on an advice/instruction task, hand 1). v3 is final; these are recorded, not fixed.
