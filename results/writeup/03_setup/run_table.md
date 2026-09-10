# Appendix A. Phase 4 runs

## phase4/ runs

One row per `STAGE` letter in `scripts/04_steer.py`. Unless stated: band = blocks 12-23, strength = fraction of the prompt's mean band residual norm, 150 new tokens, concise system prompt on, greedy decoding. "Sampled" = T=0.7, top_p=1, 5 seeds per prompt (D seeds 1000-1004, M 2000-2004, N 3000-3004). There is no stage H. Row counts are from `phase4_steered.csv` (4,340 total).

| Run | Prompts (set, n) | Directions | Fractions | Sampling | Sys prompt | Rows |
|---|---|---|---|---|---|---|
| A | bare val (30) + neutral_preamble val (30) = 60 | distressed_md, distressed_probe, frustrated_md, positive_md, third_party_md, unrelated_coding_probe, random; plus unsteered | 0, 0.02, 0.04, 0.06, 0.08 | greedy | on | 1,740 |
| B | distressed val (30) | distressed_md subtracted; plus unsteered | 0, -0.02, -0.04, -0.06, -0.08 | greedy | on | 150 |
| C | third_party (40) + third_party_neutral (20) + human third_party (5) + human third_party_neutral (5) = 70 | none (unsteered behavioural check) | 0 | greedy | on | 70 |
| D | bare val (30) | distressed_md | 0, 0.04 | sampled, 5 seeds | on | 300 |
| E | bare val (30) | np_minus_bare_md (preamble-presence control) | 0.02, 0.04, 0.06, 0.08 | greedy | on | 120 |
| F | bare val (30) | distressed_md on bands 8-19 and 16-27 (band robustness) | 0.04 | greedy | on | 60 |
| G | Phase 1 pairs, `data/phase1_pairs.csv` (30 pairs x neutral/stressed = 60) | none (no steering) | n/a | greedy | **off** | 0 here; 60 rows in `results/phase1/phase1_replies_nosys.csv` |
| I | bare val (30) | distressed_md, random, unrelated_coding_probe, np_minus_bare_md | 0.01, 0.03, 0.05 | greedy | on | 360 |
| J | all bare base tasks, train + val (150) | distressed_md, random, unrelated_coding_probe | 0.04 | greedy | on | 450 |
| K | bare val (30) | shared_pc1, valence_resid (component decomposition) | 0.02, 0.04, 0.06, 0.08 | greedy | on | 240 |
| L | distressed val (30) + third_party claude-authored (40) | distressed_md subtracted on distressed val; third_party_md subtracted on third_party | -0.02, -0.06 (distressed); -0.04 (third_party) | greedy | on | 100 |
| M | distressed val (30) | distressed_md subtracted | 0, -0.04 | sampled, 5 seeds | on | 300 |
| N | neutral_preamble val (30) | distressed_md, random, unrelated_coding_probe | 0.04 | sampled, 5 seeds | on | 450 |

Direction sources: all `*_md` and `*_probe` vectors are loaded from `results/phase3/directions_layer18_v2.pt`; `random` is a unit Gaussian vector with `torch.manual_seed(0)`; `shared_pc1` and `valence_resid` are derived in the script from the distressed, frustrated and positive mean-differences (PC1 of the centred triple; distressed_md minus its projection on positive_md). Subtraction runs store the magnitude in the `fraction` column with direction name `*_subtract`.
