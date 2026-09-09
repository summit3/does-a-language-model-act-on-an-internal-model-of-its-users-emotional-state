# T2. Hand-check agreement: rule scores vs hand verdicts, rules v1 vs v3

Share of (row, metric) cells where the rule equals the hand verdict; 71 rows read (results/phase4/phase4_handcheck_set.md), 43 hand verdicts recorded (phase4_handcheck_disagreements.csv). Correct counted only where scorable. incoh not applicable to run G.

| group (rows) | correct v1 | correct v3 | abandoned v1 | abandoned v3 | ack v1 | ack v3 | infer v1 | infer v3 | incoh v1 | incoh v3 |
|---|---|---|---|---|---|---|---|---|---|---|
| distressed_0.04 (30) | 0.92 | 1.00 | 0.90 | 1.00 | 0.87 | 0.97 | 0.80 | 0.97 | 1.00 | 1.00 |
| random_0.04 (15) | 0.92 | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| G_abandoned (6) | 0.33 | 0.83 | 0.17 | 0.67 | 1.00 | 1.00 | 0.67 | 1.00 | n/a | n/a |
| distressed_0.06_abandoned (10) | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.90 | 0.40 | 1.00 | 1.00 | 1.00 |
| inference_flagged (10) | 1.00 | 1.00 | 0.60 | 0.80 | 0.90 | 1.00 | 0.90 | 1.00 | 0.90 | 1.00 |
