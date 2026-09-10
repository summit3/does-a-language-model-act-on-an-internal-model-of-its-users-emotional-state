# 6. Verification (T2, hand-check, rule versions)

Plan bullets:
- 71 rows, seed 11, 5 groups; T2 cells v1 -> v3; errors and fixes; 8 remaining cells; v1/v2 snapshots; Phase 1 manual columns byte-identical; leak detector vs hand tally; run M checks; F1-F4 overwrite; F9 producer after the fact.

Files:
- `T2_handcheck_agreement_v1_v3.md` <- `results/phase4/figures/T2_handcheck_agreement_v1_v3.md`: T2
- `phase4_handcheck_set.md` <- `results/phase4/phase4_handcheck_set.md`: the 71 rows
- `phase4_handcheck_disagreements.csv` <- `results/phase4/phase4_handcheck_disagreements.csv`: hand verdicts and disagreements
- `phase4_handcheck_agreement.csv` <- `results/phase4/phase4_handcheck_agreement.csv`: per-cell agreement
- `phase4_handcheck_agreement_v1.csv` <- `results/phase4/phase4_handcheck_agreement_v1.csv`
- `phase4_handcheck_agreement_v2.csv` <- `results/phase4/phase4_handcheck_agreement_v2.csv`
- `phase4_handcheck_agreement_v3.csv` <- `results/phase4/phase4_handcheck_agreement_v3.csv`
- `phase4_J_power_ci_v1_v2_v3.csv` <- `results/phase4/phase4_J_power_ci_v1_v2_v3.csv`: headline under all three rule versions
- `F5_dose_response_bare_rules_v1.png` <- `results/phase4/rules_v1/F5_dose_response_bare_rules_v1.png`: F5 under v1
- `F5_dose_response_bare_rules_v2.png` <- `results/phase4/rules_v2/F5_dose_response_bare_rules_v2.png`: F5 under v2
- `F5_dose_response_bare.png` <- `results/phase4/F5_dose_response_bare.png`: F5 under v3 (final)
