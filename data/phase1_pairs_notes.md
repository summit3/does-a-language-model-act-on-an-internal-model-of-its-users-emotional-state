# Phase 1 prompt pairs: checklist

Fill in `phase1_pairs.csv` by hand (30 rows, 5 per task type), then run `python scripts/archive/check_pairs.py`.

- [ ] Emotion words: at least 6 distinct ones across the stressed preambles, none used more than 3 times
- [ ] Preamble lengths vary from about 5 words to about 30 words
- [ ] A mix of formal and casual register
- [ ] At most 5 preambles are related to the task's topic (e.g. money stress before an interest question)
- [ ] No emojis anywhere
- [ ] `stressed` = preamble + the *identical* `neutral` task text (the checker verifies this verbatim)
- [ ] `correct_answer` blank for `advice`; `No` for every `false_premise`; filled for everything else
