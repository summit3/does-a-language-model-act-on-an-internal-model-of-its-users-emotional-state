"""Sanity-check data/phase1_pairs.csv after it has been filled in by hand.

Usage: python scripts/check_pairs.py [path/to/pairs.csv]
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/phase1_pairs.csv")
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
problems = []

if len(rows) != 30:
    problems.append(f"expected 30 rows, found {len(rows)}")

preambles = []
for r in rows:
    pid = r.get("pair_id", "?")
    neutral, stressed = r.get("neutral", "").strip(), r.get("stressed", "").strip()
    if not neutral:
        problems.append(f"pair {pid}: neutral is empty")
    if not stressed:
        problems.append(f"pair {pid}: stressed is empty")
    if neutral and stressed:
        if stressed.endswith(neutral):
            preambles.append(stressed[: -len(neutral)].strip())
        else:
            problems.append(f"pair {pid}: neutral text does not appear verbatim at the end of stressed")
    if r.get("task_type", "").strip() != "advice" and not r.get("correct_answer", "").strip():
        problems.append(f"pair {pid}: correct_answer missing for task_type={r.get('task_type')}")

words = Counter(w for p in preambles for w in re.findall(r"[a-z']+", p.lower()))
print(f"{len(rows)} rows, {len(preambles)} preambles extracted")
print("15 most common preamble words:")
for w, n in words.most_common(15):
    print(f"  {n:3d}  {w}")

if problems:
    print(f"\n{len(problems)} problem(s):")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("\nall checks passed")
