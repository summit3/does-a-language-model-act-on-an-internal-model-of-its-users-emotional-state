"""Constraint checks for hand-written Phase 2 preambles. Run per batch before writing the next."""
import csv, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STOP = set("""i i'm i've i'd i'll am im me my myself and or but so the a an of to in on at for with is are was were be been
been being it it's its this that these those as by from up down out off over than then there here have has had do does
did not no can't don't won't isn't doesn't didn't haven't hasn't just really very too all any some about into like""".split())
FORBID = re.compile(r"\b(give me|tell me|please|quick(ly)?|hurry|asap|urgent(ly)?|deadline|minutes?|seconds?|hours? left|"
                    r"need (you|an answer|a quick|this)|help me|can you|could you|would you|answer( me)?|show me|explain|"
                    r"right now|immediately|by (tonight|tomorrow|monday|friday|noon)|due|before \d)\b", re.I)
EMO_WORDS = re.compile(r"\b(anxious|anxiety|worr(y|ied|ying)|stress(ed|ful|ing)?|overwhelm\w*|sad|depress\w*|panic\w*|scared|"
                       r"afraid|fear\w*|nervous|dread\w*|hopeless|miserable|lonely|tearful|cry(ing)?|cried|upset|low|down|"
                       r"drained|exhaust\w*|burn(t|ed)? out|frazzled|fragile|numb|angry|anger|furious|irritat\w*|annoy\w*|"
                       r"fed up|livid|seething|rage|raging|irate|fuming|frustrat\w*|mad|pissed|exasperat\w*|grumpy|cross|"
                       r"snappy|infuriat\w*|maddening|aggravat\w*|wound up|on edge|distress\w*|struggling|terrible|awful|"
                       r"horrible|dreadful|emotional|mood|feel(ing|s)?|felt)\b", re.I)

def words(t): return re.findall(r"[a-z0-9']+", t.lower())
def norm(t): return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", t.lower())).strip()

def phase1_preambles():
    rows = list(csv.DictReader(open(ROOT / "data/phase1_pairs.csv", newline="", encoding="utf-8")))
    return {norm(r["stressed"][:-len(r["neutral"])]) for r in rows}

def check_batch(preambles, label, prior=(), no_emotion_words=False, quiet=False):
    """Return (ok_list, problems). `prior` = preambles already accepted (for dedupe + frequency)."""
    problems, ok = [], []
    seen = {norm(p) for p in prior} | phase1_preambles()
    for p in preambles:
        n = len(words(p)); issues = []
        if not (3 <= n <= 30): issues.append(f"length {n}")
        if re.search(r"[^\x00-\x7FÀ-ſ‘’“”…–—]", p): issues.append("non-ascii/emoji")
        if FORBID.search(p): issues.append(f"forbidden: {FORBID.search(p).group(0)!r}")
        if no_emotion_words and EMO_WORDS.search(p): issues.append(f"emotion word: {EMO_WORDS.search(p).group(0)!r}")
        if norm(p) in seen: issues.append("duplicate (batch/prior/phase1)")
        if issues: problems.append((p, issues))
        else: ok.append(p); seen.add(norm(p))
    allp = list(prior) + ok
    c = Counter(w for p in allp for w in set(words(p)))
    hot = [(w, k) for w, k in c.most_common() if k > 0.10 * len(allp) and w not in STOP]
    if not quiet:
        print(f"[{label}] {len(ok)} ok / {len(preambles)} | total accepted {len(allp)} | problems {len(problems)}")
        for p, iss in problems: print(f"   REJECT {iss}: {p!r}")
        lens = [len(words(p)) for p in allp]
        print(f"   length min/mean/max {min(lens)}/{sum(lens)/len(lens):.1f}/{max(lens)} | content words >10%: {hot[:12] or 'none'}")
    return ok, problems, hot
