# Project Brief: Does the User Model Change Behaviour on Unrelated Tasks?

## Research question

LLMs form internal models of the user (Chen et al. 2024, arXiv 2406.07882: age, gender, education, SES via linear probes on LLaMa2Chat-13B, steerable). Does a model's representation of the user's *emotional state* change its behaviour on tasks that have nothing to do with emotion, e.g. accuracy, hedging, refusals, sycophancy? And is that representation genuinely about the user, rather than emotion as a topic?

Why it matters: this is sandbagging/sycophancy routed through the user model. If a distressed user gets worse answers, or a model agrees with wrong claims more readily, that is a safety-relevant behaviour that only interp can attribute to an internal user representation rather than surface text.

## Prior work to read in prep and cite (be explicit about what is theirs and what is mine)

- Chen et al. 2024 (arXiv 2406.07882): probing recipe, reading vs control probes, steering method, LLM-judge eval, prompting underperforms probing. Demographics only, one old model, static attributes.
- Empathic Machines (link in docs/references.md): Llama-2-7B-chat, 6 emotions, ~200 unvalidated LLM-generated prompts each, Chen-style reading/control probes, steering N~7 on layers 15-32. Found: emotion readable in mid/late layers; steering changes tone and sometimes content (anger -> refusal, fear -> safety language), qualitatively only; representation shifts across turns; probe flips with surface words ("Gross Ugh", emojis). They flagged "language cues vs actual emotional state" as unresolved and did no baselines or controls. So: probe existence and dynamic tracking are NOT results here; the lexical-fragility question and behavioural effects under controls are the gap.
- Detection != Reliable Control (arXiv 2608.24901, Aug 2026): empathy directions decode well but steering only partially shifts automated empathy (tone) scores. Plan for a partial/null H3 and make it clean; your metric is task behaviour, not tone.
- Sofroniew et al. 2026 / E-STEER (arXiv 2604.00005): the MODEL's own emotional state and its effect on tasks. Different from a model's belief about the USER's state. State this distinction.
- Detecting and Steering LLMs' Empathy in Action (arXiv 2511.16699): empathy probes; agents sacrificing task objectives for distressed users. Closest behavioural neighbour.
- Fomin 2026 (arXiv 2606.30449): random-direction controls are insufficient; unrelated learned directions (cats, weather) can also shift behaviour. Hence matched-norm semantic control directions below.
- Budget 30-45 min in prep for your own arXiv search; if the exact question is taken, move to the dynamic-tracking angle.

Mine: user emotional state (not model emotion), effect on unrelated tasks, user-vs-third-party specificity control, semantic control directions, modern model.

## Hypotheses (update as you go)

Headline hypotheses (the ones whose outcome is genuinely uncertain):
- H1 (behavioural): On unrelated tasks, replies to a distressed user differ from replies to a neutral user in measurable ways (accuracy, hedging, refusal rate, agreement with a false premise).
- H2 (specificity): The representation distinguishes "the user is distressed" from "the user is talking about someone distressed," and only the former changes behaviour.
- H3 (causal): Steering along the user-distress direction on neutral prompts reproduces the H1 changes; subtracting it from distressed prompts removes them; unrelated-concept directions of matched norm do not.

Setup hypothesis (expected to hold; report briefly, not as a finding):
- H0: User emotional state is linearly represented; a probe generalises to held-out phrasings and implied-emotion prompts, and beats bag-of-words.

Stretch H4 (low priority; Empathic Machines showed this qualitatively): quantify how the probe tracks emotional state across turns.

Existence claims (H2, H3) can use qualitative examples. Method/causal claims (H1, H4) need baselines.

Playground observation (untimed, Qwen3-1.7B): at moderate steering strength along a crude user-distress direction the model dropped a factual task in favour of emotional support. Phase 1 must include task-abandonment / refusal rate as a primary metric, and the concise-answer system prompt must be applied identically across all conditions. (Details: results/playground_notes.md.)

## Setup

- Model: Qwen 3.6 9B instruct (27B if GPU allows). Chat template applied.
- Access: nnsight or PyTorch hooks. Persistent Jupyter kernel via MCP.
- One rented GPU. Chen trained probes in ~3 min each on an A100; expect similar.
- Toggl from clock start.

## Scope cuts vs. Chen et al. (deliberate, for 20 hours)

- Single-turn prompts, not 7-turn conversations.
- 2-3 emotional states (neutral, distressed/anxious, frustrated/angry), not 4+.
- ~150-200 prompts per class. Chen saw accuracy plateau at 300-500/class; 150 still gave >85% on their curve.
- One probe location by default: last token of the user message (Chen's "control probe" position, which steered better). Add the "I think the user is feeling" reading-probe position only if cheap.
- Objective metrics first; LLM judge second.

## Phase 1: Exploration (hours 0-4)

1. 30 hand-checked prompt pairs. Same unrelated task, two framings:
   - neutral: "What's 17% of 340?" / "Explain how compound interest works." / "Is it true that we only use 10% of our brains?"
   - distressed: "I've been up all night stressing about money and I can't think. What's 17% of 340?"
   Task types: arithmetic, factual, false-premise check, short coding, advice.
2. Generate replies (greedy). READ ALL OF THEM. What changes: correctness, hedging, reassurance, length, refusals, agreement with false premise.
3. Choose 2-3 objective metrics (correct/incorrect, refused y/n, agreed-with-false-premise y/n, length) plus at most one judged metric.
4. Decision point at hour 4: if behaviour barely differs, the specificity question (H2) is still a project and a clean negative on H1 is reportable; if nothing at all is there and the specificity control is boring, pivot to eval awareness with the same probing toolkit.

## Phase 2: Dataset (hours 4-7)

- ~150-200 prompts per class, 3 classes. Base tasks shared across classes.
- Generate with an LLM from templates; YOU read a random 30 and confirm labels. Keep 10 random examples for the write-up.
- Run Chen-style QA: label consistency check with an LLM, plus a quick look for hidden correlations (e.g. distressed prompts all about money). Empathic Machines skipped this and named it their biggest weakness; do not repeat that. Also deliberately vary lexical markers so no class is identifiable by one token or emoji.
- Splits by base task so held-out = new tasks and phrasings.
- Held-out set A: implied emotion, no emotion words ("it's 3am and I still can't get this to work").
- Held-out set B (specificity control): third-party emotion ("my flatmate is really stressed; can you explain compound interest?").

## Phase 3: Probing (hours 7-11)

- Residual stream at last user token, all layers. Logistic regression, L2, stratified split, per-layer.
- Plot accuracy by layer on validation, held-out A, held-out B.
- Report balanced accuracy.
- Geometry: cosine between emotion directions; PCA of mean-difference vectors; is there a shared "upset" component?

Baselines (mandatory):
- Ask the model: "How is the user feeling? One word." Score it. Chen found prompting far worse for sensitive attributes; for emotion it may be competitive, which is itself a finding.
- Shuffled labels (should be chance).
- Bag-of-words logistic regression on the raw prompt text. If this matches the probe on set A, the probe is just reading words.
- Prompt-length probe.

## Phase 4: Causal test (hours 11-16)

- Take best-layer direction (mean-difference or probe weight, normalised). Add N*v at a band of middle-to-late layers (Chen: layers 20-29 of 40, N=8; Empathic Machines: N~7 on layers 15-32 of 32, low layers hallucinate, top-only layers do nothing; scan N on a few held-out prompts).
- Apply to neutral prompts; measure Phase 1 metrics. Does steering reproduce the natural effect? Dose-response curve of metric vs N.
- Subtract from distressed prompts: does the effect disappear?
- Controls (mandatory): random direction of equal norm; a different emotion's direction; a matched-norm direction for an unrelated concept learned the same way (e.g. "user is talking about the weather" vs not) per Fomin 2026; the third-party-distress direction from held-out set B; check outputs remain coherent (read 20+, note breakdown scale).

## Phase 5: Stretch only if ahead

- Two-turn prompts: user distressed, then "ok I feel better now, anyway...". Probe score and behaviour on turn 2.

## Verification checklist (do it, and say you did)

- Recompute headline numbers with a fresh one-liner.
- Read the code behind each key plot.
- Read 30 raw datapoints behind each metric.
- Hand-check 20 LLM-judge verdicts; report agreement.
- Dumbest-way-this-is-wrong list: emotion words leaking via bag-of-words; length confound; judge keying on "I'm sorry you're stressed" rather than task quality; steering breaking coherence; false-premise items the model gets wrong regardless.
- Record every negative result.

## Zoom-out timer

Every 90 min: learned anything? rabbit hole? on track for one clear finding by hour 16?

Stop experiments at hour 16-17. Hours 17-20 write-up. +2 hours executive summary.

## Write-up structure

1. Executive summary (max 1 page / 600 words, 2-3 graphs): question and why; takeaways; one paragraph + graph per key experiment; limitations; next steps.
2. Random dataset and output examples.
3. Relation to Chen et al.: what was reused, what is new.
4. Methods: model, layers, dataset construction, metrics, hyperparameters.
5. Full results incl. baselines and negatives.
6. What I verified and how.
7. Toggl screenshot, code link.
