# Claude Instructions

Use this file as the single source of truth for Claude.

## Objective

Generate respondent-level predictions for the final test using the local ML predictions as a prior, with the main goal of improving **correlation**.

Focus only on the final test in this folder.

Do not optimize for any older proxy task.

## Files To Use

Read these files:

- `final_test_questions.json`
- `data/personas_text/`
- `artifacts/final_test_ml_predictions.json`
- `artifacts/claude_blend_seed.jsonl`
- `blend_claude_with_ml.py`
- `build_family_aware_submission.py`

Ignore older experiments and any removed instruction files.

## What Claude Should Produce

Create:

- `artifacts/claude_predictions.json`

Format:

```json
[
  {
    "person_id": "example_id",
    "question_id": "T1",
    "predicted_answer": 3
  }
]
```

Rules:

- exactly one row per `(person_id, question_id)`
- numeric answers only
- no missing values
- stay inside valid ranges

## Main Modeling Strategy

Treat the final test as 3 super-blocks:

1. Stable identity/worldview
- `Personal Background`
- `Political & Social Views`

2. Observed media behavior
- `T27-T44`
- `T58-T73`

3. Derived media attitudes
- `T45-T57`
- `T74-T76`
- `T77-T84`

The key rule is:

- behavior blocks are the strongest source of ranking signal
- preference and media-trust blocks should be inferred from behavior, not answered as isolated standalone questions

## High-Level Guidance

- Use the ML prediction as a prior, not a hard rule.
- Keep answers consistent within each respondent.
- Do not collapse everyone toward the middle.
- If the ML prior looks over-smoothed, move away from it to restore ranking signal.
- For the weak blocks, prioritize ranking/coherence over trying to guess perfect raw numbers.

## Category Guidance

### 1. Personal Background

Questions:

- `T1-T11`
- `T22`

Approach:

- answer directly from persona, demographics, life circumstances
- stay fairly close to ML

### 2. Political & Social Views

Questions:

- `T12-T26`

Approach:

- answer directly from ideology, party, education, race, trust, fairness, worldview
- keep internal consistency across the whole block
- stay moderately close to ML, but use persona semantics when they clearly imply stronger ranking

### 3. News Sharing Behavior

Questions:

- `T27-T44`
- `T58-T73`

Approach:

- these are the main signal blocks for media behavior
- answer consistently across source, headline, likes, entertainment, and political lean
- these answers should imply stable respondent traits that later drive the weak blocks

Infer these latent traits while answering behavior questions:

- overall share propensity
- source sensitivity
- likes/social-proof sensitivity
- headline sensitivity
- entertainment preference
- accuracy preference
- political-lean sensitivity
- humor / weirdness attraction

### 4. News Sharing Preferences

Questions:

- `T45-T57`
- `T74-T76`

Approach:

- do not answer item by item in isolation
- derive these from the behavior blocks
- think of them as summary weights explaining why this respondent shares

Important:

- if behavior suggests the person responds strongly to source quality, then `Source` should score higher
- if behavior suggests the person responds strongly to likes/popularity, then `Number of Likes` should score higher
- if behavior suggests the person shares funny/novel stories, then entertainment/headline/content should score higher
- if behavior suggests careful sharing, then accuracy/source should score higher and entertainment should be lower

### 5. Media Trust & Accuracy

Questions:

- `T77-T84`

Approach:

- prioritize coherent source ordering over aggressive raw numbers
- think in terms of relative trust, then assign moderate numeric values
- our local blend layer will sharpen calibration afterward, so Claude does not need to force the full `0-100` range

For each respondent, infer:

- institutional trust
- mainstream-media trust
- expert/journalist trust
- peer-platform trust
- tabloid/novelty trust
- ideological trust asymmetry

Then impose a coherent ordering across:

- `The Funny Times`
- `The National Enquirer`
- `BBC News`
- `The Wall Street Journal`
- `Reddit.com`
- `The Economist`
- `Quora.com`
- `PBS News`

Practical rule:

- give useful ranking signal
- keep values moderately separated unless the persona clearly implies strong distrust or very high trust

## Recommended Workflow

1. Read `artifacts/claude_blend_seed.jsonl`.
2. Group rows by respondent.
3. Answer the full respondent coherently, not as unrelated rows.
4. Use behavior blocks to infer latent media traits.
5. Use those traits to answer preference and media-trust blocks.
6. Save the final result to `artifacts/claude_predictions.json`.

## Strong Instruction To Follow

Use the ML prediction as a prior for scale, but not as the main source of ranking. For the final test, treat news-sharing behavior as the main revealed-preference signal. Then derive news-sharing preferences and media trust from those revealed preferences. Avoid middle collapse. If the persona suggests this respondent should rank meaningfully higher or lower than peers, move away from the ML prior while staying inside valid ranges.

For media trust specifically:

Prioritize coherent source ordering and moderate separation over extreme numeric guessing. The downstream blend layer will sharpen calibration.

## Output Checklist

Before finishing, verify:

- all `(person_id, question_id)` pairs are covered
- no missing values
- outputs are numeric
- values stay in valid ranges
- weak blocks are not flat or overly compressed
- source trust ordering is coherent within each respondent

## After Claude Finishes

The local blend step should be run as one of these two variants:

Main candidate:

```bash
python blend_claude_with_ml.py --ml-json artifacts/final_test_ml_predictions.json --claude-json artifacts/claude_predictions.json --question-json final_test_questions.json --override-mode weak_blocks --output-json artifacts/final_blended_weak_blocks.json
```

Safer fallback:

```bash
python blend_claude_with_ml.py --ml-json artifacts/final_test_ml_predictions.json --claude-json artifacts/claude_predictions.json --question-json final_test_questions.json --override-mode media_only --output-json artifacts/final_blended_media_only.json
```
