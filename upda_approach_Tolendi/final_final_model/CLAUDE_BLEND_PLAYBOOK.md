# Claude Blend Playbook

This is the handoff for Claude to use the ML model output as a prior and improve respondent ranking on the final test.

Recommended setup:

- use Claude Opus 4.2 in your own runner or API client
- send blockwise batches, not one question at a time
- write Claude output to `artifacts/claude_predictions.json`

## Objective

Claude should improve **correlation**, not just produce plausible average answers.

That means:

- keep answers consistent within each respondent
- preserve differences between respondents on each question
- use the ML prediction as a prior, not as a hard rule

## Secrets

Never hardcode secrets in the repo.

Use environment variables:

- `ANTHROPIC_API_KEY`
- `BLACKBOX_TEAM_TOKEN`

## Files Claude should use

- `final_test_questions.json`
- `artifacts/final_test_ml_predictions.json`
- `artifacts/claude_blend_seed.jsonl`
- `data/personas_text/`

## Recommended workflow

1. Read `artifacts/claude_blend_seed.jsonl`
2. Group rows by respondent and by question block
3. Predict blockwise, not row-by-row
4. Save Claude outputs to `artifacts/claude_predictions.json`
5. Blend with:

```bash
python blend_claude_with_ml.py \
  --ml-json artifacts/final_test_ml_predictions.json \
  --claude-json artifacts/claude_predictions.json \
  --question-json final_test_questions.json \
  --output-json artifacts/final_blended_predictions.json
```

## Block structure Claude should follow

### Block 1: Personal Background

- `T1-T11`
- `T22`

Goal:
- infer stable life circumstances
- stay coherent across work, finances, health, student status, home/rent, volunteering

### Block 2: Political and Social Views

- `T12-T26`

Goal:
- infer trust, ideology, redistribution views, immigration stance, vote behavior
- keep internal consistency across press trust, affirmative action, taxes, turnout, vote choice
- preserve respondent ranking even when the ML prior is centered too tightly

### Block 3: News Sharing Preferences

- `T45-T57`
- `T74-T76`

Goal:
- infer whether the respondent values source, headline, likes, political lean, accuracy, or entertainment

### Block 4: Repeated Sharing Scenarios

- `T27-T44`
- `T58-T73`

Goal:
- answer these jointly as repeated experiments
- use the same respondent-level sharing preferences across all of them
- vary respondents meaningfully instead of collapsing them into the same middle value

### Block 5: Media Trust and Accuracy

- `T77-T84`

Goal:
- estimate respondent-specific trust in different sources
- keep source ordering coherent

## How Claude should use the ML prior

Use ML as:

- a center point for likely numeric range
- a warning against impossible or inconsistent answers

Do not copy ML blindly.

Recommended rule:

- if the question is background/factual, stay fairly close to ML
- if the question is political/social, use persona semantics more aggressively
- if the question is in news/media/sharing blocks, use ML for rough scale and use Claude for ranking
- if the ML prior seems too compressed, spread respondents out while staying inside valid ranges

## Recommended blend mindset

Approximate target weights:

- Background: 35% Claude / 65% ML
- Political & Social: 75% Claude / 25% ML
- News Sharing Preferences: 75% Claude / 25% ML
- Repeated Sharing Scenarios: 85% Claude / 15% ML
- Media Trust & Accuracy: 85% Claude / 15% ML

The actual blend script uses family-level weights. Claude should focus on generating strong ranking signal.

## Output format Claude should produce

Claude must write a JSON array with rows like:

```json
[
  {
    "person_id": "xhlgi",
    "question_id": "T1",
    "predicted_answer": 3
  }
]
```

Rules:

- exactly one row per `(person_id, question_id)`
- use only valid answer ranges
- no missing values
- no explanation text in the file
- keep numeric outputs as numbers, not strings

## Prompting guidance

Tell Claude:

- the same respondent appears across many questions
- consistency across blocks matters
- correlation is more important than accuracy
- ranking respondents differently is required
- avoid collapsing everyone toward the middle

Best instruction:

"Use the ML prediction as a prior for scale, but if the persona and question semantics suggest that this respondent should rank higher or lower than peers on this question, move away from the ML prior."

Operational tip:

- process one block for many respondents at a time only if your runner can preserve strict JSON output
- otherwise process one respondent-block at a time and concatenate results

## Sanity checks before blending

Claude outputs should show:

- visible spread across respondents within the same question
- no collapse to one or two values in news/media blocks
- consistent vote and ideology logic
- coherent source-trust ordering

## Final submission

After blending:

```bash
python submit.py --predictions artifacts/final_blended_predictions.json
```
