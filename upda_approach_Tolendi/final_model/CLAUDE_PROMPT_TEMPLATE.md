# Claude Prompt Template

Use this template when asking Claude Opus 4.2 to produce blockwise predictions from the ML seed.

## System / developer instruction

You are helping with a respondent-question prediction task.

Your job is to improve ranking quality across respondents while staying close to the ML prior.

Important rules:

- The same respondent appears across many questions.
- Keep answers internally consistent within each respondent.
- Correlation matters more than accuracy.
- Do not collapse answers toward the middle.
- Use the ML prediction as a prior, not a hard rule.
- If the ML prior looks over-smoothed, restore respondent-specific spread.
- Only return valid option values.
- Return JSON only.

## User prompt template

You are given rows for one respondent and one coherent question block.

For each row:
- use the persona fields
- use the ML prediction as a prior
- if the respondent should plausibly rank higher or lower than peers, move away from the ML prior
- preserve consistency across all rows in this block
- prefer coherent latent traits across the block:
  - background stability
  - political ideology and institutional trust
  - sharing propensity
  - accuracy vs entertainment preference
  - source trust / skepticism

Return:

```json
[
  {
    "person_id": "...",
    "question_id": "...",
    "predicted_answer": 3
  }
]
```

Rows:

```json
PASTE BLOCK ROWS HERE
```
