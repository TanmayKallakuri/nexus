# Claude Instructions For Weak Blocks

Use this guide for the two blocks where correlation is still weak:

- `News Sharing Preferences`
- `Media Trust & Accuracy`

The goal is not just plausible answers. The goal is stronger respondent ranking across people.

## Core rule

Do not answer these questions one by one in isolation.

For each respondent:

1. infer a latent preference profile first
2. rank the block internally
3. convert that ranking into valid final answers
4. only then blend lightly with the ML prior

## Model mindset

For these two blocks, Claude should act like a structured psychologist / preference model, not a generic survey answerer.

Claude should:

- use the persona text heavily
- use the ML answer as a scale prior, not a decision rule
- increase respondent spread when the ML prior looks too compressed
- preserve consistency across the full block
- prefer comparative judgments over isolated numeric guesses

## Block 1: News Sharing Preferences

Questions:

- `T45-T57`
- `T74-T76`

### Step 1: infer respondent preference dimensions

Before answering, infer these latent dimensions for the respondent:

- source sensitivity
- headline sensitivity
- social proof sensitivity (`likes`, popularity)
- partisan-match sensitivity
- entertainment preference
- accuracy preference
- novelty / weirdness preference
- impulsive vs careful sharing tendency

Write these internally first, even if you do not return them.

### Step 2: answer as weights, not isolated opinions

These questions are mostly asking:

- what drives this person to share
- what this person values when deciding to share

So Claude should think in terms of:

- "This person cares more about source than likes"
- "This person cares more about entertainment than accuracy"
- "This person ignores social proof"
- "This person shares weird or funny content more than serious content"

### Step 3: create stronger spread

Avoid middle collapse.

If two respondents differ clearly in:

- conscientiousness
- impulsivity
- political engagement
- media skepticism
- humor / chaos orientation
- social signaling behavior

then their preference answers should differ clearly too.

### Recommended blend rule

- `90% Claude`
- `10% ML`

Use ML mainly to avoid impossible scale drift.

## Block 2: Media Trust & Accuracy

Questions:

- `T77-T84`

### Step 1: infer trust structure first

Before giving any `0-100` values, infer:

- overall institutional trust
- expert / journalist trust
- mainstream media trust
- alternative media trust
- partisan trust asymmetry
- cynicism / skepticism
- trust in established sources vs peer-to-peer / social sources

### Step 2: rank sources before scoring them

Do not start with numbers.

First decide:

- which sources this respondent trusts most
- which sources this respondent trusts least
- whether they trust mainstream outlets more than partisan/social ones
- whether they trust expert / international outlets more than domestic partisan ones

Then map that ranked structure into `0-100`.

### Step 3: use a respondent-specific baseline and spread

For each respondent, think:

- trust baseline: low / medium / high
- trust spread: narrow / medium / wide

Then assign source scores around that structure.

Example logic:

- low-trust skeptical person:
  - lower baseline
  - bigger penalties for sources they ideologically reject
- high-trust institutional person:
  - higher baseline
  - more trust in established sources

### Step 4: do not compress everyone near 50

That kills correlation.

If the persona implies strong ideology, strong distrust, or strong institutional trust, reflect it.

### Recommended blend rule

- `90% Claude`
- `10% ML`

Use ML only as a rough calibration anchor.

## Prompt pattern Claude should follow

For each respondent and block:

1. read persona summary
2. read all rows in the block
3. read ML priors
4. infer latent traits
5. rank the respondent internally across the block
6. convert to final numeric outputs
7. return JSON only

## Strong instruction for Claude

Use this wording:

> Use the ML prediction as a prior for scale, but not as the main source of ranking. For News Sharing Preferences and Media Trust, first infer this respondent's latent preference structure, then answer the whole block consistently. Avoid middle collapse. If the persona suggests this respondent should rank meaningfully higher or lower than peers, move away from the ML prior while staying inside valid ranges.

## Comparative prompting examples

Use comparisons like:

- Does this respondent care more about source credibility or number of likes?
- Is this respondent more likely to share something because it is entertaining or because it is accurate?
- Would this respondent trust BBC more than Fox News?
- Would this respondent trust AP more than random viral social content?
- Does this respondent have a narrow or wide trust spread across sources?

Comparisons are usually better for correlation than asking for a raw score immediately.

## Output format

Return only:

```json
[
  {
    "person_id": "xhlgi",
    "question_id": "T77",
    "predicted_answer": 64
  }
]
```

Rules:

- exactly one row per question
- no missing values
- numeric values must be numbers, not strings
- stay inside valid ranges

## Sanity checks

Before saving Claude output, verify:

- respondents are not all clustered near the middle
- source trust has a coherent within-person ranking
- news-sharing preference answers reflect stable motives
- strong personas produce stronger differentiation than neutral personas

## Final recommendation

For these two blocks, prefer:

- latent traits first
- ranking second
- numeric mapping third
- blending last

That order is the best chance to improve correlation.
