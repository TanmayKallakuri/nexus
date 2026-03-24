# ML Improvement Analysis

## What the current results say

- `Tolendi ML only` scores `r=0.1587`, accuracy `26.8%` on the real blackbox API.
- `Our ML only` scores `r=0.1129`, accuracy `27.4%`.
- Adding the LLM raises correlation by about `+0.24` to `+0.26`, which is much larger than the gap between the two ML backbones.
- ML-only models are relatively strongest on `News Sharing Preferences` and weak on `Personal Background` plus `Political & Social Views`.

This strongly suggests the main bottleneck is not respondent identity alone. It is question understanding under distribution shift.

## Evidence from the blackbox questions

Blackbox set:

- `T5`: work status last week
- `T8`: poorer service in stores/restaurants
- `T10`: financial satisfaction
- `T14`: confidence in the press
- `T17`: affirmative action / preferential hiring
- `T18`: government responsibility for the poor
- `T22`: father's education
- `T25`: voted in 2020
- `T26`: voted Biden or Trump
- `T49`: importance of number of likes for sharing news

Observed pattern:

- `T22` has a clear analog in training: respondent education.
- `T49` also has some trait-style analog signal and is the category where ML-only already does best.
- Most others have very weak nearest-neighbor matches in the historical survey data. Top text similarities are often only around `0.03` to `0.06`, which means the current question representation is failing to anchor those items to useful historical signal.

## Main weak spots

### 1. Weak question semantics

Current ML works mostly because respondent features are decent. The question representation is too shallow for genuinely new political, background, and real-world behavior items.

### 2. Too much generic numeric regression

The pipeline mostly treats all items as one numeric task. But these blackbox questions mix:

- ordered categorical
- quasi-behavioral factual items
- ideology / political stance
- parental background

Those should not all be modeled the same way.

### 3. Poor mapping from persona traits to blackbox constructs

For questions like `confidence in the press`, `affirmative action`, `government responsibility`, `voted Biden or Trump`, the model needs features that connect:

- political affiliation
- ideology
- trust
- race-related attitudes
- education
- income
- media trust

The LLM appears to infer these from persona text much better than the ML-only system.

### 4. Likely over-smoothing

Our model predictions on the blackbox set have fairly small respondent spread for several questions. That can keep answers numerically plausible while hurting ranking.

## Highest-impact improvement ideas

### A. Upgrade question representation from TF-IDF to strong semantic embeddings

Use a sentence embedding model for every question:

- `question_text`
- `context`
- `question_text + context + options`

Then build:

- nearest historical question features
- weighted respondent averages over top-k similar historical questions
- cluster-level respondent priors from embedding clusters

This is likely the single biggest ML-only gain.

### B. Add category-specific models

Instead of one universal regressor, route questions into specialized heads:

- personal background
- political/social views
- news sharing / media behavior

Train a separate model or calibrator per category. The API results already show category behavior differs a lot.

### C. Build explicit politics and background features from persona text

Extract and use:

- political affiliation
- ideology
- income
- employment
- education
- race
- region
- age
- media/news trust proxies if available

Then add direct interactions:

- ideology x government-responsibility questions
- political affiliation x Biden/Trump questions
- race x affirmative-action question
- age/employment x work-status question
- education/income x financial-satisfaction question

### D. Handle question types separately

Use different modeling logic for:

- factual/background ordinal items
- ideology items
- turnout / vote-choice items
- news-sharing preference items

For example:

- `T25` turnout can be treated as a structured classification-style task.
- `T26` Biden vs Trump is closer to political-choice classification than generic regression.
- `T22` parental education should use monotone ordered modeling.

### E. Add a "don't know / nonresponse" propensity model

Several blackbox questions include explicit `Don't know` options. Current ML likely treats these as just another numeric point on the scale.

Better approach:

1. predict whether respondent chooses substantive answer vs DK/nonresponse
2. predict substantive value conditional on answering substantively

This can improve both accuracy and ranking.

### F. Use stronger respondent-history retrieval

For each test question:

1. retrieve semantically closest historical questions
2. pull each respondent's answers to those questions
3. feed them directly as features

This is stronger than only using global respondent summaries or latent factors.

### G. Calibrate per question after raw prediction

For ordinal items:

- predict continuous score
- calibrate to realistic marginal distribution
- then round/clip

For binary or quasi-binary political items:

- use probability outputs and threshold calibration

This is especially useful when the raw model collapses toward the middle.

## Useful medium-impact ideas

### H. Train with harder validation

Current CV can still be too optimistic because held-out historical questions may remain close to training distribution.

Add stress tests:

- hold out entire semantic clusters of questions
- hold out political/background categories
- hold out questions with no close nearest neighbors

That will better estimate blackbox behavior.

### I. Add confidence-aware blending

Instead of fixed `0.1/0.9` or `0.3/0.7`, use dynamic blending:

- when question retrieval similarity is high, trust ML more
- when similarity is low, trust LLM more
- when category is political/social, increase LLM weight
- when category is news-sharing and ML is strong, increase ML weight

### J. Distill the LLM into features or pseudo-labels

Since LLM adds large value, use it during training:

- generate reasoning-based question tags
- generate estimated question categories
- generate likely trait-response links
- optionally use LLM pseudo-labels on held-out-like questions as auxiliary targets

This can improve the ML backbone instead of only post-hoc blending.

## Concrete roadmap

### Phase 1: Fastest likely wins

1. Replace TF-IDF with sentence embeddings.
2. Add nearest-question respondent-answer features.
3. Add category routing and question-type handling.
4. Add explicit politics/background interactions.

### Phase 2: Better calibration

1. Add DK propensity model.
2. Add per-question calibration and discrete post-processing.
3. Add dynamic ML/LLM blend weights by category and confidence.

### Phase 3: Bigger upgrade

1. Distill LLM signals into the ML model.
2. Train category-specific or multitask models.
3. Build retrieval-augmented prediction around semantically matched historical items.

## Bottom line

The ML-only model is not failing because respondent features are useless. It is failing because the new blackbox questions require stronger semantic understanding and category-specific mapping than the current question encoder and generic regression setup can provide.

If you want the highest-probability next upgrade, start here:

1. stronger semantic embeddings
2. nearest-question respondent-history features
3. explicit politics/background feature interactions
4. category-aware modeling and calibration
