# BlackBox Hackathon — Digital Doppelganger
## Team: House of Tokens (Tanmay, Jasjyot, Tolendi)

---

## The Challenge

**Can AI predict how a specific person would answer questions they've never seen?**

Given 233 real people's complete psychological profiles, predict their answers to brand new survey questions they've never encountered.

Scored on:
- **Correlation (Pearson r)**: Did we correctly rank who answers high vs low?
- **Accuracy**: Did we get the exact right answer?

---

## The Dataset

- **233 survey respondents** from a comprehensive behavioral economics study (Twin-2K-500)
- **30+ validated psychological instruments**: Big Five, Beck Anxiety/Depression, CRT, economic games, and more
- **531 known questions** across personality, economic preferences, cognitive tests, and demographics
- **3 data formats**: structured CSV, full JSON Q&A records, natural language persona summaries
- **Final test**: 150 people x 84 unseen questions across 6 categories

---

## Final Score

| Category | Correlation | Accuracy | Status |
|---|---|---|---|
| Personal Background | 0.284 | 58.1% | **Strong** |
| Political & Social Views | 0.335 | 43.6% | **Strong** |
| News Sharing Preferences | 0.011 | 72.2% | Fail |
| News Sharing Behavior | 0.115 | 69.2% | **Mod** |
| Information Sharing Behavior | 0.113 | 68.3% | **Mod** |
| Media Trust & Accuracy | 0.087 | 77.0% | **Weak** |
| **Overall** | **0.157** | **63.9%** | **2 Strong, 2 Mod, 1 Weak, 1 Fail** |

---

## Architecture

### Hybrid ML + LLM Multi-Source Prediction System

```
Test question arrives
        |
        v
  Multiple prediction sources generated independently:
        |
        +---> Claude Sonnet (blockwise, persona text summaries)
        |       Best for: Personal Background, Political Views,
        |       News Sharing Behavior, Info Sharing Behavior
        |
        +---> Claude Sonnet (full 5000-char persona, enriched traits)
        |       Best for: Media Trust, Political & Social Views
        |
        +---> Claude Sonnet (latent-trait-first approach)
        |       Best for: News Sharing Preferences
        |
        +---> ML Backbone (Tolendi's LightGBM ensemble)
        |       Family-aware heuristics for political/background
        |
        v
  Intelligent Cherry-Pick Blend
  (best prediction source selected per category)
        |
        v
  Final prediction
```

---

## Model Components

### Model 1: Person Response Profiles (Jasjyot)
- 92 behavioral features per person from raw survey answers
- Response style, personality constructs, cognitive scores, demographics
- Sanity-checked against ground truth, broken scales repaired

### Model 2: Person Embeddings via SVD (Tanmay)
- 10 latent dimensions from SVD on 233 x 326 response matrix
- Captures cross-construct behavioral patterns invisible to summary scores

### Model 3: Question-to-Construct Mapper (Tanmay)
- 531 questions mapped to 148 constructs via sentence embeddings
- Matches unseen questions to nearest known construct + person percentile

### KNN Nearest-Question Retrieval (Tanmay)
- For each new question: find 7 most similar known questions
- Look up person's actual answers, weighted average
- Best single ML model: accuracy 0.737, r=0.319 on internal validation

### ML Backbone (Tolendi)
- Family-aware LightGBM ensemble with question-specific heuristics
- Internal validation: accuracy 0.751, r=0.390
- Final test ML-only: r=0.100, accuracy 61.1%

### Model 4: LLM Respondent Simulator (Tanmay)
- Claude Sonnet via Anthropic API
- Multiple prompt variants tested:
  - Blockwise with persona text summary (best for behavior categories)
  - Full 5000-char persona with enriched traits (best for Media Trust)
  - Latent-trait-first approach (best for News Sharing Preferences)
- Few-shot examples, anti-midpoint instructions, ML prior as anchor
- Parallel execution with dual API keys (16 workers)

---

## The Journey: How We Got Here

### Phase 1: ML-Only (r=0.100)
- Built person features, SVD embeddings, construct mapper, KNN retrieval
- Iterated LightGBM ensemble through 5 versions (V1-V4.1)
- Added family-aware heuristics and question-type routing
- **Ceiling hit at r=0.100 on final test — ML couldn't differentiate people on unseen question types**

### Phase 2: LLM Introduction (r=0.140)
- Added Claude Sonnet as respondent simulator
- Blockwise prompting: all questions for one person processed together
- LLM tripled correlation on sample test (0.113 → 0.374)
- First final test blend: r=0.140 with per-family weights

### Phase 3: Pure Claude Discovery (r=0.152)
- Discovered ML was dragging down LLM's rankings
- Pure Claude predictions outperformed all ML blends
- LLM's natural person-level variance is the key signal

### Phase 4: Multi-Source Cherry-Pick (r=0.157)
- Ran multiple Claude variants with different prompts and persona formats
- Full 5000-char persona unlocked Media Trust (r=0.010 → 0.087)
- Enriched seed traits improved News Sharing Preferences
- Category-level cherry-pick: best source per category combined
- **Final score: r=0.157, accuracy 63.9%**

---

## Key Discovery: The Variance Problem

ML models regress toward the mean. On Likert scales, 52-67% of ML predictions were the exact midpoint. When everyone gets the same answer, correlation is zero.

| Approach | Midpoint Rate | Correlation |
|---|---|---|
| ML only | 67% | 0.100 |
| Claude Sonnet (blockwise) | 26% | 0.152 |
| Final cherry-pick blend | ~25% | 0.157 |

The LLM reads each person's full profile and reasons about what THIS specific person would answer. It produces 1s, 2s, 5s, 6s — not just 3s. That spread is what Pearson r rewards.

---

## What Worked

1. **LLM as primary predictor** — the single biggest improvement (+0.05 correlation over ML)
2. **Blockwise prompting** — Claude sees all related questions together for internal consistency
3. **Multiple persona formats** — text summary for behavior, full persona for trust, enriched traits for preferences
4. **Category-level cherry-picking** — best prediction source per category, not one-size-fits-all
5. **ML family-aware heuristics** — political affiliation → vote prediction, demographics → background
6. **Dual API keys + parallel workers** — 16 concurrent workers for fast iteration

## What Didn't Work

1. **ML + LLM blending** — ML consistently dragged down Claude's rankings
2. **GPT-4.1 with JSON persona** — 48K token input overwhelmed the model (r=0.032)
3. **Calibration/variance stretching** — amplified errors as much as signal
4. **Anti-midpoint forcing** — made Claude pick wrong directions
5. **Dynamic blend weights** — performed worse than fixed cherry-pick
6. **Claude Opus** — overthought simple survey predictions (r=0.185 vs Sonnet's 0.395 on sample)

---

## Technical Stack

- **ML**: LightGBM, scikit-learn, sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Claude Sonnet 4 via Anthropic API
- **Data**: pandas, numpy, scipy
- **Collaboration**: GitHub, Claude Code
- **Tested but rejected**: GPT-4.1, Claude Opus, vast.ai

---

## Iteration Summary

| # | Approach | Overall r | Accuracy |
|---|---|---|---|
| 1 | ML only (Tolendi) | 0.100 | 61.1% |
| 2 | ML + blockwise Claude | 0.140 | 63.3% |
| 3 | Pure Claude (text persona) | 0.152 | 63.7% |
| 4 | Cherry-pick (V6 NSP + full PS) | 0.154 | 63.7% |
| **5** | **Multi-source cherry-pick** | **0.157** | **63.9%** |

35 API submissions used across the hackathon.

---

## Team Contributions

**Tanmay (Host):**
- EDA, master table (125K rows), dataset forensics
- Model 2 (SVD embeddings), Model 3 (construct mapper), KNN retrieval
- Model 4 LLM predictor — all prompt variants, parallel execution
- Final pipeline integration, cherry-pick optimization, all API submissions

**Jasjyot:**
- Model 1 (92 person features), cognitive score repair
- Submission pipeline skeleton
- LLM comparison baselines (Sonnet, GPT, Sonnet 4.6)

**Tolendi:**
- LightGBM ensemble V1-V4.1, leakage-free validation
- Family-aware ML backbone with question-specific heuristics
- Blend architecture, per-family weights
- ML improvement analysis and weak-block strategy

---

## Key Insight

> The best digital doppelganger is not a single model — it's a committee of specialized LLM prompts, each optimized for a different aspect of human behavior, intelligently combined at the category level.

ML provides stable anchoring. LLMs provide person-specific reasoning. The art is knowing when to trust each one.
