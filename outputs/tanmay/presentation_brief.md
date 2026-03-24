# BlackBox Hackathon — Digital Doppelganger
## Team: House of Tokens (Tanmay, Jasjyot, Tolendi)

---

## The Challenge

**Can AI predict how a specific person would answer questions they've never seen?**

Given 233 real people's complete psychological profiles — personality traits, cognitive abilities, economic game behavior, political views, demographics, and self-descriptions — predict their answers to brand new survey questions they've never encountered.

Scored on:
- **Accuracy**: Did we get the exact right answer?
- **Pearson Correlation**: Did we correctly rank who answers high vs low for each question?

---

## The Dataset

- **233 real survey respondents** from a comprehensive behavioral economics study
- **30+ validated psychological instruments**: Big Five personality, Beck Anxiety/Depression, Cognitive Reflection Test, Need for Cognition, Risk/Loss Aversion, Trust/Ultimatum/Dictator games, and more
- **531 known questions** across personality, economic preferences, cognitive tests, and demographics
- **3 data formats**: structured CSV survey responses, JSON with every Q&A pair, natural language persona text summaries with scores and percentiles
- Each persona file includes demographics, personality scores with percentiles, cognitive test results, economic game behavior with verbatim thought transcripts, and three open-ended self-description essays

---

## Our Architecture

### Hybrid ML + LLM Prediction System

```
Test question arrives (person_id + question_text + options)
                    |
     +--------------+--------------+
     |                             |
     v                             v
  ML BACKBONE                  LLM ENGINE
  (Tolendi's model)            (Claude Sonnet)
     |                             |
     |  Person features            |  Full persona text
     |  Question embeddings        |  Compressed trait summary
     |  Nearest-Q retrieval        |  Few-shot examples
     |  Interaction features       |  Structured answer prompt
     |                             |
     v                             v
  ML prediction              LLM prediction
     |                             |
     +---------> BLEND <-----------+
                  |
            0.2 * ML + 0.8 * LLM
                  |
                  v
           Final prediction
```

---

## Model Components

### Model 1: Person Response Profiles (Jasjyot)

**What it does:** Extracts 92 behavioral features per person from their raw survey answers.

**Features include:**
- Response style: mean, variance, midpoint tendency, extreme response style, acquiescence bias
- Personality construct averages: Big Five, empathy, need for cognition, self-monitoring, etc.
- Cognitive scores: CRT, numeracy, financial literacy, Wason selection (repaired with sanity check against ground truth)
- Economic behavior: dictator game offers, trust game sends, ultimatum responses
- Demographics: region, age, education, income, political views, employment

**Key challenge solved:** Initial cognitive score computation had broken answer keys (Beck Anxiety off by +21, Wason always zero). Identified via manual sanity check against persona text ground truth. Repaired by rebuilding scoring logic and dropping unresolvable scales (vocabulary, spatial).

---

### Model 2: Person Embeddings via SVD (Tanmay)

**What it does:** Compresses each person's full 326-question response pattern into 10 latent behavioral dimensions using Singular Value Decomposition.

**How it works:**
1. Build 233 x 326 response matrix (normalized to 0-1 across different scales)
2. Standardize per question (z-score)
3. Apply PCA/SVD to extract latent factors
4. K=10 dimensions capture 37.9% of total response variance

**What it captures:** Cross-construct behavioral patterns invisible to summary scores. Two people with identical Big Five scores may differ in how they answered economic game questions — the embeddings capture this.

**Elbow analysis:** K=18 is the mathematical elbow. K=10 chosen for parsimony. K=50 tested but showed no improvement in downstream models.

---

### Model 3: Question-to-Construct Mapper (Tanmay)

**What it does:** Maps any new question to the nearest known psychological construct, then looks up each person's percentile on that construct.

**How it works:**
1. 531 known questions grouped into 148 constructs by parent question ID
2. Each question embedded using sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
3. Construct centroids computed as mean embedding per construct
4. New question embedded → cosine similarity to all centroids → nearest match
5. Person's percentile on matched construct retrieved

**Output:** 21,436 person x construct percentile scores. Used by both ML model and as a feature signal.

---

### KNN Nearest-Question Retrieval (Tanmay)

**What it does:** For each new question, finds the 7 most similar known questions by semantic embedding, then looks up each person's actual answers to those questions.

**How it works:**
1. Embed new question (384-dim sentence transformer)
2. Cosine similarity against all 326 known question embeddings
3. Select top K=7 neighbors
4. For each person: weighted average of their real answers to those neighbors
5. Scale to target question's answer range

**Validation results:**
- Accuracy: 0.737 (best single ML model)
- Pearson r: 0.319
- Variance ratio: 0.494 (best spread of any ML model)
- Especially strong on Economic preferences (r=0.385)

---

### ML Backbone: LightGBM Ensemble (Tolendi)

**What it does:** Gradient-boosted tree ensemble that predicts normalized survey responses from person features + question features.

**Evolution through 5 versions:**

| Version | Accuracy | Pearson r | Key Change |
|---------|----------|-----------|------------|
| V1 | 0.724 | 0.259 | Single model baseline |
| V2 | 0.730 | 0.301 | 3-route split + leakage-free validation |
| V3 | 0.730 | 0.306 | + SVD person embeddings (marginal gain) |
| V4.1 | 0.731 | 0.305 | Repaired features, K=10, category features, calibration |
| New | 0.751 | 0.390 | Redesigned: semantic embeddings, interaction features, category routing |

**Final model architecture:**
- Respondent features: demographics, personality, cognitive, behavioral summaries, SVD embeddings
- Question features: sentence embeddings, nearest-neighbor retrieval, topic keywords, interaction features
- Question-type routing: ordinal attitudes, factual background, election behavior, news sharing
- Validation: GroupKFold by question_id, leakage-free construct features

**Critical insight from V2→V4.1:** Calibration (variance stretching) actively hurt correlation. The ranking wasn't strong enough to benefit from spreading predictions. This led to the LLM integration decision.

---

### Model 4: LLM Respondent Simulator (Tanmay)

**What it does:** Uses Claude Sonnet to simulate how each specific person would answer each question by reading their full psychological profile.

**Prompt structure:**
1. Full persona text (first 2000 chars): demographics, personality scores with percentiles, cognitive results, game behavior, self-description essays
2. Few-shot examples: 3 real people with known answers to a training question
3. Target question with context and answer options
4. Instruction: "Predict what THIS person would say. Do not default to average."

**Why it works:** The LLM reasons about personality-question interactions that ML can't capture. Example: "This person is a 50-64yr old moderate independent with low anxiety and high openness → on a political question about government responsibility, they would likely answer moderate-to-liberal."

**Key metrics (3-person initial test):**
- Midpoint rate: 15% (vs ML's 52%)
- The LLM produces answers across the full scale (1s, 2s, 4s, 5s), not just 3s

**API configuration:**
- Model: claude-sonnet-4-20250514
- Temperature: 0.3 (low for consistency)
- Max tokens: 10 (just a number)
- Parallel workers: 5 (30 predictions/sec)

---

## The Key Discovery: Why LLM Matters

### The Variance Compression Problem

ML models (LightGBM, KNN, Ridge) regress toward the mean. On Likert scales, 52-67% of ML predictions land on the exact midpoint. When 233 people all get the same answer, Pearson correlation is zero.

### Evidence from Live API Scoring

| Model | Correlation | Accuracy | Categories |
|-------|-------------|----------|------------|
| Our ML only | 0.113 | 27.4% | 2 Weak, 1 Strong |
| Tolendi ML only | 0.159 | 26.8% | 2 Mod, 1 Strong |
| Our ML + LLM (0.3/0.7) | 0.374 | 46.2% | All Strong |
| Tolendi ML + LLM (0.3/0.7) | 0.415 | 55.4% | All Strong |
| **Tolendi ML + LLM (0.2/0.8)** | **0.410** | **57.1%** | **All Strong** |

Adding the LLM:
- Tripled correlation (0.113 → 0.374, 3.3x)
- Doubled accuracy (27.4% → 57.1%)
- Flipped all categories from Weak/Mod to Strong

### Per-Category Performance (Best Model: 0.2/0.8)

| Category | Correlation | Accuracy | Status |
|----------|-------------|----------|--------|
| Personal Background | 0.292 | 61.2% | Strong |
| Political & Social Views | 0.541 | 46.5% | Strong |
| News Sharing Preferences | 0.478 | 78.7% | Strong |

---

## What Didn't Work

1. **Calibration / variance stretching**: Every stretch factor above 1.0 decreased Pearson r. The ML ranking wasn't strong enough to survive amplification.
2. **Higher K for SVD embeddings**: K=50 added 0.005 r over K=10. Not worth the complexity.
3. **Synthetic bootstrapped respondents**: Creating fake people by sampling within clusters destroys within-person cross-construct coherence.
4. **Claude Opus 4**: Scored r=0.185 vs Sonnet's r=0.395. Opus overthinks short survey predictions.
5. **Structured JSON LLM output with confidence**: Added complexity without improving predictions. Simpler prompt performed better.
6. **Dynamic blend weights**: Similarity-based per-question weighting scored worse than fixed weights.

---

## What Worked

1. **LLM as primary predictor**: The single biggest improvement. +0.26 correlation, +18% accuracy.
2. **Tolendi's redesigned ML model**: Proper interaction features, category routing, and semantic embeddings pushed internal r from 0.305 to 0.390.
3. **Simple fixed blend (0.2 ML / 0.8 LLM)**: Outperformed dynamic blending. Simplicity won.
4. **Few-shot prompting**: Giving the LLM 3 real examples of person-question-answer from training data calibrated its output format.
5. **Sentence-transformer question embeddings**: Enabled generalization to unseen question types.
6. **Leakage-free validation**: GroupKFold by question_id prevented optimistic evaluation. Catch from Tolendi saved us from shipping inflated metrics.

---

## Final Pipeline for Test Day

```
Step 1: Load test_questions.json
Step 2: Run Tolendi's ML model → ml_predictions.json
Step 3: Run Model 4 LLM (parallel, 5 workers) → llm_predictions.json
Step 4: Blend: 0.2 * ML + 0.8 * LLM
Step 5: Submit via scoring API
```

**Estimated for 150 people x 80 questions:**
- ML: instant (pre-trained LightGBM)
- LLM: ~$23, ~16 min with 10 workers
- Total: ~20 minutes end to end

---

## Team Contributions

**Tanmay (Host):**
- Initial EDA and dataset forensics (30+ instruments identified)
- Master table builder (125K rows from 233 persona JSONs)
- Model 2 (SVD person embeddings)
- Model 3 (question-to-construct mapper)
- KNN nearest-question retrieval
- Model 4 LLM predictor (prompt engineering, few-shot, parallel inference)
- Final pipeline integration and API submission
- All analysis and comparison reports

**Jasjyot:**
- Model 1 (person response profiles, 92 features)
- Cognitive score sanity check and repair
- Submission pipeline skeleton
- Scoring map and diagnosis tools
- LLM prediction baselines for comparison

**Tolendi:**
- LightGBM ensemble V1 through V4.1
- Leakage-free validation methodology
- Final redesigned ML backbone (r=0.390 internal)
- Model re-export for cross-machine compatibility
- ML improvement analysis and architecture recommendations

---

## Technical Stack

- **Languages:** Python 3.13
- **ML:** LightGBM, scikit-learn, sentence-transformers
- **LLM:** Claude Sonnet via Anthropic API
- **Data:** pandas, numpy, scipy
- **Embeddings:** all-MiniLM-L6-v2 (384-dim)
- **Collaboration:** GitHub, Claude Code
- **Scoring:** Live API at blackboxhackathon-production.up.railway.app

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Dataset size | 233 people, 531 questions, 125K answer rows |
| Model components | 4 (person features, SVD embeddings, construct mapper, LLM) |
| ML internal Pearson r | 0.390 |
| Final API Pearson r | **0.410** |
| Final API accuracy | **57.1%** |
| Blend weights | 0.2 ML + 0.8 LLM |
| LLM model | Claude Sonnet 4 |
| API submissions used | 15/30 |
| Total LLM cost (sample tests) | ~$12 |
