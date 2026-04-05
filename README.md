<div align="center">

# Nexus

### Behavioral Prediction Engine — BlackBox Hackathon 2026

**Predicting how real people answer unseen questions using psychological profiling, machine learning, and large language models.**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-9ACD32)](https://lightgbm.readthedocs.io/)
[![Claude API](https://img.shields.io/badge/Anthropic-Claude_Sonnet-CC785C)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

**2nd Place** · Pearson r = 0.158 · 63.9% Accuracy · 40 Submissions

</div>

---

## The Challenge

Given **233 real people** with comprehensive psychological profiles (personality traits, cognitive abilities, economic preferences, political views), predict how **150 of them** would answer **84 brand-new survey questions** they had never seen — across 6 behavioral categories.

Evaluation metric: **Pearson correlation** (ranking signal) + classification accuracy.

## Key Insight

Traditional ML models (LightGBM, KNN) achieved strong accuracy (~74%) but **collapsed on correlation** — they regressed toward midpoint answers, predicting the same safe response for everyone. When everyone gets the same prediction, correlation is zero.

Our breakthrough: **LLM-based persona simulation** using Claude Sonnet with full psychological profiles as context. The LLM produced genuine person-level variance (only 26% midpoint rate vs. ML's 52–67%), which is exactly what Pearson r rewards.

> **Final architecture:** Pure LLM prediction with per-category prompt optimization, not an ML–LLM blend. ML models informed development but the LLM alone outperformed every hybrid.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │     84 Unseen Test Questions     │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   Claude Sonnet — 3 Prompt Modes │
                    │                                  │
                    │  ┌──────────┐ ┌───────────────┐ │
                    │  │Blockwise │ │ Full Persona  │ │
                    │  │ (2K chars)│ │ (5K chars)    │ │
                    │  └──────────┘ └───────────────┘ │
                    │  ┌──────────────────────────────┐│
                    │  │ Enriched Traits + Latent-    ││
                    │  │ First Approach               ││
                    │  └──────────────────────────────┘│
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  Per-Category Cherry-Pick Blend  │
                    │                                  │
                    │  Personal Background → Blockwise │
                    │  Political Views    → Full 5K    │
                    │  News Sharing Pref  → Enriched   │
                    │  News Sharing Behav → Blockwise  │
                    │  Info Sharing Behav → Blockwise  │
                    │  Media Trust        → Full 5K    │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │     Final 150 × 84 Predictions   │
                    └─────────────────────────────────┘
```

## Results

### Final Score Breakdown

| Category | Pearson r | Accuracy | |
|---|---|---|---|
| Personal Background | **0.284** | 58.1% | Strong |
| Political & Social Views | **0.335** | 43.6% | Strong |
| News Sharing Behavior | 0.115 | 69.3% | Moderate |
| Information Sharing Behavior | 0.111 | 68.6% | Moderate |
| News Sharing Preferences | 0.063 | 71.9% | Weak |
| Media Trust & Accuracy | 0.040 | 75.9% | Weak |
| **Overall** | **0.158** | **63.9%** | **2nd Place** |

### Model Evolution

| Stage | Method | Accuracy | Pearson r |
|---|---|---|---|
| Baseline | LightGBM V1 | 0.724 | 0.259 |
| + Routing & leakage fix | LightGBM V2 | 0.730 | 0.301 |
| + SVD embeddings | LightGBM V3 | 0.730 | 0.306 |
| Nearest-question retrieval | KNN (K=7) | 0.737 | 0.319 |
| LLM simulation (3-person pilot) | Claude Sonnet | 0.833 | 0.370 |
| ML+LLM blend (pilot) | 0.3 ML + 0.7 LLM | 0.897 | 0.457 |
| **Final (full test set)** | **Multi-source LLM** | **0.639** | **0.158** |

## Dataset

**Twin-2K-500 Behavioral Economics Omnibus** — UC Davis, administered via Qualtrics on Prolific/MTurk.

- **233 respondents** with full psychological profiles
- **531 unique questions** across 30+ validated instruments (Big Five, Beck Anxiety/Depression, CRT, economic games, numeracy, political ideology)
- **125,115 observation triples** (person × question × answer) in the master table
- 3 data formats: raw text personas, structured JSON, tabular CSVs across 4 surveys
- 27% missing data (mostly text-entry fields)

## Tech Stack

| Layer | Technology |
|---|---|
| **ML Models** | LightGBM, scikit-learn, KNN retrieval |
| **Embeddings** | SVD (K=10), sentence-transformers (all-MiniLM-L6-v2) |
| **LLM** | Claude Sonnet 4 via Anthropic API |
| **Data** | pandas, NumPy, SciPy |
| **Execution** | 16 parallel workers, dual API keys |
| **Collaboration** | Claude Code with custom `/investigate`, `/sync`, `/report` commands |

## Project Structure

```
nexus/
├── data/                          # Raw survey data (4 surveys, 233 respondents)
│   ├── survey_{1-4}_labels.csv
│   └── survey_{1-4}_numbers.csv
├── scripts/                       # All model and analysis scripts
│   ├── build_master_table.py          # 125K-row dataset construction
│   ├── build_person_profiles.py       # Model 1: 92-feature person profiles
│   ├── build_person_embeddings.py     # Model 2: SVD embeddings (K=10)
│   ├── model4_llm_predictor.py        # Model 4: Claude-based persona simulation
│   ├── knn_nearest_question_retrieval.py  # KNN (K=7) question lookup
│   ├── final_predict.py               # Submission pipeline
│   └── ...
├── upda_approach_Tolendi/         # LightGBM ensemble iterations (V1–V4.1)
├── outputs/                       # Predictions, embeddings, diagnostics
├── brief.md                       # Collaborative findings log
└── requirements.txt               # Python dependencies
```

## Getting Started

```bash
git clone https://github.com/TanmayKallakuri/nexus.git
cd nexus
pip install -r requirements.txt
```

**Dependencies:** pandas, numpy, scipy, scikit-learn, lightgbm, sentence-transformers, matplotlib, seaborn, openpyxl, python-docx

## Team

<table>
<tr>
<td align="center" width="33%">

**Tanmay Kallakuri**<br>
*Lead Engineer & Systems Architect*

Built the master data pipeline (125K rows), SVD embedding model, question-to-construct mapper (531→148), KNN retrieval model, and the LLM persona simulation system. Designed the multi-source cherry-pick blending strategy and managed all 40 API submissions.

[![GitHub](https://img.shields.io/badge/GitHub-TanmayKallakuri-181717?logo=github)](https://github.com/TanmayKallakuri)

</td>
<td align="center" width="33%">

**Jasjyot Singh**<br>
*Feature Engineering & Data Quality*

Built the 92-feature person response profiles, discovered and repaired broken cognitive test scores (BAI, Wason, CRT), built the submission pipeline and scoring diagnostics, and tested LLM baselines across Claude Sonnet, GPT-4.1, and Claude Opus.

[![GitHub](https://img.shields.io/badge/GitHub-SuperfiedStudd-181717?logo=github)](https://github.com/SuperfiedStudd)

</td>
<td align="center" width="33%">

**Tolendi Tastybay**<br>
*ML Modeling & Ensemble Optimization*

Built the LightGBM ensemble through 5 iterations (V1→V4.1), implementing family-aware heuristics, question-type routing, and data leakage fixes. Ran calibration analysis and exported cross-machine-compatible model artifacts.

</td>
</tr>
</table>

## What We Learned

1. **Variance beats accuracy for correlation metrics.** A model that's right 74% of the time but gives everyone the same answer scores worse than one that's right 64% but captures individual differences.

2. **LLMs are surprisingly good behavioral simulators.** Given a rich enough persona description, Claude produced person-specific predictions that traditional ML couldn't match — not through better accuracy, but through better *spread*.

3. **Per-category optimization matters.** No single prompt or model won across all 6 categories. The winning strategy was selecting the best-performing prompt variant per category based on submission feedback.

4. **Blending can hurt.** ML+LLM blends scored worse than pure LLM because ML's midpoint regression dragged down the LLM's variance — the exact signal Pearson r rewards.

---

<div align="center">

Built in 24 hours at the **BlackBox Hackathon 2026**

</div>
