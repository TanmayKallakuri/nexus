# Nexus — Hackathon Brief

## Dataset Origin
Behavioral economics / cognitive psychology omnibus survey. 30+ validated instruments. US-only sample (Prolific/MTurk), post-2019, likely UC Davis lab. 233 respondents, Qualtrics-administered. _(filled by Tanmay)_

## Dataset Overview
- **3 data formats:** personas_text (233 .txt), personas_json (233 .json), personas_csv (8 CSVs across 4 surveys)
- **Master table:** 125,115 rows x 14 columns — every (person, question, answer) triple
- **531 unique questions** — 61% ordinal, 31% categorical, 4% text, 4% multi-select
- **Blocks:** Personality (65K rows), Economic preferences (42K), Cognitive tests (15K), Demographics (3K)
- **Missing data:** 27% overall, mostly text-entry fields coerced to NaN in numeric files
- **Person profiles:** 233 rows x 92 features (Model 1 repaired)

## Model Performance Summary

| Model | Accuracy | Mean Pearson r | Variance ratio |
|---|---|---|---|
| V1 (baseline) | 0.724 | 0.259 | 0.288 |
| V2 (routing + leakage fix) | 0.730 | 0.301 | 0.379 |
| V3 (+ embeddings K=50) | 0.730 | 0.306 | 0.383 |
| V4.1 (repaired M1 + K=10 + cats + cal) | 0.731 | 0.305 | 0.380 |
| KNN (K=7, no ML) | 0.737 | 0.319 | 0.494 |
| Model 4 LLM (3-person test) | 0.833 | 0.370 | — |
| **Blend 0.3ML + 0.7LLM (3-person test)** | **0.897** | **0.457** | — |

## Findings

### Tanmay (last synced: 6:15 PM)
- Initial EDA complete — 234 merged rows, 915 variables _(synced at 11:50 AM)_
- Dataset forensics — 30+ instruments, US-only, post-2019 _(synced at 11:50 AM)_
- Built master table — 125,115 rows, 531 unique questions _(synced at 11:50 AM)_
- Built Model 2 (SVD person embeddings) — K=10 sufficient, higher K adds no signal _(synced at 6:15 PM)_
- Built Model 3 (question-to-construct mapper) — 148 constructs, 531 questions mapped _(synced at 6:15 PM)_
- Built KNN nearest-question retrieval — best single ML model: accuracy 0.737, r=0.319 _(synced at 6:15 PM)_
- Built Model 4 LLM predictor — few-shot Claude-based, midpoint rate 15% vs ML's 52% _(synced at 6:15 PM)_
- Blend of 0.3 ML + 0.7 LLM: accuracy 0.897, r=0.457 on 3-person test _(synced at 6:15 PM)_
- ML models fail on correlation (-0.066 on 233 people) due to variance compression _(synced at 6:15 PM)_
- LLM fixes correlation by producing person-specific answers with real spread _(synced at 6:15 PM)_

### Jasjyot (last synced: 3:30 PM)
- Model 1 complete — 233 rows x 92 columns _(synced at 11:45 AM)_
- Sanity check found broken cognitive scores (BAI +21, Wason always 0, CRT off) _(synced at 3:30 PM)_
- Repaired Model 1 — fixed BAI, Wason, CRT, numeracy; dropped vocabulary/spatial _(synced at 3:30 PM)_
- Built submission pipeline (predict_submission.py) _(synced at 3:30 PM)_
- Built scoring map and diagnosis tools _(synced at 3:30 PM)_

### Tolendi (last synced: 4:30 PM)
- V1 baseline: accuracy 0.724, r=0.259, variance ratio 0.288 _(synced at 4:30 PM)_
- V2 routing + leakage fix: r improved to 0.301, Economic fixed from -0.036 to +0.118 _(synced at 4:30 PM)_
- V3 embeddings: marginal gain +0.005 r _(synced at 4:30 PM)_
- V4.1 final ML: repaired M1, K=10, category features, calibration. r=0.305 _(synced at 4:30 PM)_
- Calibration failed — every stretch factor above 1.0 decreased Pearson r _(synced at 4:30 PM)_
- Re-exported V4.1 models as LightGBM text files for cross-machine compatibility _(synced at 4:30 PM)_

## Contradictions
- Architecture shifted from "ML backbone, LLM fallback" to "LLM primary, ML secondary" after correlation analysis showed ML at -0.066 vs LLM at 0.37+ on sample questions

## Open Questions
- RESOLVED: Cognitive answer keys fixed by Jasjyot _(BAI, Wason fixed; CRT/numeracy best-effort; vocab dropped)_
- RESOLVED: Bootstrap ensemble Pearson r = 0.305 (V4.1)
- What Pearson r does Model 4 achieve on full 233 people? _(pending — needs API run)_

## Key Decisions
- Final pipeline: blend of KNN + V4.1 ensemble + Model 4 LLM
- LLM uses Claude Sonnet via Anthropic API
- Model 4 script: scripts/model4_llm_predictor.py
- Final predict script: scripts/final_predict.py
