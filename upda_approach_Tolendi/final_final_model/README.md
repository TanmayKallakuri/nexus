# BlackBox Final-Test Pipeline

This repo contains the ML-side pipeline for predicting final-test answers for the same respondents and blending those predictions with Claude for stronger correlation.

The default ML output is the calibrated family-aware model. The optional ranking residual mode exists for experiments, but it is not the default because it underperformed on the final-test API check.

## Core files

- `run_unseen_question_pipeline.py`
  - baseline respondent-question training pipeline
- `build_family_aware_submission.py`
  - main ML predictor for `final_test_questions.json`
- `export_claude_blend_inputs.py`
  - exports compact ML seed rows for Claude
- `blend_claude_with_ml.py`
  - blends Claude predictions with ML predictions
- `submit.py`
  - submits a prediction JSON to the scoring API

## Data you should keep

- `data/personas_csv/`
- `data/personas_text/`
- `final_test_questions.json`

Generated files under `artifacts/` are ignored by git and can be rebuilt at any time.

## Quickstart

1. Create the ML predictions:

```bash
python build_family_aware_submission.py \
  --input-json final_test_questions.json \
  --output-json artifacts/final_test_ml_predictions.json \
  --debug-csv artifacts/final_test_ml_predictions_debug.csv \
  --summary-csv artifacts/final_test_ml_predictions_summary.csv
```

Keep `--enable-ranking-residual` off unless you are deliberately testing an alternative variant.

2. Export compact rows for Claude:

```bash
python export_claude_blend_inputs.py \
  --question-json final_test_questions.json \
  --ml-json artifacts/final_test_ml_predictions.json \
  --output-jsonl artifacts/claude_blend_seed.jsonl
```

3. Use the instructions in `CLAUDE_INSTRUCTIONS.md` to get Claude predictions.

4. Blend Claude with ML:

```bash
python blend_claude_with_ml.py \
  --ml-json artifacts/final_test_ml_predictions.json \
  --claude-json artifacts/claude_predictions.json \
  --question-json final_test_questions.json \
  --output-json artifacts/final_blended_predictions.json
```

5. Submit:

```bash
python submit.py \
  --predictions artifacts/final_blended_predictions.json
```

Set secrets in the environment:

```bash
set ANTHROPIC_API_KEY=YOUR_CLAUDE_KEY_HERE
set BLACKBOX_TEAM_TOKEN=YOUR_TOKEN_HERE
```

If you prefer not to export the scoring token, you can also pass it directly:

```bash
python submit.py --predictions artifacts/final_blended_predictions.json --token YOUR_TOKEN_HERE
```

## Notes

- Correlation matters more than accuracy.
- Use ML as a stable anchor and Claude as the ranking/semantics specialist.
- For political, news-sharing, and media-trust blocks, Claude should have higher blend weight than ML.
- Keep the repo source-only on GitHub; regenerate `artifacts/` locally when you need a fresh run.
