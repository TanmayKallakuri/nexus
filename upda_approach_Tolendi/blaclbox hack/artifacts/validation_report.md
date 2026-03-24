# Validation Report

## Model Comparison

         model  mean_accuracy_proxy  mean_pearson_correlation    score
semantic_prior             0.699019                  0.356071 0.493250
         ridge             0.705905                  0.358029 0.497179
      lightgbm             0.751160                  0.390304 0.534646
      ensemble             0.751160                  0.390304 0.534646

## Ensemble Weights

- semantic_prior: 0.00
- ridge: 0.00
- lightgbm: 1.00

## Notes

- Validation holds out full questions with GroupKFold on `question_id`.
- Targets are modeled on normalized answer scales and converted back to raw numeric answers before scoring.
- Sparse and non-numeric historical items are excluded from the supervised target set.
- No local test/template file was detected, so final submission export is waiting on that file.