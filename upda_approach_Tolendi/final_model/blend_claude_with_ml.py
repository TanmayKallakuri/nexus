from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import build_family_aware_submission as fam


DEFAULT_ML = Path("artifacts/final_test_ml_predictions.json")
DEFAULT_QUESTION_JSON = Path("final_test_questions.json")
DEFAULT_OUTPUT = Path("artifacts/final_blended_predictions.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend Claude predictions with ML predictions.")
    parser.add_argument("--ml-json", type=Path, default=DEFAULT_ML)
    parser.add_argument("--claude-json", type=Path, required=True)
    parser.add_argument("--question-json", type=Path, default=DEFAULT_QUESTION_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_long_json(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(data)
    expected = {"person_id", "question_id", "predicted_answer"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame[["person_id", "question_id", "predicted_answer"]].copy()


def family_weight(family: str) -> float:
    weights = {
        "urbanicity": 0.35,
        "housing_tenure": 0.35,
        "business_farm": 0.30,
        "student_status": 0.35,
        "work_status": 0.35,
        "volunteering": 0.40,
        "self_rated_health": 0.40,
        "service_discrimination": 0.55,
        "internet_health_info": 0.40,
        "financial_satisfaction": 0.35,
        "layoff_risk": 0.45,
        "generalized_trust": 0.60,
        "confidence_companies": 0.65,
        "confidence_press": 0.80,
        "gender_roles": 0.75,
        "immigration_jobs": 0.80,
        "affirmative_action": 0.85,
        "government_responsibility": 0.85,
        "taxes_healthcare": 0.80,
        "hard_work_vs_luck": 0.70,
        "fairness_vs_advantage": 0.60,
        "father_education": 0.35,
        "data_privacy": 0.70,
        "tiktok_ban": 0.80,
        "turnout_2020": 0.85,
        "vote_choice_2020": 0.90,
        "share_scenario_matrix": 0.85,
        "share_attribute_importance_headline": 0.75,
        "share_attribute_importance_source": 0.75,
        "share_attribute_importance_content_type": 0.75,
        "share_attribute_importance_political_lean": 0.85,
        "share_attribute_importance_likes": 0.80,
        "accuracy_attribute_importance_headline": 0.75,
        "accuracy_attribute_importance_source": 0.80,
        "accuracy_attribute_importance_content_type": 0.75,
        "accuracy_attribute_importance_political_lean": 0.80,
        "accuracy_attribute_importance_likes": 0.75,
        "truth_vs_entertainment_share": 0.85,
        "truth_vs_entertainment_info": 0.85,
        "share_news_binary": 0.80,
        "headline_funny": 0.85,
        "share_likelihood_article": 0.85,
        "share_accuracy_norm": 0.80,
        "source_trust_100": 0.85,
    }
    return weights.get(family, 0.70)


def finalize_prediction(row: pd.Series) -> int:
    options = row["options"]
    if isinstance(options, list):
        lo, hi = 1, len(options)
    else:
        lo, hi = 0, 100
    return int(round(min(max(row["blended_prediction"], lo), hi)))


def main() -> None:
    args = parse_args()

    ml = load_long_json(args.ml_json).rename(columns={"predicted_answer": "ml_prediction"})
    claude = load_long_json(args.claude_json).rename(columns={"predicted_answer": "claude_prediction"})
    question_rows = pd.DataFrame(json.loads(args.question_json.read_text(encoding="utf-8")))
    question_meta = fam.compute_question_meta(question_rows)
    meta = question_meta[["person_id", "question_id", "options", "question_family"]].copy()

    out = meta.merge(ml, on=["person_id", "question_id"], how="left").merge(
        claude, on=["person_id", "question_id"], how="left"
    )
    if out["ml_prediction"].isna().any():
        raise ValueError("Missing ML predictions for some rows.")
    if out["claude_prediction"].isna().any():
        raise ValueError("Missing Claude predictions for some rows.")

    out["claude_weight"] = out["question_family"].map(family_weight)
    out["blended_prediction"] = (
        (1.0 - out["claude_weight"]) * out["ml_prediction"]
        + out["claude_weight"] * out["claude_prediction"]
    )
    out["predicted_answer"] = out.apply(finalize_prediction, axis=1)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            out[["person_id", "question_id", "predicted_answer"]].to_dict(orient="records"),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved blended predictions to {args.output_json}")


if __name__ == "__main__":
    main()
