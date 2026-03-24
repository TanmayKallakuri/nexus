from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import run_unseen_question_pipeline as pipeline


ROOT = Path(".")
DATA_DIR = ROOT / "data"
TEST_DIR = ROOT / "test data"
ARTIFACTS_DIR = ROOT / "artifacts"
TEMPLATE_PATH = TEST_DIR / "sample_11_questions_template.json"
LLM_FILES = [
    TEST_DIR / "predictions_233_respondents.csv",
    TEST_DIR / "respondent_predictions_233.csv",
    TEST_DIR / "predicted_answers-sonnet4.6-J.csv",
]
MODEL_NAMES = ["semantic_prior", "ridge", "lightgbm"]
ENSEMBLE_WEIGHTS = {"semantic_prior": 0.0, "ridge": 0.0, "lightgbm": 1.0}


def normalize_llm_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for column in df.columns:
        if column == "Respondent_ID":
            continue
        rename_map[column] = column.split("_", 1)[0]
    return df.rename(columns=rename_map)


def load_template() -> List[dict]:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


def build_full_test_rows(respondent_ids: List[str], template_rows: List[dict]) -> pd.DataFrame:
    expanded_rows = []
    for respondent_id in respondent_ids:
        for row in template_rows:
            expanded_rows.append(
                {
                    "person_id": respondent_id,
                    "question_id": row["question_id"],
                    "context": row.get("context"),
                    "question_text": row.get("question_text"),
                    "options": row.get("options"),
                }
            )
    return pd.DataFrame(expanded_rows)


def finalize_predictions(pred_df: pd.DataFrame, template_rows: List[dict]) -> pd.DataFrame:
    template_meta = {
        row["question_id"]: {
            "is_discrete": isinstance(row.get("options"), list),
            "min_value": 1 if isinstance(row.get("options"), list) else 0,
            "max_value": len(row["options"]) if isinstance(row.get("options"), list) else 100,
        }
        for row in template_rows
    }
    out = pred_df.copy()
    out["predicted_answer_raw"] = out["predicted_answer"]

    def finalize(row: pd.Series) -> float:
        meta = template_meta[row["question_id"]]
        value = float(np.clip(row["predicted_answer_raw"], meta["min_value"], meta["max_value"]))
        if meta["is_discrete"]:
            return float(int(round(value)))
        return float(round(value))

    out["predicted_answer"] = out.apply(finalize, axis=1)
    return out


def wide_to_long(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    long_df = df.melt(id_vars=["Respondent_ID"], var_name="question_id", value_name="reference_answer")
    long_df["source"] = source_name
    return long_df.rename(columns={"Respondent_ID": "person_id"})


def comparison_metrics(pred_long: pd.DataFrame, ref_long: pd.DataFrame, question_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pred_long.merge(ref_long, on=["person_id", "question_id"], how="inner")
    merged = merged.rename(columns={"predicted_answer": "model_answer"})
    per_question = pipeline.question_level_metrics(
        merged.rename(columns={"reference_answer": "response", "model_answer": "prediction"}),
        "prediction",
        question_meta,
    )
    summary = pipeline.summarize_metrics(per_question)
    summary_df = pd.DataFrame([summary])
    return merged, pd.concat([per_question], ignore_index=True)


def main() -> None:
    pipeline.ensure_dir(ARTIFACTS_DIR)

    llm_frames = [normalize_llm_columns(pd.read_csv(path)) for path in LLM_FILES]
    respondent_ids = llm_frames[0]["Respondent_ID"].astype(str).tolist()

    surveys = pipeline.collect_surveys(DATA_DIR)
    responses, questions = pipeline.build_historical_tables(
        surveys, min_valid_responses=pipeline.MIN_VALID_RESPONSES
    )
    questions = pipeline.add_question_text_features(questions)
    external = pipeline.encode_external_person_features(
        pipeline.parse_persona_texts(DATA_DIR / "personas_text")
    )

    template_rows = load_template()
    full_test = build_full_test_rows(respondent_ids, template_rows)
    prediction_questions = pipeline.infer_test_question_features(full_test)
    prediction_rows = full_test[["person_id", "question_id"]].copy()
    submission = pipeline.fit_full_models_and_predict(
        responses=responses,
        questions=questions,
        external_person_features=external,
        prediction_rows=prediction_rows,
        prediction_questions=prediction_questions,
        model_names=MODEL_NAMES,
        ensemble_weights=ENSEMBLE_WEIGHTS,
    )
    model_long = finalize_predictions(
        full_test.merge(
            submission[["person_id", "question_id", "predicted_answer"]],
            on=["person_id", "question_id"],
            how="left",
        ),
        template_rows,
    )
    model_long.to_csv(ARTIFACTS_DIR / "model_vs_llm_predictions_long.csv", index=False)
    model_long.pivot(index="person_id", columns="question_id", values="predicted_answer").reset_index().rename(
        columns={"person_id": "Respondent_ID"}
    ).to_csv(ARTIFACTS_DIR / "model_vs_llm_predictions_wide.csv", index=False)

    question_meta = prediction_questions[["question_id", "scale_min", "scale_max", "response_range"]].copy()

    summary_rows = []
    per_question_frames = []
    llm_long_frames = []
    for path, frame in zip(LLM_FILES, llm_frames):
        llm_long = wide_to_long(frame, path.name)
        llm_long_frames.append(llm_long)
        merged, per_question = comparison_metrics(model_long, llm_long, question_meta)
        summary = pipeline.summarize_metrics(per_question)
        summary["comparison"] = f"model_vs_{path.stem}"
        summary["n_rows"] = len(merged)
        summary_rows.append(summary)
        per_question["comparison"] = summary["comparison"]
        per_question_frames.append(per_question)

    avg_reference = (
        pd.concat(llm_long_frames, ignore_index=True)
        .groupby(["person_id", "question_id"], as_index=False)["reference_answer"]
        .mean()
    )
    merged_avg, per_question_avg = comparison_metrics(model_long, avg_reference, question_meta)
    avg_summary = pipeline.summarize_metrics(per_question_avg)
    avg_summary["comparison"] = "model_vs_average_of_3_llms"
    avg_summary["n_rows"] = len(merged_avg)
    summary_rows.append(avg_summary)
    per_question_avg["comparison"] = avg_summary["comparison"]
    per_question_frames.append(per_question_avg)

    summary_df = pd.DataFrame(summary_rows)[
        ["comparison", "mean_accuracy_proxy", "mean_pearson_correlation", "score", "n_rows"]
    ].sort_values("comparison")
    per_question_df = pd.concat(per_question_frames, ignore_index=True)[
        ["comparison", "question_id", "mad", "accuracy_proxy", "pearson_correlation", "n_participants"]
    ].sort_values(["comparison", "question_id"])

    summary_df.to_csv(ARTIFACTS_DIR / "llm_comparison_summary.csv", index=False)
    per_question_df.to_csv(ARTIFACTS_DIR / "llm_comparison_per_question.csv", index=False)

    lines = ["# Model vs LLM Comparison", ""]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- {row.comparison}: accuracy={row.mean_accuracy_proxy:.4f}, "
            f"correlation={row.mean_pearson_correlation:.4f}, score={row.score:.4f}"
        )
    (ARTIFACTS_DIR / "llm_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
