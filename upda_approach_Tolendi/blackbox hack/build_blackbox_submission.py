from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import run_unseen_question_pipeline as pipeline


ROOT = Path(".")
DATA_DIR = ROOT / "data"
BLACKBOX_PATH = ROOT / "Questions to test in blackbox.json"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_NAMES = ["semantic_prior", "ridge", "lightgbm"]
BASE_WEIGHTS = {"semantic_prior": 0.0, "ridge": 0.0, "lightgbm": 1.0}


EMPLOYED = {"Full-time employment", "Part-time employment", "Self-employed"}
INACTIVE = {"Unemployed", "Student", "Home-maker"}
IDEOLOGY_MAP = {
    "Very liberal": -2.0,
    "Liberal": -1.0,
    "Moderate": 0.0,
    "Conservative": 1.0,
    "Very conservative": 2.0,
}
PARTY_MAP = {
    "Democrat": -1.0,
    "Independent": 0.0,
    "Something else": 0.2,
    "Republican": 1.0,
}
EDUCATION_MAP = {
    "Less than high school": 0.0,
    "High school graduate": 1.0,
    "Some college, no degree": 2.0,
    "Associate's degree": 3.0,
    "College graduate/some postgrad": 4.0,
    "Postgraduate": 5.0,
}
INCOME_MAP = {
    "Less than $30,000": 0.0,
    "$30,000-$50,000": 1.0,
    "$50,000-$75,000": 2.0,
    "$75,000-$100,000": 3.0,
    "$100,000 or more": 4.0,
}


def clip_round(value: float, lo: float, hi: float) -> int:
    return int(round(float(np.clip(value, lo, hi))))


def safe_float(value, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except Exception:
        return default


def person_signal_table(raw_persona: pd.DataFrame) -> pd.DataFrame:
    df = raw_persona.copy()
    for col in [
        "political_views",
        "political_affiliation",
        "education_level",
        "income",
        "employment_status",
        "gender",
        "race",
        "geographic_region",
        "religion",
        "age_midpoint",
    ]:
        if col not in df.columns:
            df[col] = np.nan
    df["ideology_score"] = df["political_views"].map(IDEOLOGY_MAP).fillna(0.0)
    df["party_score"] = df["political_affiliation"].map(PARTY_MAP).fillna(0.0)
    df["education_score"] = df["education_level"].map(EDUCATION_MAP).fillna(2.0)
    df["income_score"] = df["income"].map(INCOME_MAP).fillna(1.5)
    df["age_midpoint"] = df["age_midpoint"].fillna(39.5)
    df["age_2020"] = df["age_midpoint"] - 6.0
    df["employed_score"] = df["employment_status"].isin(EMPLOYED).astype(float)
    df["inactive_score"] = df["employment_status"].isin(INACTIVE).astype(float)
    df["retired_score"] = (df["employment_status"] == "Retired").astype(float)
    df["female_score"] = (df["gender"] == "Female").astype(float)
    df["black_score"] = (df["race"] == "Black").astype(float)
    df["hispanic_score"] = (df["race"] == "Hispanic").astype(float)
    df["asian_score"] = (df["race"] == "Asian").astype(float)
    df["white_score"] = (df["race"] == "White").astype(float)
    df["minority_score"] = ((df["white_score"] == 0) & df["race"].notna()).astype(float)
    df["south_score"] = df["geographic_region"].fillna("").str.contains("South", regex=False).astype(float)
    df["west_score"] = df["geographic_region"].fillna("").str.contains("West", regex=False).astype(float)
    df["religious_score"] = (~df["religion"].fillna("").isin(["Atheist", "Agnostic", "Nothing in particular", ""])).astype(float)
    for col in [
        "score_extraversion",
        "score_agreeableness",
        "wave1_score_conscientiousness",
        "score_openness",
        "score_neuroticism",
        "score_needforcognition",
        "score_GREEN",
    ]:
        if col not in df.columns:
            df[col] = np.nan
    df["score_extraversion"] = df["score_extraversion"].fillna(3.0)
    df["score_agreeableness"] = df["score_agreeableness"].fillna(3.5)
    df["wave1_score_conscientiousness"] = df["wave1_score_conscientiousness"].fillna(3.5)
    df["score_openness"] = df["score_openness"].fillna(3.5)
    df["score_neuroticism"] = df["score_neuroticism"].fillna(3.0)
    df["score_needforcognition"] = df["score_needforcognition"].fillna(3.0)
    df["score_GREEN"] = df["score_GREEN"].fillna(3.0)
    return df


def turnout_heuristic(row: pd.Series) -> float:
    education = safe_float(row["education_score"])
    income = safe_float(row["income_score"])
    age2020 = safe_float(row["age_2020"], 33.5)
    ideology = safe_float(row["ideology_score"])
    employed = safe_float(row["employed_score"])
    if age2020 < 18.0:
        return 3.0
    score = 2.0 - 0.38 * min((age2020 - 18.0) / 20.0, 1.5) - 0.22 * education - 0.15 * income
    score -= 0.20 * abs(ideology) - 0.18 * employed
    return float(np.clip(score, 1, 4))


def question_specific_heuristic(row: pd.Series) -> float:
    qid = row["question_id"]
    ideology = safe_float(row["ideology_score"])
    party = safe_float(row["party_score"])
    education = safe_float(row["education_score"])
    income = safe_float(row["income_score"])
    age = safe_float(row["age_midpoint"], 39.5)
    age2020 = safe_float(row["age_2020"], 33.5)
    employed = safe_float(row["employed_score"])
    inactive = safe_float(row["inactive_score"])
    retired = safe_float(row["retired_score"])
    female = safe_float(row["female_score"])
    black = safe_float(row["black_score"])
    hispanic = safe_float(row["hispanic_score"])
    minority = safe_float(row["minority_score"])
    south = safe_float(row["south_score"])
    religious = safe_float(row["religious_score"])
    ext = safe_float(row["score_extraversion"], 3.0) - 3.0
    agr = safe_float(row["score_agreeableness"], 3.5) - 3.5
    con = safe_float(row["wave1_score_conscientiousness"], 3.5) - 3.5
    opn = safe_float(row["score_openness"], 3.5) - 3.5
    neu = safe_float(row["score_neuroticism"], 3.0) - 3.0
    nfc = safe_float(row["score_needforcognition"], 3.0) - 3.0
    green = safe_float(row["score_GREEN"], 3.0) - 3.0

    if qid == "T5":
        if retired or age >= 68:
            return 3.0
        if employed:
            return 1.0 + max(0.0, -0.15 * con)
        if inactive:
            return 2.0
        return 2.0 + 0.3 * (age < 25)

    if qid == "T8":
        score = 5.5 - 1.4 * black - 0.9 * hispanic - 0.5 * minority - 0.4 * female
        score += 0.25 * income + 0.15 * (age >= 50) + 0.15 * agr
        return float(np.clip(score, 1, 7))

    if qid == "T10":
        score = 2.45 - 0.33 * income - 0.18 * education - 0.28 * employed - 0.10 * (age >= 50)
        score += 0.30 * max(0.0, neu) - 0.08 * con
        return float(np.clip(score, 1, 4))

    if qid == "T14":
        score = 2.25 + 0.38 * ideology + 0.22 * party - 0.12 * education - 0.06 * nfc
        score += 0.08 * south
        return float(np.clip(score, 1, 4))

    if qid == "T17":
        score = 3.0 + 0.60 * ideology + 0.50 * party - 0.75 * black - 0.35 * minority
        score -= 0.12 * agr + 0.08 * opn
        return float(np.clip(score, 1, 5))

    if qid == "T18":
        score = 3.1 + 0.80 * ideology + 0.45 * party + 0.18 * income - 0.30 * minority - 0.12 * green
        return float(np.clip(score, 1, 6))

    if qid == "T22":
        score = 1.25 + 0.72 * education + 0.22 * income - 0.10 * (age >= 60) - 0.08 * minority
        return float(np.clip(score, 1, 6))

    if qid == "T25":
        return turnout_heuristic(row)

    if qid == "T26":
        turnout = turnout_heuristic(row)
        if turnout >= 2.5:
            return 4.0
        trump_score = 0.95 * party + 0.75 * ideology + 0.12 * income + 0.10 * south
        trump_score -= 0.28 * black + 0.15 * education + 0.10 * opn
        if trump_score <= -0.25:
            return 1.0
        if trump_score >= 0.25:
            return 2.0
        if abs(trump_score) < 0.10 and abs(party) < 0.25:
            return 3.0
        return 1.0 if trump_score < 0 else 2.0

    if qid == "T49":
        young = max(0.0, (45.0 - age) / 25.0)
        score = 3.6 + 0.75 * young + 0.45 * ext + 0.20 * opn - 0.18 * nfc + 0.12 * female
        return float(np.clip(score, 1, 7))

    return safe_float(row["predicted_answer"], 0.0)


def blend_weight(question_id: str) -> float:
    heur_weights = {
        "T5": 0.80,
        "T8": 0.75,
        "T10": 0.70,
        "T14": 0.80,
        "T17": 0.88,
        "T18": 0.85,
        "T22": 0.60,
        "T25": 0.82,
        "T26": 0.92,
        "T49": 0.35,
    }
    return heur_weights.get(question_id, 0.50)


def finalize_answer(row: pd.Series) -> int:
    options = row["options"]
    if isinstance(options, list):
        lo, hi = 1, len(options)
    else:
        lo, hi = 0, 100
    return clip_round(row["blended_prediction"], lo, hi)


def main() -> None:
    pipeline.ensure_dir(ARTIFACTS_DIR)

    surveys = pipeline.collect_surveys(DATA_DIR)
    responses, questions = pipeline.build_historical_tables(
        surveys, min_valid_responses=pipeline.MIN_VALID_RESPONSES
    )
    questions = pipeline.add_question_text_features(questions)
    raw_persona = pipeline.parse_persona_texts(DATA_DIR / "personas_text")
    external_persona = pipeline.encode_external_person_features(raw_persona)
    persona_signals = person_signal_table(raw_persona)

    blackbox_rows = pd.DataFrame(json.loads(BLACKBOX_PATH.read_text(encoding="utf-8")))
    prediction_questions = pipeline.infer_test_question_features(blackbox_rows)
    prediction_rows = blackbox_rows[["person_id", "question_id"]].copy()

    base_predictions = pipeline.fit_full_models_and_predict(
        responses=responses,
        questions=questions,
        external_person_features=external_persona,
        prediction_rows=prediction_rows,
        prediction_questions=prediction_questions,
        model_names=MODEL_NAMES,
        ensemble_weights=BASE_WEIGHTS,
    )[["person_id", "question_id", "predicted_answer"]]

    out = blackbox_rows.drop(columns=["predicted_answer"], errors="ignore").merge(
        base_predictions, on=["person_id", "question_id"], how="left"
    )
    out = out.merge(persona_signals, on="person_id", how="left")
    out["heuristic_prediction"] = out.apply(question_specific_heuristic, axis=1)
    out["heuristic_weight"] = out["question_id"].map(blend_weight)
    out["blended_prediction"] = (
        (1.0 - out["heuristic_weight"]) * out["predicted_answer"]
        + out["heuristic_weight"] * out["heuristic_prediction"]
    )
    out["predicted_answer"] = out.apply(finalize_answer, axis=1)

    json_rows = out[["person_id", "question_id", "predicted_answer"]].to_dict(orient="records")
    (ARTIFACTS_DIR / "blackbox_submission_predictions.json").write_text(
        json.dumps(json_rows, indent=2), encoding="utf-8"
    )
    out.to_csv(ARTIFACTS_DIR / "blackbox_submission_predictions_debug.csv", index=False)

    summary = out.groupby("question_id")["predicted_answer"].agg(["mean", "std", "min", "max"]).reset_index()
    summary.to_csv(ARTIFACTS_DIR / "blackbox_submission_prediction_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved {len(json_rows)} predictions to {ARTIFACTS_DIR / 'blackbox_submission_predictions.json'}")


if __name__ == "__main__":
    main()
