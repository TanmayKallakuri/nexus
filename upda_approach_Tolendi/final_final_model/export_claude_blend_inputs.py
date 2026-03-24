from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import build_family_aware_submission as fam
import run_unseen_question_pipeline as pipeline


ROOT = Path(".")
DATA_DIR = ROOT / "data"
DEFAULT_QUESTION_JSON = ROOT / "final_test_questions.json"
DEFAULT_ML_JSON = ROOT / "artifacts" / "final_test_ml_predictions.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "claude_blend_seed.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compact ML seed rows for Claude blending.")
    parser.add_argument("--question-json", type=Path, default=DEFAULT_QUESTION_JSON)
    parser.add_argument("--ml-json", type=Path, default=DEFAULT_ML_JSON)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = pd.DataFrame(json.loads(args.question_json.read_text(encoding="utf-8")))
    ml = pd.DataFrame(json.loads(args.ml_json.read_text(encoding="utf-8")))
    if ml.empty:
        raise ValueError("ML prediction file is empty.")

    raw_persona = pipeline.parse_persona_texts(DATA_DIR / "personas_text")
    signals = fam.person_signal_table(raw_persona)
    compact_cols = [
        "person_id",
        "age",
        "gender",
        "education_level",
        "income",
        "employment_status",
        "race",
        "geographic_region",
        "religion",
        "political_affiliation",
        "political_views",
        "social_traditionalism",
        "progressivism",
        "economic_security",
        "civic_engagement",
        "media_skepticism",
        "digital_engagement",
        "source_literacy",
        "humor_sharing_taste",
    ]
    meta = fam.compute_question_meta(questions)[
        ["person_id", "question_id", "question_family", "context", "question_text", "options"]
    ]
    seed = meta.merge(ml, on=["person_id", "question_id"], how="left").merge(
        signals[compact_cols], on="person_id", how="left"
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in seed.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Saved Claude seed rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
