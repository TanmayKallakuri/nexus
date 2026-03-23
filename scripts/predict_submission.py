"""
Predict Submission — Last-Mile Test Pipeline
=============================================
Takes a test JSON file (shaped like sample_test_questions.json), classifies
each question, routes prediction requests, fills predicted_answer for every
row, and writes the completed JSON output.

Usage:
  python scripts/predict_submission.py <test_input.json> [output.json]

If output path is omitted, writes to outputs/submission.json.

Data dependencies (loaded at startup):
  - outputs/person_response_profiles.csv     (person feature table)
  - outputs/person_construct_scores.csv      (construct-level scores)
  - outputs/construct_mapping.csv            (question→construct mapping)
  - outputs/construct_centroids.pkl          (construct centroids, optional)
  - personas_text/                           (persona text files, optional)

ML model dependencies (placeholder — awaiting Tolendi's ensemble):
  - outputs/tolendi/ensemble_model.pkl       (NOT YET AVAILABLE)

Fallback strategy:
  When ML model is not available, uses construct-score-based prediction
  for ordinal/likert/continuous questions, and midpoint fallback otherwise.

Authors: Jasjyot (pipeline), Tolendi (ensemble — pending)
"""

import json
import os
import re
import sys
import pickle
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_PATH = PROJECT_ROOT / "outputs" / "person_response_profiles_repaired.csv"
CONSTRUCT_SCORES_PATH = PROJECT_ROOT / "outputs" / "person_construct_scores.csv"
CONSTRUCT_MAPPING_PATH = PROJECT_ROOT / "outputs" / "construct_mapping.csv"
CONSTRUCT_CENTROIDS_PATH = PROJECT_ROOT / "outputs" / "construct_centroids.pkl"
PERSONAS_TEXT_DIR = PROJECT_ROOT / "data" / "personas_text"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "submission.json"

# TODO: Uncomment when Tolendi's ensemble is ready
# ENSEMBLE_MODEL_PATH = PROJECT_ROOT / "outputs" / "tolendi" / "ensemble_model.pkl"


# ============================================================
# GLOBAL STATE (loaded once at startup)
# ============================================================
PERSON_PROFILES = None       # DataFrame: person-level features
CONSTRUCT_SCORES = None      # DataFrame: person × construct scores
CONSTRUCT_MAPPING = None     # DataFrame: question_id → construct_id mapping
CONSTRUCT_CENTROIDS = None   # dict: construct_id → centroid vector (optional)
PERSONA_TEXTS = None         # dict: person_id → text summary (optional)
ENSEMBLE_MODEL = None        # placeholder — Tolendi's model (None until ready)
SENTENCE_MODEL = None        # sentence transformer model (optional)


# ============================================================
# STEP 1: LOAD TEST JSON
# ============================================================

def load_test(filepath):
    """Load test JSON file. Returns list of question dicts."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Test file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

    print(f"  Loaded {len(data)} test questions from {filepath}")
    return data


# ============================================================
# STEP 2: CLASSIFY QUESTION
# ============================================================

def classify_question(q):
    """
    Classify a question into one of:
      - continuous   (numeric range like "0 to 100")
      - binary       (exactly 2 options)
      - likert       (3-7 ordered options with scale-like text)
      - ordinal      (ordered numeric or ranked options, >7 or non-likert)
      - categorical  (unordered text options)

    Returns (question_type, parsed_range_or_options).
    """
    options = q.get("options", None)

    # --- String range format: "0 to 100", "1 to 10", etc. ---
    if isinstance(options, str):
        range_info = parse_options_range(options)
        if range_info is not None:
            return "continuous", range_info
        # Fallback: treat unknown string as categorical with one option
        return "categorical", [options]

    # --- List of options ---
    if isinstance(options, list):
        n = len(options)

        if n == 0:
            return "categorical", []

        # Binary: exactly 2 options
        if n == 2:
            return "binary", options

        # Check if options look like a Likert scale (3-7 ordered options)
        if 3 <= n <= 7:
            if _looks_likert(options):
                return "likert", options

        # Check if options are numeric or ordinal-looking
        if _looks_ordinal(options):
            return "ordinal", options

        # Default: categorical
        return "categorical", options

    # No options at all — treat as continuous with unknown range
    return "continuous", {"low": 0, "high": 100}


def _looks_likert(options):
    """Heuristic: options look like a Likert scale."""
    # Common Likert patterns: numbered prefix, agree/disagree, frequency, etc.
    likert_keywords = [
        "strongly", "agree", "disagree", "always", "never", "often",
        "rarely", "sometimes", "likely", "unlikely", "true", "untrue",
        "satisfied", "dissatisfied", "important", "unimportant",
        "extremely", "moderately", "slightly", "very", "not at all",
        "definitely", "probably", "neither",
    ]

    text_lower = " ".join(str(o).lower() for o in options)

    # If multiple Likert keywords are present, likely Likert
    keyword_hits = sum(1 for kw in likert_keywords if kw in text_lower)
    if keyword_hits >= 2:
        return True

    # If all options start with a digit followed by separator, likely ordinal/likert
    digit_pattern = re.compile(r"^\d+\s*[-–—.:)]\s*")
    if all(digit_pattern.match(str(o)) for o in options):
        return True

    return False


def _looks_ordinal(options):
    """Heuristic: options look ordinal/numeric."""
    # All options are purely numeric
    try:
        [float(str(o).strip()) for o in options]
        return True
    except (ValueError, TypeError):
        pass

    # Options start with sequential numbers
    digit_prefix = re.compile(r"^(\d+)")
    nums = []
    for o in options:
        m = digit_prefix.match(str(o))
        if m:
            nums.append(int(m.group(1)))
    if len(nums) == len(options) and nums == sorted(nums):
        return True

    return False


# ============================================================
# STEP 3: PARSE OPTIONS RANGE
# ============================================================

def parse_options_range(options):
    """
    Parse a string range like '0 to 100' or '1 to 10'.
    Returns dict {'low': int, 'high': int} or None if not a range.
    """
    if not isinstance(options, str):
        return None

    # Pattern: "<number> to <number>"
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)\s*$", options, re.IGNORECASE)
    if m:
        low = float(m.group(1))
        high = float(m.group(2))
        # Use int if both are whole numbers
        if low == int(low) and high == int(high):
            low, high = int(low), int(high)
        return {"low": low, "high": high}

    return None


# ============================================================
# STEP 4: BUILD QUESTION TEXT
# ============================================================

def build_question_text(q):
    """
    Build the full question text by combining context and question_text.
    If context is null/None, use only question_text.
    """
    context = q.get("context", None)
    question_text = q.get("question_text", "")

    if context and str(context).strip() and str(context).strip().lower() != "null":
        return f"{str(context).strip()}\n\n{str(question_text).strip()}"
    return str(question_text).strip()


# ============================================================
# STEP 5: DATA LOADING
# ============================================================

def load_data():
    """
    Load all data dependencies at startup.
    Sets global state variables. Handles missing files gracefully.
    """
    global PERSON_PROFILES, CONSTRUCT_SCORES, CONSTRUCT_MAPPING
    global CONSTRUCT_CENTROIDS, PERSONA_TEXTS, ENSEMBLE_MODEL, SENTENCE_MODEL

    print("\n[DATA] Loading data dependencies...")

    # --- Person response profiles ---
    if PROFILES_PATH.exists():
        PERSON_PROFILES = pd.read_csv(PROFILES_PATH)
        print(f"  [OK] Person profiles: {PERSON_PROFILES.shape[0]} persons x {PERSON_PROFILES.shape[1]} features")
    else:
        print(f"  [--] Person profiles NOT FOUND: {PROFILES_PATH}")
        PERSON_PROFILES = pd.DataFrame()

    # --- Construct scores ---
    if CONSTRUCT_SCORES_PATH.exists():
        CONSTRUCT_SCORES = pd.read_csv(CONSTRUCT_SCORES_PATH)
        print(f"  [OK] Construct scores: {CONSTRUCT_SCORES.shape[0]} rows")
    else:
        print(f"  [--] Construct scores NOT FOUND: {CONSTRUCT_SCORES_PATH}")
        CONSTRUCT_SCORES = pd.DataFrame()

    # --- Construct mapping ---
    if CONSTRUCT_MAPPING_PATH.exists():
        CONSTRUCT_MAPPING = pd.read_csv(CONSTRUCT_MAPPING_PATH)
        print(f"  [OK] Construct mapping: {CONSTRUCT_MAPPING.shape[0]} questions mapped")
    else:
        print(f"  [--] Construct mapping NOT FOUND: {CONSTRUCT_MAPPING_PATH}")
        CONSTRUCT_MAPPING = pd.DataFrame()

    # --- Construct centroids (optional) ---
    if CONSTRUCT_CENTROIDS_PATH.exists():
        try:
            with open(CONSTRUCT_CENTROIDS_PATH, "rb") as f:
                CONSTRUCT_CENTROIDS = pickle.load(f)
            print(f"  [OK] Construct centroids: {len(CONSTRUCT_CENTROIDS)} centroids loaded")
        except Exception as e:
            print(f"  [--] Construct centroids load error: {e}")
            CONSTRUCT_CENTROIDS = {}
    else:
        print(f"  [--] Construct centroids NOT FOUND: {CONSTRUCT_CENTROIDS_PATH}")
        CONSTRUCT_CENTROIDS = {}

    # --- Persona texts (optional) ---
    if PERSONAS_TEXT_DIR.exists() and PERSONAS_TEXT_DIR.is_dir():
        PERSONA_TEXTS = {}
        for fpath in sorted(PERSONAS_TEXT_DIR.glob("*.txt")):
            pid = fpath.stem.replace("_persona", "")
            with open(fpath, "r", encoding="utf-8") as f:
                PERSONA_TEXTS[pid] = f.read()
        print(f"  [OK] Persona texts: {len(PERSONA_TEXTS)} loaded")
    else:
        print(f"  [--] Persona texts dir NOT FOUND: {PERSONAS_TEXT_DIR}")
        PERSONA_TEXTS = {}

    # --- Sentence transformer (optional) ---
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"  [OK] Sentence transformer loaded: all-MiniLM-L6-v2")
    except ImportError:
        print(f"  [--] sentence-transformers not installed -- embedding features disabled")
        SENTENCE_MODEL = None
    except Exception as e:
        print(f"  [--] Sentence transformer load error: {e}")
        SENTENCE_MODEL = None

    # --- Tolendi's ensemble model (placeholder) ---
    # TODO: Uncomment when Tolendi's ensemble model is ready
    # ENSEMBLE_MODEL_PATH = PROJECT_ROOT / "outputs" / "tolendi" / "ensemble_model.pkl"
    # if ENSEMBLE_MODEL_PATH.exists():
    #     try:
    #         with open(ENSEMBLE_MODEL_PATH, "rb") as f:
    #             ENSEMBLE_MODEL = pickle.load(f)
    #         print(f"  [OK] Ensemble model loaded from {ENSEMBLE_MODEL_PATH}")
    #     except Exception as e:
    #         print(f"  [--] Ensemble model load error: {e}")
    #         ENSEMBLE_MODEL = None
    # else:
    #     print(f"  [--] Ensemble model NOT FOUND: {ENSEMBLE_MODEL_PATH}")
    #     ENSEMBLE_MODEL = None
    print(f"  [--] Ensemble model: placeholder -- awaiting Tolendi's build")

    print("[DATA] Loading complete.\n")


# ============================================================
# STEP 6: PREDICT LLM (placeholder)
# ============================================================

def predict_llm(person_id, question, persona_texts):
    """
    Placeholder for LLM-based prediction.

    TODO: When LLM integration is ready, implement this function to:
      1. Retrieve persona_texts[person_id] for context
      2. Build a prompt with the question text & options
      3. Call the LLM API
      4. Parse and return the predicted answer

    For now, returns None so the pipeline falls through to fallback.
    """
    # TODO: Implement LLM prediction path
    return None


# ============================================================
# PREDICTION HELPERS
# ============================================================

def predict_with_construct_scores(person_id, question, q_type, parsed_opts):
    """
    Attempt prediction using construct scores.
    If the question can be mapped to a similar construct, use the person's
    historical score for that construct as a prediction basis.

    Returns predicted answer or None if not possible.
    """
    if CONSTRUCT_SCORES is None or CONSTRUCT_SCORES.empty:
        return None
    if CONSTRUCT_MAPPING is None or CONSTRUCT_MAPPING.empty:
        return None

    # Get person's construct scores
    person_scores = CONSTRUCT_SCORES[CONSTRUCT_SCORES["person_id"] == person_id]
    if person_scores.empty:
        return None

    # For continuous questions — use the person's overall mean answer tendency
    if q_type == "continuous":
        low = parsed_opts.get("low", 0) if isinstance(parsed_opts, dict) else 0
        high = parsed_opts.get("high", 100) if isinstance(parsed_opts, dict) else 100

        # Use person's mean percentile across all constructs as a tendency proxy
        mean_percentile = person_scores["percentile"].mean()
        if not np.isnan(mean_percentile):
            predicted = low + mean_percentile * (high - low)
            return int(round(predicted))
        return None

    # For likert/ordinal — use person's mean normalized answer
    if q_type in ("likert", "ordinal"):
        if isinstance(parsed_opts, list) and len(parsed_opts) > 0:
            n_opts = len(parsed_opts)
            mean_percentile = person_scores["percentile"].mean()
            if not np.isnan(mean_percentile):
                # Map percentile to option index (1-indexed)
                predicted_pos = int(round(1 + mean_percentile * (n_opts - 1)))
                predicted_pos = max(1, min(n_opts, predicted_pos))
                return predicted_pos
        return None

    return None


def get_midpoint_fallback(q_type, parsed_opts):
    """
    Midpoint fallback: return the middle option/value.
    This is the guaranteed fallback so every row gets a prediction.
    """
    if q_type == "continuous":
        if isinstance(parsed_opts, dict):
            low = parsed_opts.get("low", 0)
            high = parsed_opts.get("high", 100)
            return int(round((low + high) / 2))
        return 50  # default midpoint

    if q_type in ("likert", "ordinal"):
        if isinstance(parsed_opts, list) and len(parsed_opts) > 0:
            n = len(parsed_opts)
            mid_idx = (n + 1) // 2  # 1-indexed midpoint
            return mid_idx
        return 3  # common Likert midpoint

    if q_type == "binary":
        if isinstance(parsed_opts, list) and len(parsed_opts) == 2:
            return 1  # default to first option for binary
        return 1

    if q_type == "categorical":
        if isinstance(parsed_opts, list) and len(parsed_opts) > 0:
            return 1  # default to first option
        return 1

    return 1  # absolute final fallback


# ============================================================
# STEP 7: PREDICT ALL
# ============================================================

def predict_all(test_data):
    """
    For each question in test_data:
      1. Classify the question
      2. Build question text
      3. Try ML ensemble (when available)
      4. Try construct-score-based prediction
      5. Try LLM placeholder
      6. Fall through to midpoint fallback

    Every item will have predicted_answer filled.
    Returns the modified test_data list and a stats dict.
    """
    stats = {
        "total": len(test_data),
        "by_type": {},
        "method_used": {"ensemble": 0, "construct_score": 0, "llm": 0, "midpoint_fallback": 0},
        "filled": 0,
        "unfilled": 0,
    }

    for i, q in enumerate(test_data):
        person_id = q.get("person_id", "")
        q_type, parsed_opts = classify_question(q)

        # Track question type stats
        stats["by_type"][q_type] = stats["by_type"].get(q_type, 0) + 1

        # Build full question text (for future model use)
        full_text = build_question_text(q)

        prediction = None

        # --- Path 1: ML ensemble (Tolendi's model, when ready) ---
        if ENSEMBLE_MODEL is not None:
            # TODO: Call ensemble model
            # prediction = ENSEMBLE_MODEL.predict(person_id, q, ...)
            pass

        if prediction is not None:
            stats["method_used"]["ensemble"] += 1
            q["predicted_answer"] = prediction
            stats["filled"] += 1
            continue

        # --- Path 2: Construct-score-based prediction ---
        if q_type in ("continuous", "likert", "ordinal"):
            prediction = predict_with_construct_scores(person_id, q, q_type, parsed_opts)

        if prediction is not None:
            stats["method_used"]["construct_score"] += 1
            q["predicted_answer"] = prediction
            stats["filled"] += 1
            continue

        # --- Path 3: LLM placeholder (for binary/categorical) ---
        if q_type in ("binary", "categorical"):
            prediction = predict_llm(person_id, q, PERSONA_TEXTS)

        if prediction is not None:
            stats["method_used"]["llm"] += 1
            q["predicted_answer"] = prediction
            stats["filled"] += 1
            continue

        # --- Path 4: Midpoint fallback (guaranteed) ---
        prediction = get_midpoint_fallback(q_type, parsed_opts)
        stats["method_used"]["midpoint_fallback"] += 1
        q["predicted_answer"] = prediction
        stats["filled"] += 1

    # Final check: everything should be filled
    stats["unfilled"] = sum(1 for q in test_data if q.get("predicted_answer") is None)

    return test_data, stats


# ============================================================
# STEP 8: RUN SUBMISSION
# ============================================================

def run_submission(test_filepath, output_filepath=None):
    """
    Full pipeline:
      1. Load test JSON
      2. Load data dependencies
      3. Classify & count question types
      4. Run predictions
      5. Save output JSON
      6. Print summary
    """
    if output_filepath is None:
        output_filepath = DEFAULT_OUTPUT_PATH

    test_filepath = Path(test_filepath)
    output_filepath = Path(output_filepath)

    print("=" * 60)
    print("PREDICT SUBMISSION PIPELINE")
    print("=" * 60)

    # --- Load test ---
    print(f"\n[1/4] Loading test file...")
    test_data = load_test(test_filepath)

    # --- Load data ---
    print(f"\n[2/4] Loading data dependencies...")
    load_data()

    # --- Classify & preview ---
    print(f"[3/4] Classifying questions and running predictions...")
    type_preview = {}
    for q in test_data:
        q_type, _ = classify_question(q)
        type_preview[q_type] = type_preview.get(q_type, 0) + 1
    print(f"  Question type distribution:")
    for qt, count in sorted(type_preview.items()):
        print(f"    {qt:15s}: {count}")

    # --- Run predictions ---
    results, stats = predict_all(test_data)

    # --- Validate: every row must have predicted_answer ---
    assert stats["unfilled"] == 0, \
        f"FATAL: {stats['unfilled']} questions have no predicted_answer!"

    # --- Save output ---
    print(f"\n[4/4] Saving submission...")
    os.makedirs(output_filepath.parent, exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    output_size = os.path.getsize(output_filepath) / 1024
    print(f"  Saved: {output_filepath}")
    print(f"  File size: {output_size:.1f} KB")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"SUBMISSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total questions:   {stats['total']}")
    print(f"  Filled:            {stats['filled']}")
    print(f"  Unfilled:          {stats['unfilled']}")
    print(f"\n  By question type:")
    for qt, count in sorted(stats["by_type"].items()):
        print(f"    {qt:15s}: {count}")
    print(f"\n  By prediction method:")
    for method, count in sorted(stats["method_used"].items()):
        pct = (count / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"    {method:20s}: {count:5d} ({pct:.1f}%)")
    print(f"\n  Output: {output_filepath}")

    return results, stats


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_submission.py <test_input.json> [output.json]")
        print("       test_input.json — path to the hidden test JSON")
        print("       output.json     — optional output path (default: outputs/submission.json)")
        sys.exit(1)

    test_filepath = sys.argv[1]
    output_filepath = sys.argv[2] if len(sys.argv) > 2 else None

    run_submission(test_filepath, output_filepath)


if __name__ == "__main__":
    main()
