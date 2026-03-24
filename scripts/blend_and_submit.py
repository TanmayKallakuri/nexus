"""
Blend ML + LLM predictions and submit to scoring API.

Usage:
  python scripts/blend_and_submit.py <ml_predictions.json> <llm_predictions.json> <output.json> [--submit]

Without --submit: just blends and saves
With --submit: blends, saves, and submits to API
"""

import json
import sys
import numpy as np
import requests

TEAM_TOKEN = "house_tokens_3f8n"
API_URL = "https://blackboxhackathon-production.up.railway.app"

ML_WEIGHT = 0.3
LLM_WEIGHT = 0.7


def parse_options(options):
    if isinstance(options, list):
        return 1, len(options)
    elif isinstance(options, str) and "to" in options.lower():
        parts = options.replace("to", " ").split()
        nums = [int(p) for p in parts if p.lstrip("-").isdigit()]
        if len(nums) == 2:
            return nums[0], nums[1]
    return 1, 5


def blend(ml_path, llm_path, output_path):
    with open(ml_path, "r") as f:
        ml_preds = json.load(f)
    with open(llm_path, "r") as f:
        llm_preds = json.load(f)

    # Build lookup: (person_id, question_id) -> prediction
    ml_lookup = {}
    for p in ml_preds:
        key = (p["person_id"], p["question_id"])
        ml_lookup[key] = p["predicted_answer"]

    llm_lookup = {}
    for p in llm_preds:
        key = (p["person_id"], p["question_id"])
        llm_lookup[key] = p["predicted_answer"]

    # Blend
    results = []
    for p in ml_preds:
        key = (p["person_id"], p["question_id"])
        ml_val = ml_lookup.get(key)
        llm_val = llm_lookup.get(key)
        lo, hi = parse_options(p.get("options", []))

        if ml_val is not None and llm_val is not None:
            blended = ML_WEIGHT * ml_val + LLM_WEIGHT * llm_val
        elif llm_val is not None:
            blended = llm_val
        elif ml_val is not None:
            blended = ml_val
        else:
            blended = (lo + hi) / 2

        blended = int(round(np.clip(blended, lo, hi)))

        result = {
            "person_id": p["person_id"],
            "question_id": p["question_id"],
            "predicted_answer": blended,
        }
        results.append(result)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Blended {len(results)} predictions (ML {ML_WEIGHT} + LLM {LLM_WEIGHT})")
    print(f"Saved: {output_path}")

    # Stats
    ml_only = sum(1 for p in ml_preds for key in [(p["person_id"], p["question_id"])] if key not in llm_lookup)
    llm_only = sum(1 for p in ml_preds for key in [(p["person_id"], p["question_id"])] if key not in ml_lookup)
    print(f"ML-only fallback: {ml_only}, LLM-only fallback: {llm_only}")

    return results


def submit(predictions):
    print(f"\nSubmitting {len(predictions)} predictions...")
    resp = requests.post(
        f"{API_URL}/score",
        headers={"Authorization": f"Bearer {TEAM_TOKEN}"},
        json=predictions,
    )
    if resp.status_code == 200:
        print(resp.text)
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python scripts/blend_and_submit.py <ml.json> <llm.json> <output.json> [--submit]")
        sys.exit(1)

    ml_path = sys.argv[1]
    llm_path = sys.argv[2]
    output_path = sys.argv[3]
    do_submit = "--submit" in sys.argv

    results = blend(ml_path, llm_path, output_path)

    if do_submit:
        submit(results)
    else:
        print("\nDry run. Add --submit to send to API.")
