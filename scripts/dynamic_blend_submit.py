"""
Dynamic blend: ML + LLM with similarity-based and confidence-based weighting.

Adapts per-question:
  - High similarity to training data -> more ML weight
  - Low similarity -> more LLM weight
  - High LLM confidence -> more LLM weight
  - Low LLM confidence -> more ML weight

Usage:
  python scripts/dynamic_blend_submit.py <ml.json> <llm.json> <output.json> [--submit]
"""

import json
import sys
import pickle
import numpy as np
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TEAM_TOKEN = "house_tokens_3f8n"
API_URL = "https://blackboxhackathon-production.up.railway.app"

# Base weights (before adjustment)
BASE_ML = 0.3
BASE_LLM = 0.7

# Weight bounds
MIN_ML = 0.05
MAX_ML = 0.50
MIN_LLM = 0.50
MAX_LLM = 0.95


def load_knn_similarity(questions):
    """Compute max cosine similarity to nearest known question for each test question."""
    from sentence_transformers import SentenceTransformer

    # Load KNN embeddings
    embeddings_384 = np.load(OUTPUT_DIR / "knn_question_embeddings_384.npy")
    norms = np.linalg.norm(embeddings_384, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_normed = embeddings_384 / norms

    # Load embedder
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Get unique question texts
    unique_texts = {}
    for q in questions:
        context = q.get("context") or ""
        qtext = q.get("question_text", "")
        full = f"{context} {qtext}".strip()
        qid = q["question_id"]
        if qid not in unique_texts:
            unique_texts[qid] = full

    # Embed test questions
    qids = list(unique_texts.keys())
    texts = [unique_texts[qid] for qid in qids]
    test_embeddings = embedder.encode(texts, batch_size=32)
    test_norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    test_norms[test_norms == 0] = 1
    test_normed = test_embeddings / test_norms

    # Compute max similarity to any known question
    sims = {}
    for i, qid in enumerate(qids):
        cos_sims = embeddings_normed @ test_normed[i]
        sims[qid] = float(np.max(cos_sims))

    return sims


def parse_options(options):
    if isinstance(options, list):
        return 1, len(options)
    elif isinstance(options, str) and "to" in options.lower():
        parts = options.replace("to", " ").split()
        nums = [int(p) for p in parts if p.lstrip("-").isdigit()]
        if len(nums) == 2:
            return nums[0], nums[1]
    return 1, 5


def dynamic_blend(ml_path, llm_path, output_path, do_submit=False):
    with open(ml_path) as f:
        ml_preds = json.load(f)
    with open(llm_path) as f:
        llm_preds = json.load(f)

    ml_lookup = {(p["person_id"], p["question_id"]): p["predicted_answer"] for p in ml_preds}
    llm_lookup = {}
    llm_conf_lookup = {}
    for p in llm_preds:
        key = (p["person_id"], p["question_id"])
        llm_lookup[key] = p["predicted_answer"]
        llm_conf_lookup[key] = p.get("llm_confidence", 0.5)

    # Compute similarities
    print("Computing question similarities...")
    sims = load_knn_similarity(ml_preds)

    # Blend
    results = []
    weight_stats = {"ml_weights": [], "llm_weights": [], "sims": [], "confs": []}

    for p in ml_preds:
        key = (p["person_id"], p["question_id"])
        qid = p["question_id"]
        ml_val = ml_lookup.get(key)
        llm_val = llm_lookup.get(key)
        llm_conf = llm_conf_lookup.get(key, 0.5)
        sim = sims.get(qid, 0.3)
        lo, hi = parse_options(p.get("options", []))

        # Dynamic weight calculation
        ml_w = BASE_ML
        llm_w = BASE_LLM

        # Similarity adjustment: high sim -> more ML, low sim -> more LLM
        if sim > 0.5:
            ml_w += 0.10  # trust ML more
        elif sim < 0.2:
            ml_w -= 0.10  # trust ML less

        # Confidence adjustment: high LLM confidence -> more LLM
        if llm_conf > 0.8:
            llm_w += 0.05
        elif llm_conf < 0.3:
            llm_w -= 0.10

        # Normalize
        total = ml_w + llm_w
        ml_w = ml_w / total
        llm_w = llm_w / total

        # Clamp
        ml_w = max(MIN_ML, min(MAX_ML, ml_w))
        llm_w = 1.0 - ml_w

        # Blend
        if ml_val is not None and llm_val is not None:
            blended = ml_w * ml_val + llm_w * llm_val
        elif llm_val is not None:
            blended = llm_val
        elif ml_val is not None:
            blended = ml_val
        else:
            blended = (lo + hi) / 2

        blended = int(round(np.clip(blended, lo, hi)))

        results.append({
            "person_id": p["person_id"],
            "question_id": qid,
            "predicted_answer": blended,
        })

        weight_stats["ml_weights"].append(ml_w)
        weight_stats["llm_weights"].append(llm_w)
        weight_stats["sims"].append(sim)
        weight_stats["confs"].append(llm_conf)

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Blended {len(results)} predictions (dynamic weights)")
    print(f"  ML weight:  mean={np.mean(weight_stats['ml_weights']):.3f} "
          f"range=[{np.min(weight_stats['ml_weights']):.3f}, {np.max(weight_stats['ml_weights']):.3f}]")
    print(f"  LLM weight: mean={np.mean(weight_stats['llm_weights']):.3f}")
    print(f"  Similarity: mean={np.mean(weight_stats['sims']):.3f}")
    print(f"  LLM confidence: mean={np.mean(weight_stats['confs']):.3f}")
    print(f"  Saved: {output_path}")

    if do_submit:
        print(f"\nSubmitting {len(results)} predictions...")
        resp = requests.post(
            f"{API_URL}/score",
            headers={"Authorization": f"Bearer {TEAM_TOKEN}"},
            json=results,
        )
        print(resp.text)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python scripts/dynamic_blend_submit.py <ml.json> <llm.json> <output.json> [--submit]")
        sys.exit(1)

    ml_path = sys.argv[1]
    llm_path = sys.argv[2]
    output_path = sys.argv[3]
    do_submit = "--submit" in sys.argv

    dynamic_blend(ml_path, llm_path, output_path, do_submit)
