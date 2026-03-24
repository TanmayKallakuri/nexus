"""
Final submission prediction script.

Usage:
  python scripts/final_predict.py <test_input.json> <output.json>

The script blends:
  1. KNN over known question embeddings
  2. Tolendi V4.1 exported LightGBM text models
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TOLENDI_DIR = OUTPUTS_DIR / "tolendi"
MODEL_DIR = TOLENDI_DIR / "v4_1_models"
META_PATH = MODEL_DIR / "meta.json"

KNN_K = 7
KNN_WEIGHT = 0.6
ENSEMBLE_WEIGHT = 0.4

ANALYTICAL_CATEGORY_MAP = {
    "verbal": {
        "QID63", "QID64", "QID65", "QID66", "QID67",
        "QID68", "QID69", "QID70", "QID71", "QID72",
        "QID74", "QID75", "QID76", "QID77", "QID78",
        "QID79", "QID80", "QID81", "QID82", "QID83",
    },
    "finlit": {"QID36", "QID37", "QID38", "QID39", "QID40", "QID41", "QID42"},
    "logical": {
        "QID217", "QID218", "QID219", "QID220",
        "QID273", "QID274", "QID275", "QID276",
        "QID277", "QID278", "QID279",
    },
    "games": {
        "QID117", "QID118", "QID119", "QID120", "QID121",
        "QID122", "QID224", "QID225", "QID226", "QID227",
        "QID228", "QID229", "QID230", "QID231",
    },
    "framing": {"QID149", "QID150", "QID151", "QID152"},
    "selfconcept": {"QID268", "QID269", "QID270", "QID271", "QID272"},
    "demographics": {
        "QID11", "QID12", "QID13", "QID14", "QID15", "QID16", "QID17",
        "QID18", "QID19", "QID20", "QID21", "QID22", "QID23", "QID24",
    },
}


def parse_options(options):
    if isinstance(options, list):
        return {
            "type": "list",
            "n_options": max(1, len(options)),
            "lo": 1.0,
            "hi": float(max(1, len(options))),
        }

    if isinstance(options, str):
        text = options.strip()
        if "to" in text.lower():
            parts = text.replace("to", " ").split()
            nums = [float(p) for p in parts if p.replace(".", "", 1).replace("-", "", 1).isdigit()]
            if len(nums) == 2:
                lo, hi = nums
                return {
                    "type": "continuous",
                    "n_options": int(round(hi - lo + 1)),
                    "lo": float(lo),
                    "hi": float(hi),
                }
        return {"type": "continuous", "n_options": 101, "lo": 0.0, "hi": 100.0}

    return {"type": "unknown", "n_options": 5, "lo": 1.0, "hi": 5.0}


def build_question_text(question: Dict[str, object]) -> str:
    context = question.get("context")
    qtext = str(question.get("question_text", "")).strip()
    if context is not None and str(context).strip() and str(context).strip().lower() != "null":
        return f"{str(context).strip()}\n\n{qtext}"
    return qtext


def get_analytical_category(parent_qid: object) -> str:
    parent_qid = str(parent_qid)
    for label, members in ANALYTICAL_CATEGORY_MAP.items():
        if parent_qid in members:
            return label
    return "other"


def infer_question_type(question: Dict[str, object], options_info: Dict[str, float]) -> str:
    if options_info["type"] in {"list", "continuous"}:
        return "MC"
    if question.get("context"):
        return "MC"
    return "TE"


def load_embedder():
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    except TypeError:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return SentenceTransformer("all-MiniLM-L6-v2")


def load_knn_assets():
    print("[1/6] Loading KNN artifacts...")

    with (OUTPUTS_DIR / "knn_response_lookup.pkl").open("rb") as f:
        knn_data = pickle.load(f)

    response_pivot = knn_data["response_pivot"]
    person_ids = set(knn_data["person_ids"])

    embeddings_384 = np.load(OUTPUTS_DIR / "knn_question_embeddings_384.npy")
    norms = np.linalg.norm(embeddings_384, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_normed = embeddings_384 / norms

    with (OUTPUTS_DIR / "knn_qid_to_idx.pkl").open("rb") as f:
        idx_data = pickle.load(f)

    ordinal_qids = idx_data["ordinal_qids"]
    print(f"  KNN loaded: {len(person_ids)} persons, {len(ordinal_qids)} known questions")

    return {
        "response_pivot": response_pivot,
        "known_person_ids": person_ids,
        "embeddings_normed": embeddings_normed,
        "ordinal_qids": ordinal_qids,
    }


def load_question_metadata() -> Dict[str, Dict[str, object]]:
    master = pd.read_csv(OUTPUTS_DIR / "master_table.csv")
    construct_mapping = pd.read_csv(OUTPUTS_DIR / "construct_mapping.csv")

    question_meta = (
        master[
            ["question_id", "parent_question_id", "block_name", "question_type", "num_options"]
        ]
        .drop_duplicates("question_id")
        .merge(construct_mapping[["question_id", "construct_id"]], on="question_id", how="left")
    )
    question_meta["construct_id"] = question_meta["construct_id"].fillna(question_meta["parent_question_id"])
    return question_meta.set_index("question_id").to_dict(orient="index")


def build_construct_lookup(question_meta: Dict[str, Dict[str, object]]):
    construct_mapping = pd.read_csv(OUTPUTS_DIR / "construct_mapping.csv")
    construct_scores = pd.read_csv(OUTPUTS_DIR / "person_construct_scores.csv")

    construct_counts = construct_mapping["construct_id"].value_counts()
    multi_item_constructs = set(construct_counts[construct_counts > 1].index.tolist())

    rows: List[Tuple[str, float]] = []
    for _, row in construct_mapping.iterrows():
        qmeta = question_meta.get(str(row["question_id"]))
        if not qmeta:
            continue
        rows.append((str(row["construct_id"]), float(qmeta.get("num_options", np.nan))))

    construct_num_options = (
        pd.DataFrame(rows, columns=["construct_id", "num_options"])
        .dropna()
        .groupby("construct_id")["num_options"]
        .median()
    )

    construct_scores["construct_num_options"] = construct_scores["construct_id"].map(construct_num_options)
    denom = (construct_scores["construct_num_options"] - 1.0).clip(lower=1.0)
    has_scale = construct_scores["construct_num_options"].fillna(0).gt(1)

    construct_scores["construct_mean_norm"] = np.where(
        has_scale,
        (construct_scores["mean_answer"] - 1.0) / denom,
        np.nan,
    )
    construct_scores["construct_std_norm"] = np.where(
        has_scale,
        construct_scores["std_answer"] / denom,
        np.nan,
    )

    score_lookup = {
        (str(row["person_id"]), str(row["construct_id"])): {
            "n_answers": float(row["n_answers"]),
            "percentile": float(row["percentile"]) if pd.notna(row["percentile"]) else np.nan,
            "construct_mean_norm": float(row["construct_mean_norm"])
            if pd.notna(row["construct_mean_norm"])
            else np.nan,
            "construct_std_norm": float(row["construct_std_norm"])
            if pd.notna(row["construct_std_norm"])
            else np.nan,
        }
        for _, row in construct_scores.iterrows()
    }

    return score_lookup, multi_item_constructs


def load_ensemble_assets(question_meta: Dict[str, Dict[str, object]]):
    print("[2/6] Loading V4.1 text-model ensemble...")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    person_feat = pd.read_csv(OUTPUTS_DIR / meta["person_feature_csv"])
    person_emb = pd.read_csv(TOLENDI_DIR / meta["person_embedding_csv"])
    person_table = person_feat.merge(person_emb, on="person_id", how="left").set_index("person_id")

    routes = {}
    for route_name, info in meta["routes"].items():
        boosters = [lgb.Booster(model_file=str(MODEL_DIR / fname)) for fname in info["model_files"]]
        imputer_statistics = np.load(MODEL_DIR / info["imputer_statistics_file"]).astype(np.float32)
        routes[route_name] = {
            "route_type": info["route_type"],
            "feature_cols": info["feature_cols"],
            "models": boosters,
            "imputer_statistics": np.where(np.isnan(imputer_statistics), 0.0, imputer_statistics),
        }

    pca_components = np.load(MODEL_DIR / "pca_components.npy").astype(np.float32)
    pca_mean = np.load(MODEL_DIR / "pca_mean.npy").astype(np.float32)
    construct_lookup, multi_item_constructs = build_construct_lookup(question_meta)

    print(f"  Ensemble loaded: {len(routes)} routes, {len(person_table)} persons")

    return {
        "meta": meta,
        "person_table": person_table,
        "routes": routes,
        "pca_components": pca_components,
        "pca_mean": pca_mean,
        "construct_lookup": construct_lookup,
        "multi_item_constructs": multi_item_constructs,
        "stretch_factor": float(meta.get("calibration_params", {}).get("stretch_factor", 1.0)),
    }


def project_question_embedding(embedding: np.ndarray, pca_mean: np.ndarray, pca_components: np.ndarray):
    centered = embedding.astype(np.float32) - pca_mean
    return centered @ pca_components.T


def predict_knn(
    person_id: str,
    question_embedding_normed: np.ndarray,
    options_info: Dict[str, float],
    knn_assets: Dict[str, object],
):
    if person_id not in knn_assets["known_person_ids"]:
        return None, 0.0, None

    sims = knn_assets["embeddings_normed"] @ question_embedding_normed
    k = min(KNN_K, len(sims))
    top_k_idx = np.argsort(sims)[-k:][::-1]
    top_k_sims = np.maximum(sims[top_k_idx], 0.0)
    top_k_qids = [knn_assets["ordinal_qids"][i] for i in top_k_idx]
    max_sim = float(top_k_sims[0]) if len(top_k_sims) else 0.0

    if float(top_k_sims.sum()) < 1e-10:
        top_k_weights = np.ones(k, dtype=np.float32) / max(k, 1)
    else:
        top_k_weights = top_k_sims / top_k_sims.sum()

    neighbor_answers = []
    neighbor_weights = []
    response_pivot = knn_assets["response_pivot"]
    for idx, qid in enumerate(top_k_qids):
        if qid in response_pivot.columns:
            val = response_pivot.loc[person_id, qid]
            if not np.isnan(val):
                neighbor_answers.append(float(val))
                neighbor_weights.append(float(top_k_weights[idx]))

    if not neighbor_answers:
        return None, max_sim, top_k_qids[0] if top_k_qids else None

    neighbor_answers = np.asarray(neighbor_answers, dtype=np.float32)
    neighbor_weights = np.asarray(neighbor_weights, dtype=np.float32)
    neighbor_weights = neighbor_weights / neighbor_weights.sum()

    pred_norm = float(np.average(neighbor_answers, weights=neighbor_weights))
    pred_raw = options_info["lo"] + pred_norm * (options_info["hi"] - options_info["lo"])
    pred_raw = float(np.clip(pred_raw, options_info["lo"], options_info["hi"]))
    return pred_raw, max_sim, top_k_qids[0] if top_k_qids else None


def infer_question_metadata(
    question: Dict[str, object],
    options_info: Dict[str, float],
    nearest_qid: str | None,
    question_meta: Dict[str, Dict[str, object]],
):
    qid = str(question.get("question_id", ""))
    exact = question_meta.get(qid, {})
    nearest = question_meta.get(str(nearest_qid), {}) if nearest_qid else {}

    parent_question_id = exact.get("parent_question_id") or nearest.get("parent_question_id") or qid or str(nearest_qid)
    block_name = exact.get("block_name") or nearest.get("block_name") or "Cognitive tests"
    question_type = exact.get("question_type") or nearest.get("question_type") or infer_question_type(question, options_info)
    construct_id = exact.get("construct_id") or nearest.get("construct_id") or parent_question_id

    return {
        "question_id": qid,
        "parent_question_id": str(parent_question_id),
        "block_name": str(block_name),
        "question_type": str(question_type),
        "construct_id": str(construct_id),
        "category": get_analytical_category(parent_question_id),
    }


def build_question_feature_values(
    pca_vector: np.ndarray,
    options_info: Dict[str, float],
    inferred_meta: Dict[str, str],
    meta: Dict[str, object],
):
    feature_values = {col: 0.0 for col in meta["question_feature_cols"]}
    for i in range(min(len(pca_vector), 50)):
        feature_values[f"qemb_pca_{i}"] = float(pca_vector[i])

    feature_values["question_num_options"] = float(options_info["n_options"])
    block_col = f"block_{inferred_meta['block_name']}"
    if block_col in feature_values:
        feature_values[block_col] = 1.0

    qtype = inferred_meta["question_type"]
    if qtype.startswith("MC"):
        qtype = "MC"
    qtype_col = f"qtype_{qtype}"
    if qtype_col in feature_values:
        feature_values[qtype_col] = 1.0

    category_cols = meta.get("analytical_category_cols", [])
    category_values = {col: 0.0 for col in category_cols}
    category_col = f"cat_{inferred_meta['category']}"
    if category_col in category_values:
        category_values[category_col] = 1.0
    elif "cat_other" in category_values:
        category_values["cat_other"] = 1.0

    return feature_values, category_values


def get_construct_feature_values(
    person_id: str,
    inferred_meta: Dict[str, str],
    construct_lookup: Dict[Tuple[str, str], Dict[str, float]],
    multi_item_constructs: set[str],
):
    construct_id = inferred_meta["construct_id"]
    values = {
        "person_construct_mean_loo": np.nan,
        "person_construct_std_loo": np.nan,
        "person_construct_percentile_loo": np.nan,
        "construct_answers_available": 0.0,
        "has_multi_item_construct": float(construct_id in multi_item_constructs),
    }

    score = construct_lookup.get((person_id, construct_id))
    if not score:
        return values

    values["construct_answers_available"] = 1.0 if score["n_answers"] > 0 else 0.0
    if construct_id in multi_item_constructs and score["n_answers"] > 0:
        values["person_construct_mean_loo"] = score["construct_mean_norm"]
        values["person_construct_std_loo"] = score["construct_std_norm"]
        values["person_construct_percentile_loo"] = score["percentile"]

    return values


def choose_route(inferred_meta: Dict[str, str], options_info: Dict[str, float]) -> str:
    if inferred_meta["block_name"] == "Personality":
        return "personality"
    if int(round(options_info["n_options"])) == 2:
        return "non_personality_binary"
    return "non_personality_multi"


def build_route_matrix(
    person_id: str,
    route_name: str,
    question_values: Dict[str, float],
    category_values: Dict[str, float],
    construct_values: Dict[str, float],
    ensemble_assets: Dict[str, object],
):
    person_table = ensemble_assets["person_table"]
    if person_id not in person_table.index:
        return None

    person_row = person_table.loc[person_id]
    route = ensemble_assets["routes"][route_name]
    feature_values = {}

    for col in ensemble_assets["meta"]["person_feature_cols"]:
        feature_values[col] = float(person_row[col]) if pd.notna(person_row[col]) else np.nan
    for col in ensemble_assets["meta"]["person_embedding_cols"]:
        feature_values[col] = float(person_row[col]) if pd.notna(person_row[col]) else np.nan

    feature_values.update(question_values)
    feature_values.update(construct_values)
    if route_name != "personality":
        feature_values.update(category_values)

    X = np.array([feature_values.get(col, np.nan) for col in route["feature_cols"]], dtype=np.float32)
    mask = np.isnan(X)
    if mask.any():
        X[mask] = route["imputer_statistics"][mask]
    X = np.nan_to_num(X, nan=0.0)
    return X.reshape(1, -1)


def predict_ensemble(
    person_id: str,
    route_name: str,
    X: np.ndarray | None,
    options_info: Dict[str, float],
    ensemble_assets: Dict[str, object],
):
    if X is None:
        return None

    route = ensemble_assets["routes"][route_name]
    preds = [float(model.predict(X)[0]) for model in route["models"]]
    pred_norm = float(np.mean(preds))
    pred_norm = float(np.clip(pred_norm, 0.0, 1.0))

    pred_raw = options_info["lo"] + pred_norm * (options_info["hi"] - options_info["lo"])
    pred_raw = float(np.clip(pred_raw, options_info["lo"], options_info["hi"]))
    return pred_raw


def resolve_paths():
    if len(sys.argv) >= 2:
        test_path = Path(sys.argv[1])
    else:
        primary = PROJECT_ROOT / "sample_test_questions.json"
        fallback = OUTPUTS_DIR / "sample_predictions.json"
        test_path = primary if primary.exists() else fallback

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = OUTPUTS_DIR / "predictions.json"

    return test_path, output_path


def main():
    test_path, output_path = resolve_paths()

    embedder = load_embedder()
    knn_assets = load_knn_assets()
    question_meta = load_question_metadata()
    ensemble_assets = load_ensemble_assets(question_meta)

    print("[3/6] Loading test questions...")
    with test_path.open("r", encoding="utf-8") as f:
        test_questions = json.load(f)

    print(f"  Loaded {len(test_questions)} questions from {test_path}")
    type_counts = {}
    for question in test_questions:
        options_info = parse_options(question.get("options", []))
        type_counts[options_info["type"]] = type_counts.get(options_info["type"], 0) + 1
    print(f"  Question types: {type_counts}")

    print("[4/6] Embedding unique question texts...")
    unique_texts = list({build_question_text(question) for question in test_questions})
    raw_embeddings = embedder.encode(unique_texts, show_progress_bar=True, batch_size=32)
    raw_embeddings = np.asarray(raw_embeddings, dtype=np.float32)
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_normed = raw_embeddings / norms

    text_features = {}
    for text, raw_embedding, normed_embedding in zip(unique_texts, raw_embeddings, embeddings_normed):
        pca_vector = project_question_embedding(
            raw_embedding,
            ensemble_assets["pca_mean"],
            ensemble_assets["pca_components"],
        )
        text_features[text] = {
            "raw_embedding": raw_embedding,
            "normed_embedding": normed_embedding,
            "pca_vector": pca_vector,
        }

    print("[5/6] Running predictions...")
    results = []
    knn_used = 0
    ensemble_used = 0
    fallback_used = 0

    for idx, question in enumerate(test_questions, start=1):
        person_id = str(question["person_id"])
        options_info = parse_options(question.get("options", []))
        question_text = build_question_text(question)
        question_cache = text_features[question_text]

        knn_pred, _, nearest_qid = predict_knn(
            person_id,
            question_cache["normed_embedding"],
            options_info,
            knn_assets,
        )

        inferred_meta = infer_question_metadata(question, options_info, nearest_qid, question_meta)
        question_values, category_values = build_question_feature_values(
            question_cache["pca_vector"],
            options_info,
            inferred_meta,
            ensemble_assets["meta"],
        )
        construct_values = get_construct_feature_values(
            person_id,
            inferred_meta,
            ensemble_assets["construct_lookup"],
            ensemble_assets["multi_item_constructs"],
        )
        route_name = choose_route(inferred_meta, options_info)
        X = build_route_matrix(
            person_id,
            route_name,
            question_values,
            category_values,
            construct_values,
            ensemble_assets,
        )
        ensemble_pred = predict_ensemble(person_id, route_name, X, options_info, ensemble_assets)

        if knn_pred is not None and ensemble_pred is not None:
            final_pred = (KNN_WEIGHT * knn_pred) + (ENSEMBLE_WEIGHT * ensemble_pred)
            knn_used += 1
            ensemble_used += 1
        elif knn_pred is not None:
            final_pred = knn_pred
            knn_used += 1
        elif ensemble_pred is not None:
            final_pred = ensemble_pred
            ensemble_used += 1
        else:
            final_pred = (options_info["lo"] + options_info["hi"]) / 2.0
            fallback_used += 1

        final_pred = float(np.clip(final_pred, options_info["lo"], options_info["hi"]))
        if options_info["type"] in {"list", "continuous", "unknown"}:
            final_answer = int(round(final_pred))
        else:
            final_answer = final_pred

        result = dict(question)
        result["predicted_answer"] = final_answer
        results.append(result)

        if idx % 100 == 0 or idx == len(test_questions):
            print(f"  [{idx}/{len(test_questions)}] predicted")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[6/6] Saving predictions...")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: {output_path}")
    print(f"  KNN available: {knn_used}/{len(test_questions)}")
    print(f"  Ensemble available: {ensemble_used}/{len(test_questions)}")
    print(f"  Midpoint fallback: {fallback_used}/{len(test_questions)}")
    print("  Sample predictions:")
    for row in results[:5]:
        print(f"    {row['person_id']} | {row['question_id']} | pred={row['predicted_answer']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
