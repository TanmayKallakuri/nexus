"""
Re-export Tolendi V4.1 models in a version-safe format.

Outputs:
  outputs/tolendi/v4_1_models/
    - *.txt LightGBM boosters
    - meta.json
    - pca_components.npy
    - pca_mean.npy
    - imputer statistics / indicator arrays

Run from repo root:
  python scripts/reexport_tolendi_v4_1.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOLENDI_DIR = PROJECT_ROOT / "outputs" / "tolendi"
MODEL_DIR = TOLENDI_DIR / "v4_1_models"

BUNDLE_PATH = TOLENDI_DIR / "bootstrap_ensemble_v4_1.pkl"
PCA_PATH = TOLENDI_DIR / "pca_model_v4_1.pkl"
QUESTION_FEATURES_PATH = TOLENDI_DIR / "question_embeddings_v4_1.csv"
PERSON_FEATURES_PATH = PROJECT_ROOT / "outputs" / "person_response_profiles_repaired.csv"
PERSON_EMBED_PATH = TOLENDI_DIR / "person_embeddings_v4_1_k10.csv"
CALIBRATION_PATH = TOLENDI_DIR / "calibration_params_v4.json"
META_PATH = MODEL_DIR / "meta.json"


def route_feature_cols(bundle: Dict[str, object], route_name: str) -> List[str]:
    shared = list(bundle["feature_cols"])
    analytical = list(bundle.get("analytical_category_cols", []))
    if route_name == "personality":
        return shared
    return shared + analytical


def save_route_imputer(route_name: str, route_data: Dict[str, object]) -> Dict[str, object]:
    imputer = route_data["imputer"]
    stats_path = MODEL_DIR / f"{route_name}_imputer_statistics.npy"
    missing_path = MODEL_DIR / f"{route_name}_imputer_missing_mask.npy"

    np.save(stats_path, imputer.statistics_)
    missing_mask = getattr(imputer, "_fit_impute_mask", None)
    if missing_mask is None:
        missing_mask = np.zeros_like(imputer.statistics_, dtype=bool)
    np.save(missing_path, np.asarray(missing_mask, dtype=bool))

    return {
        "imputer_statistics_file": stats_path.name,
        "imputer_missing_mask_file": missing_path.name,
    }


def main() -> int:
    try:
        if not BUNDLE_PATH.exists():
            raise FileNotFoundError(f"Missing bundle: {BUNDLE_PATH}")
        if not PCA_PATH.exists():
            raise FileNotFoundError(f"Missing PCA model: {PCA_PATH}")

        print("=" * 72)
        print("RE-EXPORT TOLENDI V4.1 MODELS")
        print("=" * 72)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        print("\n[1/5] Loading bundle")
        with BUNDLE_PATH.open("rb") as f:
            bundle = pickle.load(f)
        print(f"  routes: {list(bundle['routes'].keys())}")

        print("\n[2/5] Saving LightGBM boosters")
        route_meta: Dict[str, object] = {}
        total_models = 0
        for route_name, route_data in bundle["routes"].items():
            models = route_data.get("models", [])
            feature_cols = route_feature_cols(bundle, route_name)
            saved_models: List[str] = []
            for i, model in enumerate(models):
                model_path = MODEL_DIR / f"{route_name}_model_{i}.txt"
                booster = getattr(model, "booster_", None)
                if booster is None:
                    raise TypeError(f"Route {route_name} model {i} has no booster_.")
                booster.save_model(str(model_path))
                _ = lgb.Booster(model_file=str(model_path))
                saved_models.append(model_path.name)
                total_models += 1

            imputer_meta = save_route_imputer(route_name, route_data)
            route_meta[route_name] = {
                "route_type": route_data.get("route_type"),
                "feature_cols": feature_cols,
                "model_files": saved_models,
                "model_count": len(saved_models),
                "n_bootstraps": int(route_data.get("n_bootstraps", len(saved_models))),
                "n_rows": int(route_data.get("n_rows", 0)),
                "oob_mean": float(route_data.get("oob_mean", np.nan)),
                "oob_std": float(route_data.get("oob_std", np.nan)),
                **imputer_meta,
            }
            print(f"  {route_name}: saved {len(saved_models)} models")

        print(f"  total models saved: {total_models}")

        print("\n[3/5] Saving PCA arrays")
        with PCA_PATH.open("rb") as f:
            pca = pickle.load(f)
        np.save(MODEL_DIR / "pca_components.npy", pca.components_)
        np.save(MODEL_DIR / "pca_mean.npy", pca.mean_)
        print(f"  PCA components: {pca.components_.shape}")

        print("\n[4/5] Building metadata")
        question_features = pd.read_csv(QUESTION_FEATURES_PATH, nrows=1)
        person_features = pd.read_csv(PERSON_FEATURES_PATH, nrows=1)
        person_embeddings = pd.read_csv(PERSON_EMBED_PATH, nrows=1)
        calibration = json.loads(CALIBRATION_PATH.read_text()) if CALIBRATION_PATH.exists() else {}

        meta = {
            "bundle_source": BUNDLE_PATH.name,
            "question_pca_source": PCA_PATH.name,
            "question_feature_table_csv": QUESTION_FEATURES_PATH.name,
            "person_feature_csv": PERSON_FEATURES_PATH.name,
            "person_embedding_csv": PERSON_EMBED_PATH.name,
            "person_feature_cols": [c for c in person_features.columns if c != "person_id"],
            "person_embedding_cols": [c for c in person_embeddings.columns if c != "person_id"],
            "question_feature_cols": [c for c in question_features.columns if c != "question_id"],
            "construct_feature_cols": bundle.get("construct_feature_cols", []),
            "analytical_category_cols": bundle.get("analytical_category_cols", []),
            "embedding_model_name": bundle.get("embedding_model_name"),
            "person_embedding_k": int(bundle.get("person_embedding_k", 0)),
            "person_embedding_info": bundle.get("person_embedding_info", {}),
            "question_pca_components": int(bundle.get("pca_components", 0)),
            "calibration_params": calibration or bundle.get("calibration_params", {}),
            "routes": route_meta,
        }
        META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  saved metadata: {META_PATH}")

        print("\n[5/5] Verifying exports")
        model_files = sorted(p.name for p in MODEL_DIR.glob("*.txt"))
        for name in model_files[:3]:
            booster = lgb.Booster(model_file=str(MODEL_DIR / name))
            print(f"  {name}: loaded, trees={booster.num_trees()}")
        print(f"  verified {len(model_files)} text models total")

        print("\nExport complete")
        print(f"  output dir: {MODEL_DIR}")
        return 0

    except Exception as exc:
        print(f"\nRE-EXPORT FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
