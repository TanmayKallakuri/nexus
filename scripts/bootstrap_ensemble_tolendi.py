"""
Bootstrap ensemble training for Tolendi.

Builds a question-generalizing regression model using:
  - person-level features from Model 1
  - sentence embeddings of question text
  - question scale size

Artifacts are written to outputs/tolendi/.

Run from repo root:
  python scripts/bootstrap_ensemble_tolendi.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MASTER_TABLE_PATH = PROJECT_ROOT / "outputs" / "master_table.csv"
PERSON_FEATURE_PATH = PROJECT_ROOT / "outputs" / "person_response_profiles.csv"
UNIQUE_QUESTION_PATH = PROJECT_ROOT / "outputs" / "unique_questions.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tolendi"
QUESTION_EMBED_PATH = OUTPUT_DIR / "question_embeddings.csv"
MODEL_BUNDLE_PATH = OUTPUT_DIR / "bootstrap_ensemble_bundle.pkl"
FEATURE_COLS_PATH = OUTPUT_DIR / "feature_columns.pkl"
VALIDATION_SUMMARY_PATH = OUTPUT_DIR / "validation_summary.json"
VALIDATION_BY_BLOCK_PATH = OUTPUT_DIR / "validation_by_block.csv"
VALIDATION_BY_QUESTION_PATH = OUTPUT_DIR / "validation_by_question.csv"
HOLDOUT_PREDICTIONS_PATH = OUTPUT_DIR / "holdout_predictions.csv"
TRAINING_REPORT_PATH = OUTPUT_DIR / "training_report.txt"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
N_BOOTSTRAPS = 20
SEED = 42


def require_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for this build. Install it with "
            "`python -m pip install lightgbm`."
        ) from exc
    return lgb


def require_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for this build. Install it with "
            "`python -m pip install sentence-transformers`."
        ) from exc
    return SentenceTransformer


def clip_predictions(predictions: np.ndarray, num_options: Iterable[float]) -> np.ndarray:
    num_options_arr = np.asarray(num_options, dtype=np.float32)
    lower = np.ones_like(num_options_arr, dtype=np.float32)
    upper = np.maximum(num_options_arr, lower)
    return np.clip(predictions.astype(np.float32), lower, upper)


def load_ordinal_training_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("=" * 72)
    print("BOOTSTRAP ENSEMBLE BUILD - TOLENDI")
    print("=" * 72)

    for path in [MASTER_TABLE_PATH, PERSON_FEATURE_PATH, UNIQUE_QUESTION_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    print(f"\n[1/7] Loading source files")
    master_size_mb = os.path.getsize(MASTER_TABLE_PATH) / (1024 ** 2)
    print(f"  master_table.csv size: {master_size_mb:.1f} MB")

    master = pd.read_csv(MASTER_TABLE_PATH, low_memory=False)
    person_feat = pd.read_csv(PERSON_FEATURE_PATH, low_memory=False)
    unique_questions = pd.read_csv(UNIQUE_QUESTION_PATH, low_memory=False)

    print(f"  master rows: {master.shape[0]:,}")
    print(f"  master columns: {master.shape[1]}")
    print(f"  unique persons: {master['person_id'].nunique()}")
    print(f"  unique questions: {master['question_id'].nunique()}")
    print(f"  person feature shape: {person_feat.shape[0]} x {person_feat.shape[1]}")

    df = master[master["answer_type"] == "ordinal"].copy()
    df["answer_position"] = pd.to_numeric(df["answer_position"], errors="coerce")
    df["num_options"] = pd.to_numeric(df["num_options"], errors="coerce")
    df = df.dropna(subset=["answer_position", "num_options", "full_question"])
    df = df[df["num_options"] >= 2].copy()

    print(f"  trainable ordinal rows: {df.shape[0]:,}")
    print(f"  trainable question ids: {df['question_id'].nunique()}")

    return df, person_feat, unique_questions


def build_question_embeddings(
    train_df: pd.DataFrame,
    unique_questions: pd.DataFrame,
) -> pd.DataFrame:
    print(f"\n[2/7] Building question embeddings")

    ordinal_qids = pd.Index(sorted(train_df["question_id"].unique()))
    question_meta = unique_questions[unique_questions["question_id"].isin(ordinal_qids)].copy()
    question_meta = question_meta[
        ["question_id", "full_question", "num_options", "block_name"]
    ].drop_duplicates(subset=["question_id"])

    if question_meta["question_id"].nunique() != len(ordinal_qids):
        missing = sorted(set(ordinal_qids) - set(question_meta["question_id"]))
        missing_preview = ", ".join(missing[:10])
        raise ValueError(
            "Could not find metadata for every ordinal question. "
            f"Missing examples: {missing_preview}"
        )

    question_meta = question_meta.sort_values("question_id").reset_index(drop=True)

    SentenceTransformer = require_sentence_transformer()
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(
        question_meta["full_question"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    q_embed_df = pd.DataFrame(
        embeddings,
        columns=[f"qemb_{i}" for i in range(embeddings.shape[1])],
    )
    q_embed_df.insert(0, "question_id", question_meta["question_id"].values)
    q_embed_df["question_num_options"] = pd.to_numeric(
        question_meta["num_options"], errors="coerce"
    ).astype(np.float32)
    q_embed_df["block_name"] = question_meta["block_name"].values

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    q_embed_df.drop(columns=["block_name"]).to_csv(QUESTION_EMBED_PATH, index=False)
    print(
        f"  embedded {len(q_embed_df)} questions with dimension {embeddings.shape[1]}"
    )
    print(f"  saved embeddings: {QUESTION_EMBED_PATH}")

    return q_embed_df


def assemble_feature_frame(
    train_df: pd.DataFrame,
    person_feat: pd.DataFrame,
    q_embed_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    print(f"\n[3/7] Merging person and question features")

    model_df = train_df.merge(person_feat, on="person_id", how="left", validate="m:1")
    model_df = model_df.merge(
        q_embed_df.drop(columns=["block_name"]),
        on="question_id",
        how="left",
        validate="m:1",
    )

    person_cols = [col for col in person_feat.columns if col != "person_id"]
    question_cols = [col for col in q_embed_df.columns if col not in {"question_id", "block_name"}]
    feature_cols = person_cols + question_cols

    missing_feature_cols = [col for col in feature_cols if col not in model_df.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing expected feature columns: {missing_feature_cols[:10]}")

    print(f"  merged rows: {model_df.shape[0]:,}")
    print(f"  person features: {len(person_cols)}")
    print(f"  question features: {len(question_cols)}")
    print(f"  total features: {len(feature_cols)}")

    return model_df, person_cols, question_cols, feature_cols


def impute_features(
    model_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, SimpleImputer]:
    X = model_df[feature_cols].to_numpy(dtype=np.float32)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X).astype(np.float32)
    return X, imputer


def train_bootstrap_models(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstraps: int = N_BOOTSTRAPS,
    seed: int = SEED,
) -> Tuple[List[object], List[float]]:
    lgb = require_lightgbm()
    models: List[object] = []
    oob_maes: List[float] = []
    all_indices = np.arange(len(X))

    for i in range(n_bootstraps):
        model_seed = seed + i
        rng = np.random.default_rng(model_seed)
        boot_idx = rng.choice(all_indices, size=len(all_indices), replace=True)
        in_boot = np.zeros(len(all_indices), dtype=bool)
        in_boot[boot_idx] = True
        oob_idx = all_indices[~in_boot]

        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=model_seed,
            n_jobs=-1,
            verbose=-1,
        )

        print(f"  [{i + 1:02d}/{n_bootstraps}] training bootstrap model")
        model.fit(X[boot_idx], y[boot_idx])
        models.append(model)

        if len(oob_idx) > 0:
            oob_pred = model.predict(X[oob_idx]).astype(np.float32)
            oob_mae = float(np.mean(np.abs(oob_pred - y[oob_idx])))
            oob_maes.append(oob_mae)
            print(f"       OOB MAE: {oob_mae:.4f} on {len(oob_idx):,} rows")

    return models, oob_maes


def predict_ensemble(X: np.ndarray, models: List[object]) -> Tuple[np.ndarray, np.ndarray]:
    predictions = np.vstack([model.predict(X).astype(np.float32) for model in models])
    return predictions.mean(axis=0), predictions.std(axis=0)


def compute_question_metrics(
    scored_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, float, float]:
    question_rows: List[Dict[str, float]] = []

    for question_id, group in scored_df.groupby("question_id", sort=True):
        actual = group["answer_position"].to_numpy(dtype=np.float32)
        predicted = group["predicted"].to_numpy(dtype=np.float32)
        if len(actual) >= 3 and np.std(actual) > 0 and np.std(predicted) > 0:
            corr, _ = pearsonr(actual, predicted)
            corr_value = float(corr)
        else:
            corr_value = np.nan

        question_rows.append(
            {
                "question_id": question_id,
                "block_name": group["block_name"].iloc[0],
                "n_people": int(len(group)),
                "actual_std": float(np.std(actual, ddof=1)) if len(actual) > 1 else np.nan,
                "predicted_std": float(np.std(predicted, ddof=1)) if len(predicted) > 1 else np.nan,
                "mae": float(np.mean(np.abs(predicted - actual))),
                "accuracy": float(
                    1.0 - (
                        np.mean(np.abs(predicted - actual)) /
                        max(float(group["range"].iloc[0]), 1.0)
                    )
                ),
                "pearson_r": corr_value,
            }
        )

    question_metrics = pd.DataFrame(question_rows)
    valid_corr = question_metrics["pearson_r"].dropna()
    mean_r = float(valid_corr.mean()) if not valid_corr.empty else 0.0
    median_r = float(valid_corr.median()) if not valid_corr.empty else 0.0
    return question_metrics, mean_r, median_r


def compute_block_metrics(scored_df: pd.DataFrame) -> pd.DataFrame:
    block_rows: List[Dict[str, float]] = []

    for block_name, group in scored_df.groupby("block_name", dropna=False, sort=True):
        q_metrics, mean_r, median_r = compute_question_metrics(group)
        mad = float(group["abs_error"].mean())
        avg_range = float(group["range"].mean())
        accuracy = float(1.0 - (mad / avg_range)) if avg_range > 0 else np.nan

        block_rows.append(
            {
                "block_name": block_name,
                "n_rows": int(len(group)),
                "n_questions": int(group["question_id"].nunique()),
                "n_questions_with_corr": int(q_metrics["pearson_r"].notna().sum()),
                "accuracy": accuracy,
                "mean_pearson_r": mean_r,
                "median_pearson_r": median_r,
                "mad": mad,
                "avg_range": avg_range,
            }
        )

    return pd.DataFrame(block_rows).sort_values("block_name").reset_index(drop=True)


def validate_holdout_questions(
    model_df: pd.DataFrame,
    feature_cols: List[str],
    holdout_frac: float = 0.2,
    seed: int = SEED,
) -> Dict[str, object]:
    print(f"\n[4/7] Running held-out-question validation")

    all_qids = np.array(sorted(model_df["question_id"].unique()))
    holdout_size = max(1, int(len(all_qids) * holdout_frac))
    rng = np.random.default_rng(seed)
    holdout_qids = set(rng.choice(all_qids, size=holdout_size, replace=False).tolist())
    train_mask = ~model_df["question_id"].isin(holdout_qids)
    test_mask = model_df["question_id"].isin(holdout_qids)

    train_df = model_df.loc[train_mask].copy()
    test_df = model_df.loc[test_mask].copy()

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["answer_position"].to_numpy(dtype=np.float32)
    y_test = test_df["answer_position"].to_numpy(dtype=np.float32)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train).astype(np.float32)
    X_test = imputer.transform(X_test).astype(np.float32)

    print(f"  train questions: {train_df['question_id'].nunique()}")
    print(f"  holdout questions: {test_df['question_id'].nunique()}")
    print(f"  train rows: {len(train_df):,}")
    print(f"  holdout rows: {len(test_df):,}")

    models, oob_maes = train_bootstrap_models(X_train, y_train)
    mean_pred, std_pred = predict_ensemble(X_test, models)
    mean_pred = clip_predictions(mean_pred, test_df["num_options"].to_numpy())

    scored_df = test_df.copy()
    scored_df["predicted"] = mean_pred
    scored_df["prediction_std"] = std_pred
    scored_df["abs_error"] = np.abs(scored_df["predicted"] - y_test)
    scored_df["range"] = scored_df["num_options"] - 1

    mad = float(scored_df["abs_error"].mean())
    avg_range = float(scored_df["range"].mean())
    accuracy = float(1.0 - (mad / avg_range)) if avg_range > 0 else np.nan

    question_metrics, mean_r, median_r = compute_question_metrics(scored_df)
    block_metrics = compute_block_metrics(scored_df)

    actual_std = float(
        scored_df.groupby("question_id")["answer_position"].std().mean()
    )
    pred_std = float(
        scored_df.groupby("question_id")["predicted"].std().mean()
    )
    variance_ratio = float(pred_std / actual_std) if actual_std > 0 else np.nan

    question_metrics.to_csv(VALIDATION_BY_QUESTION_PATH, index=False)
    block_metrics.to_csv(VALIDATION_BY_BLOCK_PATH, index=False)
    scored_df.to_csv(HOLDOUT_PREDICTIONS_PATH, index=False)

    summary = {
        "holdout_fraction": holdout_frac,
        "holdout_question_count": int(len(holdout_qids)),
        "train_question_count": int(train_df["question_id"].nunique()),
        "train_row_count": int(len(train_df)),
        "holdout_row_count": int(len(test_df)),
        "accuracy": accuracy,
        "mad": mad,
        "avg_range": avg_range,
        "mean_pearson_r": mean_r,
        "median_pearson_r": median_r,
        "questions_with_positive_r": int((question_metrics["pearson_r"] > 0).sum()),
        "questions_with_corr": int(question_metrics["pearson_r"].notna().sum()),
        "variance_ratio": variance_ratio,
        "actual_std_mean": actual_std,
        "predicted_std_mean": pred_std,
        "validation_oob_mae_mean": float(np.mean(oob_maes)) if oob_maes else np.nan,
        "validation_oob_mae_std": float(np.std(oob_maes)) if oob_maes else np.nan,
    }

    print(f"  accuracy: {accuracy:.4f}")
    print(f"  mean Pearson r: {mean_r:.4f}")
    print(f"  median Pearson r: {median_r:.4f}")
    print(f"  variance ratio: {variance_ratio:.4f}")
    print(
        f"  positive-r questions: {summary['questions_with_positive_r']}/"
        f"{summary['questions_with_corr']}"
    )

    return {
        "summary": summary,
        "question_metrics": question_metrics,
        "block_metrics": block_metrics,
        "holdout_predictions": scored_df,
    }


def save_training_report(
    summary: Dict[str, object],
    block_metrics: pd.DataFrame,
    oob_maes: List[float],
) -> None:
    lines = [
        "BOOTSTRAP ENSEMBLE TRAINING REPORT",
        "=" * 40,
        "",
        f"Accuracy: {summary['accuracy']:.4f}",
        f"Mean Pearson r: {summary['mean_pearson_r']:.4f}",
        f"Median Pearson r: {summary['median_pearson_r']:.4f}",
        f"Variance ratio: {summary['variance_ratio']:.4f}",
        f"Validation OOB MAE mean: {summary['validation_oob_mae_mean']:.4f}",
        f"Full-train OOB MAE mean: {float(np.mean(oob_maes)):.4f}" if oob_maes else "Full-train OOB MAE mean: nan",
        "",
        "Per-block results:",
    ]

    for _, row in block_metrics.iterrows():
        lines.append(
            f"  {row['block_name']}: accuracy={row['accuracy']:.4f}, "
            f"mean_r={row['mean_pearson_r']:.4f}, questions={int(row['n_questions_with_corr'])}"
        )

    lines.extend(
        [
            "",
            "Red flag checks:",
            f"  Pearson r < 0.1: {summary['mean_pearson_r'] < 0.1}",
            f"  Variance ratio < 0.5: {summary['variance_ratio'] < 0.5}",
            f"  Accuracy < 0.67: {summary['accuracy'] < 0.67}",
        ]
    )

    TRAINING_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def train_full_ensemble(
    model_df: pd.DataFrame,
    feature_cols: List[str],
    person_cols: List[str],
    question_cols: List[str],
) -> Tuple[List[object], SimpleImputer, List[float]]:
    print(f"\n[5/7] Training full-data bootstrap ensemble")
    X, imputer = impute_features(model_df, feature_cols)
    y = model_df["answer_position"].to_numpy(dtype=np.float32)

    print(f"  feature matrix shape: {X.shape}")
    print(f"  target shape: {y.shape}")

    models, oob_maes = train_bootstrap_models(X, y)
    print(f"  mean OOB MAE: {np.mean(oob_maes):.4f}" if oob_maes else "  mean OOB MAE: nan")
    print(f"  bootstrap models saved later: {len(models)}")

    bundle = {
        "models": models,
        "imputer": imputer,
        "feature_cols": feature_cols,
        "person_feature_cols": person_cols,
        "question_feature_cols": question_cols,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "n_bootstraps": N_BOOTSTRAPS,
        "seed": SEED,
    }

    with MODEL_BUNDLE_PATH.open("wb") as f:
        pickle.dump(bundle, f)
    with FEATURE_COLS_PATH.open("wb") as f:
        pickle.dump(feature_cols, f)

    print(f"  saved model bundle: {MODEL_BUNDLE_PATH}")
    print(f"  saved feature columns: {FEATURE_COLS_PATH}")

    return models, imputer, oob_maes


def save_validation_summary(summary: Dict[str, object]) -> None:
    VALIDATION_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"  saved validation summary: {VALIDATION_SUMMARY_PATH}")


def main() -> int:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        train_df, person_feat, unique_questions = load_ordinal_training_data()
        q_embed_df = build_question_embeddings(train_df, unique_questions)
        model_df, person_cols, question_cols, feature_cols = assemble_feature_frame(
            train_df, person_feat, q_embed_df
        )

        validation = validate_holdout_questions(model_df, feature_cols)
        save_validation_summary(validation["summary"])

        _, _, full_oob_maes = train_full_ensemble(
            model_df, feature_cols, person_cols, question_cols
        )

        print(f"\n[6/7] Writing report")
        save_training_report(
            validation["summary"],
            validation["block_metrics"],
            full_oob_maes,
        )

        print(f"\n[7/7] Build complete")
        print(f"  outputs directory: {OUTPUT_DIR}")
        print(f"  key metric - mean Pearson r: {validation['summary']['mean_pearson_r']:.4f}")
        print(f"  accuracy: {validation['summary']['accuracy']:.4f}")
        print(f"  variance ratio: {validation['summary']['variance_ratio']:.4f}")
        return 0

    except Exception as exc:
        print(f"\nBUILD FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
