"""
Leakage-aware Tolendi V3 bootstrap trainer.

V3 extends V2 by adding fold-specific person embeddings derived from the
ordinal response matrix. It sweeps a few embedding sizes, selects the
best validation result, and then trains a final routed ensemble.

Run from repo root:
  python scripts/bootstrap_ensemble_tolendi_v3.py
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from build_person_profiles import (
    build_cognitive_features,
    build_coverage_features,
    build_demographic_features,
    build_economic_features,
    build_personality_construct_features,
    build_response_style_features,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MASTER_TABLE_PATH = PROJECT_ROOT / "outputs" / "master_table.csv"
PERSON_PROFILE_REFERENCE_PATH = PROJECT_ROOT / "outputs" / "person_response_profiles.csv"
UNIQUE_QUESTION_PATH = PROJECT_ROOT / "outputs" / "unique_questions.csv"
CONSTRUCT_MAPPING_PATH = PROJECT_ROOT / "outputs" / "construct_mapping.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tolendi"
QUESTION_EMBED_V3_PATH = OUTPUT_DIR / "question_embeddings_v3.csv"
QUESTION_PCA_MODEL_V3_PATH = OUTPUT_DIR / "pca_model_v3.pkl"
MODEL_BUNDLE_V3_PATH = OUTPUT_DIR / "bootstrap_ensemble_v3.pkl"
VALIDATION_SUMMARY_V3_PATH = OUTPUT_DIR / "validation_summary_v3.json"
VALIDATION_BY_BLOCK_V3_PATH = OUTPUT_DIR / "validation_by_block_v3.csv"
VALIDATION_BY_QUESTION_V3_PATH = OUTPUT_DIR / "validation_by_question_v3.csv"
HOLDOUT_PREDICTIONS_V3_PATH = OUTPUT_DIR / "holdout_predictions_v3.csv"
TRAINING_REPORT_V3_PATH = OUTPUT_DIR / "training_report_v3.txt"
V3_K_COMPARISON_PATH = OUTPUT_DIR / "v3_k_comparison.csv"
QUESTION_EMBED_V1_PATH = OUTPUT_DIR / "question_embeddings.csv"
PERSON_EMBED_FULL_PATH = PROJECT_ROOT / "outputs" / "person_embeddings.csv"
PERSON_EMBED_META_PATH = PROJECT_ROOT / "outputs" / "person_embeddings_meta.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PCA_COMPONENTS = 50
HOLDOUT_FRAC = 0.2
SEED = 42
PERSON_EMBEDDING_KS = [10, 20, 30, 50]

PERSONALITY_BOOTSTRAPS = 12
OTHER_BOOTSTRAPS = 10
BINARY_BOOTSTRAPS = 10

CONSTRUCT_FEATURE_COLS = [
    "person_construct_mean_loo",
    "person_construct_std_loo",
    "person_construct_percentile_loo",
    "construct_answers_available",
    "has_multi_item_construct",
]


def person_embedding_cols(k: int) -> List[str]:
    return [f"emb_{i}" for i in range(k)]


@dataclass
class ConstantPredictionModel:
    value: float
    mode: str = "regression"

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value, dtype=np.float32)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        p1 = float(np.clip(self.value, 0.0, 1.0))
        p0 = 1.0 - p1
        return np.column_stack(
            [
                np.full(len(X), p0, dtype=np.float32),
                np.full(len(X), p1, dtype=np.float32),
            ]
        )


def require_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required. Install with `python -m pip install lightgbm`."
        ) from exc
    return lgb


def require_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install with "
            "`python -m pip install sentence-transformers`."
        ) from exc
    return SentenceTransformer


def normalize_target(answer_position: pd.Series, num_options: pd.Series) -> pd.Series:
    denom = (pd.to_numeric(num_options, errors="coerce") - 1).replace(0, np.nan)
    return ((pd.to_numeric(answer_position, errors="coerce") - 1) / denom).clip(0, 1)


def denormalize_target(pred_norm: np.ndarray, num_options: pd.Series) -> np.ndarray:
    opts = np.asarray(num_options, dtype=np.float32)
    preds = np.clip(np.asarray(pred_norm, dtype=np.float32), 0.0, 1.0)
    return 1.0 + preds * np.maximum(opts - 1.0, 0.0)


def clip_position_predictions(predictions: np.ndarray, num_options: pd.Series) -> np.ndarray:
    opts = np.asarray(num_options, dtype=np.float32)
    lower = np.ones_like(opts, dtype=np.float32)
    upper = np.maximum(opts, lower)
    return np.clip(np.asarray(predictions, dtype=np.float32), lower, upper)


def load_source_frames() -> Dict[str, object]:
    print("=" * 72)
    print("TOLENDI V3 - LEAKAGE-AWARE BOOTSTRAP BUILD")
    print("=" * 72)

    required_paths = [
        MASTER_TABLE_PATH,
        PERSON_PROFILE_REFERENCE_PATH,
        UNIQUE_QUESTION_PATH,
        CONSTRUCT_MAPPING_PATH,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    print("\n[1/8] Loading source files")
    master = pd.read_csv(MASTER_TABLE_PATH, low_memory=False)
    unique_questions = pd.read_csv(UNIQUE_QUESTION_PATH, low_memory=False)
    construct_mapping = pd.read_csv(CONSTRUCT_MAPPING_PATH, low_memory=False)
    reference_profile_cols = pd.read_csv(
        PERSON_PROFILE_REFERENCE_PATH,
        nrows=0,
    ).columns.tolist()

    ordinal = master[master["answer_type"] == "ordinal"].copy()
    ordinal["answer_position"] = pd.to_numeric(ordinal["answer_position"], errors="coerce")
    ordinal["num_options"] = pd.to_numeric(ordinal["num_options"], errors="coerce")
    ordinal = ordinal.dropna(subset=["answer_position", "num_options", "full_question"])
    ordinal = ordinal[ordinal["num_options"] >= 2].copy()
    ordinal = ordinal.merge(
        construct_mapping[["question_id", "construct_id"]],
        on="question_id",
        how="left",
        validate="m:1",
    )
    ordinal["construct_id"] = ordinal["construct_id"].fillna(ordinal["parent_question_id"])
    ordinal["answer_norm"] = normalize_target(
        ordinal["answer_position"],
        ordinal["num_options"],
    ).astype(np.float32)

    construct_counts = construct_mapping.groupby("construct_id")["question_id"].nunique()
    multi_item_constructs = set(construct_counts[construct_counts > 1].index)
    all_person_ids = sorted(master["person_id"].unique().tolist())

    print(f"  master rows: {master.shape[0]:,}")
    print(f"  ordinal rows: {ordinal.shape[0]:,}")
    print(f"  ordinal questions: {ordinal['question_id'].nunique()}")
    print(f"  unique persons: {len(all_person_ids)}")
    print(f"  multi-item constructs: {len(multi_item_constructs)}")

    return {
        "master": master,
        "ordinal": ordinal,
        "unique_questions": unique_questions,
        "reference_profile_cols": reference_profile_cols,
        "all_person_ids": all_person_ids,
        "multi_item_constructs": multi_item_constructs,
    }


def build_question_feature_table(
    ordinal: pd.DataFrame,
    unique_questions: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], PCA]:
    print("\n[2/8] Building PCA-reduced question features")

    ordinal_qids = sorted(ordinal["question_id"].unique().tolist())
    question_meta = unique_questions[unique_questions["question_id"].isin(ordinal_qids)].copy()
    question_meta = question_meta[
        ["question_id", "full_question", "num_options", "block_name", "question_type"]
    ].drop_duplicates(subset=["question_id"])
    question_meta = question_meta.sort_values("question_id").reset_index(drop=True)

    if question_meta["question_id"].nunique() != len(ordinal_qids):
        missing = sorted(set(ordinal_qids) - set(question_meta["question_id"]))
        raise ValueError(f"Missing question metadata for: {missing[:10]}")

    embeddings = None
    if QUESTION_EMBED_V1_PATH.exists():
        existing = pd.read_csv(QUESTION_EMBED_V1_PATH, low_memory=False)
        embed_cols = [col for col in existing.columns if col.startswith("qemb_")]
        fallback = existing[["question_id"] + embed_cols].copy()
        fallback = question_meta[["question_id"]].merge(
            fallback,
            on="question_id",
            how="left",
            validate="1:1",
        )
        if fallback[embed_cols].isna().any().any():
            missing = fallback.loc[fallback[embed_cols].isna().any(axis=1), "question_id"].tolist()
            raise ValueError(f"Missing fallback embeddings for question ids: {missing[:10]}")
        embeddings = fallback[embed_cols].to_numpy(dtype=np.float32)
        print(f"  source: local fallback embeddings from {QUESTION_EMBED_V1_PATH}")
    else:
        SentenceTransformer = require_sentence_transformer()
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = embedder.encode(
            question_meta["full_question"].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        print("  source: fresh sentence-transformer embeddings")

    n_components = min(PCA_COMPONENTS, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    reduced = pca.fit_transform(embeddings)

    q_feat = pd.DataFrame(
        reduced,
        columns=[f"qemb_pca_{i}" for i in range(reduced.shape[1])],
    )
    q_feat.insert(0, "question_id", question_meta["question_id"].values)
    q_feat["question_num_options"] = pd.to_numeric(
        question_meta["num_options"],
        errors="coerce",
    ).astype(np.float32)

    block_dummies = pd.get_dummies(
        question_meta["block_name"].fillna("unknown"),
        prefix="block",
        dtype=np.float32,
    )
    qtype_dummies = pd.get_dummies(
        question_meta["question_type"].fillna("unknown"),
        prefix="qtype",
        dtype=np.float32,
    )
    q_feat = pd.concat([q_feat, block_dummies, qtype_dummies], axis=1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    q_feat.to_csv(QUESTION_EMBED_V3_PATH, index=False)
    with QUESTION_PCA_MODEL_V3_PATH.open("wb") as f:
        pickle.dump(pca, f)

    question_feature_cols = [col for col in q_feat.columns if col != "question_id"]

    print(f"  embedded {len(q_feat)} questions")
    print(f"  PCA components: {n_components}")
    print(f"  variance explained: {float(pca.explained_variance_ratio_.sum()):.3f}")
    print(f"  saved question features: {QUESTION_EMBED_V3_PATH}")
    print(f"  saved PCA model: {QUESTION_PCA_MODEL_V3_PATH}")

    return q_feat, question_feature_cols, pca


def build_person_feature_frame(
    master_subset: pd.DataFrame,
    all_person_ids: Sequence[str],
    reference_profile_cols: Sequence[str],
) -> pd.DataFrame:
    grouped = {
        person_id: group.copy()
        for person_id, group in master_subset.groupby("person_id", sort=False)
    }
    profiles: List[Dict[str, object]] = []

    for person_id in all_person_ids:
        person_df = grouped.get(person_id)
        profile: Dict[str, object] = {"person_id": person_id}

        if person_df is not None and not person_df.empty:
            profile.update(build_coverage_features(person_df))
            profile.update(build_response_style_features(person_df))
            profile.update(build_personality_construct_features(person_df))
            profile.update(build_economic_features(person_df))
            profile.update(build_cognitive_features(person_df))
            profile.update(build_demographic_features(person_df))

        profiles.append(profile)

    person_feat = pd.DataFrame(profiles)
    for col in reference_profile_cols:
        if col not in person_feat.columns:
            person_feat[col] = np.nan

    ordered_cols = [col for col in reference_profile_cols if col in person_feat.columns]
    extra_cols = [col for col in person_feat.columns if col not in ordered_cols]
    return person_feat[ordered_cols + extra_cols]


def compute_person_embeddings(
    ordinal_rows: pd.DataFrame,
    all_person_ids: Sequence[str],
    k: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    response_matrix = (
        ordinal_rows.pivot_table(
            index="person_id",
            columns="question_id",
            values="answer_norm",
            aggfunc="mean",
        )
        .reindex(all_person_ids)
        .sort_index()
    )

    if response_matrix.shape[1] == 0:
        emb_df = pd.DataFrame({"person_id": list(all_person_ids)})
        for col in person_embedding_cols(k):
            emb_df[col] = 0.0
        return emb_df, {
            "k_requested": int(k),
            "k_used": 0,
            "n_questions_used": 0,
            "explained_variance": 0.0,
            "zero_variance_questions": [],
        }

    col_means = response_matrix.mean(axis=0)
    filled = response_matrix.fillna(col_means)
    col_stds = filled.std(axis=0, ddof=0).replace(0, np.nan)
    keep_cols = col_stds[col_stds.notna()].index.tolist()
    zero_var_cols = sorted(set(filled.columns) - set(keep_cols))

    if not keep_cols:
        emb_df = pd.DataFrame({"person_id": list(all_person_ids)})
        for col in person_embedding_cols(k):
            emb_df[col] = 0.0
        return emb_df, {
            "k_requested": int(k),
            "k_used": 0,
            "n_questions_used": 0,
            "explained_variance": 0.0,
            "zero_variance_questions": zero_var_cols,
        }

    standardized = (filled[keep_cols] - col_means[keep_cols]) / col_stds[keep_cols]

    n_components = min(k, standardized.shape[0], standardized.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    transformed = pca.fit_transform(standardized.to_numpy(dtype=np.float32))

    emb_df = pd.DataFrame(
        transformed,
        columns=[f"emb_{i}" for i in range(n_components)],
        index=standardized.index,
    ).reset_index()

    for col in person_embedding_cols(k):
        if col not in emb_df.columns:
            emb_df[col] = 0.0

    emb_df = emb_df[["person_id"] + person_embedding_cols(k)].copy()
    for col in person_embedding_cols(k):
        emb_df[col] = emb_df[col].astype(np.float32)

    info = {
        "k_requested": int(k),
        "k_used": int(n_components),
        "n_questions_used": int(len(keep_cols)),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "zero_variance_questions": zero_var_cols,
    }
    return emb_df, info


def combine_person_features(
    person_feat: pd.DataFrame,
    person_emb: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    merged = person_feat.merge(person_emb, on="person_id", how="left", validate="1:1")
    for col in person_embedding_cols(k):
        if col not in merged.columns:
            merged[col] = 0.0
    return merged


def compute_construct_aggregate_table(train_rows: pd.DataFrame) -> pd.DataFrame:
    construct_rows = train_rows[
        train_rows["construct_id"].notna() & train_rows["answer_norm"].notna()
    ].copy()

    if construct_rows.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "construct_id",
                "sum_norm",
                "sum_sq_norm",
                "n_answers",
                "construct_mean_full",
                "construct_std_full",
                "construct_percentile_full",
            ]
        )

    construct_rows["answer_norm_sq"] = np.square(construct_rows["answer_norm"])
    agg = construct_rows.groupby(["person_id", "construct_id"], as_index=False).agg(
        sum_norm=("answer_norm", "sum"),
        sum_sq_norm=("answer_norm_sq", "sum"),
        n_answers=("answer_norm", "size"),
    )
    agg["construct_mean_full"] = agg["sum_norm"] / agg["n_answers"]
    agg["construct_std_full"] = np.sqrt(
        np.maximum(
            (agg["sum_sq_norm"] / agg["n_answers"]) - np.square(agg["construct_mean_full"]),
            0.0,
        )
    )
    agg["construct_percentile_full"] = agg.groupby("construct_id")[
        "construct_mean_full"
    ].rank(method="average", pct=True)
    return agg


def add_construct_features(
    rows: pd.DataFrame,
    train_rows: pd.DataFrame,
    multi_item_constructs: set[str],
    training_mode: bool,
) -> pd.DataFrame:
    out = rows.copy()
    agg = compute_construct_aggregate_table(train_rows)

    out = out.merge(
        agg,
        on=["person_id", "construct_id"],
        how="left",
        validate="m:1",
    )
    out["has_multi_item_construct"] = out["construct_id"].isin(multi_item_constructs).astype(
        np.float32
    )
    out["construct_answers_available"] = (
        out["n_answers"].fillna(0).gt(0).astype(np.float32)
    )

    out["person_construct_mean_loo"] = np.nan
    out["person_construct_std_loo"] = np.nan
    out["person_construct_percentile_loo"] = np.nan

    if training_mode:
        valid = (
            out["has_multi_item_construct"].eq(1)
            & out["n_answers"].fillna(0).gt(1)
            & out["answer_norm"].notna()
        )
        if valid.any():
            loo_n = out.loc[valid, "n_answers"] - 1
            loo_sum = out.loc[valid, "sum_norm"] - out.loc[valid, "answer_norm"]
            loo_sq = out.loc[valid, "sum_sq_norm"] - np.square(out.loc[valid, "answer_norm"])
            loo_mean = loo_sum / loo_n
            loo_var = np.maximum((loo_sq / loo_n) - np.square(loo_mean), 0.0)

            out.loc[valid, "person_construct_mean_loo"] = loo_mean.astype(np.float32)
            out.loc[valid, "person_construct_std_loo"] = np.sqrt(loo_var).astype(np.float32)
            out.loc[valid, "person_construct_percentile_loo"] = (
                out.loc[valid]
                .groupby("construct_id")["person_construct_mean_loo"]
                .rank(method="average", pct=True)
                .astype(np.float32)
            )
    else:
        use_full = out["has_multi_item_construct"].eq(1) & out["n_answers"].fillna(0).gt(0)
        out.loc[use_full, "person_construct_mean_loo"] = out.loc[
            use_full, "construct_mean_full"
        ].astype(np.float32)
        out.loc[use_full, "person_construct_std_loo"] = out.loc[
            use_full, "construct_std_full"
        ].astype(np.float32)
        out.loc[use_full, "person_construct_percentile_loo"] = out.loc[
            use_full, "construct_percentile_full"
        ].astype(np.float32)

    return out.drop(
        columns=[
            col
            for col in [
                "sum_norm",
                "sum_sq_norm",
                "n_answers",
                "construct_mean_full",
                "construct_std_full",
                "construct_percentile_full",
            ]
            if col in out.columns
        ]
    )


def assemble_model_frame(
    rows: pd.DataFrame,
    person_feat: pd.DataFrame,
    question_feat: pd.DataFrame,
    train_construct_rows: pd.DataFrame,
    multi_item_constructs: set[str],
    training_mode: bool,
) -> pd.DataFrame:
    model_df = rows.merge(person_feat, on="person_id", how="left", validate="m:1")
    model_df = model_df.merge(question_feat, on="question_id", how="left", validate="m:1")
    return add_construct_features(
        model_df,
        train_construct_rows,
        multi_item_constructs,
        training_mode=training_mode,
    )


def build_feature_columns(person_feat: pd.DataFrame, question_feat_cols: Sequence[str]) -> List[str]:
    person_cols = [col for col in person_feat.columns if col != "person_id"]
    return person_cols + list(question_feat_cols) + CONSTRUCT_FEATURE_COLS


def fit_imputer(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, SimpleImputer]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    transformed = imputer.fit_transform(df[feature_cols])
    X_df = pd.DataFrame(transformed, columns=feature_cols, index=df.index).astype(np.float32)
    return X_df, imputer


def transform_with_imputer(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    imputer: SimpleImputer,
) -> pd.DataFrame:
    transformed = imputer.transform(df[feature_cols])
    return pd.DataFrame(transformed, columns=feature_cols, index=df.index).astype(np.float32)


def train_bootstrap_regressors(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    n_bootstraps: int,
    seed: int,
) -> Tuple[List[object], SimpleImputer, List[float]]:
    lgb = require_lightgbm()
    X_df, imputer = fit_imputer(train_df, feature_cols)
    y = train_df[target_col].to_numpy(dtype=np.float32)

    if float(np.nanstd(y)) == 0.0:
        constant = float(np.nanmean(y))
        return [ConstantPredictionModel(constant, mode="regression")], imputer, [0.0]

    models: List[object] = []
    oob_maes: List[float] = []
    all_indices = np.arange(len(train_df))

    for i in range(n_bootstraps):
        model_seed = seed + i
        rng = np.random.default_rng(model_seed)
        boot_idx = rng.choice(all_indices, size=len(all_indices), replace=True)
        in_boot = np.zeros(len(all_indices), dtype=bool)
        in_boot[boot_idx] = True
        oob_idx = all_indices[~in_boot]

        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=model_seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_df.iloc[boot_idx], y[boot_idx])
        models.append(model)

        if len(oob_idx) > 0:
            oob_pred = model.predict(X_df.iloc[oob_idx]).astype(np.float32)
            oob_maes.append(float(np.mean(np.abs(oob_pred - y[oob_idx]))))

    return models, imputer, oob_maes


def train_bootstrap_binary_classifiers(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    n_bootstraps: int,
    seed: int,
) -> Tuple[List[object], SimpleImputer, List[float]]:
    lgb = require_lightgbm()
    X_df, imputer = fit_imputer(train_df, feature_cols)
    y = train_df[target_col].to_numpy(dtype=np.int32)

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        constant = float(unique_classes[0]) if len(unique_classes) == 1 else 0.5
        return [ConstantPredictionModel(constant, mode="binary")], imputer, [0.0]

    models: List[object] = []
    oob_maes: List[float] = []
    all_indices = np.arange(len(train_df))

    for i in range(n_bootstraps):
        model_seed = seed + i
        rng = np.random.default_rng(model_seed)
        boot_idx = rng.choice(all_indices, size=len(all_indices), replace=True)
        in_boot = np.zeros(len(all_indices), dtype=bool)
        in_boot[boot_idx] = True
        oob_idx = all_indices[~in_boot]

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=0.1,
            random_state=model_seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_df.iloc[boot_idx], y[boot_idx])
        models.append(model)

        if len(oob_idx) > 0:
            oob_pred = model.predict_proba(X_df.iloc[oob_idx])[:, 1].astype(np.float32)
            oob_maes.append(float(np.mean(np.abs(oob_pred - y[oob_idx]))))

    return models, imputer, oob_maes


def predict_regression_ensemble(
    models: Sequence[object],
    imputer: SimpleImputer,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X_df = transform_with_imputer(df, feature_cols, imputer)
    preds = np.vstack([model.predict(X_df).astype(np.float32) for model in models])
    return preds.mean(axis=0), preds.std(axis=0)


def predict_binary_ensemble(
    models: Sequence[object],
    imputer: SimpleImputer,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X_df = transform_with_imputer(df, feature_cols, imputer)
    probs = np.vstack([model.predict_proba(X_df)[:, 1].astype(np.float32) for model in models])
    return probs.mean(axis=0), probs.std(axis=0)


def baseline_norm_prediction(df: pd.DataFrame) -> np.ndarray:
    if "person_construct_percentile_loo" in df.columns:
        baseline = df["person_construct_percentile_loo"].astype(float)
    else:
        baseline = pd.Series(np.nan, index=df.index, dtype=float)

    if "ordinal_mean_normalized" in df.columns:
        baseline = baseline.fillna(df["ordinal_mean_normalized"].astype(float))

    return baseline.fillna(0.5).clip(0, 1).to_numpy(dtype=np.float32)


def score_predictions(scored_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    scored = scored_df.copy()
    scored["abs_error"] = np.abs(scored["predicted"] - scored["answer_position"])
    scored["range"] = scored["num_options"] - 1

    question_rows: List[Dict[str, float]] = []
    for question_id, group in scored.groupby("question_id", sort=True):
        actual = group["answer_position"].to_numpy(dtype=np.float32)
        predicted = group["predicted"].to_numpy(dtype=np.float32)

        if len(actual) >= 3 and np.std(actual) > 0 and np.std(predicted) > 0:
            corr, _ = pearsonr(actual, predicted)
            corr_value = float(corr)
        else:
            corr_value = np.nan

        rng = float(max(group["range"].iloc[0], 1.0))
        mae = float(np.mean(np.abs(predicted - actual)))
        question_rows.append(
            {
                "question_id": question_id,
                "block_name": group["block_name"].iloc[0],
                "question_type": group["question_type"].iloc[0],
                "route_name": group["route_name"].iloc[0],
                "n_people": int(len(group)),
                "actual_std": float(np.std(actual, ddof=1)) if len(actual) > 1 else np.nan,
                "predicted_std": float(np.std(predicted, ddof=1)) if len(predicted) > 1 else np.nan,
                "mae": mae,
                "accuracy": float(1.0 - (mae / rng)),
                "pearson_r": corr_value,
            }
        )

    question_metrics = pd.DataFrame(question_rows)

    block_rows: List[Dict[str, float]] = []
    for block_name, group in scored.groupby("block_name", sort=True):
        q_metrics = question_metrics[question_metrics["block_name"] == block_name]
        valid_corr = q_metrics["pearson_r"].dropna()
        mad = float(group["abs_error"].mean())
        avg_range = float(group["range"].mean())
        block_rows.append(
            {
                "block_name": block_name,
                "n_rows": int(len(group)),
                "n_questions": int(group["question_id"].nunique()),
                "n_questions_with_corr": int(q_metrics["pearson_r"].notna().sum()),
                "accuracy": float(1.0 - (mad / avg_range)) if avg_range > 0 else np.nan,
                "mean_pearson_r": float(valid_corr.mean()) if not valid_corr.empty else np.nan,
                "median_pearson_r": float(valid_corr.median()) if not valid_corr.empty else np.nan,
                "mad": mad,
                "avg_range": avg_range,
            }
        )

    block_metrics = pd.DataFrame(block_rows).sort_values("block_name").reset_index(drop=True)

    valid_corr = question_metrics["pearson_r"].dropna()
    actual_std = float(scored.groupby("question_id")["answer_position"].std().mean())
    pred_std = float(scored.groupby("question_id")["predicted"].std().mean())
    summary = {
        "accuracy": float(1.0 - (scored["abs_error"].mean() / scored["range"].mean())),
        "mad": float(scored["abs_error"].mean()),
        "avg_range": float(scored["range"].mean()),
        "mean_pearson_r": float(valid_corr.mean()) if not valid_corr.empty else 0.0,
        "median_pearson_r": float(valid_corr.median()) if not valid_corr.empty else 0.0,
        "questions_with_corr": int(question_metrics["pearson_r"].notna().sum()),
        "questions_with_positive_r": int((question_metrics["pearson_r"] > 0).sum()),
        "actual_std_mean": actual_std,
        "predicted_std_mean": pred_std,
        "variance_ratio": float(pred_std / actual_std) if actual_std > 0 else np.nan,
    }

    return question_metrics, block_metrics, summary


def route_label(df: pd.DataFrame) -> pd.Series:
    labels = pd.Series("other_multi", index=df.index, dtype="object")
    labels[df["block_name"] == "Personality"] = "personality"
    labels[(df["block_name"] != "Personality") & (df["num_options"] == 2)] = "binary_non_personality"
    return labels


def run_validation(
    master: pd.DataFrame,
    ordinal: pd.DataFrame,
    question_feat: pd.DataFrame,
    question_feature_cols: Sequence[str],
    reference_profile_cols: Sequence[str],
    all_person_ids: Sequence[str],
    multi_item_constructs: set[str],
    person_embedding_k: int,
) -> Dict[str, object]:
    print(f"\n[3/8] Running leakage-aware held-out-question validation (K={person_embedding_k})")

    all_qids = np.array(sorted(ordinal["question_id"].unique().tolist()))
    holdout_size = max(1, int(len(all_qids) * HOLDOUT_FRAC))
    rng = np.random.default_rng(SEED)
    holdout_qids = set(rng.choice(all_qids, size=holdout_size, replace=False).tolist())

    master_train = master[~master["question_id"].isin(holdout_qids)].copy()
    ordinal_train = ordinal[~ordinal["question_id"].isin(holdout_qids)].copy()
    ordinal_test = ordinal[ordinal["question_id"].isin(holdout_qids)].copy()

    person_feat_base = build_person_feature_frame(
        master_train,
        all_person_ids,
        reference_profile_cols,
    )
    person_emb_train, embedding_info = compute_person_embeddings(
        ordinal_train,
        all_person_ids,
        person_embedding_k,
    )
    person_feat_train = combine_person_features(
        person_feat_base,
        person_emb_train,
        person_embedding_k,
    )
    feature_cols = build_feature_columns(person_feat_train, question_feature_cols)

    train_routed = ordinal_train.copy()
    test_routed = ordinal_test.copy()
    train_routed["route_name"] = route_label(train_routed)
    test_routed["route_name"] = route_label(test_routed)

    scored_parts: List[pd.DataFrame] = []
    model_reports: Dict[str, Dict[str, float]] = {}
    route_specs = [
        ("personality", train_routed["route_name"] == "personality", PERSONALITY_BOOTSTRAPS, "regression"),
        (
            "binary_non_personality",
            train_routed["route_name"] == "binary_non_personality",
            BINARY_BOOTSTRAPS,
            "binary",
        ),
        ("other_multi", train_routed["route_name"] == "other_multi", OTHER_BOOTSTRAPS, "regression"),
    ]

    for route_name, train_mask, n_bootstraps, route_type in route_specs:
        route_train = train_routed.loc[train_mask].copy()
        route_test = test_routed.loc[test_routed["route_name"] == route_name].copy()
        if route_test.empty:
            continue

        print(f"  route={route_name} train_rows={len(route_train):,} test_rows={len(route_test):,}")

        train_model_df = assemble_model_frame(
            route_train,
            person_feat_train,
            question_feat,
            ordinal_train,
            multi_item_constructs,
            training_mode=True,
        )
        test_model_df = assemble_model_frame(
            route_test,
            person_feat_train,
            question_feat,
            ordinal_train,
            multi_item_constructs,
            training_mode=False,
        )

        if route_type == "binary":
            train_model_df["binary_target"] = (
                train_model_df["answer_position"].round().astype(int) - 1
            )
            models, imputer, oob_scores = train_bootstrap_binary_classifiers(
                train_model_df,
                feature_cols,
                "binary_target",
                n_bootstraps=n_bootstraps,
                seed=SEED + 1000,
            )
            pred_norm, pred_std = predict_binary_ensemble(models, imputer, test_model_df, feature_cols)
        else:
            models, imputer, oob_scores = train_bootstrap_regressors(
                train_model_df,
                feature_cols,
                "answer_norm",
                n_bootstraps=n_bootstraps,
                seed=SEED + 100,
            )
            pred_norm, pred_std = predict_regression_ensemble(models, imputer, test_model_df, feature_cols)

        baseline_norm = baseline_norm_prediction(test_model_df)
        pred_norm = np.where(np.isnan(pred_norm), baseline_norm, pred_norm)
        pred_norm = np.clip(pred_norm, 0.0, 1.0)

        route_scored = route_test.copy()
        route_scored["route_name"] = route_name
        route_scored["prediction_norm"] = pred_norm
        route_scored["prediction_norm_std"] = pred_std
        route_scored["predicted"] = clip_position_predictions(
            denormalize_target(pred_norm, route_scored["num_options"]),
            route_scored["num_options"],
        )
        route_scored["baseline_predicted"] = clip_position_predictions(
            denormalize_target(baseline_norm, route_scored["num_options"]),
            route_scored["num_options"],
        )
        scored_parts.append(route_scored)

        model_reports[route_name] = {
            "train_rows": int(len(route_train)),
            "test_rows": int(len(route_test)),
            "n_bootstraps": int(n_bootstraps),
            "oob_mean": float(np.mean(oob_scores)) if oob_scores else np.nan,
            "oob_std": float(np.std(oob_scores)) if oob_scores else np.nan,
        }

    scored_df = pd.concat(scored_parts, ignore_index=True)
    question_metrics, block_metrics, summary = score_predictions(scored_df)
    summary.update(
        {
            "holdout_fraction": HOLDOUT_FRAC,
            "holdout_question_count": int(len(holdout_qids)),
            "train_question_count": int(ordinal_train["question_id"].nunique()),
            "train_row_count": int(len(ordinal_train)),
            "holdout_row_count": int(len(ordinal_test)),
            "person_embedding_k": int(person_embedding_k),
            "person_embedding_info": embedding_info,
            "route_reports": model_reports,
        }
    )

    print(f"  accuracy: {summary['accuracy']:.4f}")
    print(f"  mean Pearson r: {summary['mean_pearson_r']:.4f}")
    print(f"  variance ratio: {summary['variance_ratio']:.4f}")

    return {
        "summary": summary,
        "question_metrics": question_metrics,
        "block_metrics": block_metrics,
        "scored_df": scored_df,
    }


def train_full_models(
    master: pd.DataFrame,
    ordinal: pd.DataFrame,
    question_feat: pd.DataFrame,
    question_feature_cols: Sequence[str],
    reference_profile_cols: Sequence[str],
    all_person_ids: Sequence[str],
    multi_item_constructs: set[str],
    person_embedding_k: int,
) -> Dict[str, object]:
    print(f"\n[4/8] Training full routed ensemble (best K={person_embedding_k})")

    person_feat_base = build_person_feature_frame(
        master,
        all_person_ids,
        reference_profile_cols,
    )
    person_emb_full, embedding_info = compute_person_embeddings(
        ordinal,
        all_person_ids,
        person_embedding_k,
    )
    person_feat_full = combine_person_features(
        person_feat_base,
        person_emb_full,
        person_embedding_k,
    )
    person_emb_output_path = OUTPUT_DIR / f"person_embeddings_v3_k{person_embedding_k}.csv"
    person_emb_full.to_csv(person_emb_output_path, index=False)
    feature_cols = build_feature_columns(person_feat_full, question_feature_cols)

    routed = ordinal.copy()
    routed["route_name"] = route_label(routed)

    bundle: Dict[str, object] = {
        "feature_cols": feature_cols,
        "person_feature_cols": [col for col in person_feat_full.columns if col != "person_id"],
        "question_feature_cols": list(question_feature_cols),
        "construct_feature_cols": CONSTRUCT_FEATURE_COLS,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "person_embedding_k": int(person_embedding_k),
        "person_embedding_info": embedding_info,
        "pca_components": int(
            len([col for col in question_feat.columns if col.startswith("qemb_pca_")])
        ),
        "seed": SEED,
        "routes": {},
        "question_feature_table": question_feat,
        "person_feature_table": person_feat_full,
    }

    route_specs = [
        ("personality", routed["route_name"] == "personality", PERSONALITY_BOOTSTRAPS, "regression"),
        (
            "binary_non_personality",
            routed["route_name"] == "binary_non_personality",
            BINARY_BOOTSTRAPS,
            "binary",
        ),
        ("other_multi", routed["route_name"] == "other_multi", OTHER_BOOTSTRAPS, "regression"),
    ]

    for route_name, train_mask, n_bootstraps, route_type in route_specs:
        route_train = routed.loc[train_mask].copy()
        if route_train.empty:
            continue

        print(f"  route={route_name} full_train_rows={len(route_train):,}")
        train_model_df = assemble_model_frame(
            route_train,
            person_feat_full,
            question_feat,
            ordinal,
            multi_item_constructs,
            training_mode=True,
        )

        if route_type == "binary":
            train_model_df["binary_target"] = (
                train_model_df["answer_position"].round().astype(int) - 1
            )
            models, imputer, oob_scores = train_bootstrap_binary_classifiers(
                train_model_df,
                feature_cols,
                "binary_target",
                n_bootstraps=n_bootstraps,
                seed=SEED + 3000,
            )
        else:
            models, imputer, oob_scores = train_bootstrap_regressors(
                train_model_df,
                feature_cols,
                "answer_norm",
                n_bootstraps=n_bootstraps,
                seed=SEED + 2000,
            )

        bundle["routes"][route_name] = {
            "route_type": route_type,
            "models": models,
            "imputer": imputer,
            "oob_mean": float(np.mean(oob_scores)) if oob_scores else np.nan,
            "oob_std": float(np.std(oob_scores)) if oob_scores else np.nan,
            "n_rows": int(len(route_train)),
            "n_bootstraps": int(n_bootstraps),
        }

    with MODEL_BUNDLE_V3_PATH.open("wb") as f:
        pickle.dump(bundle, f)

    print(f"  saved model bundle: {MODEL_BUNDLE_V3_PATH}")
    print(f"  saved full person embeddings: {person_emb_output_path}")
    return bundle


def save_validation_outputs(validation: Dict[str, object]) -> None:
    validation["question_metrics"].to_csv(VALIDATION_BY_QUESTION_V3_PATH, index=False)
    validation["block_metrics"].to_csv(VALIDATION_BY_BLOCK_V3_PATH, index=False)
    validation["scored_df"].to_csv(HOLDOUT_PREDICTIONS_V3_PATH, index=False)
    VALIDATION_SUMMARY_V3_PATH.write_text(
        json.dumps(validation["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_training_report(validation: Dict[str, object], comparison_df: pd.DataFrame) -> None:
    summary = validation["summary"]
    block_metrics = validation["block_metrics"]

    lines = [
        "TOLENDI V3 TRAINING REPORT",
        "=" * 40,
        "",
        f"Best embedding K: {summary['person_embedding_k']}",
        f"Accuracy: {summary['accuracy']:.4f}",
        f"Mean Pearson r: {summary['mean_pearson_r']:.4f}",
        f"Median Pearson r: {summary['median_pearson_r']:.4f}",
        f"Variance ratio: {summary['variance_ratio']:.4f}",
        f"Holdout questions: {summary['holdout_question_count']}",
        "",
        "Per-block results:",
    ]

    for _, row in block_metrics.iterrows():
        lines.append(
            f"  {row['block_name']}: accuracy={row['accuracy']:.4f}, "
            f"mean_r={row['mean_pearson_r']:.4f}, questions={int(row['n_questions_with_corr'])}"
        )

    lines.extend(["", "Route diagnostics:"])
    for route_name, route_info in summary["route_reports"].items():
        lines.append(
            f"  {route_name}: train_rows={route_info['train_rows']}, "
            f"test_rows={route_info['test_rows']}, "
            f"oob_mean={route_info['oob_mean']:.4f}"
        )

    lines.extend(["", "K comparison:"])
    for _, row in comparison_df.sort_values("mean_pearson_r", ascending=False).iterrows():
        lines.append(
            f"  K={int(row['person_embedding_k'])}: "
            f"accuracy={row['accuracy']:.4f}, "
            f"mean_r={row['mean_pearson_r']:.4f}, "
            f"variance_ratio={row['variance_ratio']:.4f}"
        )

    lines.extend(
        [
            "",
            "Red flag checks:",
            f"  Mean Pearson r <= V2 (0.3009): {summary['mean_pearson_r'] <= 0.3009}",
            f"  Variance ratio <= V2 (0.3786): {summary['variance_ratio'] <= 0.3786}",
            f"  Accuracy < 0.67: {summary['accuracy'] < 0.67}",
        ]
    )

    TRAINING_REPORT_V3_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        frames = load_source_frames()
        question_feat, question_feature_cols, _ = build_question_feature_table(
            frames["ordinal"],
            frames["unique_questions"],
        )

        validations: List[Dict[str, object]] = []
        comparison_rows: List[Dict[str, float]] = []
        print(f"\nSweeping person embedding sizes: {PERSON_EMBEDDING_KS}")
        for k in PERSON_EMBEDDING_KS:
            validation = run_validation(
                frames["master"],
                frames["ordinal"],
                question_feat,
                question_feature_cols,
                frames["reference_profile_cols"],
                frames["all_person_ids"],
                frames["multi_item_constructs"],
                person_embedding_k=k,
            )
            validations.append(validation)
            comparison_rows.append(
                {
                    "person_embedding_k": int(k),
                    "accuracy": validation["summary"]["accuracy"],
                    "mean_pearson_r": validation["summary"]["mean_pearson_r"],
                    "median_pearson_r": validation["summary"]["median_pearson_r"],
                    "variance_ratio": validation["summary"]["variance_ratio"],
                    "positive_questions": validation["summary"]["questions_with_positive_r"],
                }
            )

        comparison_df = pd.DataFrame(comparison_rows).sort_values(
            ["mean_pearson_r", "variance_ratio", "accuracy"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        comparison_df.to_csv(V3_K_COMPARISON_PATH, index=False)

        best_k = int(comparison_df.iloc[0]["person_embedding_k"])
        best_validation = next(
            validation
            for validation in validations
            if int(validation["summary"]["person_embedding_k"]) == best_k
        )
        save_validation_outputs(best_validation)

        train_full_models(
            frames["master"],
            frames["ordinal"],
            question_feat,
            question_feature_cols,
            frames["reference_profile_cols"],
            frames["all_person_ids"],
            frames["multi_item_constructs"],
            person_embedding_k=best_k,
        )

        print("\n[5/8] Writing report")
        write_training_report(best_validation, comparison_df)

        print("\n[6/8] Key outputs")
        print(f"  summary: {VALIDATION_SUMMARY_V3_PATH}")
        print(f"  by block: {VALIDATION_BY_BLOCK_V3_PATH}")
        print(f"  by question: {VALIDATION_BY_QUESTION_V3_PATH}")
        print(f"  holdout predictions: {HOLDOUT_PREDICTIONS_V3_PATH}")
        print(f"  bundle: {MODEL_BUNDLE_V3_PATH}")
        print(f"  K comparison: {V3_K_COMPARISON_PATH}")
        print(f"  report: {TRAINING_REPORT_V3_PATH}")

        print("\n[7/8] Validation snapshot")
        print(f"  best K: {best_k}")
        print(f"  accuracy: {best_validation['summary']['accuracy']:.4f}")
        print(f"  mean Pearson r: {best_validation['summary']['mean_pearson_r']:.4f}")
        print(f"  variance ratio: {best_validation['summary']['variance_ratio']:.4f}")

        print("\n[8/8] Build complete")
        return 0

    except Exception as exc:
        print(f"\nBUILD FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
