from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


MIN_VALID_RESPONSES = 100
RANDOM_STATE = 42
TARGET_SCORE_WEIGHTS = {"accuracy": 0.4, "correlation": 0.6}
TOPIC_KEYWORDS = {
    "risk": ["risk", "lottery", "probability", "chance", "unsafe", "gamble", "insurance"],
    "future": ["future", "tomorrow", "later", "delay", "wait", "years", "months", "next"],
    "social": ["friend", "family", "people", "social", "others", "community", "coworker", "partner"],
    "money": ["$", "money", "price", "cost", "income", "pay", "financial", "dollar"],
    "identity": ["identity", "personality", "self", "aspire", "ought", "actual", "me "],
    "habit": ["habit", "routine", "phone", "work", "sleep", "daily", "practice"],
    "emotion": ["happy", "sad", "angry", "anxiety", "fear", "emotion", "nervous"],
    "politics": ["politic", "government", "election", "liberal", "conservative"],
    "environment": ["green", "environment", "climate", "recycle", "sustainable"],
    "probabilistic": ["probability", "percent", "%", "likely", "chance", "odds"],
}


@dataclass
class SurveyData:
    survey_name: str
    numbers_raw: pd.DataFrame
    labels_raw: pd.DataFrame


@dataclass
class FeatureBundle:
    train_matrix: pd.DataFrame
    target_matrix: pd.DataFrame
    train_features: pd.DataFrame
    target_features: pd.DataFrame
    question_features: pd.DataFrame
    dynamic_person_features: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict unseen question responses for the same respondents."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-valid-responses", type=int, default=MIN_VALID_RESPONSES)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return re.sub(r"\s+", " ", text)


def safe_json_import_id(value: object) -> str:
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, dict) and "ImportId" in parsed:
            return str(parsed["ImportId"])
    except Exception:
        pass
    return clean_text(value)


def load_survey_pair(numbers_path: Path, labels_path: Path) -> SurveyData:
    return SurveyData(
        survey_name=numbers_path.stem.replace("_numbers", ""),
        numbers_raw=pd.read_csv(numbers_path, header=None),
        labels_raw=pd.read_csv(labels_path, header=None),
    )


def collect_surveys(data_dir: Path) -> List[SurveyData]:
    surveys: List[SurveyData] = []
    csv_dir = data_dir / "personas_csv"
    for numbers_path in sorted(csv_dir.glob("survey_*_numbers.csv")):
        labels_path = numbers_path.with_name(numbers_path.name.replace("_numbers", "_labels"))
        if labels_path.exists():
            surveys.append(load_survey_pair(numbers_path, labels_path))
    if not surveys:
        raise FileNotFoundError(f"No survey CSV pairs found under {csv_dir}")
    return surveys


def ordered_label_map(question_values: pd.Series, label_values: pd.Series) -> Tuple[Dict[float, str], List[str]]:
    pairs = pd.DataFrame(
        {
            "value": pd.to_numeric(question_values, errors="coerce"),
            "label": label_values.astype("string"),
        }
    )
    pairs = pairs.dropna(subset=["value"])
    pairs["label"] = pairs["label"].fillna("").map(clean_text)
    pairs = pairs[pairs["label"] != ""]
    if pairs.empty:
        return {}, []
    label_map = (
        pairs.groupby("value")["label"]
        .agg(lambda x: x.value_counts().idxmax())
        .sort_index()
        .to_dict()
    )
    ordered = [label_map[key] for key in sorted(label_map)]
    return label_map, ordered


def maybe_numeric_span_from_options(options_texts: Sequence[str]) -> Tuple[Optional[float], Optional[float]]:
    found: List[float] = []
    for option in options_texts:
        for match in re.findall(r"-?\d+(?:\.\d+)?", option):
            try:
                found.append(float(match))
            except Exception:
                continue
    if not found:
        return None, None
    return min(found), max(found)


def build_historical_tables(
    surveys: Sequence[SurveyData], min_valid_responses: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    response_rows: List[Dict[str, object]] = []
    question_rows: List[Dict[str, object]] = []

    for survey in surveys:
        columns = survey.numbers_raw.iloc[0].astype(str).tolist()
        question_texts = survey.numbers_raw.iloc[1].tolist()
        import_ids = survey.numbers_raw.iloc[2].tolist()
        num_df = survey.numbers_raw.iloc[3:].copy()
        label_df = survey.labels_raw.iloc[3:].copy()
        num_df.columns = columns
        label_df.columns = columns
        person_ids = num_df["person_id"].astype(str).map(clean_text)

        for column_name in columns[1:]:
            question_id = f"{survey.survey_name}::{column_name}"
            response_values = pd.to_numeric(num_df[column_name], errors="coerce")
            valid_count = int(response_values.notna().sum())
            if valid_count < min_valid_responses:
                continue

            _, ordered_options = ordered_label_map(num_df[column_name], label_df[column_name])
            observed_min = float(response_values.min())
            observed_max = float(response_values.max())
            q_idx = columns.index(column_name)
            option_min, option_max = maybe_numeric_span_from_options(ordered_options)
            scale_min = observed_min if option_min is None else min(observed_min, option_min)
            scale_max = observed_max if option_max is None else max(observed_max, option_max)

            question_rows.append(
                {
                    "question_id": question_id,
                    "survey_name": survey.survey_name,
                    "column_name": column_name,
                    "question_text": clean_text(question_texts[q_idx]),
                    "import_id": safe_json_import_id(import_ids[q_idx]),
                    "option_text": " | ".join(ordered_options),
                    "option_count": len(ordered_options),
                    "observed_min": observed_min,
                    "observed_max": observed_max,
                    "scale_min": scale_min,
                    "scale_max": scale_max,
                    "response_range": scale_max - scale_min,
                    "n_responses": valid_count,
                }
            )

            valid_mask = response_values.notna()
            block = pd.DataFrame(
                {
                    "person_id": person_ids[valid_mask].values,
                    "question_id": question_id,
                    "response": response_values[valid_mask].astype(float).values,
                    "response_label": label_df.loc[valid_mask, column_name].map(clean_text).values,
                }
            )
            response_rows.extend(block.to_dict(orient="records"))

    responses = pd.DataFrame(response_rows)
    questions = pd.DataFrame(question_rows).drop_duplicates("question_id").reset_index(drop=True)
    questions["response_range"] = questions["response_range"].replace(0, np.nan)
    response_meta = questions.set_index("question_id")[["scale_min", "scale_max"]]
    responses = responses.merge(response_meta, left_on="question_id", right_index=True, how="left")
    denom = (responses["scale_max"] - responses["scale_min"]).replace(0, np.nan)
    responses["response_norm"] = np.where(
        denom.notna(), (responses["response"] - responses["scale_min"]) / denom, 0.5
    )
    responses["response_norm"] = responses["response_norm"].clip(0, 1)
    responses = responses.drop(columns=["scale_min", "scale_max"])
    return responses, questions


def parse_persona_texts(persona_dir: Path) -> pd.DataFrame:
    feature_rows: List[Dict[str, object]] = []
    score_pattern = re.compile(
        r"^\s*([A-Za-z0-9_]+)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*\((\d+)(?:st|nd|rd|th) percentile\)\s*$"
    )
    simple_score_pattern = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*$")

    for path in sorted(persona_dir.glob("*_persona.txt")):
        person_id = path.stem.replace("_persona", "")
        row: Dict[str, object] = {"person_id": person_id}
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            score_match = score_pattern.match(stripped)
            if score_match:
                key, value, percentile = score_match.groups()
                row[key] = float(value)
                row[f"{key}_percentile"] = float(percentile)
                continue
            simple_score_match = simple_score_pattern.match(stripped)
            if simple_score_match:
                key, value = simple_score_match.groups()
                if key.lower().startswith("score_") or "crt" in key.lower():
                    row[key] = float(value)
                continue
            if ":" in stripped:
                left, right = stripped.split(":", 1)
                key = re.sub(r"[^A-Za-z0-9]+", "_", left.strip().lower()).strip("_")
                row[key] = clean_text(right)
        age_label = str(row.get("age", ""))
        age_numbers = re.findall(r"\d+", age_label)
        if len(age_numbers) >= 2:
            row["age_midpoint"] = (float(age_numbers[0]) + float(age_numbers[1])) / 2.0
        elif len(age_numbers) == 1:
            row["age_midpoint"] = float(age_numbers[0])
        feature_rows.append(row)
    return pd.DataFrame(feature_rows).drop_duplicates("person_id")


def add_question_text_features(questions: pd.DataFrame) -> pd.DataFrame:
    out = questions.copy()
    doc = (
        out["question_text"].fillna("")
        + " "
        + out["option_text"].fillna("")
        + " "
        + out["column_name"].fillna("")
    ).map(clean_text)
    out["question_document"] = doc
    out["text_char_len"] = doc.str.len()
    out["text_word_len"] = doc.str.split().map(len)
    out["has_question_mark"] = doc.str.contains(r"\?", regex=True).astype(int)
    out["has_percent_symbol"] = doc.str.contains("%", regex=False).astype(int)
    out["has_money_symbol"] = doc.str.contains("$", regex=False).astype(int)
    out["is_binary"] = (out["option_count"].fillna(0) == 2).astype(int)
    out["is_wide_scale"] = (out["response_range"].fillna(0) >= 10).astype(int)
    out["is_continuousish"] = (out["observed_max"] - out["observed_min"] > 10).astype(int)
    lower_doc = doc.str.lower()
    for topic, words in TOPIC_KEYWORDS.items():
        pattern = "|".join(re.escape(word) for word in words)
        out[f"topic_{topic}"] = lower_doc.str.contains(pattern, regex=True).astype(int)
    return out


def build_question_embeddings(
    train_questions: pd.DataFrame, target_questions: pd.DataFrame
) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        max_features=8000,
        min_df=1,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    vectorizer.fit(train_questions["question_document"].fillna(""))
    target_matrix = vectorizer.transform(target_questions["question_document"].fillna(""))
    embedding_df = target_questions[["question_id"]].copy()
    if target_matrix.shape[1] >= 2 and target_questions.shape[0] >= 3:
        n_components = int(min(32, target_matrix.shape[0] - 1, target_matrix.shape[1] - 1))
    else:
        n_components = 0
    if n_components >= 2:
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        dense = svd.fit_transform(target_matrix)
        for idx in range(dense.shape[1]):
            embedding_df[f"q_emb_{idx:02d}"] = dense[:, idx]
    return embedding_df, vectorizer


def encode_external_person_features(person_features: pd.DataFrame) -> pd.DataFrame:
    if person_features.empty:
        return pd.DataFrame(columns=["person_id"])
    out = person_features.copy()
    numeric_cols = [c for c in out.columns if c != "person_id" and pd.api.types.is_numeric_dtype(out[c])]
    categorical_cols = [c for c in out.columns if c != "person_id" and c not in numeric_cols]
    cat = pd.get_dummies(out[categorical_cols], prefix=categorical_cols, dummy_na=True)
    return pd.concat([out[["person_id"]], out[numeric_cols], cat], axis=1)


def compute_dynamic_person_features(
    train_responses: pd.DataFrame, target_questions: pd.DataFrame, question_embeddings: pd.DataFrame
) -> pd.DataFrame:
    persons = sorted(train_responses["person_id"].unique())
    pivot = train_responses.pivot_table(
        index="person_id", columns="question_id", values="response_norm", aggfunc="mean"
    ).reindex(persons)

    person_stats = pd.DataFrame({"person_id": persons})
    person_stats["person_mean_norm"] = pivot.mean(axis=1).values
    person_stats["person_std_norm"] = pivot.std(axis=1).fillna(0).values
    person_stats["person_extreme_ratio"] = (
        ((pivot <= 0.1) | (pivot >= 0.9)).sum(axis=1) / pivot.notna().sum(axis=1).replace(0, np.nan)
    ).fillna(0).values
    person_stats["person_mid_ratio"] = (
        ((pivot >= 0.4) & (pivot <= 0.6)).sum(axis=1) / pivot.notna().sum(axis=1).replace(0, np.nan)
    ).fillna(0).values

    survey_means = (
        train_responses.groupby(["person_id", "survey_name"])["response_norm"]
        .mean()
        .unstack(fill_value=np.nan)
        .add_prefix("survey_mean_")
        .reset_index()
    )
    person_stats = person_stats.merge(survey_means, on="person_id", how="left")

    filled = pivot.apply(lambda col: col.fillna(col.mean()), axis=0).fillna(0.5)
    n_components = int(min(12, filled.shape[0] - 1, filled.shape[1] - 1)) if filled.shape[1] > 1 else 0
    if n_components >= 2:
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        latent = svd.fit_transform(filled.values)
        latent_df = pd.DataFrame(latent, columns=[f"person_latent_{i:02d}" for i in range(latent.shape[1])])
        latent_df.insert(0, "person_id", filled.index.values)
        person_stats = person_stats.merge(latent_df, on="person_id", how="left")

    q_emb_cols = [c for c in question_embeddings.columns if c.startswith("q_emb_")]
    cluster_lookup = target_questions[["question_id"]].copy()
    if q_emb_cols and train_responses["question_id"].nunique() >= 8:
        train_q = target_questions[target_questions["question_id"].isin(train_responses["question_id"].unique())]
        n_clusters = int(max(4, min(12, len(train_q) // 20 + 1)))
        n_clusters = min(n_clusters, max(2, len(train_q)))
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=RANDOM_STATE)
        train_embed = question_embeddings[
            question_embeddings["question_id"].isin(train_q["question_id"])
        ][q_emb_cols].fillna(0)
        model.fit(train_embed.values)
        cluster_lookup = target_questions[["question_id"]].merge(question_embeddings, on="question_id", how="left")
        cluster_lookup["question_cluster"] = model.predict(cluster_lookup[q_emb_cols].fillna(0).values)
    else:
        cluster_lookup["question_cluster"] = 0

    cluster_means = (
        train_responses.merge(cluster_lookup[["question_id", "question_cluster"]], on="question_id", how="left")
        .groupby(["person_id", "question_cluster"])["response_norm"]
        .mean()
        .unstack(fill_value=np.nan)
        .add_prefix("cluster_mean_")
        .reset_index()
    )
    return person_stats.merge(cluster_means, on="person_id", how="left")


def compute_nearest_question_prior(
    train_responses: pd.DataFrame,
    train_questions: pd.DataFrame,
    target_questions: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    k: int = 8,
) -> pd.DataFrame:
    persons = sorted(train_responses["person_id"].unique())
    prior_rows: List[Dict[str, object]] = []
    train_docs = train_questions["question_document"].fillna("")
    target_docs = target_questions["question_document"].fillna("")
    train_matrix = vectorizer.transform(train_docs)
    target_matrix = vectorizer.transform(target_docs)
    sim = cosine_similarity(target_matrix, train_matrix)
    train_qids = train_questions["question_id"].tolist()
    target_qids = target_questions["question_id"].tolist()
    response_pivot = train_responses.pivot_table(
        index="person_id", columns="question_id", values="response_norm", aggfunc="mean"
    )
    global_person_mean = response_pivot.mean(axis=1).fillna(0.5).to_dict()

    for row_idx, question_id in enumerate(target_qids):
        similarities = sim[row_idx].copy()
        if question_id in train_qids:
            similarities[train_qids.index(question_id)] = -1.0
        neighbor_idx = np.argsort(similarities)[::-1][:k]
        valid_neighbors = [(train_qids[idx], float(similarities[idx])) for idx in neighbor_idx if similarities[idx] > 0]
        for person_id in persons:
            weighted_sum = 0.0
            total_weight = 0.0
            for neighbor_qid, weight in valid_neighbors:
                if neighbor_qid not in response_pivot.columns:
                    continue
                person_value = response_pivot.at[person_id, neighbor_qid]
                if pd.notna(person_value):
                    weighted_sum += float(person_value) * weight
                    total_weight += weight
            nn_value = weighted_sum / total_weight if total_weight > 0 else float(global_person_mean.get(person_id, 0.5))
            prior_rows.append(
                {
                    "person_id": person_id,
                    "question_id": question_id,
                    "nn_prior_norm": nn_value,
                    "nn_neighbor_count": len(valid_neighbors),
                }
            )
    return pd.DataFrame(prior_rows)


def assemble_feature_bundle(
    train_responses: pd.DataFrame,
    target_rows: pd.DataFrame,
    all_questions: pd.DataFrame,
    external_person_features: pd.DataFrame,
) -> FeatureBundle:
    target_qids = target_rows["question_id"].drop_duplicates().tolist()
    train_questions = all_questions[all_questions["question_id"].isin(train_responses["question_id"].unique())].copy()
    relevant_qids = set(train_responses["question_id"].unique()) | set(target_qids)
    relevant_questions = all_questions[all_questions["question_id"].isin(relevant_qids)].copy()
    question_embeddings, vectorizer = build_question_embeddings(train_questions, relevant_questions)
    dynamic_person_features = compute_dynamic_person_features(train_responses, relevant_questions, question_embeddings)
    nearest_prior = compute_nearest_question_prior(train_responses, train_questions, relevant_questions, vectorizer)
    question_features = relevant_questions.merge(question_embeddings, on="question_id", how="left")

    train_matrix = train_responses[["person_id", "question_id", "response_norm", "response"]].copy()
    target_matrix = target_rows.copy()
    train_features = (
        train_matrix[["person_id", "question_id"]]
        .merge(external_person_features, on="person_id", how="left")
        .merge(dynamic_person_features, on="person_id", how="left")
        .merge(question_features, on="question_id", how="left")
        .merge(nearest_prior, on=["person_id", "question_id"], how="left")
    )
    target_features = (
        target_matrix[["person_id", "question_id"]]
        .merge(external_person_features, on="person_id", how="left")
        .merge(dynamic_person_features, on="person_id", how="left")
        .merge(question_features, on="question_id", how="left")
        .merge(nearest_prior, on=["person_id", "question_id"], how="left")
    )

    numeric_person_cols = [
        c
        for c in external_person_features.columns
        if c != "person_id" and pd.api.types.is_numeric_dtype(external_person_features[c])
    ]
    topic_cols = [c for c in question_features.columns if c.startswith("topic_")]
    preferred_numeric = [
        "score_extraversion",
        "score_agreeableness",
        "wave1_score_conscientiousness",
        "score_openness",
        "score_neuroticism",
        "score_needforcognition",
        "score_GREEN",
        "crt2_score",
        "score_fluid",
        "score_crystallized",
        "age_midpoint",
    ]
    selected_numeric = [c for c in preferred_numeric if c in numeric_person_cols][:6]
    if not selected_numeric:
        selected_numeric = numeric_person_cols[:6]
    for p_col, q_col in product(selected_numeric, topic_cols[:8]):
        name = f"int__{p_col}__{q_col}"
        train_features[name] = train_features[p_col].fillna(0) * train_features[q_col].fillna(0)
        target_features[name] = target_features[p_col].fillna(0) * target_features[q_col].fillna(0)

    return FeatureBundle(
        train_matrix=train_matrix,
        target_matrix=target_matrix,
        train_features=train_features,
        target_features=target_features,
        question_features=question_features,
        dynamic_person_features=dynamic_person_features,
    )


def pearson_safe(y_true: pd.Series, y_pred: pd.Series) -> float:
    if y_true.nunique(dropna=True) <= 1 or pd.Series(y_pred).nunique(dropna=True) <= 1:
        return 0.0
    try:
        return float(pearsonr(y_true, y_pred)[0])
    except Exception:
        return 0.0


def question_level_metrics(
    truth_df: pd.DataFrame, pred_col: str, question_meta: pd.DataFrame
) -> pd.DataFrame:
    meta = question_meta.set_index("question_id")
    rows = []
    for question_id, group in truth_df.groupby("question_id"):
        q_meta = meta.loc[question_id]
        actual = group["response"]
        pred = group[pred_col]
        question_range = q_meta["response_range"]
        if pd.isna(question_range) or float(question_range) <= 0:
            observed_range = float(actual.max() - actual.min())
            question_range = observed_range if observed_range > 0 else 1.0
        mad = float(np.mean(np.abs(actual - pred)))
        accuracy = 1.0 - (mad / float(question_range))
        correlation = pearson_safe(actual, pred)
        rows.append(
            {
                "question_id": question_id,
                "mad": mad,
                "accuracy_proxy": accuracy,
                "pearson_correlation": correlation,
                "n_participants": len(group),
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(per_question: pd.DataFrame) -> Dict[str, float]:
    accuracy = float(per_question["accuracy_proxy"].mean())
    correlation = float(per_question["pearson_correlation"].mean())
    score = TARGET_SCORE_WEIGHTS["accuracy"] * accuracy + TARGET_SCORE_WEIGHTS["correlation"] * correlation
    return {
        "mean_accuracy_proxy": accuracy,
        "mean_pearson_correlation": correlation,
        "score": score,
    }


def feature_columns(frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    exclude = {
        "person_id",
        "question_id",
        "response",
        "response_norm",
        "response_label",
        "question_document",
        "question_text",
        "option_text",
        "column_name",
        "import_id",
        "survey_name",
    }
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for column in frame.columns:
        if column in exclude:
            continue
        series = frame[column]
        if series.notna().sum() == 0:
            continue
        if not pd.api.types.is_numeric_dtype(series):
            cleaned = series.fillna("").astype(str).map(clean_text)
            if (cleaned == "").all():
                continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)
    return numeric_cols, categorical_cols


def build_ridge_pipeline(feature_frame: pd.DataFrame) -> Pipeline:
    numeric_cols, categorical_cols = feature_columns(feature_frame)
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ]
    )


def prepare_lightgbm_matrix(
    train_df: pd.DataFrame, target_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [
            train_df.assign(_dataset="train"),
            target_df.assign(_dataset="target"),
        ],
        ignore_index=True,
    )
    feature_df = combined.drop(columns=["person_id", "question_id"], errors="ignore").copy()
    for column in feature_df.columns:
        if pd.api.types.is_object_dtype(feature_df[column]) or str(feature_df[column].dtype).startswith("string"):
            feature_df[column] = feature_df[column].fillna("__MISSING__").astype(str)
    feature_df = pd.get_dummies(feature_df, dummy_na=True)
    cleaned_columns = []
    seen: Dict[str, int] = {}
    for column in feature_df.columns:
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(column)).strip("_")
        clean = clean or "feature"
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        cleaned_columns.append(clean)
    feature_df.columns = cleaned_columns
    train_matrix = feature_df.loc[combined["_dataset"] == "train"].reset_index(drop=True)
    target_matrix = feature_df.loc[combined["_dataset"] == "target"].reset_index(drop=True)
    return train_matrix, target_matrix


def clip_from_question_meta(pred_norm: np.ndarray, question_ids: Sequence[str], question_meta: pd.DataFrame) -> np.ndarray:
    meta = question_meta.set_index("question_id")[["scale_min", "scale_max"]]
    result = []
    for value, question_id in zip(pred_norm, question_ids):
        scale_min = float(meta.loc[question_id, "scale_min"])
        scale_max = float(meta.loc[question_id, "scale_max"])
        raw = scale_min + float(np.clip(value, 0, 1)) * (scale_max - scale_min)
        result.append(float(np.clip(raw, scale_min, scale_max)))
    return np.array(result, dtype=float)


def fit_predict_models(bundle: FeatureBundle, question_meta: pd.DataFrame) -> Dict[str, np.ndarray]:
    predictions: Dict[str, np.ndarray] = {}
    y_train = bundle.train_matrix["response_norm"].values

    base_person_mean = bundle.target_features.get(
        "person_mean_norm", pd.Series(0.5, index=bundle.target_features.index)
    )
    baseline_norm = (
        0.7 * bundle.target_features["nn_prior_norm"].fillna(base_person_mean)
        + 0.3 * base_person_mean.fillna(0.5)
    ).clip(0, 1)
    predictions["semantic_prior"] = clip_from_question_meta(
        baseline_norm.values, bundle.target_matrix["question_id"].values, question_meta
    )

    ridge = build_ridge_pipeline(bundle.train_features)
    ridge.fit(bundle.train_features, y_train)
    ridge_pred_norm = np.clip(ridge.predict(bundle.target_features), 0, 1)
    predictions["ridge"] = clip_from_question_meta(
        ridge_pred_norm, bundle.target_matrix["question_id"].values, question_meta
    )

    if HAS_LIGHTGBM:
        x_train, x_target = prepare_lightgbm_matrix(bundle.train_features, bundle.target_features)
        lgbm = LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=RANDOM_STATE,
            min_child_samples=20,
            verbose=-1,
        )
        lgbm.fit(x_train, y_train)
        lgbm_pred_norm = np.clip(lgbm.predict(x_target), 0, 1)
        predictions["lightgbm"] = clip_from_question_meta(
            lgbm_pred_norm, bundle.target_matrix["question_id"].values, question_meta
        )
    return predictions


def fit_model_objects(bundle: FeatureBundle) -> Dict[str, object]:
    models: Dict[str, object] = {
        "semantic_prior": {"nn_weight": 0.7, "person_mean_weight": 0.3},
    }
    y_train = bundle.train_matrix["response_norm"].values

    ridge = build_ridge_pipeline(bundle.train_features)
    ridge.fit(bundle.train_features, y_train)
    models["ridge"] = ridge

    if HAS_LIGHTGBM:
        x_train, _ = prepare_lightgbm_matrix(bundle.train_features, bundle.train_features.iloc[0:0].copy())
        lgbm = LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=RANDOM_STATE,
            min_child_samples=20,
            verbose=-1,
        )
        lgbm.fit(x_train, y_train)
        models["lightgbm"] = lgbm
    return models


def optimize_ensemble_weights(
    oof_frame: pd.DataFrame, question_meta: pd.DataFrame, model_names: Sequence[str]
) -> Tuple[Dict[str, float], pd.DataFrame]:
    comparison_rows: List[Dict[str, object]] = []
    for name in model_names:
        summary = summarize_metrics(question_level_metrics(oof_frame, name, question_meta))
        comparison_rows.append({"model": name, **summary})

    if len(model_names) == 1:
        return {model_names[0]: 1.0}, pd.DataFrame(comparison_rows)

    best_weights: Dict[str, float] = {}
    best_score = -np.inf
    grid = np.arange(0.0, 1.01, 0.1)
    for weights in product(grid, repeat=len(model_names)):
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            continue
        ensemble = sum(oof_frame[name].values * weight for name, weight in zip(model_names, weights))
        summary = summarize_metrics(
            question_level_metrics(oof_frame.assign(ensemble=ensemble), "ensemble", question_meta)
        )
        if summary["score"] > best_score:
            best_score = summary["score"]
            best_weights = {name: float(weight) for name, weight in zip(model_names, weights)}

    ensemble_summary = summarize_metrics(
        question_level_metrics(
            oof_frame.assign(
                ensemble=sum(oof_frame[name] * best_weights.get(name, 0.0) for name in model_names)
            ),
            "ensemble",
            question_meta,
        )
    )
    comparison_rows.append({"model": "ensemble", **ensemble_summary})
    return best_weights, pd.DataFrame(comparison_rows)


def discover_test_file(root: Path) -> Optional[Path]:
    candidate_paths = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "artifacts" in {part.lower() for part in path.parts}:
            continue
        lower_name = path.name.lower()
        if "persona" in lower_name:
            continue
        if path.suffix.lower() not in {".json", ".csv", ".tsv", ".xlsx", ".parquet"}:
            continue
        if any(token in lower_name for token in ["test", "submission", "template", "unseen"]):
            candidate_paths.append(path)
    return candidate_paths[0] if candidate_paths else None


def load_test_table(test_path: Path) -> pd.DataFrame:
    suffix = test_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(test_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data", [data])
        df = pd.DataFrame(data)
    elif suffix == ".csv":
        df = pd.read_csv(test_path)
    elif suffix == ".tsv":
        df = pd.read_csv(test_path, sep="\t")
    elif suffix == ".xlsx":
        df = pd.read_excel(test_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(test_path)
    else:
        raise ValueError(f"Unsupported test file: {test_path}")

    rename_map = {}
    for column in df.columns:
        lower = column.lower()
        if column == "Respondent_ID" or lower in {"respondent_id", "id"}:
            rename_map[column] = "person_id"
    df = df.rename(columns=rename_map)
    if "person_id" not in df.columns or "question_id" not in df.columns:
        raise ValueError("Test file must contain person_id/Respondent_ID and question_id columns.")
    return df


def infer_test_question_features(test_df: pd.DataFrame) -> pd.DataFrame:
    out = test_df.copy()
    for column in ["question_text", "context", "options"]:
        if column not in out.columns:
            out[column] = ""

    question_df = (
        out.groupby("question_id", as_index=False)
        .agg({"question_text": "first", "context": "first", "options": "first"})
        .assign(survey_name="test", column_name=lambda x: x["question_id"])
    )
    question_df["question_text"] = (
        question_df["question_text"].fillna("") + " " + question_df["context"].fillna("")
    ).map(clean_text)
    question_df["option_text"] = question_df["options"].apply(
        lambda value: " | ".join(value) if isinstance(value, list) else clean_text(value)
    )
    question_df["option_count"] = question_df["options"].apply(
        lambda value: len(value)
        if isinstance(value, list)
        else (0 if clean_text(value) == "" else len(str(value).split("|")))
    )
    bounds = question_df["option_text"].apply(lambda x: maybe_numeric_span_from_options([x]))
    question_df["scale_min"] = bounds.apply(lambda x: 0.0 if x[0] is None else float(x[0]))
    question_df["scale_max"] = bounds.apply(lambda x: 1.0 if x[1] is None else float(x[1]))
    question_df["observed_min"] = question_df["scale_min"]
    question_df["observed_max"] = question_df["scale_max"]
    question_df["response_range"] = (question_df["scale_max"] - question_df["scale_min"]).replace(0, 1.0)
    question_df["import_id"] = question_df["question_id"]
    question_df["n_responses"] = out.groupby("question_id")["person_id"].size().values
    return add_question_text_features(question_df)


def run_group_validation(
    responses: pd.DataFrame, questions: pd.DataFrame, external_person_features: pd.DataFrame, folds: int
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:
    splitter = GroupKFold(n_splits=folds)
    oof = responses[["person_id", "question_id", "response"]].copy()
    model_names: List[str] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(
        splitter.split(responses, groups=responses["question_id"]), start=1
    ):
        train_responses = responses.iloc[train_idx].copy().merge(
            questions[["question_id", "survey_name"]], on="question_id", how="left"
        )
        valid_rows = responses.iloc[valid_idx][["person_id", "question_id", "response"]].copy()
        bundle = assemble_feature_bundle(
            train_responses=train_responses,
            target_rows=valid_rows,
            all_questions=questions,
            external_person_features=external_person_features,
        )
        fold_predictions = fit_predict_models(bundle, questions)
        if not model_names:
            model_names = list(fold_predictions.keys())
        for name, values in fold_predictions.items():
            oof.loc[oof.index[valid_idx], name] = values
        print(
            f"Completed fold {fold_idx}/{folds} with "
            f"{valid_rows['question_id'].nunique()} held-out questions."
        )

    weights, comparison = optimize_ensemble_weights(oof, questions, model_names)
    oof["ensemble"] = sum(oof[name] * weights.get(name, 0.0) for name in model_names)
    return oof, comparison, weights, model_names


def fit_full_models_and_predict(
    responses: pd.DataFrame,
    questions: pd.DataFrame,
    external_person_features: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    prediction_questions: pd.DataFrame,
    model_names: Sequence[str],
    ensemble_weights: Dict[str, float],
) -> pd.DataFrame:
    train_responses = responses.merge(questions[["question_id", "survey_name"]], on="question_id", how="left")
    all_questions = pd.concat([questions, prediction_questions], ignore_index=True).drop_duplicates("question_id")
    bundle = assemble_feature_bundle(
        train_responses=train_responses,
        target_rows=prediction_rows[["person_id", "question_id"]].copy(),
        all_questions=all_questions,
        external_person_features=external_person_features,
    )
    raw_predictions = fit_predict_models(bundle, all_questions)
    submission = prediction_rows.copy()
    for name in model_names:
        if name in raw_predictions:
            submission[name] = raw_predictions[name]
    submission["predicted_answer"] = sum(
        submission[name] * ensemble_weights.get(name, 0.0)
        for name in model_names
        if name in submission.columns
    )
    meta = prediction_questions.set_index("question_id")[["scale_min", "scale_max"]]
    submission["predicted_answer"] = submission.apply(
        lambda row: float(
            np.clip(
                row["predicted_answer"],
                meta.loc[row["question_id"], "scale_min"],
                meta.loc[row["question_id"], "scale_max"],
            )
        ),
        axis=1,
    )
    return submission


def write_validation_report(
    output_dir: Path,
    summary_table: pd.DataFrame,
    per_question: pd.DataFrame,
    ensemble_weights: Dict[str, float],
    test_path: Optional[Path],
) -> None:
    summary_table.to_csv(output_dir / "model_comparison.csv", index=False)
    per_question.to_csv(output_dir / "per_question_validation.csv", index=False)
    lines = [
        "# Validation Report",
        "",
        "## Model Comparison",
        "",
        summary_table.to_string(index=False),
        "",
        "## Ensemble Weights",
        "",
    ]
    for name, weight in ensemble_weights.items():
        lines.append(f"- {name}: {weight:.2f}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Validation holds out full questions with GroupKFold on `question_id`.",
            "- Targets are modeled on normalized answer scales and converted back to raw numeric answers before scoring.",
            "- Sparse and non-numeric historical items are excluded from the supervised target set.",
        ]
    )
    if test_path is None:
        lines.append("- No local test/template file was detected, so final submission export is waiting on that file.")
    else:
        lines.append(f"- Test file detected: `{test_path}`")
    (output_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def save_core_artifacts(
    output_dir: Path,
    responses: pd.DataFrame,
    questions: pd.DataFrame,
    external_person_features: pd.DataFrame,
    oof: pd.DataFrame,
) -> None:
    responses.to_csv(output_dir / "cleaned_train_long.csv", index=False)
    questions.to_csv(output_dir / "question_features_base.csv", index=False)
    external_person_features.to_csv(output_dir / "respondent_features.csv", index=False)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)


def save_full_training_artifacts(
    output_dir: Path,
    responses: pd.DataFrame,
    questions: pd.DataFrame,
    external_person_features: pd.DataFrame,
    ensemble_weights: Dict[str, float],
    model_names: Sequence[str],
) -> None:
    full_train_responses = responses.merge(
        questions[["question_id", "survey_name"]], on="question_id", how="left"
    )
    full_target_rows = responses[["person_id", "question_id"]].copy()
    full_bundle = assemble_feature_bundle(
        train_responses=full_train_responses,
        target_rows=full_target_rows,
        all_questions=questions,
        external_person_features=external_person_features,
    )
    full_bundle.dynamic_person_features.to_csv(output_dir / "respondent_history_features.csv", index=False)
    full_bundle.question_features.to_csv(output_dir / "question_features_full.csv", index=False)
    model_objects = fit_model_objects(full_bundle)
    joblib.dump(
        {
            "models": model_objects,
            "ensemble_weights": ensemble_weights,
            "model_names": list(model_names),
            "question_meta": questions,
        },
        output_dir / "models" / "trained_models.joblib",
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "models")

    print(f"Scanning data under {args.data_dir.resolve()}")
    surveys = collect_surveys(args.data_dir)
    responses, questions = build_historical_tables(surveys, min_valid_responses=args.min_valid_responses)
    questions = add_question_text_features(questions)
    external_person_features = encode_external_person_features(parse_persona_texts(args.data_dir / "personas_text"))

    print(
        f"Loaded {responses['person_id'].nunique()} respondents, "
        f"{questions['question_id'].nunique()} usable historical questions, "
        f"and {len(responses):,} training rows."
    )

    oof, comparison, ensemble_weights, model_names = run_group_validation(
        responses=responses,
        questions=questions,
        external_person_features=external_person_features,
        folds=args.folds,
    )
    per_question = question_level_metrics(oof, "ensemble", questions)
    save_core_artifacts(args.output_dir, responses, questions, external_person_features, oof)
    save_full_training_artifacts(
        args.output_dir,
        responses,
        questions,
        external_person_features,
        ensemble_weights,
        model_names,
    )
    test_path = discover_test_file(args.data_dir.parent)
    write_validation_report(args.output_dir, comparison, per_question, ensemble_weights, test_path)

    joblib.dump(
        {
            "ensemble_weights": ensemble_weights,
            "model_names": model_names,
            "question_meta": questions,
        },
        args.output_dir / "models" / "pipeline_metadata.joblib",
    )

    if test_path is None:
        (args.output_dir / "submission_status.txt").write_text(
            "No test/template file detected. Add the test file to the workspace and rerun the pipeline.",
            encoding="utf-8",
        )
        print("No test/template file detected. Validation artifacts were written successfully.")
        return

    print(f"Detected test file at {test_path}")
    raw_test = load_test_table(test_path)
    prediction_questions = infer_test_question_features(raw_test)
    prediction_rows = raw_test[["person_id", "question_id"]].copy()
    submission = fit_full_models_and_predict(
        responses=responses,
        questions=questions,
        external_person_features=external_person_features,
        prediction_rows=prediction_rows,
        prediction_questions=prediction_questions,
        model_names=model_names,
        ensemble_weights=ensemble_weights,
    )

    submission_cols = ["person_id", "question_id", "predicted_answer"]
    if "Respondent_ID" in raw_test.columns:
        submission = submission.rename(columns={"person_id": "Respondent_ID"})
        submission_cols = ["Respondent_ID", "question_id", "predicted_answer"]
    submission[submission_cols].to_csv(args.output_dir / "final_submission.csv", index=False)
    submission.to_csv(args.output_dir / "final_predictions_full.csv", index=False)
    print(f"Wrote submission to {args.output_dir / 'final_submission.csv'}")


if __name__ == "__main__":
    main()
