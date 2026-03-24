from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_family_aware_submission as fam


DEFAULT_ML = Path("artifacts/final_test_ml_predictions.json")
DEFAULT_QUESTION_JSON = Path("final_test_questions.json")
DEFAULT_OUTPUT = Path("artifacts/final_blended_predictions.json")
OVERRIDE_CHOICES = ("none", "media_only", "weak_blocks")
PREFERENCE_QIDS = {f"T{i}" for i in range(45, 58)} | {"T74", "T75", "T76"}
MEDIA_TRUST_QIDS = {f"T{i}" for i in range(77, 85)}

MAINSTREAM_SOURCES = {"bbc news", "bbcnews", "pbs news", "the economist", "the wall street journal"}
PEER_SOURCES = {"reddit.com", "quora.com", "reddit", "quora"}
TABLOID_SOURCES = {"the funny times", "the national enquirer"}


def canonical_source(source: str) -> str:
    return fam.SOURCE_ALIASES.get(source, source)


QID_TO_SOURCE = {qid: canonical_source(source) for source, qid in fam.SOURCE_QID_MAP.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend Claude predictions with ML predictions.")
    parser.add_argument("--ml-json", type=Path, default=DEFAULT_ML)
    parser.add_argument("--claude-json", type=Path, required=True)
    parser.add_argument("--question-json", type=Path, default=DEFAULT_QUESTION_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--override-mode", choices=OVERRIDE_CHOICES, default="weak_blocks")
    return parser.parse_args()


def load_long_json(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(data)
    expected = {"person_id", "question_id", "predicted_answer"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame[["person_id", "question_id", "predicted_answer"]].copy()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def bounded(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def option_bounds(options: object) -> tuple[float, float]:
    if isinstance(options, list):
        return 1.0, float(len(options))
    return 0.0, 100.0


def family_weight(family: str) -> float:
    weights = {
        "urbanicity": 0.35,
        "housing_tenure": 0.35,
        "business_farm": 0.30,
        "student_status": 0.35,
        "work_status": 0.35,
        "volunteering": 0.40,
        "self_rated_health": 0.40,
        "service_discrimination": 0.55,
        "internet_health_info": 0.40,
        "financial_satisfaction": 0.35,
        "layoff_risk": 0.45,
        "generalized_trust": 0.60,
        "confidence_companies": 0.65,
        "confidence_press": 0.80,
        "gender_roles": 0.75,
        "immigration_jobs": 0.80,
        "affirmative_action": 0.85,
        "government_responsibility": 0.85,
        "taxes_healthcare": 0.80,
        "hard_work_vs_luck": 0.70,
        "fairness_vs_advantage": 0.60,
        "father_education": 0.35,
        "data_privacy": 0.70,
        "tiktok_ban": 0.80,
        "turnout_2020": 0.85,
        "vote_choice_2020": 0.90,
        "share_scenario_matrix": 0.85,
        "share_attribute_importance_headline": 0.62,
        "share_attribute_importance_source": 0.60,
        "share_attribute_importance_content_type": 0.62,
        "share_attribute_importance_political_lean": 0.70,
        "share_attribute_importance_likes": 0.68,
        "accuracy_attribute_importance_headline": 0.60,
        "accuracy_attribute_importance_source": 0.62,
        "accuracy_attribute_importance_content_type": 0.58,
        "accuracy_attribute_importance_political_lean": 0.64,
        "accuracy_attribute_importance_likes": 0.60,
        "truth_vs_entertainment_share": 0.74,
        "truth_vs_entertainment_info": 0.72,
        "share_news_binary": 0.70,
        "headline_funny": 0.82,
        "share_likelihood_article": 0.85,
        "share_accuracy_norm": 0.72,
        "source_trust_100": 0.20,
    }
    return weights.get(family, 0.70)


def finalize_prediction(row: pd.Series) -> int:
    lo, hi = option_bounds(row["options"])
    return int(round(min(max(row["blended_prediction"], lo), hi)))


def response_norm(series: pd.Series, lo: float, hi: float, default: float = 0.5) -> pd.Series:
    denom = max(hi - lo, 1e-6)
    values = pd.to_numeric(series, errors="coerce")
    out = ((values - lo) / denom).clip(0.0, 1.0)
    return out.fillna(default)


def mapped_signal(series: pd.Series, mapping: dict[int, float], default: float = 0.5) -> pd.Series:
    rounded = pd.to_numeric(series, errors="coerce").round()
    out = pd.Series(default, index=series.index, dtype=float)
    for key, value in mapping.items():
        out = out.mask(rounded == key, value)
    return out


def zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std < 1e-8:
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std


def question_rank_adjust(
    current: pd.Series,
    derived: pd.Series,
    lo: float,
    hi: float,
    current_weight: float,
    spread_floor_ratio: float,
    gain: float,
) -> pd.Series:
    cur = pd.to_numeric(current, errors="coerce").astype(float)
    der = pd.to_numeric(derived, errors="coerce").astype(float).reindex(cur.index)
    if der.isna().all():
        return cur
    der = der.fillna(float(der.mean()) if not der.dropna().empty else float(cur.mean()))
    cur_mean = float(cur.mean())
    cur_std = float(cur.std(ddof=0))
    target_std = max(cur_std * 1.08, (hi - lo) * spread_floor_ratio)
    combined = gain * zscore(der) + current_weight * zscore(cur)
    comb_std = float(combined.std(ddof=0))
    if comb_std < 1e-8:
        return cur.clip(lo, hi)
    adjusted = cur_mean + target_std * (combined / comb_std)
    return adjusted.clip(lo, hi)


def _group_mean(group: pd.DataFrame, mask: pd.Series, fallback: float) -> float:
    if bool(mask.any()):
        return float(group.loc[mask, "share_norm"].mean())
    return fallback


def build_person_latents(out: pd.DataFrame) -> pd.DataFrame:
    pivot = out.pivot(index="person_id", columns="question_id", values="blended_prediction")
    person = pd.DataFrame(index=pivot.index)

    def get(qid: str, default: float = np.nan) -> pd.Series:
        if qid in pivot.columns:
            return pd.to_numeric(pivot[qid], errors="coerce")
        return pd.Series(default, index=pivot.index, dtype=float)

    trust_others = mapped_signal(get("T12"), {1: 0.90, 2: 0.12, 3: 0.50, 4: 0.45})
    company_conf = mapped_signal(get("T13"), {1: 0.92, 2: 0.56, 3: 0.10, 4: 0.45})
    press_conf = mapped_signal(get("T14"), {1: 0.96, 2: 0.58, 3: 0.06, 4: 0.42})
    fairness = mapped_signal(get("T21"), {1: 0.08, 2: 0.90, 3: 0.50, 4: 0.45})
    vote_con = mapped_signal(get("T26"), {1: 0.05, 2: 0.95, 3: 0.65, 4: 0.50, 5: 0.55})
    gov_con = response_norm(get("T18").clip(upper=5.0), 1.0, 5.0)
    taxes_con = response_norm(get("T19").clip(upper=5.0), 1.0, 5.0)
    tiktok_con = 1.0 - response_norm(get("T24").clip(upper=5.0), 1.0, 5.0)
    truth_share = response_norm(get("T55"), 1.0, 6.0)
    truth_info = response_norm(get("T75"), 1.0, 6.0)
    accuracy_commit = 1.0 - response_norm(get("T74"), 1.0, 6.0)
    share_binary = (
        mapped_signal(get("T56"), {1: 0.95, 2: 0.15, 3: 0.00}) + mapped_signal(get("T76"), {1: 0.95, 2: 0.15, 3: 0.00})
    ) / 2.0
    funny_signal = response_norm(get("T57"), 1.0, 6.0)

    person["trust_others"] = trust_others
    person["company_conf"] = company_conf
    person["press_conf"] = press_conf
    person["fairness"] = fairness
    person["institutional_trust"] = (0.30 * trust_others + 0.20 * company_conf + 0.30 * press_conf + 0.20 * fairness).clip(0.0, 1.0)
    person["conservative_tilt"] = (0.30 * gov_con + 0.25 * taxes_con + 0.20 * tiktok_con + 0.25 * vote_con).clip(0.0, 1.0)
    person["party_strength"] = (person["conservative_tilt"] - 0.5).abs() * 2.0
    person["entertainment_pref_direct"] = (truth_share + truth_info) / 2.0
    person["accuracy_commit_direct"] = accuracy_commit.clip(0.0, 1.0)
    person["share_binary_direct"] = share_binary.clip(0.0, 1.0)
    person["funny_direct"] = funny_signal.clip(0.0, 1.0)

    scenario = out[out["question_family"].isin({"share_scenario_matrix", "share_likelihood_article"})].copy()
    if scenario.empty:
        person = person.fillna(0.5).reset_index().rename(columns={"index": "person_id"})
        return person

    scenario["lo"] = scenario["options"].apply(lambda x: option_bounds(x)[0])
    scenario["hi"] = scenario["options"].apply(lambda x: option_bounds(x)[1])
    denom = (scenario["hi"] - scenario["lo"]).replace(0.0, 1.0)
    scenario["share_norm"] = ((scenario["blended_prediction"] - scenario["lo"]) / denom).clip(0.0, 1.0)
    scenario["source_key"] = scenario["source_name"].fillna("").astype(str).str.lower().map(canonical_source)
    scenario["is_mainstream"] = scenario["source_key"].isin(MAINSTREAM_SOURCES)
    scenario["is_peer"] = scenario["source_key"].isin(PEER_SOURCES)
    scenario["is_tabloid"] = scenario["source_key"].isin(TABLOID_SOURCES)
    scenario["is_lowcred"] = scenario["is_peer"] | scenario["is_tabloid"]
    scenario["high_likes"] = pd.to_numeric(scenario["like_count"], errors="coerce").fillna(0.0) >= 1000.0
    scenario["low_likes"] = pd.to_numeric(scenario["like_count"], errors="coerce").fillna(0.0) <= 25.0
    scenario["entertaining"] = scenario["content_type_meta"].fillna("").eq("entertaining")
    scenario["informative"] = scenario["content_type_meta"].fillna("").eq("informative")
    scenario["liberal"] = scenario["political_lean_meta"].fillna("").eq("liberal")
    scenario["conservative"] = scenario["political_lean_meta"].fillna("").eq("conservative")
    scenario["headline_signal"] = scenario.apply(fam.headline_appeal_score, axis=1)
    scenario["headline_high"] = scenario["headline_signal"] >= 0.45
    scenario["headline_low"] = scenario["headline_signal"] <= 0.15

    behavior_rows: list[dict[str, float | str]] = []
    source_keys = [
        "the funny times",
        "the national enquirer",
        "bbc news",
        "the wall street journal",
        "reddit.com",
        "the economist",
        "quora.com",
        "pbs news",
    ]
    for person_id, group in scenario.groupby("person_id", sort=False):
        overall = float(group["share_norm"].mean())
        mainstream = _group_mean(group, group["is_mainstream"], overall)
        lowcred = _group_mean(group, group["is_lowcred"], overall)
        peer = _group_mean(group, group["is_peer"], overall)
        tabloid = _group_mean(group, group["is_tabloid"], overall)
        high_likes = _group_mean(group, group["high_likes"], overall)
        low_likes = _group_mean(group, group["low_likes"], overall)
        entertaining = _group_mean(group, group["entertaining"], overall)
        informative = _group_mean(group, group["informative"], overall)
        liberal = _group_mean(group, group["liberal"], overall)
        conservative = _group_mean(group, group["conservative"], overall)
        headline_high = _group_mean(group, group["headline_high"], overall)
        headline_low = _group_mean(group, group["headline_low"], overall)
        behavior_rows.append(
            {
                "person_id": person_id,
                "scenario_share_mean": overall,
                "mainstream_mean": mainstream,
                "lowcred_mean": lowcred,
                "peer_mean": peer,
                "tabloid_mean": tabloid,
                "high_likes_mean": high_likes,
                "low_likes_mean": low_likes,
                "entertaining_mean": entertaining,
                "informative_mean": informative,
                "liberal_mean": liberal,
                "conservative_mean": conservative,
                "headline_high_mean": headline_high,
                "headline_low_mean": headline_low,
                **{
                    f"source_mean__{source.replace('.', '').replace(' ', '_')}": _group_mean(
                        group,
                        group["source_key"].eq(source),
                        overall,
                    )
                    for source in source_keys
                },
            }
        )

    behavior = pd.DataFrame(behavior_rows).set_index("person_id")
    person = person.join(behavior, how="left")
    person["scenario_share_mean"] = person["scenario_share_mean"].fillna(0.35)
    person["mainstream_mean"] = person["mainstream_mean"].fillna(person["scenario_share_mean"])
    person["lowcred_mean"] = person["lowcred_mean"].fillna(person["scenario_share_mean"])
    person["peer_mean"] = person["peer_mean"].fillna(person["scenario_share_mean"])
    person["tabloid_mean"] = person["tabloid_mean"].fillna(person["scenario_share_mean"])
    person["high_likes_mean"] = person["high_likes_mean"].fillna(person["scenario_share_mean"])
    person["low_likes_mean"] = person["low_likes_mean"].fillna(person["scenario_share_mean"])
    person["entertaining_mean"] = person["entertaining_mean"].fillna(person["scenario_share_mean"])
    person["informative_mean"] = person["informative_mean"].fillna(person["scenario_share_mean"])
    person["liberal_mean"] = person["liberal_mean"].fillna(person["scenario_share_mean"])
    person["conservative_mean"] = person["conservative_mean"].fillna(person["scenario_share_mean"])
    person["headline_high_mean"] = person["headline_high_mean"].fillna(person["scenario_share_mean"])
    person["headline_low_mean"] = person["headline_low_mean"].fillna(person["scenario_share_mean"])
    source_cols = [c for c in person.columns if c.startswith("source_mean__")]
    person["source_dispersion"] = person[source_cols].std(axis=1, ddof=0).fillna(0.0)

    source_split = (person["mainstream_mean"] - person["lowcred_mean"]).abs()
    likes_split = (person["high_likes_mean"] - person["low_likes_mean"]).abs()
    content_split = (person["entertaining_mean"] - person["informative_mean"]).abs()
    political_split = (person["liberal_mean"] - person["conservative_mean"]).abs()
    headline_split = (person["headline_high_mean"] - person["headline_low_mean"]).abs()

    person["peer_affinity"] = (0.50 + 0.95 * (person["peer_mean"] - person["mainstream_mean"])).clip(0.0, 1.0)
    person["tabloid_affinity"] = (0.50 + 1.10 * (person["tabloid_mean"] - person["mainstream_mean"])).clip(0.0, 1.0)
    person["humor_affinity"] = (
        0.35
        + 0.50 * person["entertaining_mean"]
        + 0.35 * person["headline_high_mean"]
        + 0.25 * person["funny_direct"]
        - 0.20 * person["informative_mean"]
    ).clip(0.0, 1.0)
    person["source_focus"] = (
        0.14 + 0.80 * source_split + 1.35 * person["source_dispersion"] + 0.15 * person["institutional_trust"]
    ).clip(0.0, 1.0)
    person["likes_focus"] = (0.18 + 1.15 * likes_split + 0.18 * person["peer_affinity"]).clip(0.0, 1.0)
    person["content_focus"] = (0.20 + 1.00 * content_split + 0.25 * person["humor_affinity"]).clip(0.0, 1.0)
    person["political_focus"] = (0.12 + 0.95 * political_split + 0.30 * person["party_strength"]).clip(0.0, 1.0)
    person["headline_focus"] = (0.20 + 1.05 * headline_split + 0.20 * person["humor_affinity"]).clip(0.0, 1.0)
    person["entertainment_pref"] = (
        0.55 * person["entertainment_pref_direct"] + 0.30 * person["humor_affinity"] + 0.15 * person["tabloid_affinity"]
    ).clip(0.0, 1.0)
    person["accuracy_orientation"] = (
        0.45 * person["accuracy_commit_direct"] + 0.35 * person["source_focus"] + 0.20 * person["institutional_trust"]
    ).clip(0.0, 1.0)
    person["share_propensity"] = (0.65 * person["scenario_share_mean"] + 0.35 * person["share_binary_direct"]).clip(0.0, 1.0)

    person = person.fillna(0.5).reset_index().rename(columns={"index": "person_id"})
    return person


def media_source_signal(person: pd.DataFrame, source_key: str) -> pd.Series:
    mainstream_trait = (
        0.45 * person["institutional_trust"] + 0.35 * person["source_focus"] + 0.20 * person["accuracy_orientation"]
    ).clip(0.0, 1.0)
    peer_trait = (0.55 * person["peer_affinity"] + 0.25 * person["likes_focus"] + 0.20 * person["share_propensity"]).clip(0.0, 1.0)
    humor_trait = (0.60 * person["entertainment_pref"] + 0.40 * person["tabloid_affinity"]).clip(0.0, 1.0)
    conservative = person["conservative_tilt"]

    source_slug = source_key.replace(".", "").replace(" ", "_")
    direct = person.get(f"source_mean__{source_slug}", person["scenario_share_mean"]).fillna(person["scenario_share_mean"])
    direct_delta = direct - person["scenario_share_mean"]

    if source_key == "the funny times":
        score = 22.0 + 22.0 * (humor_trait - 0.5) + 8.0 * (peer_trait - 0.5) - 18.0 * (mainstream_trait - 0.5) + 42.0 * direct_delta
    elif source_key == "the national enquirer":
        score = 18.0 + 16.0 * (humor_trait - 0.5) + 6.0 * (peer_trait - 0.5) - 22.0 * (mainstream_trait - 0.5) + 40.0 * direct_delta
    elif source_key == "bbc news":
        score = 69.0 + 28.0 * (mainstream_trait - 0.5) - 6.0 * (peer_trait - 0.5) - 8.0 * (conservative - 0.5) + 44.0 * direct_delta
    elif source_key == "pbs news":
        score = 72.0 + 30.0 * (mainstream_trait - 0.5) - 6.0 * (peer_trait - 0.5) - 6.0 * (conservative - 0.5) + 46.0 * direct_delta
    elif source_key == "the economist":
        score = 66.0 + 24.0 * (mainstream_trait - 0.5) - 4.0 * (peer_trait - 0.5) - 3.0 * (conservative - 0.5) + 42.0 * direct_delta
    elif source_key == "the wall street journal":
        score = 63.0 + 20.0 * (mainstream_trait - 0.5) - 3.0 * (peer_trait - 0.5) + 10.0 * (conservative - 0.5) + 40.0 * direct_delta
    elif source_key == "reddit.com":
        score = 38.0 - 8.0 * (mainstream_trait - 0.5) + 22.0 * (peer_trait - 0.5) + 8.0 * (humor_trait - 0.5) + 42.0 * direct_delta
    elif source_key == "quora.com":
        score = 34.0 - 10.0 * (mainstream_trait - 0.5) + 18.0 * (peer_trait - 0.5) + 4.0 * (humor_trait - 0.5) + 40.0 * direct_delta
    else:
        score = 50.0 + 12.0 * (mainstream_trait - 0.5) - 4.0 * (peer_trait - 0.5) + 36.0 * direct_delta
    return score.clip(0.0, 100.0)


def apply_question_calibrations(
    out: pd.DataFrame,
    derived_signals: dict[str, pd.Series],
    calibrations: dict[str, dict[str, float]],
) -> pd.DataFrame:
    out = out.copy()
    for qid, settings in calibrations.items():
        mask = out["question_id"].eq(qid)
        if not bool(mask.any()):
            continue
        current = out.loc[mask].set_index("person_id")["blended_prediction"]
        derived = derived_signals.get(qid)
        if derived is None:
            continue
        lo, hi = option_bounds(out.loc[mask, "options"].iloc[0])
        adjusted = question_rank_adjust(
            current=current,
            derived=derived,
            lo=lo,
            hi=hi,
            current_weight=settings["current"],
            spread_floor_ratio=settings["floor"],
            gain=settings["gain"],
        )
        out.loc[mask, "blended_prediction"] = out.loc[mask, "person_id"].map(adjusted)
    return out


def apply_media_trust_override(
    out: pd.DataFrame,
    person: pd.DataFrame,
    current_mix: float = 0.16,
) -> pd.DataFrame:
    out = out.copy()
    for qid in sorted(MEDIA_TRUST_QIDS, key=lambda x: int(x[1:])):
        mask = out["question_id"].eq(qid)
        if not bool(mask.any()):
            continue
        signal = media_source_signal(person, QID_TO_SOURCE[qid])
        current = out.loc[mask].set_index("person_id")["blended_prediction"]
        base = current_mix * current + (1.0 - current_mix) * signal.reindex(current.index).fillna(float(signal.mean()))
        adjusted = question_rank_adjust(
            current=base,
            derived=signal,
            lo=0.0,
            hi=100.0,
            current_weight=0.04,
            spread_floor_ratio=0.22,
            gain=1.30,
        )
        out.loc[mask, "blended_prediction"] = out.loc[mask, "person_id"].map(adjusted)
    return out


def apply_final_focus_calibration(out: pd.DataFrame, override_mode: str) -> pd.DataFrame:
    if override_mode == "none":
        return out

    person = build_person_latents(out).set_index("person_id")

    derived_signals: dict[str, pd.Series] = {
        "T45": 0.55 * person["headline_focus"] + 0.20 * person["humor_affinity"] + 0.15 * person["share_propensity"] + 0.10 * person["peer_affinity"],
        "T46": 0.50 * person["source_focus"] + 0.20 * person["accuracy_orientation"] + 0.15 * person["institutional_trust"] + 0.15 * (person["source_dispersion"] * 2.0).clip(0.0, 1.0),
        "T47": 0.55 * person["content_focus"] + 0.20 * person["entertainment_pref"] + 0.15 * person["share_propensity"] + 0.10 * person["humor_affinity"],
        "T48": 0.50 * person["political_focus"] + 0.30 * person["party_strength"] + 0.20 * (person["conservative_tilt"] - 0.5).abs() * 2.0,
        "T49": 0.65 * person["likes_focus"] + 0.20 * person["peer_affinity"] + 0.15 * person["share_propensity"],
        "T50": 0.25 + 0.45 * person["headline_focus"] + 0.20 * person["institutional_trust"] - 0.10 * person["entertainment_pref"],
        "T51": 0.28 + 0.46 * person["source_focus"] + 0.26 * person["accuracy_orientation"] + 0.10 * (person["source_dispersion"] * 2.0).clip(0.0, 1.0),
        "T52": 0.20 + 0.25 * person["content_focus"] + 0.25 * person["institutional_trust"] - 0.10 * person["entertainment_pref"],
        "T53": 0.15 + 0.45 * person["political_focus"] + 0.30 * person["party_strength"] + 0.10 * (1.0 - person["institutional_trust"]),
        "T54": 0.10 + 0.50 * person["likes_focus"] + 0.15 * person["peer_affinity"] - 0.15 * person["source_focus"],
        "T55": 0.65 * person["entertainment_pref"] + 0.20 * person["tabloid_affinity"] + 0.15 * person["likes_focus"] - 0.25 * person["accuracy_orientation"],
        "T56": 0.85 * person["share_propensity"] + 0.15 * person["peer_affinity"],
        "T57": 0.60 * person["humor_affinity"] + 0.25 * person["headline_focus"] + 0.15 * person["share_propensity"],
        "T74": 1.0 - (0.55 * person["accuracy_orientation"] + 0.25 * person["source_focus"] + 0.20 * person["institutional_trust"]),
        "T75": 0.55 * person["entertainment_pref"] + 0.15 * person["tabloid_affinity"] + 0.10 * person["likes_focus"] - 0.20 * person["accuracy_orientation"],
        "T76": 0.85 * person["share_propensity"] + 0.15 * person["peer_affinity"],
    }

    for qid, source in QID_TO_SOURCE.items():
        derived_signals[qid] = media_source_signal(person, source)

    preference_calibrations = {
        "T45": {"floor": 0.20, "gain": 1.18, "current": 0.10},
        "T46": {"floor": 0.20, "gain": 1.22, "current": 0.08},
        "T47": {"floor": 0.20, "gain": 1.18, "current": 0.10},
        "T48": {"floor": 0.22, "gain": 1.24, "current": 0.06},
        "T49": {"floor": 0.20, "gain": 1.22, "current": 0.08},
        "T50": {"floor": 0.18, "gain": 1.12, "current": 0.14},
        "T51": {"floor": 0.20, "gain": 1.22, "current": 0.08},
        "T52": {"floor": 0.18, "gain": 1.10, "current": 0.14},
        "T53": {"floor": 0.22, "gain": 1.20, "current": 0.08},
        "T54": {"floor": 0.20, "gain": 1.16, "current": 0.12},
        "T55": {"floor": 0.22, "gain": 1.24, "current": 0.06},
        "T56": {"floor": 0.30, "gain": 1.12, "current": 0.12},
        "T57": {"floor": 0.20, "gain": 1.14, "current": 0.12},
        "T74": {"floor": 0.22, "gain": 1.24, "current": 0.06},
        "T75": {"floor": 0.20, "gain": 1.20, "current": 0.08},
        "T76": {"floor": 0.30, "gain": 1.12, "current": 0.12},
    }

    out = apply_media_trust_override(out, person, current_mix=0.16)
    if override_mode == "weak_blocks":
        out = apply_question_calibrations(out, derived_signals, preference_calibrations)
    return out


def main() -> None:
    args = parse_args()

    ml = load_long_json(args.ml_json).rename(columns={"predicted_answer": "ml_prediction"})
    claude = load_long_json(args.claude_json).rename(columns={"predicted_answer": "claude_prediction"})
    question_rows = pd.DataFrame(json.loads(args.question_json.read_text(encoding="utf-8")))
    question_meta = fam.compute_question_meta(question_rows)
    keep_cols = [
        "person_id",
        "question_id",
        "options",
        "question_family",
        "source_name",
        "like_count",
        "content_type_meta",
        "political_lean_meta",
        "headline_title",
    ]
    meta = question_meta[keep_cols].copy()

    out = meta.merge(ml, on=["person_id", "question_id"], how="left").merge(
        claude, on=["person_id", "question_id"], how="left"
    )
    if out["ml_prediction"].isna().any():
        raise ValueError("Missing ML predictions for some rows.")
    if out["claude_prediction"].isna().any():
        raise ValueError("Missing Claude predictions for some rows.")

    out["claude_weight"] = out["question_family"].map(family_weight)
    out["blended_prediction"] = (
        (1.0 - out["claude_weight"]) * out["ml_prediction"]
        + out["claude_weight"] * out["claude_prediction"]
    )
    out = apply_final_focus_calibration(out, args.override_mode)
    out["predicted_answer"] = out.apply(finalize_prediction, axis=1)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            out[["person_id", "question_id", "predicted_answer"]].to_dict(orient="records"),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved blended predictions to {args.output_json}")


if __name__ == "__main__":
    main()
