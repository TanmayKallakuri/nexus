from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import run_unseen_question_pipeline as pipeline


ROOT = Path('.')
DATA_DIR = ROOT / 'data'
DEFAULT_INPUT = ROOT / 'final_test_questions.json'
DEFAULT_OUTPUT = ROOT / 'artifacts' / 'final_test_ml_predictions.json'
DEFAULT_DEBUG = ROOT / 'artifacts' / 'final_test_ml_predictions_debug.csv'
DEFAULT_SUMMARY = ROOT / 'artifacts' / 'final_test_ml_predictions_summary.csv'

MODEL_NAMES = ['semantic_prior', 'ridge', 'lightgbm']
BASE_WEIGHTS = {'semantic_prior': 0.0, 'ridge': 0.0, 'lightgbm': 1.0}

EMPLOYED = {'Full-time employment', 'Part-time employment', 'Self-employed'}
INACTIVE = {'Unemployed', 'Student', 'Home-maker'}
IDEOLOGY_MAP = {
    'Very liberal': -2.0,
    'Liberal': -1.0,
    'Moderate': 0.0,
    'Conservative': 1.0,
    'Very conservative': 2.0,
}
PARTY_MAP = {
    'Democrat': -1.0,
    'Independent': 0.0,
    'Something else': 0.2,
    'Republican': 1.0,
}
EDUCATION_MAP = {
    'Less than high school': 0.0,
    'High school graduate': 1.0,
    'Some college, no degree': 2.0,
    "Associate's degree": 3.0,
    'College graduate/some postgrad': 4.0,
    'Postgraduate': 5.0,
}
INCOME_MAP = {
    'Less than $30,000': 0.0,
    '$30,000-$50,000': 1.0,
    '$50,000-$75,000': 2.0,
    '$75,000-$100,000': 3.0,
    '$100,000 or more': 4.0,
}
SOURCE_TRUST_BASE = {
    'the funny times': 25.0,
    'the national enquirer': 18.0,
    'bbc news': 82.0,
    'pbsnews': 85.0,
    'the wall street journal': 76.0,
    'reddit.com': 44.0,
    'the economist': 84.0,
    'quora.com': 34.0,
    'pbs news': 85.0,
    'apnews.com': 84.0,
    'bbcnews': 82.0,
    'reddit': 44.0,
    'quora': 34.0,
}
SOURCE_QID_MAP = {
    'the funny times': 'T77',
    'the national enquirer': 'T78',
    'bbc news': 'T79',
    'bbcnews': 'T79',
    'the wall street journal': 'T80',
    'reddit.com': 'T81',
    'reddit': 'T81',
    'the economist': 'T82',
    'quora.com': 'T83',
    'quora': 'T83',
    'pbs news': 'T84',
    'pbsnews': 'T84',
}

SOURCE_ALIASES = {
    'bbcnews': 'bbc news',
    'pbsnews': 'pbs news',
    'reddit': 'reddit.com',
    'quora': 'quora.com',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build family-aware submission predictions.')
    parser.add_argument('--input-json', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--debug-csv', type=Path, default=DEFAULT_DEBUG)
    parser.add_argument('--summary-csv', type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument('--enable-ranking-residual', action='store_true')
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_float(value, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except Exception:
        return default


def clip_round(value: float, lo: float, hi: float) -> int:
    return int(round(float(np.clip(value, lo, hi))))


def bounded(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def lower_join(*parts: object) -> str:
    joined = ' '.join('' if p is None else str(p) for p in parts)
    return ' '.join(joined.lower().split())


def option_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [text]


def find_option_index(options: object, keywords: Iterable[str]) -> Optional[int]:
    texts = [str(opt).lower() for opt in option_list(options)]
    needles = [kw.lower() for kw in keywords]
    for idx, text in enumerate(texts, start=1):
        if any(needle in text for needle in needles):
            return idx
    return None


def option_upper_bound(options: object) -> int:
    if not isinstance(options, list):
        return 100
    texts = option_list(options)
    return len(texts) if texts else 100


def option_lower_bound(options: object) -> int:
    if not isinstance(options, list):
        return 0
    return 1 if option_list(options) else 0


def has_dk_option(options: object) -> int:
    return int(find_option_index(options, ["don't know", 'cant remember', "can't remember", "can't choose", 'depends']) is not None)


def parse_like_count(text: object) -> float:
    if text is None:
        return 0.0
    match = re.search(r'([0-9][0-9,]*)\s+likes', str(text), flags=re.IGNORECASE)
    if not match:
        return 0.0
    return float(match.group(1).replace(',', ''))


def parse_content_type(text: object) -> str:
    lower = str(text or '').lower()
    if 'content type: entertaining' in lower:
        return 'entertaining'
    if 'content type: informative' in lower:
        return 'informative'
    return ''


def parse_political_lean(text: object) -> str:
    lower = str(text or '').lower()
    if 'political lean: liberal' in lower:
        return 'liberal'
    if 'political lean: conservative' in lower:
        return 'conservative'
    return ''


def parse_headline_title(text: object) -> str:
    raw = str(text or '')
    match = re.search(r'headline:\s*"([^"]+)"', raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    source_suffix = re.search(
        r'^(.*?)(?:\s+(?:THE FUNNY TIMES|THE NATIONAL ENQUIRER|BBC NEWS|PBS NEWS|THE WALL STREET JOURNAL|THE ECONOMIST|REDDIT\.COM|QUORA\.COM|APNEWS\.COM))\s*$',
        raw.strip(),
        flags=re.IGNORECASE,
    )
    if source_suffix:
        return source_suffix.group(1).strip()
    upper_chunks = re.split(r'\s+[A-Z][A-Z.\s]+$', raw.strip())
    return upper_chunks[0].strip()


def source_in_text(text: str) -> str:
    lower = text.lower()
    normalized = re.sub(r'[^a-z0-9]+', '', lower)
    for source in SOURCE_TRUST_BASE:
        if source in lower:
            return SOURCE_ALIASES.get(source, source)
    for source in SOURCE_TRUST_BASE:
        source_norm = re.sub(r'[^a-z0-9]+', '', source)
        if source_norm and source_norm in normalized:
            return SOURCE_ALIASES.get(source, source)
    return ''


def person_signal_table(raw_persona: pd.DataFrame) -> pd.DataFrame:
    df = raw_persona.copy()
    for col in [
        'political_views', 'political_affiliation', 'education_level', 'income', 'employment_status',
        'gender', 'race', 'geographic_region', 'religion', 'marital_status', 'age_midpoint'
    ]:
        if col not in df.columns:
            df[col] = np.nan

    score_defaults = {
        'score_extraversion': 3.0,
        'score_agreeableness': 3.5,
        'wave1_score_conscientiousness': 3.5,
        'score_openness': 3.5,
        'score_neuroticism': 3.0,
        'score_needforcognition': 3.0,
        'score_GREEN': 3.0,
        'score_selfmonitor': 3.0,
        'score_needforclosure': 3.0,
        'score_finliteracy': 5.0,
        'score_numeracy': 4.0,
        'score_anxiety': 10.0,
        'score_depression': 10.0,
        'crt2_score': 3.0,
        'score_trustgame_sender': 50.0,
        'score_trustgame_receiver': 50.0,
    }
    for col, default in score_defaults.items():
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].fillna(default)

    df['ideology_score'] = df['political_views'].map(IDEOLOGY_MAP).fillna(0.0)
    df['party_score'] = df['political_affiliation'].map(PARTY_MAP).fillna(0.0)
    df['education_score'] = df['education_level'].map(EDUCATION_MAP).fillna(2.0)
    df['income_score'] = df['income'].map(INCOME_MAP).fillna(1.5)
    df['age_midpoint'] = df['age_midpoint'].fillna(39.5)
    df['age_2020'] = df['age_midpoint'] - 6.0
    df['age_norm'] = ((df['age_midpoint'] - 45.0) / 20.0).clip(-1.5, 1.5)
    df['education_norm'] = (df['education_score'] - 2.5) / 2.0
    df['income_norm'] = (df['income_score'] - 2.0) / 2.0

    df['employed_score'] = df['employment_status'].isin(EMPLOYED).astype(float)
    df['inactive_score'] = df['employment_status'].isin(INACTIVE).astype(float)
    df['retired_score'] = (df['employment_status'] == 'Retired').astype(float)
    df['student_score'] = (df['employment_status'] == 'Student').astype(float)
    df['female_score'] = (df['gender'] == 'Female').astype(float)
    df['male_score'] = (df['gender'] == 'Male').astype(float)
    df['black_score'] = (df['race'] == 'Black').astype(float)
    df['hispanic_score'] = (df['race'] == 'Hispanic').astype(float)
    df['white_score'] = (df['race'] == 'White').astype(float)
    df['minority_score'] = ((df['white_score'] == 0) & df['race'].notna()).astype(float)
    df['south_score'] = df['geographic_region'].fillna('').str.contains('South', regex=False).astype(float)
    df['urban_proxy'] = df['geographic_region'].fillna('').str.contains('Northeast|West', regex=True).astype(float)
    df['religious_score'] = (~df['religion'].fillna('').isin(['Atheist', 'Agnostic', 'Nothing in particular', ''])).astype(float)
    df['married_score'] = df['marital_status'].fillna('').str.contains('Married', regex=False).astype(float)

    ext = df['score_extraversion'] - 3.0
    agr = df['score_agreeableness'] - 3.5
    con = df['wave1_score_conscientiousness'] - 3.5
    opn = df['score_openness'] - 3.5
    neu = df['score_neuroticism'] - 3.0
    nfc = df['score_needforcognition'] - 3.0
    green = df['score_GREEN'] - 3.0
    selfmonitor = df['score_selfmonitor'] - 3.0
    closure = df['score_needforclosure'] - 3.0
    finlit = (df['score_finliteracy'] - 5.0) / 3.0
    numeracy = (df['score_numeracy'] - 4.0) / 2.0
    crt = (df['crt2_score'] - 3.0) / 2.0
    trust_game = ((df['score_trustgame_sender'] - 50.0) + (df['score_trustgame_receiver'] - 50.0)) / 60.0
    anxiety = (df['score_anxiety'] - 10.0) / 10.0
    depression = (df['score_depression'] - 10.0) / 10.0

    df['social_traditionalism'] = (
        0.70 * df['party_score'] + 0.55 * df['ideology_score'] + 0.18 * df['religious_score']
        + 0.12 * df['age_norm'] - 0.15 * opn + 0.08 * closure
    )
    df['progressivism'] = (
        -0.75 * df['party_score'] - 0.60 * df['ideology_score'] + 0.32 * df['minority_score']
        + 0.18 * df['female_score'] + 0.15 * green + 0.10 * agr + 0.10 * opn
    )
    df['economic_security'] = (
        0.45 * df['income_norm'] + 0.30 * df['education_norm'] + 0.25 * df['employed_score']
        + 0.18 * df['married_score'] - 0.18 * anxiety - 0.12 * depression
    )
    df['civic_engagement'] = (
        0.35 * df['age_norm'] + 0.28 * df['education_norm'] + 0.18 * con
        + 0.12 * df['religious_score'] + 0.12 * nfc
    )
    df['institutional_trust'] = 0.35 * agr + 0.25 * trust_game + 0.15 * con - 0.10 * neu
    df['media_skepticism'] = (
        0.50 * df['party_score'] + 0.40 * df['ideology_score'] + 0.12 * df['south_score']
        - 0.16 * df['education_norm'] - 0.18 * nfc - 0.12 * crt
    )
    df['digital_engagement'] = -0.35 * df['age_norm'] + 0.24 * ext + 0.18 * selfmonitor + 0.12 * opn - 0.10 * con
    df['source_literacy'] = 0.30 * df['education_norm'] + 0.25 * nfc + 0.20 * finlit + 0.15 * numeracy + 0.15 * crt
    df['humor_sharing_taste'] = 0.30 * ext + 0.18 * opn + 0.18 * df['digital_engagement'] - 0.14 * nfc
    return df


def infer_question_family(row: pd.Series) -> str:
    qid = str(row.get('question_id', ''))
    text = lower_join(row.get('context'), row.get('question_text'))
    specific = {
        'T45': 'share_attribute_importance_headline', 'T46': 'share_attribute_importance_source',
        'T47': 'share_attribute_importance_content_type', 'T48': 'share_attribute_importance_political_lean',
        'T49': 'share_attribute_importance_likes', 'T50': 'accuracy_attribute_importance_headline',
        'T51': 'accuracy_attribute_importance_source', 'T52': 'accuracy_attribute_importance_content_type',
        'T53': 'accuracy_attribute_importance_political_lean', 'T54': 'accuracy_attribute_importance_likes',
        'T55': 'truth_vs_entertainment_share', 'T56': 'share_news_binary', 'T57': 'headline_funny',
        'T58': 'share_likelihood_article', 'T59': 'share_likelihood_article', 'T60': 'share_likelihood_article',
        'T61': 'share_likelihood_article', 'T62': 'share_likelihood_article', 'T63': 'share_likelihood_article',
        'T64': 'share_likelihood_article', 'T65': 'share_likelihood_article', 'T66': 'share_likelihood_article',
        'T67': 'share_likelihood_article', 'T68': 'share_likelihood_article', 'T69': 'share_likelihood_article',
        'T70': 'share_likelihood_article', 'T71': 'share_likelihood_article', 'T72': 'share_likelihood_article',
        'T73': 'share_likelihood_article', 'T74': 'share_accuracy_norm', 'T75': 'truth_vs_entertainment_info',
        'T76': 'share_news_binary', 'T77': 'source_trust_100', 'T78': 'source_trust_100', 'T79': 'source_trust_100',
        'T80': 'source_trust_100', 'T81': 'source_trust_100', 'T82': 'source_trust_100', 'T83': 'source_trust_100',
        'T84': 'source_trust_100',
    }
    if qid in specific:
        return specific[qid]
    if 'headline:' in text and 'source:' in text and 'number of likes:' in text:
        return 'share_scenario_matrix'
    if 'best describes the area where you currently live' in text:
        return 'urbanicity'
    if 'own your home' in text or 'pay rent' in text:
        return 'housing_tenure'
    if 'business or a farm' in text:
        return 'business_farm'
    if 'enrolled in a high school, college, or university' in text:
        return 'student_status'
    if 'did you do any work for either pay or profit' in text:
        return 'work_status'
    if 'volunteer activities' in text:
        return 'volunteering'
    if 'your own health, in general' in text:
        return 'self_rated_health'
    if 'receive poorer service than other people' in text:
        return 'service_discrimination'
    if 'look for health or medical information' in text:
        return 'internet_health_info'
    if 'present financial situation' in text:
        return 'financial_satisfaction'
    if 'lose your job or be laid off' in text:
        return 'layoff_risk'
    if 'most people can be trusted' in text:
        return 'generalized_trust'
    if 'major companies' in text:
        return 'confidence_companies'
    if 'running the press' in text:
        return 'confidence_press'
    if 'family life suffers when the woman has a full-time job' in text:
        return 'gender_roles'
    if 'immigrants take jobs away' in text:
        return 'immigration_jobs'
    if 'preference in hiring and promotion of black people' in text:
        return 'affirmative_action'
    if 'government in washington should do everything possible' in text:
        return 'government_responsibility'
    if 'higher taxes to improve the level of health care' in text:
        return 'taxes_healthcare'
    if 'hard work' in text and 'lucky breaks' in text:
        return 'hard_work_vs_luck'
    if 'take advantage of you' in text and 'try to be fair' in text:
        return 'fairness_vs_advantage'
    if 'highest level of education your father completed' in text:
        return 'father_education'
    if 'companies are using the data they collect online' in text:
        return 'data_privacy'
    if 'tiktok' in text and 'ban' in text:
        return 'tiktok_ban'
    if 'whether or not you voted in that election' in text:
        return 'turnout_2020'
    if 'did you vote for joe biden or donald trump' in text:
        return 'vote_choice_2020'
    if 'how trustworthy (from 0-100%) do you think this source is' in text:
        return 'source_trust_100'
    if 'how likely would you be to share it' in text:
        return 'share_likelihood_article'
    return 'generic'


def sanitize_prediction_questions(question_df: pd.DataFrame) -> pd.DataFrame:
    out = question_df.copy()
    is_list_scale = out['option_count'].fillna(0) >= 2
    out.loc[is_list_scale, 'scale_min'] = 1.0
    out.loc[is_list_scale, 'scale_max'] = out.loc[is_list_scale, 'option_count'].astype(float)
    numeric_text = out['option_text'].fillna('').str.lower()
    numeric_100 = ~is_list_scale & numeric_text.str.contains('0 to 100', regex=False)
    out.loc[numeric_100, 'scale_min'] = 0.0
    out.loc[numeric_100, 'scale_max'] = 100.0
    out['observed_min'] = out['scale_min']
    out['observed_max'] = out['scale_max']
    out['response_range'] = (out['scale_max'] - out['scale_min']).replace(0, 1.0)
    return out


def compute_question_meta(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['question_family'] = out.apply(infer_question_family, axis=1)
    out['has_dk'] = out['options'].apply(has_dk_option)
    out['dk_option_index'] = out['options'].apply(lambda x: find_option_index(x, ["don't know", "can't remember", "can't choose", 'depends']))
    out['nonuser_option_index'] = out['options'].apply(lambda x: find_option_index(x, ["don't use social media"]))
    context_text = out['context'].fillna('') if 'context' in out.columns else pd.Series('', index=out.index)
    question_text = out['question_text'].fillna('') if 'question_text' in out.columns else pd.Series('', index=out.index)
    combined = context_text + ' ' + question_text
    title_text = context_text.where(context_text.str.strip().ne(''), question_text)
    out['source_name'] = combined.map(source_in_text)
    out['like_count'] = combined.map(parse_like_count)
    out['content_type_meta'] = combined.map(parse_content_type)
    out['political_lean_meta'] = combined.map(parse_political_lean)
    out['headline_title'] = title_text.map(parse_headline_title)
    return out


def turnout_score(row: pd.Series) -> float:
    age2020 = safe_float(row['age_2020'], 33.5)
    if age2020 < 18.0:
        return 3.0
    vote_propensity = (
        0.40 + 0.18 * safe_float(row['age_norm']) + 0.12 * safe_float(row['education_norm'])
        + 0.08 * safe_float(row['income_norm']) + 0.12 * safe_float(row['civic_engagement'])
        + 0.08 * (safe_float(row['wave1_score_conscientiousness'], 3.5) - 3.5) - 0.10 * safe_float(row['student_score'])
    )
    if vote_propensity >= 0.54:
        return 1.0
    if age2020 <= 21.0 and vote_propensity < 0.10:
        return 3.0
    return 2.0


def vote_choice_score(row: pd.Series) -> float:
    turnout = turnout_score(row)
    if turnout >= 2.5:
        return 4.0
    partisan = (
        0.95 * safe_float(row['party_score']) + 0.70 * safe_float(row['ideology_score'])
        + 0.10 * safe_float(row['south_score']) + 0.08 * safe_float(row['religious_score'])
        - 0.15 * safe_float(row['minority_score']) - 0.12 * safe_float(row['education_norm'])
        - 0.10 * (safe_float(row['score_openness'], 3.5) - 3.5)
    )
    certainty = abs(safe_float(row['party_score'])) + 0.7 * abs(safe_float(row['ideology_score']))
    if certainty < 0.35 and abs(partisan) < 0.18:
        return 3.0
    return 1.0 if partisan < 0 else 2.0


def source_trust_score(row: pd.Series) -> float:
    source = str(row.get('source_name', ''))
    base = SOURCE_TRUST_BASE.get(source, 55.0)
    trust = base
    trust += 6.0 * safe_float(row['source_literacy'])
    trust += 5.0 * safe_float(row['institutional_trust'])
    trust -= 4.0 * safe_float(row['media_skepticism'])
    if source in {'bbc news', 'pbs news', 'the economist', 'apnews.com', 'bbcnews'}:
        trust -= 5.0 * safe_float(row['social_traditionalism'])
    if source == 'the wall street journal':
        trust += 4.0 * safe_float(row['social_traditionalism'])
    if source in {'reddit.com', 'quora.com', 'reddit', 'quora'}:
        trust -= 8.0 * safe_float(row['source_literacy'])
    return bounded(trust, 0.0, 100.0)


def headline_appeal_score(row: pd.Series) -> float:
    title = str(row.get('headline_title', '')).lower()
    score = 0.0
    if any(term in title for term in ['gold toilet', 'sword fight', 'girlfriend have topped 20,000', 'bathroom']):
        score += 0.55
    if any(term in title for term in ['heavy metal festival', 'spotify launches playlists for dogs', 'stolen solid gold toilet']):
        score += 0.40
    if any(term in title for term in ['bermuda triangle', 'glow-in-the-dark', 'singing spider', 'dinosaur dna', 'rat-sized elephants', 'pink bananas', 'dolphins can be trained']):
        score += 0.30
    if any(term in title for term in ['nasa internship', 'bbc', 'economist']):
        score -= 0.05
    return score


def family_heuristic(row: pd.Series) -> float:
    family = row['question_family']
    trad = safe_float(row['social_traditionalism'])
    econ = safe_float(row['economic_security'])
    trust = safe_float(row['institutional_trust'])
    skeptic = safe_float(row['media_skepticism'])
    digital = safe_float(row['digital_engagement'])
    literacy = safe_float(row['source_literacy'])
    humor = safe_float(row['humor_sharing_taste'])
    age = safe_float(row['age_midpoint'], 39.5)
    age_norm = safe_float(row['age_norm'])
    income = safe_float(row['income_norm'])
    education = safe_float(row['education_norm'])
    employed = safe_float(row['employed_score'])
    inactive = safe_float(row['inactive_score'])
    retired = safe_float(row['retired_score'])
    student = safe_float(row['student_score'])
    female = safe_float(row['female_score'])
    male = safe_float(row['male_score'])
    black = safe_float(row['black_score'])
    minority = safe_float(row['minority_score'])
    south = safe_float(row['south_score'])
    religious = safe_float(row['religious_score'])
    agree = safe_float(row['score_agreeableness'], 3.5) - 3.5
    cons = safe_float(row['wave1_score_conscientiousness'], 3.5) - 3.5
    open_ = safe_float(row['score_openness'], 3.5) - 3.5
    neuro = safe_float(row['score_neuroticism'], 3.0) - 3.0
    nfc = safe_float(row['score_needforcognition'], 3.0) - 3.0
    green = safe_float(row['score_GREEN'], 3.0) - 3.0
    selfmonitor = safe_float(row['score_selfmonitor'], 3.0) - 3.0
    trust_game = ((safe_float(row['score_trustgame_sender'], 50.0) - 50.0) + (safe_float(row['score_trustgame_receiver'], 50.0) - 50.0)) / 60.0

    if family == 'urbanicity':
        return bounded(3.4 - 0.75 * safe_float(row['urban_proxy']) - 0.35 * income - 0.25 * education + 0.08 * age_norm, 1.0, 5.0)
    if family == 'housing_tenure':
        return bounded(2.0 - 0.55 * income - 0.22 * age_norm - 0.18 * safe_float(row['married_score']) + 0.12 * student, 1.0, 3.0)
    if family == 'business_farm':
        if str(row.get('employment_status', '')) == 'Self-employed':
            return 1.0
        return bounded(2.2 - 0.20 * south - 0.10 * age_norm + 0.08 * male, 1.0, 2.0)
    if family == 'student_status':
        return 1.0 if student or age <= 23.0 else 2.0
    if family == 'work_status':
        if retired or age >= 68:
            return 3.0
        if employed:
            return 1.0
        if inactive:
            return 2.0
        return 2.0 + 0.25 * (age < 25)
    if family == 'volunteering':
        return bounded(1.9 - 0.28 * religious - 0.18 * retired - 0.12 * female - 0.10 * agree - 0.08 * safe_float(row['civic_engagement']), 1.0, 2.0)
    if family == 'self_rated_health':
        return bounded(2.1 + 0.55 * age_norm + 0.25 * neuro - 0.18 * income - 0.12 * cons, 1.0, 4.0)
    if family == 'service_discrimination':
        return bounded(5.8 - 1.45 * black - 0.95 * safe_float(row['hispanic_score']) - 0.55 * minority - 0.25 * female + 0.15 * age_norm + 0.18 * income + 0.08 * agree, 1.0, 6.0)
    if family == 'internet_health_info':
        return bounded(4.2 - 0.55 * digital - 0.10 * age_norm - 0.18 * female - 0.10 * neuro, 1.0, 6.0)
    if family == 'financial_satisfaction':
        return bounded(2.1 - 0.45 * econ - 0.12 * employed + 0.22 * max(0.0, neuro) + 0.08 * inactive, 1.0, 3.0)
    if family == 'layoff_risk':
        return bounded(3.0 - 0.18 * employed - 0.18 * econ + 0.12 * student + 0.12 * neuro, 1.0, 4.0)
    if family == 'generalized_trust':
        return bounded(2.0 - 0.35 * trust - 0.22 * trust_game - 0.10 * agree + 0.12 * neuro, 1.0, 3.0)
    if family == 'confidence_companies':
        return bounded(2.15 + 0.10 * trad - 0.15 * trust + 0.08 * econ, 1.0, 3.0)
    if family == 'confidence_press':
        return bounded(2.10 + 0.38 * trad + 0.22 * safe_float(row['party_score']) - 0.16 * trust - 0.08 * literacy + 0.06 * south, 1.0, 3.0)
    if family == 'gender_roles':
        return bounded(3.0 - 0.75 * trad - 0.10 * male - 0.10 * religious + 0.15 * education + 0.08 * open_, 1.0, 5.0)
    if family == 'immigration_jobs':
        return bounded(3.1 - 0.78 * trad - 0.10 * econ + 0.15 * education + 0.10 * minority, 1.0, 5.0)
    if family == 'affirmative_action':
        return bounded(3.0 + 0.55 * trad - 0.75 * black - 0.32 * minority - 0.12 * female - 0.12 * agree - 0.08 * open_, 1.0, 4.0)
    if family == 'government_responsibility':
        return bounded(3.0 + 0.82 * trad + 0.16 * income - 0.28 * minority - 0.12 * green, 1.0, 5.0)
    if family == 'taxes_healthcare':
        return bounded(3.0 + 0.75 * trad + 0.12 * income - 0.18 * minority - 0.10 * agree, 1.0, 5.0)
    if family == 'hard_work_vs_luck':
        return bounded(2.0 - 0.55 * trad + 0.12 * minority - 0.08 * neuro, 1.0, 3.0)
    if family == 'fairness_vs_advantage':
        return bounded(2.0 - 0.28 * trust - 0.22 * trust_game - 0.06 * agree + 0.12 * neuro, 1.0, 3.0)
    if family == 'father_education':
        return bounded(3.0 + 1.00 * education + 0.18 * income - 0.08 * minority - 0.05 * (age >= 60), 1.0, 6.0)
    if family == 'data_privacy':
        return bounded(2.2 - 0.28 * age_norm - 0.12 * digital - 0.22 * skeptic, 1.0, 4.0)
    if family == 'tiktok_ban':
        return bounded(3.0 - 0.52 * trad - 0.12 * age_norm - 0.15 * skeptic + 0.08 * open_, 1.0, 5.0)
    if family == 'turnout_2020':
        return turnout_score(row)
    if family == 'vote_choice_2020':
        return vote_choice_score(row)
    if family == 'share_attribute_importance_headline':
        return bounded(4.1 + 0.25 * digital + 0.12 * humor, 1.0, 7.0)
    if family == 'share_attribute_importance_source':
        return bounded(4.8 + 0.45 * literacy - 0.10 * humor, 1.0, 7.0)
    if family == 'share_attribute_importance_content_type':
        return bounded(4.0 + 0.35 * humor + 0.10 * digital - 0.10 * literacy, 1.0, 7.0)
    if family == 'share_attribute_importance_political_lean':
        return bounded(3.6 + 0.20 * abs(safe_float(row['party_score'])) + 0.08 * safe_float(row['civic_engagement']), 1.0, 7.0)
    if family == 'share_attribute_importance_likes':
        return bounded(3.0 + 0.72 * digital + 0.20 * selfmonitor + 0.12 * female - 0.10 * nfc, 1.0, 7.0)
    if family == 'accuracy_attribute_importance_headline':
        return bounded(4.1 + 0.18 * literacy, 1.0, 7.0)
    if family == 'accuracy_attribute_importance_source':
        return bounded(5.2 + 0.55 * literacy + 0.10 * trust, 1.0, 7.0)
    if family == 'accuracy_attribute_importance_content_type':
        return bounded(3.1 + 0.10 * literacy - 0.10 * humor, 1.0, 7.0)
    if family == 'accuracy_attribute_importance_political_lean':
        return bounded(3.0 + 0.16 * abs(safe_float(row['party_score'])) + 0.08 * literacy, 1.0, 7.0)
    if family == 'accuracy_attribute_importance_likes':
        return bounded(2.4 + 0.10 * digital - 0.18 * literacy, 1.0, 7.0)
    if family == 'truth_vs_entertainment_share':
        return bounded(2.5 + 0.65 * humor + 0.20 * digital - 0.35 * literacy, 1.0, 6.0)
    if family == 'truth_vs_entertainment_info':
        return bounded(2.2 + 0.35 * humor + 0.15 * digital - 0.45 * literacy, 1.0, 6.0)
    if family == 'share_news_binary':
        if digital < -0.55 and age > 62:
            return 3.0
        return bounded(1.9 - 0.42 * digital - 0.12 * humor + 0.10 * literacy, 1.0, 2.0)
    if family == 'share_scenario_matrix':
        likes = safe_float(row.get('like_count'))
        like_boost = 0.0 if likes <= 25 else (0.18 if likes <= 250 else 0.34)
        source_trust = source_trust_score(row) / 100.0
        entertaining = 1.0 if str(row.get('content_type_meta', '')) == 'entertaining' else 0.0
        lean = str(row.get('political_lean_meta', ''))
        lean_match = 0.0
        if lean == 'liberal':
            lean_match = max(0.0, -safe_float(row['party_score']))
        elif lean == 'conservative':
            lean_match = max(0.0, safe_float(row['party_score']))
        headline = headline_appeal_score(row)
        score = 3.0 + 0.65 * humor + 0.35 * digital + 0.30 * source_trust + like_boost
        score += 0.18 * entertaining + 0.12 * lean_match + headline - 0.20 * literacy
        return bounded(score, 1.0, 7.0)
    if family == 'headline_funny':
        return bounded(3.6 + 0.48 * humor + 0.10 * digital - 0.10 * literacy, 1.0, 6.0)
    if family == 'share_likelihood_article':
        source_trust = source_trust_score(row) / 100.0
        headline = headline_appeal_score(row)
        score = 3.0 + 1.05 * humor + 0.40 * digital + 0.28 * source_trust - 0.22 * literacy - 0.12 * nfc + 0.35 * headline
        return bounded(score, 1.0, 6.0)
    if family == 'share_accuracy_norm':
        return bounded(2.0 - 0.45 * literacy - 0.18 * digital + 0.08 * humor, 1.0, 6.0)
    if family == 'source_trust_100':
        return source_trust_score(row)
    return safe_float(row['predicted_answer'], 0.0)


def heuristic_confidence(row: pd.Series) -> float:
    family = row['question_family']
    partisan_strength = abs(safe_float(row['party_score'])) + 0.6 * abs(safe_float(row['ideology_score']))
    economic_strength = abs(safe_float(row['economic_security']))
    digital_strength = abs(safe_float(row['digital_engagement']))
    literacy_strength = abs(safe_float(row['source_literacy']))
    if family in {'work_status', 'father_education', 'turnout_2020', 'vote_choice_2020', 'housing_tenure', 'student_status'}:
        return bounded(0.58 + 0.14 * partisan_strength + 0.10 * economic_strength, 0.35, 0.92)
    if family in {'confidence_press', 'affirmative_action', 'government_responsibility', 'taxes_healthcare', 'immigration_jobs', 'gender_roles', 'tiktok_ban'}:
        return bounded(0.42 + 0.18 * partisan_strength, 0.25, 0.88)
    if family in {'urbanicity', 'generalized_trust', 'fairness_vs_advantage', 'confidence_companies'}:
        return bounded(0.26 + 0.10 * economic_strength, 0.18, 0.45)
    if family in {'share_scenario_matrix', 'share_likelihood_article'}:
        return bounded(0.42 + 0.16 * digital_strength + 0.10 * literacy_strength, 0.24, 0.78)
    if family.startswith('share_') or family.startswith('accuracy_') or family in {'truth_vs_entertainment_share', 'truth_vs_entertainment_info', 'headline_funny', 'source_trust_100'}:
        return bounded(0.32 + 0.15 * digital_strength + 0.12 * literacy_strength, 0.20, 0.75)
    return bounded(0.40 + 0.12 * economic_strength, 0.25, 0.78)


def dk_probability(row: pd.Series) -> float:
    if not safe_float(row['has_dk']):
        return 0.0
    family = row['question_family']
    partisan_strength = abs(safe_float(row['party_score'])) + 0.6 * abs(safe_float(row['ideology_score']))
    low_engagement = max(0.0, -safe_float(row['civic_engagement'])) + max(0.0, -safe_float(row['digital_engagement']))
    low_literacy = max(0.0, -safe_float(row['source_literacy']))
    base = 0.02
    if family in {'turnout_2020', 'vote_choice_2020'}:
        base = 0.04 + 0.10 * low_engagement + 0.06 * max(0.0, 0.3 - partisan_strength)
        age2020 = safe_float(row['age_2020'], 33.5)
        if 18.0 <= age2020 <= 21.0:
            base += 0.05
    elif family in {'confidence_press', 'confidence_companies', 'generalized_trust', 'fairness_vs_advantage'}:
        base = 0.04 + 0.06 * max(0.0, 0.4 - partisan_strength) + 0.04 * low_literacy
    elif family.startswith('share_') or family.startswith('accuracy_') or family in {'source_trust_100'}:
        base = 0.01 + 0.04 * low_engagement + 0.04 * low_literacy
    return bounded(base, 0.0, 0.35)


def blend_weight(row: pd.Series) -> float:
    family = row['question_family']
    conf = heuristic_confidence(row)
    family_base = {
        'work_status': 0.78, 'father_education': 0.64, 'turnout_2020': 0.74, 'vote_choice_2020': 0.82,
        'financial_satisfaction': 0.62, 'service_discrimination': 0.68, 'confidence_press': 0.64,
        'affirmative_action': 0.74, 'government_responsibility': 0.72, 'share_attribute_importance_likes': 0.42,
        'source_trust_100': 0.36,
    }.get(family, 0.46)
    if family in {'urbanicity', 'generalized_trust', 'fairness_vs_advantage', 'confidence_companies'}:
        family_base = 0.18
    if family == 'share_scenario_matrix':
        family_base = 0.42
    if family == 'share_likelihood_article':
        family_base = 0.38
    if family.startswith('share_') or family.startswith('accuracy_') or family in {'truth_vs_entertainment_share', 'truth_vs_entertainment_info', 'headline_funny'}:
        family_base = 0.34
    if family in {'turnout_2020', 'vote_choice_2020'} and abs(safe_float(row['party_score'])) < 0.35 and abs(safe_float(row['ideology_score'])) < 0.5:
        family_base -= 0.10
    return bounded(0.20 + 0.55 * family_base + 0.35 * conf, 0.15, 0.92)


def add_person_block_calibration(out: pd.DataFrame) -> pd.DataFrame:
    pivot = out.pivot(index='person_id', columns='question_id', values='blended_prediction')
    person = pd.DataFrame(index=pivot.index)

    def norm_from_question(qid: str, lo: float, hi: float, invert: bool = False, default: float = 0.5) -> pd.Series:
        if qid not in pivot.columns:
            return pd.Series(default, index=pivot.index)
        values = (pivot[qid] - lo) / max(hi - lo, 1e-6)
        values = values.clip(0.0, 1.0).fillna(default)
        return (1.0 - values) if invert else values

    person['share_headline_pref'] = norm_from_question('T45', 1.0, 7.0)
    person['share_source_pref'] = norm_from_question('T46', 1.0, 7.0)
    person['share_content_pref'] = norm_from_question('T47', 1.0, 7.0)
    person['share_political_pref'] = norm_from_question('T48', 1.0, 7.0)
    person['share_likes_pref'] = norm_from_question('T49', 1.0, 7.0)
    person['acc_headline_pref'] = norm_from_question('T50', 1.0, 7.0)
    person['acc_source_pref'] = norm_from_question('T51', 1.0, 7.0)
    person['acc_content_pref'] = norm_from_question('T52', 1.0, 7.0)
    person['acc_political_pref'] = norm_from_question('T53', 1.0, 7.0)
    person['acc_likes_pref'] = norm_from_question('T54', 1.0, 7.0)
    person['entertainment_pref'] = (
        norm_from_question('T55', 1.0, 6.0) + norm_from_question('T75', 1.0, 6.0)
    ) / 2.0
    person['share_accuracy_commitment'] = norm_from_question('T74', 1.0, 6.0, invert=True)
    person['share_binary_yes'] = (
        norm_from_question('T56', 1.0, 3.0, invert=True) + norm_from_question('T76', 1.0, 3.0, invert=True)
    ) / 2.0

    for source_name, qid in SOURCE_QID_MAP.items():
        col = f"src__{source_name.replace('.', '').replace(' ', '_')}"
        if qid in pivot.columns:
            person[col] = (pivot[qid] / 100.0).clip(0.0, 1.0)
        else:
            person[col] = 0.5

    person = person.reset_index().rename(columns={'index': 'person_id'})
    enriched = out.merge(person, on='person_id', how='left')

    def calibrated_value(row: pd.Series) -> float:
        family = row['question_family']
        current = safe_float(row['blended_prediction'])
        source_key = str(row.get('source_name', '')).replace('.', '').replace(' ', '_')
        source_pref = safe_float(row.get(f'src__{source_key}', np.nan), default=np.nan)
        if np.isnan(source_pref):
            source_pref = source_trust_score(row) / 100.0
        likes = safe_float(row.get('like_count'))
        like_norm = 0.0 if likes <= 25 else (0.45 if likes <= 250 else 1.0)
        entertaining = 1.0 if str(row.get('content_type_meta', '')) == 'entertaining' else 0.0
        informative = 1.0 - entertaining if str(row.get('content_type_meta', '')) else 0.0
        lean = str(row.get('political_lean_meta', ''))
        lean_match = 0.5
        if lean == 'liberal':
            lean_match = bounded(0.5 + 0.35 * max(0.0, -safe_float(row['party_score'])), 0.0, 1.0)
        elif lean == 'conservative':
            lean_match = bounded(0.5 + 0.35 * max(0.0, safe_float(row['party_score'])), 0.0, 1.0)
        headline = bounded(0.5 + headline_appeal_score(row), 0.0, 1.2)

        if family == 'share_scenario_matrix':
            score = 1.8
            score += 1.5 * safe_float(row['share_binary_yes'])
            score += 0.9 * safe_float(row['entertainment_pref'])
            score += 0.7 * safe_float(row['share_headline_pref']) * headline
            score += 0.7 * safe_float(row['share_source_pref']) * source_pref
            score += 0.35 * safe_float(row['share_content_pref']) * entertaining
            score += 0.25 * safe_float(row['share_political_pref']) * lean_match
            score += 0.45 * safe_float(row['share_likes_pref']) * like_norm
            score -= 0.5 * safe_float(row['share_accuracy_commitment']) * (1.0 - source_pref)
            return bounded(0.45 * current + 0.55 * score, 1.0, 7.0)

        if family == 'share_likelihood_article':
            score = 1.4
            score += 1.35 * safe_float(row['share_binary_yes'])
            score += 0.85 * safe_float(row['entertainment_pref']) * headline
            score += 0.90 * safe_float(row['share_source_pref']) * source_pref
            score += 0.30 * safe_float(row['share_content_pref']) * (0.5 * entertaining + 0.2 * informative)
            score += 0.35 * safe_float(row['share_likes_pref']) * like_norm
            score -= 0.45 * safe_float(row['share_accuracy_commitment']) * (1.0 - source_pref)
            return bounded(0.40 * current + 0.60 * score, 1.0, 6.0)

        if family == 'source_trust_100':
            baseline = source_trust_score(row)
            source_weight = 0.55 + 0.20 * safe_float(row['acc_source_pref']) + 0.10 * safe_float(row['acc_headline_pref'])
            score = baseline + 8.0 * (source_pref - 0.5) * source_weight - 6.0 * safe_float(row['entertainment_pref'])
            return bounded(0.55 * current + 0.45 * score, 0.0, 100.0)

        if family == 'share_news_binary':
            score = 1.7 + 0.85 * safe_float(row['share_binary_yes']) + 0.20 * safe_float(row['entertainment_pref'])
            return bounded(0.55 * current + 0.45 * score, 1.0, 3.0)

        return current

    enriched['blended_prediction'] = enriched.apply(calibrated_value, axis=1)
    return enriched


def add_ranking_residuals(out: pd.DataFrame) -> pd.DataFrame:
    enriched = out.copy()
    if 'raw_model_prediction' not in enriched.columns:
        return enriched

    family_alpha = {
        'urbanicity': 0.55,
        'housing_tenure': 0.45,
        'business_farm': 0.35,
        'student_status': 0.45,
        'work_status': 0.50,
        'volunteering': 0.35,
        'self_rated_health': 0.35,
        'service_discrimination': 0.55,
        'internet_health_info': 0.40,
        'financial_satisfaction': 0.40,
        'layoff_risk': 0.35,
        'generalized_trust': 0.65,
        'confidence_companies': 0.60,
        'confidence_press': 0.75,
        'gender_roles': 0.70,
        'immigration_jobs': 0.80,
        'affirmative_action': 0.95,
        'government_responsibility': 0.90,
        'taxes_healthcare': 0.80,
        'hard_work_vs_luck': 0.70,
        'fairness_vs_advantage': 0.65,
        'father_education': 0.35,
        'data_privacy': 0.60,
        'tiktok_ban': 0.75,
        'turnout_2020': 0.85,
        'vote_choice_2020': 1.05,
        'share_scenario_matrix': 1.20,
        'share_attribute_importance_headline': 0.70,
        'share_attribute_importance_source': 0.70,
        'share_attribute_importance_content_type': 0.70,
        'share_attribute_importance_political_lean': 0.75,
        'share_attribute_importance_likes': 0.80,
        'accuracy_attribute_importance_headline': 0.65,
        'accuracy_attribute_importance_source': 0.70,
        'accuracy_attribute_importance_content_type': 0.65,
        'accuracy_attribute_importance_political_lean': 0.70,
        'accuracy_attribute_importance_likes': 0.75,
        'truth_vs_entertainment_share': 0.85,
        'truth_vs_entertainment_info': 0.85,
        'share_news_binary': 0.95,
        'headline_funny': 0.90,
        'share_likelihood_article': 1.15,
        'share_accuracy_norm': 0.80,
        'source_trust_100': 1.05,
    }
    family_beta = {
        'share_scenario_matrix': 0.45,
        'share_likelihood_article': 0.35,
        'source_trust_100': 0.50,
        'share_news_binary': 0.55,
    }

    def apply_group(group: pd.DataFrame) -> pd.DataFrame:
        family = str(group['question_family'].iloc[0])
        alpha = family_alpha.get(family, 0.60)
        beta = family_beta.get(family, 0.75)
        calibrated_mean = float(group['blended_prediction'].mean())
        raw_mean = float(group['raw_model_prediction'].mean())
        raw_centered = group['raw_model_prediction'] - raw_mean
        calib_centered = group['blended_prediction'] - calibrated_mean
        raw_std = float(raw_centered.std(ddof=0))
        calib_std = float(calib_centered.std(ddof=0))
        if raw_std > 1e-8 and calib_std > 1e-8:
            raw_centered = raw_centered * min(1.75, max(0.65, calib_std / raw_std))
        elif raw_std <= 1e-8:
            raw_centered = raw_centered * 0.0
        group = group.copy()
        group['blended_prediction'] = calibrated_mean + beta * calib_centered + alpha * raw_centered
        lo = option_lower_bound(group['options'].iloc[0])
        hi = option_upper_bound(group['options'].iloc[0])
        group['blended_prediction'] = group['blended_prediction'].clip(lo, hi)
        return group

    parts = []
    for _, group in enriched.groupby('question_id', sort=False, observed=False):
        parts.append(apply_group(group))
    return pd.concat(parts, ignore_index=True)


def finalize_answer(row: pd.Series) -> int:
    options = row['options']
    family = row['question_family']
    dk_idx = row['dk_option_index']
    nonuser_idx = row['nonuser_option_index']
    if pd.isna(dk_idx):
        dk_idx = None
    if pd.isna(nonuser_idx):
        nonuser_idx = None
    if not option_list(options):
        return clip_round(row['blended_prediction'], 0, 100)
    if family == 'turnout_2020':
        base = clip_round(row['heuristic_prediction'], 1, option_upper_bound(options))
        if dk_idx and row['dk_probability'] >= 0.22 and base in {1, 2}:
            return int(dk_idx)
        return base
    if family == 'vote_choice_2020':
        base = clip_round(row['heuristic_prediction'], 1, option_upper_bound(options))
        if dk_idx and row['dk_probability'] >= 0.22 and base in {1, 2, 3}:
            return int(dk_idx)
        return base
    if family == 'share_news_binary' and nonuser_idx and safe_float(row['digital_engagement']) < -0.55 and safe_float(row['age_midpoint']) > 62:
        return int(nonuser_idx)
    if dk_idx and row['dk_probability'] >= 0.24 and row['heuristic_confidence'] < 0.45:
        return int(dk_idx)
    return clip_round(row['blended_prediction'], option_lower_bound(options), option_upper_bound(options))


def main() -> None:
    args = parse_args()
    ensure_parent(args.output_json)
    ensure_parent(args.debug_csv)
    ensure_parent(args.summary_csv)

    surveys = pipeline.collect_surveys(DATA_DIR)
    responses, questions = pipeline.build_historical_tables(surveys, min_valid_responses=pipeline.MIN_VALID_RESPONSES)
    questions = pipeline.add_question_text_features(questions)
    raw_persona = pipeline.parse_persona_texts(DATA_DIR / 'personas_text')
    external_persona = pipeline.encode_external_person_features(raw_persona)
    persona_signals = person_signal_table(raw_persona)

    target_rows_raw = pd.DataFrame(json.loads(args.input_json.read_text(encoding='utf-8')))
    prediction_questions = pipeline.infer_test_question_features(target_rows_raw)
    prediction_questions = sanitize_prediction_questions(prediction_questions)
    prediction_rows = target_rows_raw[['person_id', 'question_id']].copy()

    base_predictions = pipeline.fit_full_models_and_predict(
        responses=responses,
        questions=questions,
        external_person_features=external_persona,
        prediction_rows=prediction_rows,
        prediction_questions=prediction_questions,
        model_names=MODEL_NAMES,
        ensemble_weights=BASE_WEIGHTS,
    )

    out = target_rows_raw.drop(columns=['predicted_answer'], errors='ignore').merge(
        base_predictions[['person_id', 'question_id', 'predicted_answer']], on=['person_id', 'question_id'], how='left'
    )
    out = out.merge(
        base_predictions[['person_id', 'question_id', 'predicted_answer']].rename(
            columns={'predicted_answer': 'raw_model_prediction'}
        ),
        on=['person_id', 'question_id'],
        how='left',
    )
    out = compute_question_meta(out)
    out = out.merge(persona_signals, on='person_id', how='left')
    out['heuristic_prediction'] = out.apply(family_heuristic, axis=1)
    out['heuristic_confidence'] = out.apply(heuristic_confidence, axis=1)
    out['dk_probability'] = out.apply(dk_probability, axis=1)
    out['heuristic_weight'] = out.apply(blend_weight, axis=1)
    out['blended_prediction'] = (1.0 - out['heuristic_weight']) * out['predicted_answer'] + out['heuristic_weight'] * out['heuristic_prediction']
    out = add_person_block_calibration(out)
    if args.enable_ranking_residual:
        out = add_ranking_residuals(out)
    out['predicted_answer'] = out.apply(finalize_answer, axis=1)

    json_rows = out[['person_id', 'question_id', 'predicted_answer']].to_dict(orient='records')
    args.output_json.write_text(json.dumps(json_rows, indent=2), encoding='utf-8')
    out.to_csv(args.debug_csv, index=False)
    summary = out.groupby(['question_id', 'question_family'])['predicted_answer'].agg(['mean', 'std', 'min', 'max']).reset_index()
    summary.to_csv(args.summary_csv, index=False)
    print(summary.to_string(index=False))
    print(f'\nSaved {len(json_rows)} predictions to {args.output_json}')


if __name__ == '__main__':
    main()
