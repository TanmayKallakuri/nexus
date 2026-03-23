"""
Build Person Response Profiles — Model 1
=========================================
Creates a one-row-per-person engineered feature table from the master_table.csv
produced by build_master_table.py.

Data Sources:
  - outputs/master_table.csv   (125K rows, one per person×question×answer)
  - outputs/unique_questions.csv (531 unique questions, used for reference)

Output:
  - outputs/person_response_profiles.csv   (233 rows × ~60 feature columns)
  - outputs/person_response_profiles_data_dictionary.csv

Feature Groups:
  1. Coverage & completeness       — answer counts per block and overall
  2. Response style                — ordinal tendencies (mean, variance, extremes, midpoint, straightlining)
  3. Personality construct scores  — BFI, BAI, values, self-monitoring, need for closure, etc.
  4. Economic preference features  — game behavior (dictator, trust, ultimatum), risk/loss aversion
  5. Cognitive test scores         — CRT, numeracy, vocabulary, spatial reasoning, financial literacy
  6. Demographics (encoded)        — ordinal positions for demographic items
  7. Behavioral/spending style     — spending habits, tightwad-spendthrift scale

Assumptions:
  - answer_position is 1-indexed for ordinal/categorical items
  - Correct answers for CRT, numeracy, vocabulary, and financial literacy are
    hard-coded from validated psychometric literature
  - Ordinal features are normalized to [0, 1] via answer_position / num_options
  - Missing values are np.nan, never fabricated

How to run:
  python scripts/build_person_profiles.py
"""

import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_TABLE_PATH = PROJECT_ROOT / "outputs" / "master_table.csv"
UNIQUE_Q_PATH = PROJECT_ROOT / "outputs" / "unique_questions.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "person_response_profiles.csv"
DATA_DICT_PATH = PROJECT_ROOT / "outputs" / "person_response_profiles_data_dictionary.csv"


# ============================================================
# CORRECT ANSWER KEYS (from validated psychometric tests)
# ============================================================

# Cognitive Reflection Test (CRT) — QID49-QID55
# Standard CRT items with known correct answers
CRT_ANSWERS = {
    "QID49": "5",       # 5 minutes for 100 machines
    "QID50": "0.05",    # ball costs 5 cents
    "QID51": "47",      # 47 days for half the lake
    "QID52": "emily",   # Emily's father's third daughter
    "QID53": "0",       # no dirt in a hole
    "QID54": "2",       # 2nd place after passing 2nd
    "QID55": "8",       # 8 sheep left
}

# Numeracy items — QID43-QID48
# Berlin Numeracy Test + Schwartz/Woloshin items
NUMERACY_ANSWERS = {
    "QID43": "never",   # credit card minimum payment — never paid off
    "QID44": "167",     # 1000/6 ≈ 167 (accept 166-167)
    "QID45": "10",      # 1% of BIG BUCKS = 10 winners per 1000
    "QID46": "20",      # 20 out of 100 = 20%
    "QID47": "0.1",     # 1/1000 = 0.1%
    "QID48": "100",     # 10% of 1000 = 100
}

# Financial literacy — QID38-QID42 (Lusardi & Mitchell)
FINANCIAL_LITERACY_ANSWERS = {
    "QID38": "Stocks",       # more return than savings/bonds over long period
    "QID39": "True",         # interest rate + inflation understanding
    "QID40": "Decreases",    # diversification reduces risk
    "QID41": "Stocks",       # stocks give highest return over 10-20 years
    "QID42": "True",         # must withdraw from 401k after 70.5
}

# Vocabulary — Synonyms (QID63-72), correct answer positions
# Verified from the option lists in the data
SYNONYM_CORRECT_POS = {
    "QID63": "1",   # CONCUR → acquiesce
    "QID64": "4",   # CONFISCATE → appropriate
    "QID65": "5",   # SOLICIT → beseech
    "QID66": "3",   # FURTIVE → stealthy
    "QID67": "3",   # ASTUTE → sagacious
    "QID68": "1",   # COVET → crave
    "QID69": "3",   # OSCILLATE → vacillate
    "QID70": "5",   # INDOLENT → slothful
    "QID71": "4",   # DISPARITY → incongruity
    "QID72": "3",   # INDIGENT → destitute
}

# Vocabulary — Antonyms (QID74-83), correct answer positions
ANTONYM_CORRECT_POS = {
    "QID74": "1",   # SATED opposite → famished
    "QID75": "5",   # COMPLAISANT opposite → recalcitrant
    "QID76": "3",   # ALOOF opposite → gregarious
    "QID77": "1",   # ABOMINATE opposite → adore
    "QID78": "4",   # VERBOSE opposite → taciturn
    "QID79": "3",   # DEARTH opposite → abundance
    "QID80": "3",   # SPORADIC opposite → incessant
    "QID81": "3",   # CORPULENT opposite → emaciated
    "QID82": "3",   # GERMANE opposite → irrelevant
    "QID83": "5",   # VACUOUS opposite → profound
}

# Wason Card Task — correct answer is A and 7 (positions 1 and 4)
WASON_CORRECT = {1, 4}


# ============================================================
# CONSTRUCT MAPPINGS — parent_question_id → construct name
# ============================================================
PERSONALITY_CONSTRUCTS = {
    "QID25": "bfi_44",
    "QID26": "bfi_18",
    "QID28": "autonomy_cognition",
    "QID29": "personal_values",
    "QID30": "self_monitoring",
    "QID27": "self_monitoring_binary",
    "QID35": "regulatory_focus",
    "QID125": "bai_anxiety",
    "QID232": "self_concept",
    "QID233": "need_uniqueness",
    "QID234": "need_closure",
    "QID235": "personality_traits_5pt",
    "QID236": "maximization",
    "QID237": "uncertainty_beliefs",
    "QID238": "risk_sensitivity_personality",
    "QID239": "self_regulation",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(val):
    """Convert a value to float, returning NaN on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def normalize_text_answer(text):
    """Lowercase and strip a text answer for comparison."""
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


def check_crt_correct(qid, answer_text):
    """Check if a CRT answer is correct, with fuzzy numeric matching."""
    expected = CRT_ANSWERS.get(qid)
    if expected is None or pd.isna(answer_text):
        return np.nan

    ans = normalize_text_answer(answer_text)
    if not ans:
        return np.nan

    # Special case: QID52 (Emily) — text match
    if qid == "QID52":
        return 1.0 if "emily" in ans else 0.0

    # Special case: QID43 (never) — text match
    if qid == "QID43":
        return 1.0 if "never" in ans else 0.0

    # Numeric comparison
    try:
        ans_num = float(ans)
        exp_num = float(expected)
        # Allow small tolerance for float representations
        if abs(ans_num - exp_num) < 0.01:
            return 1.0
        return 0.0
    except ValueError:
        return 0.0


def check_numeracy_correct(qid, answer_text):
    """Check if a numeracy answer is correct, with fuzzy matching."""
    expected = NUMERACY_ANSWERS.get(qid)
    if expected is None or pd.isna(answer_text):
        return np.nan

    ans = normalize_text_answer(answer_text)
    if not ans:
        return np.nan

    # QID43 — text "never"
    if qid == "QID43":
        return 1.0 if "never" in ans else 0.0

    # QID44 — accept 166 or 167
    if qid == "QID44":
        try:
            val = float(ans)
            return 1.0 if 166 <= val <= 167 else 0.0
        except ValueError:
            return 0.0

    # Numeric comparison for the rest
    try:
        ans_num = float(ans)
        exp_num = float(expected)
        if abs(ans_num - exp_num) < 0.01:
            return 1.0
        return 0.0
    except ValueError:
        return 0.0


def count_forward_flow_words(answer_text):
    """Count words in the forward flow task (QID10)."""
    if pd.isna(answer_text):
        return np.nan
    text = str(answer_text)
    # Format is "word N: <word> | word N+1: <word> | ..."
    # Count the number of "word N:" segments
    segments = [s.strip() for s in text.split("|") if s.strip()]
    return len(segments)


def parse_multi_select_positions(val):
    """Parse a multi-select answer_position string like '[1, 3]' into a set of ints."""
    if pd.isna(val):
        return set()
    try:
        parsed = json.loads(str(val))
        if isinstance(parsed, list):
            return set(int(x) for x in parsed)
    except (json.JSONDecodeError, ValueError):
        pass
    return set()


# ============================================================
# FEATURE BUILDING FUNCTIONS
# ============================================================

def build_coverage_features(person_df):
    """Group 1: Coverage & completeness features."""
    features = {}
    features["n_total_answers"] = len(person_df)

    # Count missing answer positions for non-text items
    non_text = person_df[person_df["answer_type"] != "text"]
    features["n_missing_answers"] = int(non_text["answer_position"].isna().sum())

    # Per-block counts
    block_counts = person_df["block_name"].value_counts()
    features["n_answered_personality"] = int(block_counts.get("Personality", 0))
    features["n_answered_economic"] = int(block_counts.get("Economic preferences", 0))
    features["n_answered_cognitive"] = int(block_counts.get("Cognitive tests", 0))
    features["n_answered_demographics"] = int(block_counts.get("Demographics", 0))

    # Answer type counts
    type_counts = person_df["answer_type"].value_counts()
    features["n_ordinal_items"] = int(type_counts.get("ordinal", 0))
    features["n_categorical_items"] = int(type_counts.get("categorical", 0))
    features["n_text_items"] = int(type_counts.get("text", 0))
    features["n_multi_select_items"] = int(type_counts.get("multi_select", 0))

    return features


def build_response_style_features(person_df):
    """Group 2: Response style features from ordinal items."""
    features = {}

    ordinal = person_df[person_df["answer_type"] == "ordinal"].copy()
    ordinal["pos_float"] = ordinal["answer_position"].apply(safe_float)
    ordinal["num_opts"] = ordinal["num_options"].apply(safe_float)

    # Filter to valid rows
    valid = ordinal.dropna(subset=["pos_float", "num_opts"])
    valid = valid[valid["num_opts"] > 0]

    if len(valid) == 0:
        features["ordinal_mean_normalized"] = np.nan
        features["ordinal_std_normalized"] = np.nan
        features["ordinal_midpoint_rate"] = np.nan
        features["ordinal_extreme_rate"] = np.nan
        features["ordinal_acquiescence"] = np.nan
        features["straightlining_max_repeat_rate"] = np.nan
        return features

    # Normalized position: pos / num_options → [0, 1] range (roughly)
    valid = valid.copy()
    valid["norm_pos"] = valid["pos_float"] / valid["num_opts"]

    features["ordinal_mean_normalized"] = float(valid["norm_pos"].mean())
    features["ordinal_std_normalized"] = float(valid["norm_pos"].std())

    # Midpoint rate: position at exact middle
    midpoints = valid["num_opts"].apply(lambda n: (n + 1) / 2.0)
    is_midpoint = (valid["pos_float"] == midpoints)
    features["ordinal_midpoint_rate"] = float(is_midpoint.mean())

    # Extreme rate: position 1 or position == num_options
    is_extreme = (valid["pos_float"] == 1) | (valid["pos_float"] == valid["num_opts"])
    features["ordinal_extreme_rate"] = float(is_extreme.mean())

    # Acquiescence: mean raw position (higher = more agreement for agree/disagree scales)
    features["ordinal_acquiescence"] = float(valid["pos_float"].mean())

    # Straightlining: for each parent question with >2 sub-items,
    # find max fraction of identical responses
    matrix_items = valid[valid["question_type"] == "Matrix"]
    if len(matrix_items) > 0:
        parent_groups = matrix_items.groupby("parent_question_id")
        max_repeat_rates = []
        for _, grp in parent_groups:
            if len(grp) >= 3:  # only meaningful for multi-item scales
                pos_counts = grp["pos_float"].value_counts()
                max_repeat_rate = pos_counts.iloc[0] / len(grp)
                max_repeat_rates.append(max_repeat_rate)
        if max_repeat_rates:
            features["straightlining_max_repeat_rate"] = float(max(max_repeat_rates))
        else:
            features["straightlining_max_repeat_rate"] = np.nan
    else:
        features["straightlining_max_repeat_rate"] = np.nan

    return features


def build_personality_construct_features(person_df):
    """Group 3: Personality construct aggregate scores."""
    features = {}

    ordinal = person_df[person_df["answer_type"] == "ordinal"].copy()
    ordinal["pos_float"] = ordinal["answer_position"].apply(safe_float)
    ordinal["num_opts"] = ordinal["num_options"].apply(safe_float)

    for parent_qid, construct_name in PERSONALITY_CONSTRUCTS.items():
        items = ordinal[ordinal["parent_question_id"] == parent_qid]
        valid = items.dropna(subset=["pos_float", "num_opts"])
        valid = valid[valid["num_opts"] > 0]

        if len(valid) == 0:
            features[f"{construct_name}_mean_norm"] = np.nan
            features[f"{construct_name}_n_items"] = 0
            continue

        valid = valid.copy()
        valid["norm_pos"] = valid["pos_float"] / valid["num_opts"]

        features[f"{construct_name}_mean_norm"] = float(valid["norm_pos"].mean())
        features[f"{construct_name}_n_items"] = int(len(valid))

        # BAI: also compute sum score (standard clinical scoring)
        if parent_qid == "QID125":
            features["bai_sum_score"] = float(valid["pos_float"].sum())

    # Spending habits (categorical MC items QID31-34)
    for qid, name in [("QID31", "spending_scale"), ("QID32", "spending_difficulty"),
                       ("QID33", "spending_anxiety"), ("QID34", "shopper_type")]:
        row = person_df[person_df["question_id"] == qid]
        if len(row) > 0:
            features[name] = safe_float(row.iloc[0]["answer_position"])
        else:
            features[name] = np.nan

    # Weight loss intention (QID148)
    row = person_df[person_df["question_id"] == "QID148"]
    if len(row) > 0:
        features["weight_loss_intention"] = safe_float(row.iloc[0]["answer_position"])
    else:
        features["weight_loss_intention"] = np.nan

    return features


def build_economic_features(person_df):
    """Group 4: Economic preference features."""
    features = {}

    # Dictator game — QID117: how much to send (position = generosity)
    row = person_df[person_df["question_id"] == "QID117"]
    features["dictator_game_position"] = safe_float(row.iloc[0]["answer_position"]) if len(row) > 0 else np.nan

    # Ultimatum game offer — QID224
    row = person_df[person_df["question_id"] == "QID224"]
    features["ultimatum_offer_position"] = safe_float(row.iloc[0]["answer_position"]) if len(row) > 0 else np.nan

    # Trust game — QID118: how much to send back (trust)
    row = person_df[person_df["question_id"] == "QID118"]
    features["trust_game_send_position"] = safe_float(row.iloc[0]["answer_position"]) if len(row) > 0 else np.nan

    # Trust game return rates — QID119-122: mean return position
    return_positions = []
    for qid in ["QID119", "QID120", "QID121", "QID122"]:
        row = person_df[person_df["question_id"] == qid]
        if len(row) > 0:
            val = safe_float(row.iloc[0]["answer_position"])
            if not np.isnan(val):
                return_positions.append(val)
    features["trust_return_mean_position"] = float(np.mean(return_positions)) if return_positions else np.nan

    # Ultimatum rejection tendency — QID225-230: fraction of offers rejected
    # Position 2 = reject in these questions
    reject_count = 0
    total_ultimatum = 0
    for qid in ["QID225", "QID226", "QID227", "QID228", "QID229", "QID230"]:
        row = person_df[person_df["question_id"] == qid]
        if len(row) > 0:
            pos = safe_float(row.iloc[0]["answer_position"])
            if not np.isnan(pos):
                total_ultimatum += 1
                if pos == 2:
                    reject_count += 1
    features["ultimatum_reject_rate"] = reject_count / total_ultimatum if total_ultimatum > 0 else np.nan

    # Silver lining / hedonic framing — QID149-152
    # Person A vs Person B — encode as binary happiness tendency
    happiness_sum = 0
    happiness_count = 0
    for qid in ["QID149", "QID150", "QID151", "QID152"]:
        row = person_df[person_df["question_id"] == qid]
        if len(row) > 0:
            pos = safe_float(row.iloc[0]["answer_position"])
            if not np.isnan(pos):
                happiness_count += 1
                # Position 1 = Person A (the scenario person), position 2 = Person B
                happiness_sum += pos
    features["hedonic_framing_mean"] = happiness_sum / happiness_count if happiness_count > 0 else np.nan

    # Loss aversion lottery — QID84_r1 to QID84_r9
    # Position 1 → safe option, Position 2 → risky option (taking the gamble)
    risky_count = 0
    total_lotteries = 0
    for i in range(1, 10):
        qid = f"QID84_r{i}"
        row = person_df[person_df["question_id"] == qid]
        if len(row) > 0:
            pos = safe_float(row.iloc[0]["answer_position"])
            if not np.isnan(pos):
                total_lotteries += 1
                if pos == 2:
                    risky_count += 1
    features["loss_aversion_risky_choices"] = risky_count if total_lotteries > 0 else np.nan
    features["loss_aversion_risky_rate"] = risky_count / total_lotteries if total_lotteries > 0 else np.nan

    return features


def build_cognitive_features(person_df):
    """Group 5: Cognitive test scores."""
    features = {}

    # CRT score — QID49-55
    crt_correct = 0
    crt_total = 0
    for qid in CRT_ANSWERS:
        row = person_df[person_df["question_id"] == qid]
        if len(row) > 0:
            correct = check_crt_correct(qid, row.iloc[0]["answer_text"])
            if not np.isnan(correct):
                crt_total += 1
                crt_correct += int(correct)
    features["crt_score"] = crt_correct if crt_total > 0 else np.nan
    features["crt_total_attempted"] = crt_total

    # Numeracy score — QID43-48
    num_correct = 0
    num_total = 0
    for qid in NUMERACY_ANSWERS:
        row = person_df[person_df["question_id"] == qid]
        if len(row) > 0:
            correct = check_numeracy_correct(qid, row.iloc[0]["answer_text"])
            if not np.isnan(correct):
                num_total += 1
                num_correct += int(correct)
    features["numeracy_score"] = num_correct if num_total > 0 else np.nan
    features["numeracy_total_attempted"] = num_total

    # Financial literacy — QID38-42
    fin_correct = 0
    fin_total = 0
    for qid, correct_text in FINANCIAL_LITERACY_ANSWERS.items():
        row = person_df[person_df["question_id"] == qid]
        if len(row) > 0:
            ans = normalize_text_answer(row.iloc[0]["answer_text"])
            if ans:
                fin_total += 1
                if ans == correct_text.lower():
                    fin_correct += 1
    features["financial_literacy_score"] = fin_correct if fin_total > 0 else np.nan
    features["financial_literacy_total"] = fin_total

    # Vocabulary — Synonyms: QID63-72
    syn_correct = 0
    syn_total = 0
    for qid, correct_pos in SYNONYM_CORRECT_POS.items():
        row = person_df[person_df["question_id"] == qid]
        if len(row) > 0:
            pos = str(row.iloc[0]["answer_position"]).strip()
            if pos and pos != "nan":
                syn_total += 1
                if pos == correct_pos:
                    syn_correct += 1
    features["vocabulary_synonym_score"] = syn_correct if syn_total > 0 else np.nan
    features["vocabulary_synonym_total"] = syn_total

    # Vocabulary — Antonyms: QID74-83
    ant_correct = 0
    ant_total = 0
    for qid, correct_pos in ANTONYM_CORRECT_POS.items():
        row = person_df[person_df["question_id"] == qid]
        if len(row) > 0:
            pos = str(row.iloc[0]["answer_position"]).strip()
            if pos and pos != "nan":
                ant_total += 1
                if pos == correct_pos:
                    ant_correct += 1
    features["vocabulary_antonym_score"] = ant_correct if ant_total > 0 else np.nan
    features["vocabulary_antonym_total"] = ant_total

    # Combined vocabulary
    if syn_total > 0 or ant_total > 0:
        features["vocabulary_total_score"] = syn_correct + ant_correct
    else:
        features["vocabulary_total_score"] = np.nan

    # Spatial reasoning / pattern completion — QID56-61
    # These are IQ-type items; we store the answer position
    # Without a validated key we just capture the raw positions
    spatial_positions = []
    for qid in ["QID56", "QID57", "QID58", "QID59", "QID60", "QID61"]:
        row = person_df[person_df["question_id"] == qid]
        if len(row) > 0:
            pos = safe_float(row.iloc[0]["answer_position"])
            if not np.isnan(pos):
                spatial_positions.append(pos)
    features["spatial_n_attempted"] = len(spatial_positions)

    # Wason card task — QID221
    row = person_df[person_df["question_id"] == "QID221"]
    if len(row) > 0:
        selected = parse_multi_select_positions(row.iloc[0]["answer_position"])
        features["wason_correct"] = 1.0 if selected == WASON_CORRECT else 0.0
        features["wason_n_selected"] = len(selected)
    else:
        features["wason_correct"] = np.nan
        features["wason_n_selected"] = np.nan

    # Forward flow — QID10: count of words generated
    row = person_df[person_df["question_id"] == "QID10"]
    if len(row) > 0:
        features["forward_flow_word_count"] = count_forward_flow_words(row.iloc[0]["answer_text"])
    else:
        features["forward_flow_word_count"] = np.nan

    return features


def build_demographic_features(person_df):
    """Group 6: Demographic item positions (ordinal encoding)."""
    features = {}

    demo_map = {
        "QID11": "demo_region",
        "QID12": "demo_sex",
        "QID13": "demo_age",
        "QID14": "demo_education",
        "QID15": "demo_race",
        "QID16": "demo_citizenship",
        "QID17": "demo_occupation",
        "QID18": "demo_religion",
        "QID19": "demo_religiosity",
        "QID20": "demo_political_party",
        "QID21": "demo_income",
        "QID22": "demo_political_views",
        "QID23": "demo_household_size",
        "QID24": "demo_employment",
    }

    for qid, col_name in demo_map.items():
        row = person_df[person_df["question_id"] == qid]
        if len(row) > 0:
            features[col_name] = safe_float(row.iloc[0]["answer_position"])
        else:
            features[col_name] = np.nan

    return features


def build_data_dictionary(columns):
    """Build a data dictionary describing each column."""
    descriptions = {
        "person_id": ("identifier", "Unique person identifier", "all"),
        "n_total_answers": ("coverage", "Total number of answered items", "all"),
        "n_missing_answers": ("coverage", "Count of non-text items with missing answer position", "all"),
        "n_answered_personality": ("coverage", "Items answered in Personality block", "Personality"),
        "n_answered_economic": ("coverage", "Items answered in Economic preferences block", "Economic preferences"),
        "n_answered_cognitive": ("coverage", "Items answered in Cognitive tests block", "Cognitive tests"),
        "n_answered_demographics": ("coverage", "Items answered in Demographics block", "Demographics"),
        "n_ordinal_items": ("coverage", "Count of ordinal-type answers", "all"),
        "n_categorical_items": ("coverage", "Count of categorical-type answers", "all"),
        "n_text_items": ("coverage", "Count of text-entry answers", "all"),
        "n_multi_select_items": ("coverage", "Count of multi-select answers", "all"),
        "ordinal_mean_normalized": ("response_style", "Mean of answer_position/num_options across ordinal items", "ordinal"),
        "ordinal_std_normalized": ("response_style", "Std dev of normalized ordinal responses", "ordinal"),
        "ordinal_midpoint_rate": ("response_style", "Fraction of ordinal responses at exact midpoint", "ordinal"),
        "ordinal_extreme_rate": ("response_style", "Fraction of ordinal responses at min or max", "ordinal"),
        "ordinal_acquiescence": ("response_style", "Mean raw ordinal position (agreement tendency)", "ordinal"),
        "straightlining_max_repeat_rate": ("response_style", "Max fraction of identical responses within any matrix question", "ordinal"),
        "bai_sum_score": ("personality", "Beck Anxiety Inventory sum score (QID125, 21 items)", "QID125"),
        "spending_scale": ("personality", "Tightwad-spendthrift self-placement QID31 (1-11 scale)", "QID31"),
        "spending_difficulty": ("personality", "Difficulty controlling spending QID32", "QID32"),
        "spending_anxiety": ("personality", "Anxiety about spending QID33", "QID33"),
        "shopper_type": ("personality", "Shopper type self-identification QID34", "QID34"),
        "weight_loss_intention": ("personality", "Weight loss intention QID148", "QID148"),
        "dictator_game_position": ("economic", "Dictator game offer position QID117", "QID117"),
        "ultimatum_offer_position": ("economic", "Ultimatum game offer position QID224", "QID224"),
        "trust_game_send_position": ("economic", "Trust game send-back position QID118", "QID118"),
        "trust_return_mean_position": ("economic", "Mean trust game return position QID119-122", "QID119-122"),
        "ultimatum_reject_rate": ("economic", "Fraction of ultimatum offers rejected QID225-230", "QID225-230"),
        "hedonic_framing_mean": ("economic", "Mean hedonic framing tendency QID149-152", "QID149-152"),
        "loss_aversion_risky_choices": ("economic", "Count of risky choices in loss aversion lottery QID84", "QID84"),
        "loss_aversion_risky_rate": ("economic", "Rate of risky choices in loss aversion lottery QID84", "QID84"),
        "crt_score": ("cognitive", "Cognitive Reflection Test correct answers (0-7)", "QID49-55"),
        "crt_total_attempted": ("cognitive", "CRT items attempted", "QID49-55"),
        "numeracy_score": ("cognitive", "Numeracy test correct answers (0-6)", "QID43-48"),
        "numeracy_total_attempted": ("cognitive", "Numeracy items attempted", "QID43-48"),
        "financial_literacy_score": ("cognitive", "Financial literacy correct answers (0-5)", "QID38-42"),
        "financial_literacy_total": ("cognitive", "Financial literacy items attempted", "QID38-42"),
        "vocabulary_synonym_score": ("cognitive", "Synonym vocabulary correct (0-10)", "QID63-72"),
        "vocabulary_synonym_total": ("cognitive", "Synonym items attempted", "QID63-72"),
        "vocabulary_antonym_score": ("cognitive", "Antonym vocabulary correct (0-10)", "QID74-83"),
        "vocabulary_antonym_total": ("cognitive", "Antonym items attempted", "QID74-83"),
        "vocabulary_total_score": ("cognitive", "Total vocabulary correct (synonyms+antonyms, 0-20)", "QID63-83"),
        "spatial_n_attempted": ("cognitive", "Spatial reasoning items attempted", "QID56-61"),
        "wason_correct": ("cognitive", "Wason card task correct (1) or not (0)", "QID221"),
        "wason_n_selected": ("cognitive", "Number of cards selected in Wason task", "QID221"),
        "forward_flow_word_count": ("cognitive", "Number of words generated in forward flow task", "QID10"),
    }

    # Add personality construct entries
    for parent_qid, construct_name in PERSONALITY_CONSTRUCTS.items():
        descriptions[f"{construct_name}_mean_norm"] = (
            "personality", f"Mean normalized score for {construct_name} ({parent_qid})", parent_qid
        )
        descriptions[f"{construct_name}_n_items"] = (
            "personality", f"Number of items answered for {construct_name}", parent_qid
        )

    # Add demographics
    demo_names = {
        "demo_region": "US region", "demo_sex": "Sex at birth", "demo_age": "Age bracket",
        "demo_education": "Education level", "demo_race": "Race/origin",
        "demo_citizenship": "US citizenship", "demo_occupation": "Occupation type",
        "demo_religion": "Religion", "demo_religiosity": "Religious service attendance",
        "demo_political_party": "Political party", "demo_income": "Family income bracket",
        "demo_political_views": "Political views", "demo_household_size": "Household size",
        "demo_employment": "Employment status",
    }
    for col, desc in demo_names.items():
        descriptions[col] = ("demographics", f"{desc} (ordinal position encoding)", "Demographics")

    rows = []
    for col in columns:
        if col in descriptions:
            group, desc, source = descriptions[col]
        else:
            group, desc, source = "unknown", col, ""
        rows.append({"column": col, "group": group, "description": desc, "source_qids": source})

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("BUILD PERSON RESPONSE PROFILES — Model 1")
    print("=" * 60)

    # --- Load data ---
    print(f"\n[1/5] Loading master table from {MASTER_TABLE_PATH}...")
    file_size_mb = os.path.getsize(MASTER_TABLE_PATH) / (1024 ** 2)
    print(f"  File size: {file_size_mb:.1f} MB")

    df = pd.read_csv(MASTER_TABLE_PATH, low_memory=False)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Unique persons: {df['person_id'].nunique()}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # --- Build features per person ---
    print(f"\n[2/5] Building features for {df['person_id'].nunique()} persons...")

    person_ids = sorted(df["person_id"].unique())
    all_profiles = []
    errors = []

    for i, pid in enumerate(person_ids):
        person_df = df[df["person_id"] == pid]

        try:
            profile = {"person_id": pid}
            profile.update(build_coverage_features(person_df))
            profile.update(build_response_style_features(person_df))
            profile.update(build_personality_construct_features(person_df))
            profile.update(build_economic_features(person_df))
            profile.update(build_cognitive_features(person_df))
            profile.update(build_demographic_features(person_df))
            all_profiles.append(profile)
        except Exception as e:
            errors.append(f"Error processing {pid}: {str(e)}")
            # Still add a minimal row
            all_profiles.append({"person_id": pid})

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(person_ids)} persons...")

    print(f"  Processed all {len(person_ids)} persons")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"    {e}")

    # --- Build DataFrame ---
    print(f"\n[3/5] Assembling profile DataFrame...")
    profiles_df = pd.DataFrame(all_profiles)
    print(f"  Shape: {profiles_df.shape[0]} rows × {profiles_df.shape[1]} columns")

    # --- Validation ---
    print(f"\n[4/5] Validating...")

    assert len(profiles_df) == len(person_ids), \
        f"Expected {len(person_ids)} rows, got {len(profiles_df)}"
    assert profiles_df["person_id"].nunique() == len(person_ids), \
        "Duplicate person_ids found"

    # Check for all-NaN columns
    all_nan_cols = profiles_df.columns[profiles_df.isna().all()].tolist()
    if all_nan_cols:
        print(f"  WARNING: {len(all_nan_cols)} columns are entirely NaN: {all_nan_cols}")

    # Missing value summary by feature group
    print(f"\n  Missing value rates by group:")
    coverage_cols = [c for c in profiles_df.columns if c.startswith("n_")]
    style_cols = [c for c in profiles_df.columns if c.startswith("ordinal_") or c == "straightlining_max_repeat_rate"]
    personality_cols = [c for c in profiles_df.columns if any(c.startswith(cn) for cn in
        [v for v in PERSONALITY_CONSTRUCTS.values()]) or c in
        ["bai_sum_score", "spending_scale", "spending_difficulty", "spending_anxiety",
         "shopper_type", "weight_loss_intention"]]
    economic_cols = [c for c in profiles_df.columns if c.startswith(("dictator", "ultimatum", "trust", "hedonic", "loss_aversion"))]
    cognitive_cols = [c for c in profiles_df.columns if c.startswith(("crt", "numeracy", "financial", "vocabulary", "spatial", "wason", "forward"))]
    demo_cols = [c for c in profiles_df.columns if c.startswith("demo_")]

    for group_name, group_cols in [("coverage", coverage_cols), ("response_style", style_cols),
                                     ("personality", personality_cols), ("economic", economic_cols),
                                     ("cognitive", cognitive_cols), ("demographics", demo_cols)]:
        if group_cols:
            group_df = profiles_df[group_cols]
            missing_rate = group_df.isna().mean().mean()
            print(f"    {group_name:20s}: {len(group_cols):3d} cols, {missing_rate:.1%} missing")

    # Ordinal normalization range check
    norm_cols = [c for c in profiles_df.columns if "mean_norm" in c]
    for c in norm_cols:
        vals = profiles_df[c].dropna()
        if len(vals) > 0:
            if vals.min() < 0 or vals.max() > 1.1:  # small tolerance above 1
                print(f"  WARNING: {c} has values outside [0, 1]: [{vals.min():.3f}, {vals.max():.3f}]")

    # --- Save ---
    print(f"\n[5/5] Saving outputs...")

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    profiles_df.to_csv(OUTPUT_PATH, index=False)
    output_size = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"  Saved: {OUTPUT_PATH}")
    print(f"  File size: {output_size:.1f} KB")

    # Data dictionary
    dd = build_data_dictionary(profiles_df.columns.tolist())
    dd.to_csv(DATA_DICT_PATH, index=False)
    print(f"  Saved data dictionary: {DATA_DICT_PATH}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"PERSON RESPONSE PROFILES BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Shape: {profiles_df.shape[0]} persons × {profiles_df.shape[1]} features")
    print(f"\n  Feature groups:")
    print(f"    Coverage:       {len(coverage_cols)} columns")
    print(f"    Response style: {len(style_cols)} columns")
    print(f"    Personality:    {len(personality_cols)} columns")
    print(f"    Economic:       {len(economic_cols)} columns")
    print(f"    Cognitive:      {len(cognitive_cols)} columns")
    print(f"    Demographics:   {len(demo_cols)} columns")

    print(f"\n  Sample scores (first 5 persons):")
    sample_cols = ["person_id", "n_total_answers", "ordinal_mean_normalized",
                   "crt_score", "numeracy_score", "vocabulary_total_score",
                   "dictator_game_position", "bai_sum_score"]
    sample_cols = [c for c in sample_cols if c in profiles_df.columns]
    print(profiles_df[sample_cols].head().to_string(index=False))

    return profiles_df


if __name__ == "__main__":
    main()
