import os

with open('scripts/build_person_profiles.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Output files
text = text.replace('person_response_profiles.csv', 'person_response_profiles_repaired.csv')
text = text.replace('person_response_profiles_data_dictionary.csv', 'person_response_profiles_repaired_data_dictionary.csv')

# 2. Dictionaries
old_dicts = """# Cognitive Reflection Test (CRT) — QID49-QID55
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
}"""

new_dicts = """# Cognitive Reflection Test (CRT) — QID49-QID55
# Standard CRT items with known correct answers
CRT_ANSWERS = {
    "QID52": "emily",
    "QID53": "0",
    "QID54": "2",
    "QID55": "8",
}

# Numeracy items — QID43-QID48
# Berlin Numeracy Test + Schwartz/Woloshin items
NUMERACY_ANSWERS = {
    "QID44": "500",
    "QID45": "10",
    "QID46": "20",
    "QID47": "0.1",
    "QID48": "100",
    "QID49": "5",
    "QID50": "0.05",
    "QID51": "47",
}

# Financial literacy — QID38-QID42 (Lusardi & Mitchell)
FINANCIAL_LITERACY_ANSWERS = {
    "QID36": "true",
    "QID37": "less than today with the money in this account",
    "QID38": "stocks",
    "QID39": "true",
    "QID40": "decreases",
    "QID41": "stocks",
    "QID42": "false",
    "QID43": "never",
}"""

text = text.replace(old_dicts, new_dicts)

# 3. Validation functions
old_funcs = """def check_crt_correct(qid, answer_text):
    \"\"\"Check if a CRT answer is correct, with fuzzy numeric matching.\"\"\"
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
    \"\"\"Check if a numeracy answer is correct, with fuzzy matching.\"\"\"
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
        return 0.0"""

new_funcs = """def check_crt_correct(qid, answer_text):
    expected = CRT_ANSWERS.get(qid)
    if expected is None or pd.isna(answer_text):
        return np.nan
    ans = str(answer_text).strip().lower()
    return 1.0 if expected == ans else 0.0

def check_numeracy_correct(qid, answer_text):
    expected = NUMERACY_ANSWERS.get(qid)
    if expected is None or pd.isna(answer_text):
        return np.nan
    ans = str(answer_text).strip().lower()
    return 1.0 if expected == ans else 0.0"""

text = text.replace(old_funcs, new_funcs)

# 4. Finlit string matching adjustment
old_finlit = """                if ans == correct_text.lower():
                    fin_correct += 1"""
new_finlit = """                if correct_text.lower() in ans:
                    fin_correct += 1"""
text = text.replace(old_finlit, new_finlit)

# 5. Wason Task
old_wason = """    # Wason card task — QID221
    row = person_df[person_df["question_id"] == "QID221"]
    if len(row) > 0:
        selected = parse_multi_select_positions(row.iloc[0]["answer_position"])
        features["wason_correct"] = 1.0 if selected == WASON_CORRECT else 0.0
        features["wason_n_selected"] = len(selected)
    else:
        features["wason_correct"] = np.nan
        features["wason_n_selected"] = np.nan"""

new_wason = """    # Wason card task — QID221
    row = person_df[person_df["question_id"] == "QID221"]
    if len(row) > 0:
        selected = parse_multi_select_positions(row.iloc[0]["answer_position"])
        wason_score = 0
        if 1 in selected: wason_score += 1
        if 2 not in selected: wason_score += 1
        if 3 not in selected: wason_score += 1
        if 4 in selected: wason_score += 1
        features["wason_correct"] = float(wason_score)
        features["wason_n_selected"] = len(selected)
    else:
        features["wason_correct"] = np.nan
        features["wason_n_selected"] = np.nan"""

text = text.replace(old_wason, new_wason)

# 6. BAI Anxiety sum (0-indexed instead of 1-indexed)
old_bai = """        if parent_qid == "QID125":
            features["bai_sum_score"] = float(valid["pos_float"].sum())"""
new_bai = """        if parent_qid == "QID125":
            features["bai_sum_score"] = float((valid["pos_float"] - 1).sum())"""
text = text.replace(old_bai, new_bai)

# 7. Exclude Vocab and Spatial logic completely
text = text.replace('features["vocabulary_total_score"] = syn_correct + ant_correct', 'pass # Vocab unresolved')
text = text.replace('features["vocabulary_synonym_score"] = syn_correct if syn_total > 0 else np.nan', '')
text = text.replace('features["vocabulary_antonym_score"] = ant_correct if ant_total > 0 else np.nan', '')

# Ensure we rewrite
with open('scripts/build_person_profiles.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Safely patched.")
