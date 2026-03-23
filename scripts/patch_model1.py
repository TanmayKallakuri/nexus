import re

with open('scripts/build_person_profiles.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Output Files
code = code.replace('person_response_profiles.csv', 'person_response_profiles_repaired.csv')
code = code.replace('person_response_profiles_data_dictionary.csv', 'person_response_profiles_repaired_data_dictionary.csv')

# 2. Key dictionaries
old_crt = """CRT_ANSWERS = {
    "QID49": "5",       # 5 minutes for 100 machines
    "QID50": "0.05",    # ball costs 5 cents
    "QID51": "47",      # 47 days for half the lake
    "QID52": "emily",   # Emily's father's third daughter
    "QID53": "0",       # no dirt in a hole
    "QID54": "2",       # 2nd place after passing 2nd
    "QID55": "8",       # 8 sheep left
}"""
new_crt = """CRT_ANSWERS = {
    "QID52": "emily",
    "QID53": "0",
    "QID54": "2",
    "QID55": "8",
}"""
code = code.replace(old_crt, new_crt)

old_num = """NUMERACY_ANSWERS = {
    "QID43": "never",   # credit card minimum payment — never paid off
    "QID44": "167",     # 1000/6 ≈ 167 (accept 166-167)
    "QID45": "10",      # 1% of BIG BUCKS = 10 winners per 1000
    "QID46": "20",      # 20 out of 100 = 20%
    "QID47": "0.1",     # 1/1000 = 0.1%
    "QID48": "100",     # 10% of 1000 = 100
}"""
new_num = """NUMERACY_ANSWERS = {
    "QID44": "500",
    "QID45": "10",
    "QID46": "20",
    "QID47": "0.1",
    "QID48": "100",
    "QID49": "5",
    "QID50": "0.05",
    "QID51": "47",
}"""
code = code.replace(old_num, new_num)

old_fin = """FINANCIAL_LITERACY_ANSWERS = {
    "QID38": "Stocks",       # more return than savings/bonds over long period
    "QID39": "True",         # interest rate + inflation understanding
    "QID40": "Decreases",    # diversification reduces risk
    "QID41": "Stocks",       # stocks give highest return over 10-20 years
    "QID42": "True",         # must withdraw from 401k after 70.5
}"""
new_fin = """FINANCIAL_LITERACY_ANSWERS = {
    "QID36": "true",
    "QID37": "less than today with the money in this account",
    "QID38": "stocks",
    "QID39": "true",
    "QID40": "decreases",
    "QID41": "stocks",
    "QID42": "false",
    "QID43": "never",
}"""
code = code.replace(old_fin, new_fin)

# 3. Helper tweaks (remove the hardcoded text matches since check_numeracy_correct uses fuzzy but our strings are exact now)
code = re.sub(r'# Special case: QID43 \(never\) — text match(.*?)return 1\.0 if "never" in ans else 0\.0', 
              r'# QID43 handled below', code, flags=re.DOTALL)
code = re.sub(r'# QID44 — accept 166 or 167(.*?)return 0\.0',
              r'# QID44 handled normally below', code, flags=re.DOTALL)

# 4. BAI logic
old_bai = """        if parent_qid == "QID125":
            features["bai_sum_score"] = float(valid["pos_float"].sum())"""
new_bai = """        if parent_qid == "QID125":
            features["bai_sum_score"] = float((valid["pos_float"] - 1).sum())"""
code = code.replace(old_bai, new_bai)

# 5. Wason logic
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
code = code.replace(old_wason, new_wason)

# 6. Remove vocab and spatial code
vocab_regex = r'# Vocabulary — Synonyms.*?vocabulary_total_score"] = np\.nan'
code = re.sub(vocab_regex, '# Vocabulary excluded (unresolved ground truth)', code, flags=re.DOTALL)

spatial_regex = r'# Spatial reasoning.*?features\["spatial_n_attempted"\] = len\(spatial_positions\)'
code = re.sub(spatial_regex, '# Spatial excluded', code, flags=re.DOTALL)

# Also remove them from exact text matches
code = code.replace('if ans == correct_text.lower():', 'if correct_text.lower() in ans:')

with open('scripts/build_person_profiles.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Patch applied.")
