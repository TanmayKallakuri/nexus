import pandas as pd
import re
import os

# Load Model 1
profiles = pd.read_csv('outputs/person_response_profiles.csv')

# People to check
check_ids = ['00a1r', '7hiu4', 'k8gd7', 'sw55g', 'ridfr']

output_lines = []
output_lines.append("============================================================")
output_lines.append("SANITY CHECK RESULTS: MODEL 1 COGNITIVE ANSWER KEYS")
output_lines.append("============================================================\n")

all_matched = True
suspect_keys = set()
matched_features = set()

# Map features to the expected column names in Model 1
feature_map = {
    'CRT': 'crt_score',
    'Numeracy': 'numeracy_score',
    'Financial Literacy': 'financial_literacy_score',
    'Beck Anxiety': 'bai_sum_score',
    'Wason': 'wason_correct',
    'Crystallized Intel': 'vocabulary_total_score'
}
# Note: fluid and syllogism features were not built in Model 1, so they won't be checked 
# unless we see columns for them.

for pid in check_ids:
    output_lines.append(f"\nPerson: {pid}")
    output_lines.append(f"{'-'*50}")

    # Read persona text
    txt_path = f'data/personas_text/{pid}_persona.txt'
    if not os.path.exists(txt_path):
        output_lines.append(f"  File not found: {txt_path}")
        all_matched = False
        continue

    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Extract scores from persona text using regex
    ground_truth = {}
    patterns = {
        'CRT': r'crt2_score\s*=\s*(\d+)',
        'Numeracy': r'score_numeracy\s*=\s*(\d+)',
        'Financial Literacy': r'score_finliteracy\s*=\s*(\d+)',
        'Fluid Intelligence': r'score_fluid\s*=\s*(\d+)',
        'Crystallized Intel': r'score_crystallized\s*=\s*(\d+)',
        'Syllogism': r'score_syllogism_merged\s*=\s*(\d+)',
        'Wason': r'score_wason\s*=\s*(\d+)',
        'Beck Anxiety': r'score_anxiety\s*=\s*(\d+)',
        'Beck Depression': r'score_depression\s*=\s*(\d+)',
    }

    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            ground_truth[name] = int(match.group(1))

    # Get Model 1 row
    row = profiles[profiles['person_id'] == pid]
    if row.empty:
        output_lines.append(f"  Not found in Model 1 CSV")
        all_matched = False
        continue
    row = row.iloc[0]

    output_lines.append(f"  Feature              | Persona Text  | Model 1 CSV  | Match?")
    output_lines.append(f"  ---------------------|---------------|---------------|-------")

    for name, gt_val in ground_truth.items():
        m1_col = feature_map.get(name)
        if m1_col and m1_col in row.index:
            m1_val = row[m1_col]
            if pd.isna(m1_val):
                m1_val = "NaN"
                match_status = "MISSING"
                all_matched = False
            else:
                m1_val = round(float(m1_val)) if isinstance(m1_val, float) else int(m1_val)
                diff = abs(m1_val - gt_val)
                if diff <= 1:
                    match_status = "YES"
                    matched_features.add(name)
                else:
                    match_status = f"NO (diff {diff})"
                    all_matched = False
                    suspect_keys.add(name)
            
            output_lines.append(f"  {name:20s} | {str(gt_val):13s} | {str(m1_val):13s} | {match_status}")
        else:
            # Not implemented in Model 1
            pass

output_lines.append(f"\n============================================================")
output_lines.append("SUMMARY")
output_lines.append(f"============================================================")

if all_matched and len(suspect_keys) == 0:
    output_lines.append("Model 1 cognitive scores validated. Safe to use.")
else:
    output_lines.append("MISMATCHES FOUND. Model 1 Answer Keys require fixing before test time!")
    if suspect_keys:
        output_lines.append(f"Suspect Keys: {', '.join(suspect_keys)}")

result_text = '\n'.join(output_lines)
print(result_text)

os.makedirs('outputs/jasjyot', exist_ok=True)
with open('outputs/jasjyot/sanity_check_results.txt', 'w', encoding='utf-8') as f:
    f.write(result_text + '\n')
