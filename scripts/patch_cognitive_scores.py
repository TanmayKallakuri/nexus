"""
Quick fix: Extract correct cognitive scores from persona text files
and patch them into person_response_profiles.csv.

Overwrites the broken Model 1 scored columns with ground truth.

Author: Tanmay
"""

import pandas as pd
import re
import os
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# [1/3] EXTRACT SCORES FROM PERSONA TEXT FILES
# ============================================================
print("=" * 60)
print("[1/3] Extracting scores from persona text files...")
print("=" * 60)

text_dir = os.path.join(PROJECT_ROOT, 'personas_text')
files = sorted(glob.glob(os.path.join(text_dir, '*_persona.txt')))
print(f"  Found {len(files)} persona files")

# All score patterns from the persona text files
patterns = {
    'score_extraversion': r'score_extraversion\s*=\s*([\d.]+)',
    'score_agreeableness': r'score_agreeableness\s*=\s*([\d.]+)',
    'score_conscientiousness_w1': r'wave1_score_conscientiousness\s*=\s*([\d.]+)',
    'score_conscientiousness_w2': r'wave2_score_conscientiousness\s*=\s*([\d.]+)',
    'score_openness': r'score_openness\s*=\s*([\d.]+)',
    'score_neuroticism': r'score_neuroticism\s*=\s*([\d.]+)',
    'score_needforcognition': r'score_needforcognition\s*=\s*([\d.]+)',
    'score_agency': r'score_agency\s*=\s*([\d.]+)',
    'score_communion': r'score_communion\s*=\s*([\d.]+)',
    'score_minimalism': r'score_minimalism\s*=\s*([\d.]+)',
    'score_BES': r'score_BES\s*=\s*([\d.]+)',
    'score_GREEN': r'score_GREEN\s*=\s*([\d.]+)',
    'crt_score': r'crt2_score\s*=\s*([\d.]+)',
    'score_fluid': r'score_fluid\s*=\s*([\d.]+)',
    'score_crystallized': r'score_crystallized\s*=\s*([\d.]+)',
    'score_syllogism': r'score_syllogism_merged\s*=\s*([\d.]+)',
    'score_actual_total': r'score_actual_total\s*=\s*([\d.]+)',
    'score_overconfidence': r'score_overconfidence\s*=\s*([\d.-]+)',
    'score_overplacement': r'score_overplacement\s*=\s*([\d.-]+)',
    'score_ultimatum_sender': r'score_ultimatum_sender\s*=\s*([\d.]+)',
    'score_ultimatum_accepted': r'score_ultimatum_accepted\s*=\s*([\d.]+)',
    'score_mentalaccounting': r'score_mentalaccounting\s*=\s*([\d.]+)',
    'score_socialdesirability': r'score_socialdesirability\s*=\s*([\d.]+)',
    'score_anxiety': r'score_anxiety\s*=\s*([\d.]+)',
    'score_HI': r'score_HI\s*=\s*([\d.]+)',
    'score_HC': r'score_HC\s*=\s*([\d.]+)',
    'score_VI': r'score_VI\s*=\s*([\d.]+)',
    'score_VC': r'score_VC\s*=\s*([\d.]+)',
    'score_finliteracy': r'score_finliteracy\s*=\s*([\d.]+)',
    'score_numeracy': r'score_numeracy\s*=\s*([\d.]+)',
    'score_deductive_certainty': r'score_deductive_certainty\s*=\s*([\d.]+)',
    'score_forwardflow': r'score_forwardflow\s*=\s*([\d.]+)',
    'score_discount': r'score_discount\s*=\s*([\d.-]+)',
    'score_presentbias': r'score_presentbias\s*=\s*([\d.-]+)',
    'score_riskaversion': r'score_riskaversion\s*=\s*([\d.-]+)',
    'score_trustgame_sender': r'score_trustgame_sender\s*=\s*([\d.]+)',
    'score_trustgame_receiver': r'score_trustgame_receiver\s*=\s*([\d.]+)',
    'score_RFS': r'score_RFS\s*=\s*([\d.]+)',
    'score_ST_TW': r'score_ST-TW\s*=\s*([\d.]+)',
    'score_depression': r'score_depression\s*=\s*([\d.]+)',
    'score_CNFU': r'score_CNFU-S\s*=\s*([\d.]+)',
    'score_selfmonitor': r'score_selfmonitor\s*=\s*([\d.]+)',
    'score_SCC': r'score_SCC\s*=\s*([\d.]+)',
    'score_needforclosure': r'score_needforclosure\s*=\s*([\d.]+)',
    'score_maximization': r'score_maximization\s*=\s*([\d.]+)',
    'score_wason': r'score_wason\s*=\s*([\d.]+)',
    'score_dictator_sender': r'score_dictator_sender\s*=\s*([\d.]+)',
}

rows = []
for fpath in files:
    pid = os.path.basename(fpath).replace('_persona.txt', '')
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()

    row = {'person_id': pid}
    for score_name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            row[score_name] = float(match.group(1))
        else:
            row[score_name] = None

    rows.append(row)

ground_truth = pd.DataFrame(rows)
print(f"  Extracted {len(ground_truth)} people x {len(ground_truth.columns)} columns")
print(f"  Score columns: {len(ground_truth.columns) - 1}")

# Show completeness
null_pcts = ground_truth.isnull().mean()
missing = null_pcts[null_pcts > 0]
if len(missing) > 0:
    print(f"\n  Columns with missing values:")
    for col, pct in missing.items():
        print(f"    {col}: {pct*100:.0f}% missing")
else:
    print(f"  All scores extracted successfully (0 missing)")

# ============================================================
# [2/3] SAVE GROUND TRUTH SCORES
# ============================================================
print("\n" + "=" * 60)
print("[2/3] Saving ground truth scores...")
print("=" * 60)

gt_path = os.path.join(PROJECT_ROOT, 'outputs', 'persona_ground_truth_scores.csv')
ground_truth.to_csv(gt_path, index=False)
print(f"  Saved: {gt_path}")

# ============================================================
# [3/3] PATCH person_response_profiles.csv
# ============================================================
print("\n" + "=" * 60)
print("[3/3] Patching person_response_profiles.csv...")
print("=" * 60)

profiles_path = os.path.join(PROJECT_ROOT, 'outputs', 'person_response_profiles.csv')
profiles = pd.read_csv(profiles_path)
print(f"  Original profiles: {profiles.shape}")

# Merge ground truth scores into profiles
# Add all ground truth columns as new columns (prefixed with gt_)
gt_cols = [c for c in ground_truth.columns if c != 'person_id']
gt_renamed = ground_truth.rename(columns={c: f'gt_{c}' for c in gt_cols})

patched = profiles.merge(gt_renamed, on='person_id', how='left')
print(f"  Patched profiles: {patched.shape}")
print(f"  Added {len(gt_cols)} ground truth score columns (prefixed gt_)")

# Save patched version
patched_path = os.path.join(PROJECT_ROOT, 'outputs', 'person_response_profiles_patched.csv')
patched.to_csv(patched_path, index=False)
print(f"  Saved: {patched_path}")

# ============================================================
# VERIFY: Spot check against known values
# ============================================================
print("\n" + "=" * 60)
print("VERIFICATION: Spot check person 00a1r")
print("=" * 60)

p = ground_truth[ground_truth['person_id'] == '00a1r'].iloc[0]
print(f"  CRT:                {p['crt_score']}")
print(f"  Numeracy:           {p['score_numeracy']}")
print(f"  Financial Literacy: {p['score_finliteracy']}")
print(f"  Crystallized:       {p['score_crystallized']}")
print(f"  Wason:              {p['score_wason']}")
print(f"  Beck Anxiety:       {p['score_anxiety']}")
print(f"  Beck Depression:    {p['score_depression']}")
print(f"  Risk Aversion:      {p['score_riskaversion']}")
print(f"  Trust Sender:       {p['score_trustgame_sender']}")
print(f"  Dictator Sender:    {p['score_dictator_sender']}")

print(f"\n  Expected from persona text: CRT=3, Numeracy=7, FinLit=4, Crystal=7, Wason=2, Anxiety=1")

print("\n" + "=" * 60)
print("PATCH COMPLETE!")
print("=" * 60)
print(f"\nFiles created:")
print(f"  outputs/persona_ground_truth_scores.csv     — all 46 scores from persona text")
print(f"  outputs/person_response_profiles_patched.csv — original Model 1 + ground truth columns")
print(f"\nTolendi: use person_response_profiles_patched.csv instead of person_response_profiles.csv")
print(f"The gt_ prefixed columns are the correct scores. Use those for cognitive features.")
