import os
import re
import pandas as pd
import glob

# Search data/personas_text for all txt files
persona_dir = os.path.join('data', 'personas_text')
persona_files = glob.glob(os.path.join(persona_dir, '*_persona.txt'))

records = []
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

for file_path in persona_files:
    filename = os.path.basename(file_path)
    person_id = filename.replace('_persona.txt', '')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    record = {'person_id': person_id}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            record[name] = int(match.group(1))
        else:
            record[name] = None
    records.append(record)

df = pd.DataFrame(records)
os.makedirs('outputs/jasjyot', exist_ok=True)
df.to_csv('outputs/jasjyot/ground_truth_scores_from_personas.csv', index=False)
print(f"Extracted ground truth for {len(df)} personas. Saved to outputs/jasjyot/ground_truth_scores_from_personas.csv")
