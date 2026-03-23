import pandas as pd

# Load data
master = pd.read_csv('outputs/master_table.csv', low_memory=False)
patched = pd.read_csv('outputs/person_response_profiles_patched.csv')
gt = pd.read_csv('outputs/jasjyot/ground_truth_scores_from_personas.csv')

users = ['00a1r', '7hiu4', 'k8gd7']

scales = {
    'CRT': {
        'qids': ['QID49', 'QID50', 'QID51', 'QID52', 'QID53', 'QID54', 'QID55'],
        'patched_col': 'crt_score',
        'gt_col': 'CRT'
    },
    'Numeracy': {
        'qids': ['QID43', 'QID44', 'QID45', 'QID46', 'QID47', 'QID48'],
        'patched_col': 'numeracy_score',
        'gt_col': 'Numeracy'
    },
    'Financial Literacy': {
        'qids': ['QID38', 'QID39', 'QID40', 'QID41', 'QID42'],
        'patched_col': 'financial_literacy_score',
        'gt_col': 'Financial Literacy'
    },
    'Crystallized Intel': {
        'qids': [f'QID{i}' for i in range(63, 73)] + [f'QID{i}' for i in range(74, 84)],
        'patched_col': 'vocabulary_total_score',
        'gt_col': 'Crystallized Intel'
    },
    'Wason': {
        'qids': ['QID221'],
        'patched_col': 'wason_correct',
        'gt_col': 'Wason'
    },
    'Beck Anxiety': {
        'qids': ['QID125'], # Note: BAI items usually have same parent QID in master_table.csv
        'patched_col': 'bai_sum_score',
        'gt_col': 'Beck Anxiety'
    }
}

lines = []
for scale_name, info in scales.items():
    lines.append(f"============================================================")
    lines.append(f"SCALE: {scale_name}")
    lines.append(f"============================================================")
    
    for uid in users:
        # Get Patch value
        p_row = patched[patched['person_id'] == uid]
        p_val = p_row.iloc[0][info['patched_col']] if not p_row.empty else "N/A"
        
        # Get GT value
        g_row = gt[gt['person_id'] == uid]
        g_val = g_row.iloc[0][info['gt_col']] if not g_row.empty else "N/A"
        
        lines.append(f"\nUser: {uid} | GT: {g_val} | Patched CSV: {p_val}")
        lines.append(f"Raw answers:")
        
        # Get raw answers
        if scale_name == 'Beck Anxiety':
            m_rows = master[(master['person_id'] == uid) & (master['parent_question_id'] == 'QID125')]
        else:
            m_rows = master[(master['person_id'] == uid) & (master['question_id'].isin(info['qids']))]
            
        if m_rows.empty:
            lines.append("  [No answers found]")
        else:
            for _, row in m_rows.iterrows():
                qid = row['question_id']
                ans_text = row['answer_text'] if not pd.isna(row['answer_text']) else "NaN"
                ans_pos = row['answer_position'] if not pd.isna(row['answer_position']) else "NaN"
                options = row['options'] if not pd.isna(row['options']) else ""
                lines.append(f"  {qid}: text='{ans_text}', pos={ans_pos}, options='{options}'")

with open('outputs/jasjyot/model1_score_repair_diagnosis.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
    
print("Diagnosis complete. Saved to outputs/jasjyot/model1_score_repair_diagnosis.txt")
