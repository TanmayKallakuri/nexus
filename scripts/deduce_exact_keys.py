import pandas as pd
import numpy as np

master = pd.read_csv('outputs/master_table.csv', low_memory=False)
gt = pd.read_csv('outputs/person_response_profiles_patched.csv')

def find_best_exact_keys(target_col, candidate_qids):
    gt_dict = dict(zip(gt['person_id'], gt[target_col]))
    
    # 1. Gather all unique answers for all candidate QIDs
    unique_ans = {}
    for qid in candidate_qids:
        q_df = master[master['question_id'] == qid]
        answers = {}
        for _, row in q_df.iterrows():
            # If answer is text, use it. Else use position.
            val = row['answer_text'] if not pd.isna(row['answer_text']) else row['answer_position']
            answers[row['person_id']] = str(val).strip().lower()
        unique_ans[qid] = answers
        
    keys = {}
    
    # 2. For each QID, find which answer gives us +1 point (has high mean GT)
    for qid, p_ans in unique_ans.items():
        ans_list = list(set(p_ans.values()))
        best_corr = -1
        best_ans = None
        for cand_ans in ans_list:
            x, y = [], []
            for pid, gt_val in gt_dict.items():
                if pd.isna(gt_val): continue
                usr_ans = p_ans.get(pid, "missing")
                x.append(1 if usr_ans == cand_ans else 0)
                y.append(gt_val)
            if len(set(x)) > 1:
                corr = np.corrcoef(x, y)[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_ans = cand_ans
        # We only keep the key if it has a decent positive correlation
        if best_corr > 0.15:
            keys[qid] = {'answer': best_ans, 'corr': best_corr}
            
    # Remove least correlated if we found too many
    sorted_keys = sorted(keys.items(), key=lambda kv: kv[1]['corr'], reverse=True)
    return sorted_keys

print("--- FINLIT (Target 8) ---")
qids = [f'QID{i}' for i in range(36, 44)]
fk = find_best_exact_keys('gt_score_finliteracy', qids)
for q, meta in fk: print(f"{q}: {meta['answer']} (corr={meta['corr']:.2f})")

print("\n--- NUMERACY (Target 8) ---")
qids = [f'QID{i}' for i in range(43, 52)] # 43-51 are 9 items, let's see which 8
nk = find_best_exact_keys('gt_score_numeracy', qids)
for q, meta in nk: print(f"{q}: {meta['answer']} (corr={meta['corr']:.2f})")

print("\n--- CRT 2 (Target 4) ---")
qids = [f'QID{i}' for i in range(52, 56)]
ck = find_best_exact_keys('gt_crt_score', qids)
for q, meta in ck: print(f"{q}: {meta['answer']} (corr={meta['corr']:.2f})")

print("\n--- VOCAB (Target 20) ---")
qids = [f'QID{i}' for i in range(63, 84) if i != 73]
vk = find_best_exact_keys('gt_score_crystallized', qids)
for q, meta in vk: print(f"{q}: {meta['answer']} (corr={meta['corr']:.2f})")
