import pandas as pd
import numpy as np
from collections import defaultdict
import json

df = pd.read_csv('outputs/master_table.csv', low_memory=False)
gt = pd.read_csv('outputs/person_response_profiles_patched.csv')

def get_answer_matrix(qids):
    # rows: person_id, cols: qids, vals: answer_text or answer_position
    matrix = {}
    for qid in qids:
        qdf = df[df['question_id'] == qid]
        # map person -> answer
        mapping = {}
        for _, row in qdf.iterrows():
            ans = str(row['answer_text']).strip().lower() if not pd.isna(row['answer_text']) else str(row['answer_position']).strip()
            mapping[row['person_id']] = ans
        matrix[qid] = mapping
    return matrix

# Let's get GT scores
numeracy_gt = dict(zip(gt['person_id'], gt['gt_score_numeracy']))
finlit_gt = dict(zip(gt['person_id'], gt['gt_score_finliteracy']))
vocab_gt = dict(zip(gt['person_id'], gt['gt_score_crystallized']))

# Numeracy candidate questions: Q43-Q55
num_cands = [f'QID{i}' for i in range(43, 56)]
num_mat = get_answer_matrix(num_cands)

# FinLit candidate questions: Q36-Q42
fin_cands = [f'QID{i}' for i in range(36, 43)]
fin_mat = get_answer_matrix(fin_cands)

def find_key(matrix, gt_dict, num_target_items):
    # Find the most frequent "correct" answer for each item that maximizes correlation with GT
    # A simple way: for each item, the answer that correlates highest with the total score is the correct answer
    best_key = {}
    for qid, p_ans in matrix.items():
        ans_scores = defaultdict(list)
        for pid, ans in p_ans.items():
            if pid in gt_dict and not pd.isna(gt_dict[pid]):
                ans_scores[ans].append(gt_dict[pid])
        
        # calculate mean GT score for people who gave each answer
        ans_means = {}
        for ans, scores in ans_scores.items():
            if len(scores) >= 5: # min support
                ans_means[ans] = np.mean(scores)
        
        if ans_means:
            best_ans = max(ans_means.items(), key=lambda x: x[1])[0]
            best_key[qid] = best_ans

    # Now we have a hypothesized best answer for every candidate question.
    # Let's see which subset of candidates perfectly recreates the GT.
    # Actually, we can just look at the item-total correlations.
    
    corrs = []
    for qid in matrix.keys():
        key_ans = best_key.get(qid)
        if not key_ans: continue
        
        x = []
        y = []
        for pid, gt_val in gt_dict.items():
            if pd.isna(gt_val): continue
            ans = matrix[qid].get(pid, "missing")
            x.append(1 if ans == key_ans else 0)
            y.append(gt_val)
        
        if len(set(x)) > 1:
            corr = np.corrcoef(x, y)[0,1]
            corrs.append((qid, key_ans, corr))
            
    corrs.sort(key=lambda x: x[2], reverse=True)
    top_qids = [c[0] for c in corrs[:num_target_items]]
    
    # Verify perfectly
    errors = 0
    for pid, gt_val in gt_dict.items():
        if pd.isna(gt_val): continue
        score = 0
        for qid in top_qids:
            if matrix[qid].get(pid, "missing") == best_key[qid]:
                score += 1
        if score != gt_val:
            errors += 1
            
    return top_qids, {q: best_key[q] for q in top_qids}, errors

print("--- NUMERACY (Target: 8 items) ---")
num_q, num_k, num_e = find_key(num_mat, numeracy_gt, 8)
print(f"Top QIDs: {num_q}")
print(f"Keys: {num_k}")
print(f"Errors: {num_e}")

print("\n--- FINLIT (Target: 8 items) ---")
# Let's add more candidates for FinLit, maybe up to Q30?
fin_cands_ext = [f'QID{i}' for i in range(25, 43)]
fin_mat_ext = get_answer_matrix(fin_cands_ext)
fin_q, fin_k, fin_e = find_key(fin_mat_ext, finlit_gt, 8)
print(f"Top QIDs: {fin_q}")
print(f"Keys: {fin_k}")
print(f"Errors: {fin_e}")

print("\n--- VOCAB (Target: 20 items, Q63-Q83 exclude 73) ---")
voc_cands = [f'QID{i}' for i in range(63, 84) if i != 73]
voc_mat = get_answer_matrix(voc_cands)
voc_q, voc_k, voc_e = find_key(voc_mat, vocab_gt, 20)
print(f"Top QIDs: {voc_q}")
print(f"Keys: {voc_k}")
print(f"Errors: {voc_e}")
