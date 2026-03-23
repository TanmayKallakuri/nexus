"""
Nearest-Question Retrieval Model
For a new question, find the K most similar known questions,
look up this person's actual answers, and predict from those.

No ML training. No leakage risk from answer data.
Produces person-specific predictions with natural variance.

Input:  outputs/master_table.csv, outputs/unique_questions.csv
Output: outputs/knn_predictions_validation.csv
        outputs/knn_validation_summary.json
        outputs/knn_question_embeddings_384.npy
        outputs/knn_response_lookup.pkl

Author: Tanmay
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# ============================================================
# [1/7] LOAD DATA
# ============================================================
print("=" * 60)
print("[1/7] Loading data...")
print("=" * 60)

master = pd.read_csv(os.path.join(OUTPUT_DIR, 'master_table.csv'))
unique_q = pd.read_csv(os.path.join(OUTPUT_DIR, 'unique_questions.csv'))

ordinal = master[master['answer_type'] == 'ordinal'].copy()
ordinal['answer_position'] = pd.to_numeric(ordinal['answer_position'], errors='coerce')
ordinal['num_options'] = pd.to_numeric(ordinal['num_options'], errors='coerce')
ordinal = ordinal.dropna(subset=['answer_position', 'num_options'])
ordinal = ordinal[ordinal['num_options'] > 1]

# Normalize answers to 0-1
ordinal['answer_norm'] = (ordinal['answer_position'] - 1) / (ordinal['num_options'] - 1)

print(f"  Ordinal rows: {len(ordinal)}")
print(f"  Persons: {ordinal['person_id'].nunique()}")
print(f"  Questions: {ordinal['question_id'].nunique()}")

# ============================================================
# [2/7] BUILD RESPONSE LOOKUP
# ============================================================
print("\n" + "=" * 60)
print("[2/7] Building response lookup...")
print("=" * 60)

# Create a fast lookup: (person_id, question_id) -> answer_norm
response_pivot = ordinal.pivot_table(
    index='person_id',
    columns='question_id',
    values='answer_norm',
    aggfunc='first'
)

person_ids = response_pivot.index.tolist()
question_ids = response_pivot.columns.tolist()

print(f"  Response matrix: {response_pivot.shape}")
print(f"  Missing cells: {response_pivot.isnull().sum().sum()}")

# Also build num_options lookup per question
qid_to_numopts = ordinal.drop_duplicates('question_id').set_index('question_id')['num_options'].to_dict()

# Save response lookup
with open(os.path.join(OUTPUT_DIR, 'knn_response_lookup.pkl'), 'wb') as f:
    pickle.dump({
        'response_pivot': response_pivot,
        'qid_to_numopts': qid_to_numopts,
        'person_ids': person_ids,
        'question_ids': question_ids,
    }, f)
print(f"  Saved: outputs/knn_response_lookup.pkl")

# ============================================================
# [3/7] EMBED ALL QUESTIONS (full 384-dim)
# ============================================================
print("\n" + "=" * 60)
print("[3/7] Embedding all questions (384-dim)...")
print("=" * 60)

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Build question texts — use full_question from unique_questions
# Map question_id to full text
qid_to_text = {}
qid_to_block = {}
for _, row in unique_q.iterrows():
    qid = row['question_id']
    text = str(row.get('full_question', ''))
    if text == 'nan' or not text:
        text = str(row.get('question_text', ''))
    qid_to_text[qid] = text
    qid_to_block[qid] = row.get('block_name', '')

# Only embed questions that are in our ordinal set
ordinal_qids = sorted(question_ids)
ordinal_texts = [qid_to_text.get(qid, '') for qid in ordinal_qids]

print(f"  Embedding {len(ordinal_texts)} questions...")
embeddings_384 = embedder.encode(ordinal_texts, show_progress_bar=True, batch_size=64)
print(f"  Embedding shape: {embeddings_384.shape}")

# Normalize embeddings for cosine similarity (so dot product = cosine)
norms = np.linalg.norm(embeddings_384, axis=1, keepdims=True)
norms[norms == 0] = 1  # avoid division by zero
embeddings_normed = embeddings_384 / norms

# Build lookup: qid -> index in embedding matrix
qid_to_idx = {qid: i for i, qid in enumerate(ordinal_qids)}

# Save embeddings
np.save(os.path.join(OUTPUT_DIR, 'knn_question_embeddings_384.npy'), embeddings_384)
with open(os.path.join(OUTPUT_DIR, 'knn_qid_to_idx.pkl'), 'wb') as f:
    pickle.dump({'qid_to_idx': qid_to_idx, 'ordinal_qids': ordinal_qids}, f)
print(f"  Saved: outputs/knn_question_embeddings_384.npy")

# ============================================================
# [4/7] KNN PREDICTION FUNCTION
# ============================================================
print("\n" + "=" * 60)
print("[4/7] Defining KNN prediction function...")
print("=" * 60)

def knn_predict(target_qid, target_embedding_normed, train_qids,
                embeddings_normed, qid_to_idx, response_pivot,
                qid_to_numopts, K=5):
    """
    For each person, predict the answer to target_qid by finding
    K nearest known questions and averaging their answers.

    Args:
        target_qid: the question to predict
        target_embedding_normed: L2-normalized 384-dim embedding
        train_qids: list of question_ids to search (excludes holdout)
        embeddings_normed: (n_questions, 384) normalized matrix
        qid_to_idx: question_id -> row index in embedding matrix
        response_pivot: person_id x question_id -> answer_norm
        qid_to_numopts: question_id -> num_options
        K: number of neighbors

    Returns:
        predictions: dict of person_id -> predicted answer_position (raw scale)
        details: dict of metadata (neighbors found, similarities, etc.)
    """
    # Get indices of training questions
    train_indices = [qid_to_idx[q] for q in train_qids if q in qid_to_idx]
    train_embeddings = embeddings_normed[train_indices]
    train_qid_list = [train_qids[i] if isinstance(train_qids, list)
                      else list(train_qids)[i] for i in range(len(train_indices))]
    # Rebuild the mapping for train-only
    train_qid_list = [q for q in train_qids if q in qid_to_idx]

    if len(train_qid_list) == 0:
        return {}, {}

    # Compute cosine similarities (dot product since both are L2-normed)
    sims = train_embeddings @ target_embedding_normed
    # Get top K
    k_actual = min(K, len(sims))
    top_k_local_idx = np.argsort(sims)[-k_actual:][::-1]
    top_k_qids = [train_qid_list[i] for i in top_k_local_idx]
    top_k_sims = sims[top_k_local_idx]

    # Clamp negative similarities to 0 (don't use anti-correlated questions)
    top_k_sims = np.maximum(top_k_sims, 0.0)

    # If all similarities are 0, fall back to uniform weights
    if top_k_sims.sum() < 1e-10:
        top_k_weights = np.ones(k_actual) / k_actual
    else:
        top_k_weights = top_k_sims / top_k_sims.sum()

    target_numopts = qid_to_numopts.get(target_qid, 5)

    predictions = {}
    for person_id in response_pivot.index:
        # Get this person's normalized answers to the K nearest questions
        neighbor_answers = []
        neighbor_weights = []
        for i, nq in enumerate(top_k_qids):
            if nq in response_pivot.columns:
                val = response_pivot.loc[person_id, nq]
                if not np.isnan(val):
                    neighbor_answers.append(val)
                    neighbor_weights.append(top_k_weights[i])

        if len(neighbor_answers) == 0:
            # No data — predict midpoint
            predictions[person_id] = (target_numopts + 1) / 2
            continue

        neighbor_answers = np.array(neighbor_answers)
        neighbor_weights = np.array(neighbor_weights)
        # Re-normalize weights after dropping missing
        w_sum = neighbor_weights.sum()
        if w_sum < 1e-10:
            neighbor_weights = np.ones(len(neighbor_weights)) / len(neighbor_weights)
        else:
            neighbor_weights = neighbor_weights / w_sum

        # Weighted average in normalized space (0-1)
        pred_norm = np.average(neighbor_answers, weights=neighbor_weights)

        # Scale back to target question's scale
        pred_raw = 1 + pred_norm * (target_numopts - 1)
        pred_raw = np.clip(pred_raw, 1, target_numopts)

        predictions[person_id] = pred_raw

    details = {
        'target_qid': target_qid,
        'top_k_qids': top_k_qids,
        'top_k_sims': top_k_sims.tolist(),
        'top_k_weights': top_k_weights.tolist(),
    }

    return predictions, details

print("  KNN prediction function defined.")

# ============================================================
# [5/7] VALIDATION — LEAVE-OUT-QUESTION
# ============================================================
print("\n" + "=" * 60)
print("[5/7] Running validation (leave-out-question)...")
print("=" * 60)

from scipy.stats import pearsonr

np.random.seed(42)
all_qids = sorted(question_ids)
holdout_size = int(len(all_qids) * 0.2)
holdout_qids = set(np.random.choice(all_qids, size=holdout_size, replace=False))
train_qids = [q for q in all_qids if q not in holdout_qids]

print(f"  Train questions: {len(train_qids)}")
print(f"  Holdout questions: {len(holdout_qids)}")

# Test multiple K values
K_VALUES = [3, 5, 7, 10, 15]
results_by_k = {}

for K in K_VALUES:
    print(f"\n  --- K={K} ---")

    all_predictions = []  # list of (person_id, question_id, actual, predicted, num_options, block)

    for qi, target_qid in enumerate(sorted(holdout_qids)):
        if target_qid not in qid_to_idx:
            continue

        target_idx = qid_to_idx[target_qid]
        target_emb = embeddings_normed[target_idx]

        preds, details = knn_predict(
            target_qid, target_emb, train_qids,
            embeddings_normed, qid_to_idx, response_pivot,
            qid_to_numopts, K=K
        )

        numopts = qid_to_numopts.get(target_qid, 5)
        block = qid_to_block.get(target_qid, '')

        for person_id, pred_val in preds.items():
            actual_norm = response_pivot.loc[person_id, target_qid] if target_qid in response_pivot.columns else np.nan
            if np.isnan(actual_norm):
                continue
            actual_raw = 1 + actual_norm * (numopts - 1)
            all_predictions.append({
                'person_id': person_id,
                'question_id': target_qid,
                'actual': actual_raw,
                'predicted': pred_val,
                'num_options': numopts,
                'block_name': block,
            })

    pred_df = pd.DataFrame(all_predictions)

    # Accuracy = 1 - (MAD / range)
    pred_df['abs_error'] = np.abs(pred_df['predicted'] - pred_df['actual'])
    pred_df['range'] = pred_df['num_options'] - 1
    mad = pred_df['abs_error'].mean()
    avg_range = pred_df['range'].mean()
    accuracy = 1 - (mad / avg_range)

    # Pearson r per question
    correlations = []
    for qid in holdout_qids:
        q_data = pred_df[pred_df['question_id'] == qid]
        if len(q_data) < 3:
            continue
        a = q_data['actual'].values
        p = q_data['predicted'].values
        if np.std(a) > 0 and np.std(p) > 0:
            r, _ = pearsonr(a, p)
            correlations.append(r)

    mean_r = np.mean(correlations) if correlations else 0
    median_r = np.median(correlations) if correlations else 0
    positive_r = sum(1 for r in correlations if r > 0)

    # Variance ratio
    actual_std = pred_df.groupby('question_id')['actual'].std().mean()
    pred_std = pred_df.groupby('question_id')['predicted'].std().mean()
    var_ratio = pred_std / actual_std if actual_std > 0 else 0

    # Per-block
    block_results = {}
    for block in pred_df['block_name'].unique():
        b_data = pred_df[pred_df['block_name'] == block]
        b_qids = b_data['question_id'].unique()
        b_corrs = []
        for qid in b_qids:
            qd = b_data[b_data['question_id'] == qid]
            if len(qd) < 3:
                continue
            a, p = qd['actual'].values, qd['predicted'].values
            if np.std(a) > 0 and np.std(p) > 0:
                r, _ = pearsonr(a, p)
                b_corrs.append(r)
        b_mean_r = np.mean(b_corrs) if b_corrs else 0
        b_mad = b_data['abs_error'].mean()
        b_range = b_data['range'].mean()
        b_acc = 1 - (b_mad / b_range) if b_range > 0 else 0
        block_results[block] = {
            'accuracy': round(b_acc, 4),
            'mean_r': round(b_mean_r, 4),
            'n_questions': len(b_corrs),
        }

    results_by_k[K] = {
        'accuracy': round(accuracy, 4),
        'mean_r': round(mean_r, 4),
        'median_r': round(median_r, 4),
        'var_ratio': round(var_ratio, 4),
        'positive_r': positive_r,
        'total_corr_questions': len(correlations),
        'block_results': block_results,
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Mean Pearson r: {mean_r:.4f}")
    print(f"  Median Pearson r: {median_r:.4f}")
    print(f"  Variance ratio: {var_ratio:.4f}")
    print(f"  Questions with r>0: {positive_r}/{len(correlations)}")
    for block, br in sorted(block_results.items()):
        print(f"    {block}: acc={br['accuracy']:.4f} r={br['mean_r']:.4f} ({br['n_questions']} Qs)")

# ============================================================
# [6/7] FIND BEST K
# ============================================================
print("\n" + "=" * 60)
print("[6/7] Selecting best K...")
print("=" * 60)

print(f"\n  {'K':>3s} | {'Accuracy':>8s} | {'Mean r':>8s} | {'Median r':>8s} | {'Var ratio':>9s} | {'r>0':>4s}")
print(f"  {'-'*3} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*4}")

best_k = None
best_r = -1
for K in K_VALUES:
    res = results_by_k[K]
    print(f"  {K:3d} | {res['accuracy']:8.4f} | {res['mean_r']:8.4f} | {res['median_r']:8.4f} | {res['var_ratio']:9.4f} | {res['positive_r']:4d}")
    if res['mean_r'] > best_r:
        best_r = res['mean_r']
        best_k = K

print(f"\n  Best K: {best_k} (mean r = {best_r:.4f})")

# ============================================================
# [7/7] SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("[7/7] Saving results...")
print("=" * 60)

# Save validation summary
summary = {
    'best_k': best_k,
    'results_by_k': results_by_k,
    'holdout_fraction': 0.2,
    'holdout_questions': len(holdout_qids),
    'train_questions': len(train_qids),
    'embedding_dim': 384,
    'embedding_model': 'all-MiniLM-L6-v2',
}

with open(os.path.join(OUTPUT_DIR, 'knn_validation_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: outputs/knn_validation_summary.json")

# Save the best K's predictions for potential blending
best_res = results_by_k[best_k]
print(f"\n  Best K={best_k} results:")
print(f"    Accuracy:       {best_res['accuracy']}")
print(f"    Mean Pearson r: {best_res['mean_r']}")
print(f"    Variance ratio: {best_res['var_ratio']}")

print("\n" + "=" * 60)
print("NEAREST-QUESTION RETRIEVAL COMPLETE!")
print("=" * 60)
