"""
Model 2: Person Embeddings from Response Matrix
Uses SVD to learn latent behavioral factors from how 233 people
answered 326 ordinal questions.

Input:  outputs/master_table.csv
Output: outputs/person_embeddings.csv
        outputs/person_embeddings_meta.json
        outputs/question_loadings.csv
        outputs/svd_explained_variance.png

Author: Tanmay
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# ============================================================
# [1/7] LOAD AND FILTER ORDINAL DATA
# ============================================================
print("=" * 60)
print("[1/7] Loading ordinal responses from master table...")
print("=" * 60)

master = pd.read_csv(os.path.join(OUTPUT_DIR, 'master_table.csv'))
ordinal = master[master['answer_type'] == 'ordinal'].copy()
ordinal['answer_position'] = pd.to_numeric(ordinal['answer_position'], errors='coerce')
ordinal['num_options'] = pd.to_numeric(ordinal['num_options'], errors='coerce')
ordinal = ordinal.dropna(subset=['answer_position', 'num_options'])

print(f"  Ordinal rows: {ordinal.shape[0]}")
print(f"  Unique persons: {ordinal['person_id'].nunique()}")
print(f"  Unique questions: {ordinal['question_id'].nunique()}")

# ============================================================
# [2/7] NORMALIZE TO 0-1 SCALE
# ============================================================
print("\n" + "=" * 60)
print("[2/7] Normalizing answer positions to 0-1...")
print("=" * 60)

# (answer_position - 1) / (num_options - 1)
# This maps 1 → 0.0, num_options → 1.0
# A 5-point scale: 1→0, 2→0.25, 3→0.5, 4→0.75, 5→1.0
# A 2-point scale: 1→0, 2→1.0
ordinal['answer_normalized'] = (ordinal['answer_position'] - 1) / (ordinal['num_options'] - 1)

# Verify range
assert ordinal['answer_normalized'].min() >= 0.0, "Normalization error: values below 0"
assert ordinal['answer_normalized'].max() <= 1.0, "Normalization error: values above 1"

print(f"  Normalized range: {ordinal['answer_normalized'].min():.3f} to {ordinal['answer_normalized'].max():.3f}")
print(f"  Mean: {ordinal['answer_normalized'].mean():.3f}")
print(f"  Std: {ordinal['answer_normalized'].std():.3f}")

# Check for any num_options == 1 (would cause division by zero)
single_option = ordinal[ordinal['num_options'] == 1]
if len(single_option) > 0:
    print(f"  WARNING: {len(single_option)} rows with num_options=1 — dropping these")
    ordinal = ordinal[ordinal['num_options'] > 1]

# ============================================================
# [3/7] PIVOT TO PERSON × QUESTION MATRIX
# ============================================================
print("\n" + "=" * 60)
print("[3/7] Pivoting to person x question matrix...")
print("=" * 60)

# If a person answered the same question twice (shouldn't happen), take the first
response_matrix = ordinal.pivot_table(
    index='person_id',
    columns='question_id',
    values='answer_normalized',
    aggfunc='first'
)

print(f"  Matrix shape: {response_matrix.shape}")
print(f"  Missing cells: {response_matrix.isnull().sum().sum()} / {response_matrix.size} "
      f"({response_matrix.isnull().sum().sum()/response_matrix.size*100:.2f}%)")

# Store person_ids and question_ids for later
person_ids = response_matrix.index.tolist()
question_ids = response_matrix.columns.tolist()
print(f"  Persons: {len(person_ids)}")
print(f"  Questions: {len(question_ids)}")

# ============================================================
# [4/7] IMPUTE MISSING VALUES AND STANDARDIZE
# ============================================================
print("\n" + "=" * 60)
print("[4/7] Imputing missing values and standardizing...")
print("=" * 60)

# Impute with column mean (question average)
# This is the least-informative imputation — doesn't bias any person
col_means = response_matrix.mean()
response_filled = response_matrix.fillna(col_means)

n_imputed = response_matrix.isnull().sum().sum()
print(f"  Imputed {n_imputed} missing values with column means")

# Standardize: z-score per column (mean=0, std=1)
# This ensures all questions contribute equally to SVD
# regardless of their original scale or variance
col_stds = response_filled.std()

# Check for zero-variance columns (everyone gave the same answer)
zero_var_cols = col_stds[col_stds < 1e-10].index.tolist()
if zero_var_cols:
    print(f"  WARNING: {len(zero_var_cols)} zero-variance questions — dropping them")
    response_filled = response_filled.drop(columns=zero_var_cols)
    col_means = response_filled.mean()
    col_stds = response_filled.std()
    question_ids = response_filled.columns.tolist()

response_standardized = (response_filled - col_means) / col_stds

print(f"  Standardized matrix shape: {response_standardized.shape}")
print(f"  Verify: column means ~ 0: {response_standardized.mean().mean():.6f}")
print(f"  Verify: column stds ~ 1: {response_standardized.std().mean():.6f}")

# ============================================================
# [5/7] SVD — COMPUTE EXPLAINED VARIANCE TO CHOOSE K
# ============================================================
print("\n" + "=" * 60)
print("[5/7] Running SVD...")
print("=" * 60)

from sklearn.decomposition import PCA

# First: run full PCA to see explained variance curve
# n_components = min(n_persons, n_questions) - 1
max_components = min(response_standardized.shape) - 1
print(f"  Max possible components: {max_components}")

# Fit PCA with enough components to analyze the curve
n_analyze = min(100, max_components)
pca_full = PCA(n_components=n_analyze, random_state=42)
pca_full.fit(response_standardized.values)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"\n  Explained variance by component:")
for k in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
    if k <= len(cumulative):
        print(f"    K={k:3d}: {cumulative[k-1]*100:.1f}% cumulative")

# Find elbow: K where adding one more component gives < 1% marginal gain
marginal_gains = explained
elbow_k = None
for i in range(1, len(marginal_gains)):
    if marginal_gains[i] < 0.01:
        elbow_k = i
        break

if elbow_k:
    print(f"\n  Elbow point (marginal gain < 1%): K={elbow_k}")
else:
    print(f"\n  No clear elbow found in first {n_analyze} components")

# ============================================================
# [5.5/7] PLOT EXPLAINED VARIANCE
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(range(1, min(51, len(explained)+1)), explained[:50], color='steelblue', alpha=0.8)
ax1.set_xlabel('Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Individual Explained Variance (first 50)')
ax1.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='1% threshold')
ax1.legend()

ax2.plot(range(1, len(cumulative)+1), cumulative * 100, 'b-o', markersize=3)
ax2.set_xlabel('Number of Components (K)')
ax2.set_ylabel('Cumulative Explained Variance (%)')
ax2.set_title('Cumulative Explained Variance')
ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
for k in [10, 20, 30, 50]:
    if k <= len(cumulative):
        ax2.axvline(x=k, color='gray', linestyle=':', alpha=0.3)
        ax2.annotate(f'K={k}: {cumulative[k-1]*100:.1f}%', xy=(k, cumulative[k-1]*100),
                    fontsize=8, ha='left')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'svd_explained_variance.png'), dpi=300, bbox_inches='tight')
plt.close('all')
print(f"  Saved: outputs/svd_explained_variance.png")

# ============================================================
# [6/7] CHOOSE K AND EXTRACT FINAL EMBEDDINGS
# ============================================================
print("\n" + "=" * 60)
print("[6/7] Extracting final embeddings...")
print("=" * 60)

# Strategy: save multiple K values so Tolendi can test
# Primary: K=30 (good balance of information and dimensionality)
# Also save K=10, K=20, K=50 as alternatives

K_OPTIONS = [10, 20, 30, 50]

for K in K_OPTIONS:
    if K > max_components:
        print(f"  K={K}: skipped (exceeds max components {max_components})")
        continue

    pca_k = PCA(n_components=K, random_state=42)
    embeddings = pca_k.fit_transform(response_standardized.values)

    # Create DataFrame
    embed_df = pd.DataFrame(
        embeddings,
        columns=[f'emb_{i}' for i in range(K)]
    )
    embed_df.insert(0, 'person_id', person_ids)

    # Save
    fname = f'person_embeddings_k{K}.csv'
    embed_df.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
    print(f"  K={K}: {cumulative[K-1]*100:.1f}% variance, saved to outputs/{fname}")

# Save the primary choice (K=30) as the default file
K_PRIMARY = 30
pca_primary = PCA(n_components=K_PRIMARY, random_state=42)
embeddings_primary = pca_primary.fit_transform(response_standardized.values)

embed_primary_df = pd.DataFrame(
    embeddings_primary,
    columns=[f'emb_{i}' for i in range(K_PRIMARY)]
)
embed_primary_df.insert(0, 'person_id', person_ids)
embed_primary_df.to_csv(os.path.join(OUTPUT_DIR, 'person_embeddings.csv'), index=False)
print(f"\n  Primary file: outputs/person_embeddings.csv (K={K_PRIMARY})")

# ============================================================
# [7/7] SAVE METADATA AND QUESTION LOADINGS
# ============================================================
print("\n" + "=" * 60)
print("[7/7] Saving metadata and question loadings...")
print("=" * 60)

# Question loadings: how much each question contributes to each component
# components_ has shape (K, n_questions)
loadings = pca_primary.components_.T  # (n_questions, K)
loadings_df = pd.DataFrame(
    loadings,
    columns=[f'loading_{i}' for i in range(K_PRIMARY)]
)
loadings_df.insert(0, 'question_id', question_ids)
loadings_df.to_csv(os.path.join(OUTPUT_DIR, 'question_loadings.csv'), index=False)
print(f"  Saved: outputs/question_loadings.csv")

# Metadata
meta = {
    'K_primary': K_PRIMARY,
    'K_options_saved': K_OPTIONS,
    'n_persons': len(person_ids),
    'n_questions': len(question_ids),
    'n_imputed_cells': int(n_imputed),
    'n_zero_variance_dropped': len(zero_var_cols) if zero_var_cols else 0,
    'zero_variance_questions': zero_var_cols if zero_var_cols else [],
    'explained_variance_by_K': {str(k): float(cumulative[k-1]) for k in K_OPTIONS if k <= len(cumulative)},
    'elbow_k': elbow_k,
    'normalization': '(answer_position - 1) / (num_options - 1)',
    'standardization': 'z-score per column (mean=0, std=1)',
    'imputation': 'column mean',
    'method': 'PCA (centered SVD)',
}

with open(os.path.join(OUTPUT_DIR, 'person_embeddings_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
print(f"  Saved: outputs/person_embeddings_meta.json")

# ============================================================
# INTERPRETATION: Top questions per component (first 5 components)
# ============================================================
print("\n" + "=" * 60)
print("INTERPRETATION: Top questions per component")
print("=" * 60)

# Load question text for interpretation
unique_q = pd.read_csv(os.path.join(OUTPUT_DIR, 'unique_questions.csv'))
qid_to_text = dict(zip(unique_q['question_id'], unique_q['full_question'].fillna('')))
qid_to_block = dict(zip(unique_q['question_id'], unique_q['block_name'].fillna('')))

for comp in range(min(5, K_PRIMARY)):
    loading_vals = loadings[:, comp]
    # Top 3 positive and top 3 negative loadings
    top_pos_idx = np.argsort(loading_vals)[-3:][::-1]
    top_neg_idx = np.argsort(loading_vals)[:3]

    print(f"\n  Component {comp} ({explained[comp]*100:.1f}% variance):")
    print(f"    Positive pole (high score = ...):")
    for idx in top_pos_idx:
        qid = question_ids[idx]
        qtext = str(qid_to_text.get(qid, ''))[:70]
        block = qid_to_block.get(qid, '?')
        print(f"      {loading_vals[idx]:+.3f} | {qid} ({block}) | {qtext}")

    print(f"    Negative pole (low score = ...):")
    for idx in top_neg_idx:
        qid = question_ids[idx]
        qtext = str(qid_to_text.get(qid, ''))[:70]
        block = qid_to_block.get(qid, '?')
        print(f"      {loading_vals[idx]:+.3f} | {qid} ({block}) | {qtext}")

# ============================================================
# SANITY CHECK: embedding variance
# ============================================================
print("\n" + "=" * 60)
print("SANITY CHECK")
print("=" * 60)

print(f"  Embedding shape: {embeddings_primary.shape}")
print(f"  Per-person embedding norm: mean={np.linalg.norm(embeddings_primary, axis=1).mean():.2f}, "
      f"std={np.linalg.norm(embeddings_primary, axis=1).std():.2f}")
print(f"  Per-component std: {embeddings_primary.std(axis=0).mean():.3f} (should be > 0)")
print(f"  Any NaN: {np.isnan(embeddings_primary).any()}")
print(f"  Any Inf: {np.isinf(embeddings_primary).any()}")

# Check that embeddings actually differ between people
# Compute pairwise distances between first 10 people
from scipy.spatial.distance import pdist
dists = pdist(embeddings_primary[:10])
print(f"  Pairwise distances (first 10 people): mean={dists.mean():.2f}, min={dists.min():.2f}, max={dists.max():.2f}")
if dists.min() < 0.01:
    print("  WARNING: Some people have nearly identical embeddings!")
else:
    print("  People are well-separated in embedding space.")

print("\n" + "=" * 60)
print("MODEL 2 COMPLETE!")
print(f"Primary output: outputs/person_embeddings.csv (233 x {K_PRIMARY+1})")
print("=" * 60)
