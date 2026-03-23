"""
Question-to-Construct Mapper
Maps each question to its psychological construct using embeddings.
For new unseen questions, finds the nearest construct and retrieves
each person's known scores on that construct.

Author: Tanmay
Output: outputs/construct_mapping.csv, outputs/construct_centroids.pkl
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os
import json
import pickle
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# ============================================================
# [1/5] LOAD DATA
# ============================================================
print("=" * 60)
print("[1/5] Loading data...")
print("=" * 60)

master = pd.read_csv(os.path.join(OUTPUT_DIR, 'master_table.csv'))
unique_q = pd.read_csv(os.path.join(OUTPUT_DIR, 'unique_questions.csv'))

print(f"  Master table: {master.shape[0]} rows")
print(f"  Unique questions: {len(unique_q)}")

# ============================================================
# [2/5] DEFINE CONSTRUCTS FROM BLOCK + PARENT QUESTION GROUPINGS
# ============================================================
print("\n" + "=" * 60)
print("[2/5] Defining constructs...")
print("=" * 60)

# Group questions by parent_question_id — each parent QID is a construct/scale
# For Matrix questions, all sub-items share the same parent QID and measure the same thing
construct_map = {}
for _, row in unique_q.iterrows():
    qid = row['question_id']
    parent = row['parent_question_id']
    block = row['block_name']
    qtype = row['question_type']

    # Use parent_question_id as construct identifier
    # This naturally groups matrix sub-items together
    construct_map[qid] = {
        'construct_id': parent,
        'block_name': block,
        'question_type': qtype,
    }

# Count questions per construct
construct_sizes = defaultdict(int)
for v in construct_map.values():
    construct_sizes[v['construct_id']] += 1

print(f"  Found {len(construct_sizes)} constructs")
print(f"  Construct sizes: min={min(construct_sizes.values())}, "
      f"max={max(construct_sizes.values())}, "
      f"mean={np.mean(list(construct_sizes.values())):.1f}")

# Show largest constructs
sorted_constructs = sorted(construct_sizes.items(), key=lambda x: -x[1])
print(f"\n  Top 15 constructs by size:")
for cid, size in sorted_constructs[:15]:
    block = [v['block_name'] for v in construct_map.values() if v['construct_id'] == cid][0]
    print(f"    {cid} ({block}): {size} questions")

# ============================================================
# [3/5] EMBED ALL QUESTIONS
# ============================================================
print("\n" + "=" * 60)
print("[3/5] Embedding questions...")
print("=" * 60)

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Build full question text (context-aware)
q_texts = unique_q['full_question'].fillna('').tolist()
q_ids = unique_q['question_id'].tolist()

print(f"  Embedding {len(q_texts)} questions...")
q_embeddings = embedder.encode(q_texts, show_progress_bar=True, batch_size=64)
print(f"  Embedding dim: {q_embeddings.shape[1]}")

# Create lookup: question_id → embedding
qid_to_embedding = {qid: emb for qid, emb in zip(q_ids, q_embeddings)}

# ============================================================
# [4/5] COMPUTE CONSTRUCT CENTROIDS
# ============================================================
print("\n" + "=" * 60)
print("[4/5] Computing construct centroids...")
print("=" * 60)

# For each construct, compute the centroid (mean embedding) of its questions
construct_centroids = {}
construct_metadata = {}

for cid in construct_sizes:
    # Get all question_ids in this construct
    member_qids = [qid for qid, info in construct_map.items() if info['construct_id'] == cid]
    member_embeddings = [qid_to_embedding[qid] for qid in member_qids if qid in qid_to_embedding]

    if member_embeddings:
        centroid = np.mean(member_embeddings, axis=0)
        construct_centroids[cid] = centroid

        # Get metadata
        block = [v['block_name'] for v in construct_map.values() if v['construct_id'] == cid][0]
        # Get a sample question text for labeling
        sample_qid = member_qids[0]
        sample_row = unique_q[unique_q['question_id'] == sample_qid]
        sample_text = str(sample_row['full_question'].iloc[0]) if len(sample_row) > 0 else ''

        construct_metadata[cid] = {
            'block_name': block,
            'n_questions': len(member_qids),
            'sample_question': sample_text[:100],
            'member_qids': member_qids,
        }

print(f"  Computed {len(construct_centroids)} construct centroids")

# ============================================================
# [4.5/5] COMPUTE PER-PERSON CONSTRUCT SCORES
# ============================================================
print("\n  Computing per-person construct averages...")

# For each person × construct, compute their average answer_position
ordinal = master[master['answer_type'] == 'ordinal'].copy()
ordinal['answer_position'] = pd.to_numeric(ordinal['answer_position'], errors='coerce')

# Map each question to its construct
ordinal['construct_id'] = ordinal['question_id'].map(
    lambda qid: construct_map.get(qid, {}).get('construct_id', None)
)
ordinal = ordinal.dropna(subset=['construct_id', 'answer_position'])

# Compute person × construct averages
person_construct = ordinal.groupby(['person_id', 'construct_id']).agg(
    mean_answer=('answer_position', 'mean'),
    std_answer=('answer_position', 'std'),
    n_answers=('answer_position', 'count'),
).reset_index()

# Compute percentile rank per construct
def compute_percentile(group):
    group['percentile'] = group['mean_answer'].rank(pct=True)
    return group

person_construct = person_construct.groupby('construct_id', group_keys=False).apply(compute_percentile)

print(f"  Person × construct scores: {person_construct.shape[0]} rows")
print(f"  Unique persons: {person_construct['person_id'].nunique()}")
print(f"  Unique constructs: {person_construct['construct_id'].nunique()}")

# ============================================================
# [5/5] BUILD MATCHING FUNCTION AND SAVE
# ============================================================
print("\n" + "=" * 60)
print("[5/5] Saving outputs...")
print("=" * 60)

# Save construct mapping
construct_df = pd.DataFrame([
    {'question_id': qid, 'construct_id': info['construct_id'],
     'block_name': info['block_name'], 'question_type': info['question_type']}
    for qid, info in construct_map.items()
])
construct_df.to_csv(os.path.join(OUTPUT_DIR, 'construct_mapping.csv'), index=False)
print(f"  Saved: outputs/construct_mapping.csv")

# Save person × construct scores
person_construct.to_csv(os.path.join(OUTPUT_DIR, 'person_construct_scores.csv'), index=False)
print(f"  Saved: outputs/person_construct_scores.csv")

# Save centroids + metadata for runtime matching
with open(os.path.join(OUTPUT_DIR, 'construct_centroids.pkl'), 'wb') as f:
    pickle.dump({
        'centroids': construct_centroids,
        'metadata': construct_metadata,
        'embedder_name': 'all-MiniLM-L6-v2',
    }, f)
print(f"  Saved: outputs/construct_centroids.pkl")

# ============================================================
# TEST: Match sample questions to constructs
# ============================================================
print("\n" + "=" * 60)
print("TEST: Matching sample questions to nearest constructs")
print("=" * 60)

sample_path = os.path.join(PROJECT_ROOT, 'sample_test_questions.json')
if os.path.exists(sample_path):
    with open(sample_path, 'r') as f:
        sample_qs = json.load(f)

    # Build centroid matrix for fast matching
    centroid_ids = list(construct_centroids.keys())
    centroid_matrix = np.array([construct_centroids[cid] for cid in centroid_ids])

    # Normalize for cosine similarity
    centroid_norms = centroid_matrix / np.linalg.norm(centroid_matrix, axis=1, keepdims=True)

    for sq in sample_qs:
        context = sq.get('context', '') or ''
        qtext = sq.get('question_text', '')
        full = f"{context} {qtext}".strip()

        # Embed
        emb = embedder.encode([full])[0]
        emb_norm = emb / np.linalg.norm(emb)

        # Cosine similarity to all centroids
        sims = centroid_norms @ emb_norm
        top3_idx = np.argsort(sims)[-3:][::-1]

        print(f"\n  Q: {qtext[:60]}...")
        for idx in top3_idx:
            cid = centroid_ids[idx]
            sim = sims[idx]
            meta = construct_metadata[cid]
            print(f"    -> {cid} ({meta['block_name']}, {meta['n_questions']} items) "
                  f"sim={sim:.3f}")
            print(f"      Sample: {meta['sample_question'][:80]}...")
else:
    print("  sample_test_questions.json not found, skipping test.")

print("\n" + "=" * 60)
print("CONSTRUCT MAPPER COMPLETE!")
print("=" * 60)
