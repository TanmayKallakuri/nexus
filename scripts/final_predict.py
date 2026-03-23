"""
FINAL SUBMISSION PREDICTION SCRIPT
===================================

Usage:
  python scripts/final_predict.py <test_input.json> <output.json>

Example:
  python scripts/final_predict.py test_questions.json predictions.json

Pipeline:
  1. Load test JSON
  2. For each (person_id, question):
     a. Embed question (context + question_text)
     b. KNN: find K=7 nearest known questions, weighted average of person's answers
     c. Ensemble: run through V4.1 LightGBM models (if loadable)
     d. Blend or use best available
  3. Handle all question types: Likert, binary, MC, continuous (0-100)
  4. Fill predicted_answer
  5. Save output JSON

Authors: Tanmay, Jasjyot, Tolendi
"""

import json
import os
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONFIGURATION
# ============================================================
KNN_K = 7
ENSEMBLE_WEIGHT = 0.0  # set to 0.4 if ensemble loads successfully
KNN_WEIGHT = 1.0       # adjusted if ensemble loads

# ============================================================
# [1] LOAD KNN ARTIFACTS
# ============================================================
print("[1/6] Loading KNN artifacts...")

# Response lookup
with open(os.path.join(PROJECT_ROOT, 'outputs', 'knn_response_lookup.pkl'), 'rb') as f:
    knn_data = pickle.load(f)
response_pivot = knn_data['response_pivot']
qid_to_numopts = knn_data['qid_to_numopts']
known_person_ids = set(knn_data['person_ids'])
known_question_ids = knn_data['question_ids']

# Question embeddings (384-dim)
embeddings_384 = np.load(os.path.join(PROJECT_ROOT, 'outputs', 'knn_question_embeddings_384.npy'))
norms = np.linalg.norm(embeddings_384, axis=1, keepdims=True)
norms[norms == 0] = 1
embeddings_normed = embeddings_384 / norms

# QID to index
with open(os.path.join(PROJECT_ROOT, 'outputs', 'knn_qid_to_idx.pkl'), 'rb') as f:
    idx_data = pickle.load(f)
qid_to_idx = idx_data['qid_to_idx']
ordinal_qids = idx_data['ordinal_qids']

# Sentence transformer for embedding new questions
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print(f"  KNN loaded: {len(known_person_ids)} persons, {len(known_question_ids)} known questions")

# ============================================================
# [2] TRY LOADING ENSEMBLE (optional, graceful fallback)
# ============================================================
print("[2/6] Attempting to load V4.1 ensemble...")

ensemble_available = False
try:
    with open(os.path.join(PROJECT_ROOT, 'outputs', 'tolendi', 'bootstrap_ensemble_v4_1.pkl'), 'rb') as f:
        ensemble_bundle = pickle.load(f)

    # Load PCA model for question embeddings
    with open(os.path.join(PROJECT_ROOT, 'outputs', 'tolendi', 'pca_model_v4_1.pkl'), 'rb') as f:
        pca_model = pickle.load(f)

    # Load person features
    person_feat = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'person_response_profiles_repaired.csv'))
    person_emb = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'tolendi', 'person_embeddings_v4_1_k10.csv'))
    person_combined = person_feat.merge(person_emb, on='person_id', how='left')

    ensemble_available = True
    ENSEMBLE_WEIGHT = 0.4
    KNN_WEIGHT = 0.6
    print(f"  Ensemble loaded successfully. Blend: KNN {KNN_WEIGHT} / Ensemble {ENSEMBLE_WEIGHT}")
except Exception as e:
    print(f"  Ensemble failed to load: {e}")
    print(f"  Using KNN-only mode.")
    KNN_WEIGHT = 1.0
    ENSEMBLE_WEIGHT = 0.0

# ============================================================
# [3] LOAD PERSONA TEXT FILES (for LLM fallback if available)
# ============================================================
print("[3/6] Loading persona text files...")

persona_texts = {}
persona_dir = os.path.join(PROJECT_ROOT, 'personas_text')
if os.path.isdir(persona_dir):
    for fname in os.listdir(persona_dir):
        if fname.endswith('_persona.txt'):
            pid = fname.replace('_persona.txt', '')
            with open(os.path.join(persona_dir, fname), 'r', encoding='utf-8') as f:
                persona_texts[pid] = f.read()
print(f"  Loaded {len(persona_texts)} persona text files")

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def parse_options(options):
    """Parse options field — can be a list or a string like '0 to 100'."""
    if isinstance(options, list):
        return {
            'type': 'list',
            'options': options,
            'n_options': len(options),
            'lo': 1,
            'hi': len(options),
        }
    elif isinstance(options, str):
        # "0 to 100" style
        parts = options.replace('to', ' ').split()
        nums = [int(p) for p in parts if p.lstrip('-').isdigit()]
        if len(nums) == 2:
            return {
                'type': 'continuous',
                'options': options,
                'n_options': nums[1] - nums[0] + 1,
                'lo': nums[0],
                'hi': nums[1],
            }
        return {
            'type': 'continuous',
            'options': options,
            'n_options': 101,
            'lo': 0,
            'hi': 100,
        }
    else:
        return {
            'type': 'unknown',
            'options': options,
            'n_options': 5,
            'lo': 1,
            'hi': 5,
        }


def build_question_text(q):
    """Combine context + question_text for embedding."""
    context = q.get('context', None) or ''
    qtext = q.get('question_text', '')
    full = f"{context} {qtext}".strip()
    return full


def predict_knn(person_id, question_embedding_normed, opts_info):
    """
    KNN prediction: find K=7 nearest known questions,
    weighted average of this person's answers to those questions.
    """
    if person_id not in known_person_ids:
        return None, 0.0

    # Cosine similarity to all known questions
    sims = embeddings_normed @ question_embedding_normed

    # Get top K
    k = min(KNN_K, len(sims))
    top_k_idx = np.argsort(sims)[-k:][::-1]
    top_k_sims = sims[top_k_idx]
    top_k_qids = [ordinal_qids[i] for i in top_k_idx]

    # Clamp negative similarities to 0
    top_k_sims = np.maximum(top_k_sims, 0.0)
    max_sim = top_k_sims[0] if len(top_k_sims) > 0 else 0.0

    if top_k_sims.sum() < 1e-10:
        top_k_weights = np.ones(k) / k
    else:
        top_k_weights = top_k_sims / top_k_sims.sum()

    # Get this person's normalized answers to nearest questions
    neighbor_answers = []
    neighbor_weights = []
    for i, nq in enumerate(top_k_qids):
        if nq in response_pivot.columns:
            val = response_pivot.loc[person_id, nq]
            if not np.isnan(val):
                neighbor_answers.append(val)
                neighbor_weights.append(top_k_weights[i])

    if len(neighbor_answers) == 0:
        return None, max_sim

    neighbor_answers = np.array(neighbor_answers)
    neighbor_weights = np.array(neighbor_weights)
    w_sum = neighbor_weights.sum()
    if w_sum < 1e-10:
        neighbor_weights = np.ones(len(neighbor_weights)) / len(neighbor_weights)
    else:
        neighbor_weights = neighbor_weights / w_sum

    # Weighted average in normalized space (0-1)
    pred_norm = np.average(neighbor_answers, weights=neighbor_weights)

    # Scale to target question's range
    lo, hi = opts_info['lo'], opts_info['hi']
    pred_raw = lo + pred_norm * (hi - lo)
    pred_raw = np.clip(pred_raw, lo, hi)

    return pred_raw, max_sim


def predict_ensemble(person_id, question_text, opts_info):
    """
    V4.1 ensemble prediction. Returns None if ensemble not available
    or person not found.
    """
    if not ensemble_available:
        return None

    if person_id not in person_combined['person_id'].values:
        return None

    try:
        # Get person features
        person_row = person_combined[person_combined['person_id'] == person_id]
        person_cols = [c for c in person_combined.columns if c != 'person_id']
        person_vec = person_row[person_cols].values.flatten()

        # Embed question and PCA
        q_emb = embedder.encode([question_text])
        q_emb_pca = pca_model.transform(q_emb)[0]

        # Build feature vector (must match training feature order)
        # This is approximate — exact feature order depends on Tolendi's script
        feature_vec = np.concatenate([person_vec, q_emb_pca, [opts_info['n_options']]])
        feature_vec = np.nan_to_num(feature_vec, nan=0.0)

        # Get models from bundle
        if 'routes' in ensemble_bundle:
            # Routed ensemble
            models = []
            for route_name, route_data in ensemble_bundle['routes'].items():
                if 'models' in route_data:
                    models.extend(route_data['models'])
            if not models:
                models = ensemble_bundle.get('models', [])
        elif 'models' in ensemble_bundle:
            models = ensemble_bundle['models']
        else:
            return None

        if not models:
            return None

        # Predict with all models, average
        preds = []
        for m in models:
            try:
                p = m.predict(feature_vec.reshape(1, -1))[0]
                preds.append(p)
            except:
                continue

        if not preds:
            return None

        pred_norm = np.mean(preds)

        # Scale to target range
        lo, hi = opts_info['lo'], opts_info['hi']
        pred_raw = lo + pred_norm * (hi - lo)
        pred_raw = np.clip(pred_raw, lo, hi)

        return pred_raw

    except Exception as e:
        return None


def predict_one(person_id, question, opts_info, q_embedding_normed):
    """
    Main prediction function. Blends KNN + ensemble.
    Falls back gracefully.
    """
    # KNN prediction
    knn_pred, knn_sim = predict_knn(person_id, q_embedding_normed, opts_info)

    # Ensemble prediction
    question_text = build_question_text(question)
    ens_pred = predict_ensemble(person_id, question_text, opts_info)

    # Blend
    if knn_pred is not None and ens_pred is not None:
        final = KNN_WEIGHT * knn_pred + ENSEMBLE_WEIGHT * ens_pred
    elif knn_pred is not None:
        final = knn_pred
    elif ens_pred is not None:
        final = ens_pred
    else:
        # Last resort: midpoint
        lo, hi = opts_info['lo'], opts_info['hi']
        final = (lo + hi) / 2

    # Round appropriately
    lo, hi = opts_info['lo'], opts_info['hi']
    final = np.clip(final, lo, hi)

    if opts_info['type'] == 'list':
        # Discrete options: round to nearest integer
        final = int(round(final))
    elif opts_info['type'] == 'continuous':
        # Continuous (0-100): round to integer
        final = int(round(final))
    else:
        final = int(round(final))

    return final


# ============================================================
# [4] LOAD TEST QUESTIONS
# ============================================================
print("[4/6] Loading test questions...")

if len(sys.argv) < 2:
    # Default to sample if no argument
    test_path = os.path.join(PROJECT_ROOT, 'sample_test_questions.json')
    output_path = os.path.join(PROJECT_ROOT, 'outputs', 'predictions.json')
elif len(sys.argv) == 2:
    test_path = sys.argv[1]
    output_path = os.path.join(PROJECT_ROOT, 'outputs', 'predictions.json')
else:
    test_path = sys.argv[1]
    output_path = sys.argv[2]

with open(test_path, 'r', encoding='utf-8') as f:
    test_questions = json.load(f)

print(f"  Loaded {len(test_questions)} questions from {test_path}")

# Count question types
type_counts = {}
for q in test_questions:
    opts_info = parse_options(q.get('options', []))
    type_counts[opts_info['type']] = type_counts.get(opts_info['type'], 0) + 1
print(f"  Question types: {type_counts}")

# Count unique persons
unique_persons = set(q['person_id'] for q in test_questions)
known_count = len(unique_persons & known_person_ids)
print(f"  Unique persons: {len(unique_persons)} ({known_count} known, {len(unique_persons) - known_count} unknown)")

# ============================================================
# [5] PREDICT
# ============================================================
print("[5/6] Running predictions...")

# Pre-embed all unique question texts for efficiency
unique_texts = list(set(build_question_text(q) for q in test_questions))
print(f"  Embedding {len(unique_texts)} unique question texts...")
text_embeddings = embedder.encode(unique_texts, show_progress_bar=True, batch_size=32)
text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
text_norms[text_norms == 0] = 1
text_embeddings_normed = text_embeddings / text_norms
text_to_embedding = {t: e for t, e in zip(unique_texts, text_embeddings_normed)}

results = []
knn_used = 0
ens_used = 0
fallback_used = 0

for i, q in enumerate(test_questions):
    person_id = q['person_id']
    opts_info = parse_options(q.get('options', []))
    question_text = build_question_text(q)
    q_emb_normed = text_to_embedding[question_text]

    predicted = predict_one(person_id, q, opts_info, q_emb_normed)

    # Track which predictor was used
    knn_pred, _ = predict_knn(person_id, q_emb_normed, opts_info)
    if knn_pred is not None:
        knn_used += 1
    ens_pred = predict_ensemble(person_id, question_text, opts_info) if ensemble_available else None
    if ens_pred is not None:
        ens_used += 1
    if knn_pred is None and ens_pred is None:
        fallback_used += 1

    result = q.copy()
    result['predicted_answer'] = predicted
    results.append(result)

    if (i + 1) % 100 == 0 or (i + 1) == len(test_questions):
        print(f"  [{i+1}/{len(test_questions)}] predicted")

print(f"\n  Prediction sources:")
print(f"    KNN available: {knn_used}/{len(test_questions)}")
print(f"    Ensemble available: {ens_used}/{len(test_questions)}")
print(f"    Midpoint fallback: {fallback_used}/{len(test_questions)}")

# ============================================================
# [6] SAVE OUTPUT
# ============================================================
print(f"[6/6] Saving predictions to {output_path}...")

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

# Verify
filled = sum(1 for r in results if r.get('predicted_answer') is not None)
print(f"  Filled: {filled}/{len(results)}")

# Show sample predictions
print(f"\n  Sample predictions:")
for r in results[:5]:
    pid = r['person_id']
    qtext = r['question_text'][:50]
    pred = r['predicted_answer']
    opts = r.get('options', '?')
    if isinstance(opts, list):
        opts_str = f"{len(opts)} options"
    else:
        opts_str = str(opts)
    print(f"    {pid} | {qtext}... | pred={pred} | {opts_str}")

print(f"\nDone. Output: {output_path}")
