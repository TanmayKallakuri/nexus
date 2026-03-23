"""
Build Master Table — Flatten all persona JSONs into one table.
Every (person, question, answer) becomes one row.

Output: outputs/master_table.csv

Authors: Tanmay, Jasjyot
"""

import json
import os
import pandas as pd
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = os.path.join(PROJECT_ROOT, 'personas_json')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# [1/4] LOAD ALL PERSONA JSONs
# ============================================================
print("=" * 60)
print("[1/4] Loading persona JSONs...")
print("=" * 60)

json_files = sorted(glob.glob(os.path.join(JSON_DIR, '*_persona.json')))
print(f"  Found {len(json_files)} persona files")

# ============================================================
# [2/4] FLATTEN INTO ROWS
# ============================================================
print("\n" + "=" * 60)
print("[2/4] Flattening all questions into rows...")
print("=" * 60)

rows = []
skipped_db = 0
skipped_empty = 0
errors = []

for fpath in json_files:
    # Extract person_id from filename: "00a1r_persona.json" → "00a1r"
    fname = os.path.basename(fpath)
    person_id = fname.replace('_persona.json', '')

    with open(fpath, 'r', encoding='utf-8') as f:
        blocks = json.load(f)

    for block in blocks:
        block_name = block.get('BlockName', 'Unknown').strip()
        questions = block.get('Questions', [])

        for q in questions:
            qid = q.get('QuestionID', '')
            qtype = q.get('QuestionType', '')
            qtext = q.get('QuestionText', '')
            answers = q.get('Answers', {})
            settings = q.get('Settings', {})
            selector = settings.get('Selector', '')

            # --- SKIP display blocks (DB) — no answers ---
            if qtype == 'DB':
                skipped_db += 1
                continue

            # --- SKIP questions with no answers ---
            if not answers:
                skipped_empty += 1
                continue

            try:
                # ==================================================
                # MATRIX questions — one row per sub-item
                # ==================================================
                if qtype == 'Matrix':
                    sub_items = q.get('Rows', [])
                    columns = q.get('Columns', [])
                    positions = answers.get('SelectedByPosition', [])
                    texts = answers.get('SelectedText', [])

                    # Determine answer type from selector
                    if selector == 'Likert':
                        answer_type = 'ordinal'
                    else:
                        answer_type = 'categorical'

                    num_options = len(columns)
                    options_str = json.dumps(columns)

                    for idx, sub_item in enumerate(sub_items):
                        if idx < len(positions):
                            pos = positions[idx]
                            txt = texts[idx] if idx < len(texts) else ''
                        else:
                            pos = None
                            txt = ''

                        rows.append({
                            'person_id': person_id,
                            'question_id': f"{qid}_r{idx+1}",
                            'parent_question_id': qid,
                            'block_name': block_name,
                            'question_type': qtype,
                            'selector': selector,
                            'answer_type': answer_type,
                            'question_text': qtext.strip(),
                            'sub_item_text': sub_item.strip(),
                            'full_question': f"{qtext.strip()} - {sub_item.strip()}",
                            'answer_position': pos,
                            'answer_text': txt,
                            'options': options_str,
                            'num_options': num_options,
                        })

                # ==================================================
                # MC questions — single or multi-select
                # ==================================================
                elif qtype == 'MC':
                    options = q.get('Options', [])
                    num_options = len(options)
                    options_str = json.dumps(options)
                    pos = answers.get('SelectedByPosition', None)
                    txt = answers.get('SelectedText', None)

                    # Check if multi-select (lists) or single (scalar)
                    is_multi = isinstance(pos, list)

                    # Determine answer type
                    # If Likert-like selector → ordinal
                    if selector in ('Likert', 'SAVR'):
                        answer_type = 'ordinal'
                    elif is_multi:
                        answer_type = 'multi_select'
                    else:
                        answer_type = 'categorical'

                    if is_multi:
                        # Multi-select: store as JSON lists
                        rows.append({
                            'person_id': person_id,
                            'question_id': qid,
                            'parent_question_id': qid,
                            'block_name': block_name,
                            'question_type': 'MC_multi',
                            'selector': selector,
                            'answer_type': answer_type,
                            'question_text': qtext.strip(),
                            'sub_item_text': '',
                            'full_question': qtext.strip(),
                            'answer_position': json.dumps(pos),
                            'answer_text': json.dumps(txt) if isinstance(txt, list) else txt,
                            'options': options_str,
                            'num_options': num_options,
                        })
                    else:
                        # Single-select
                        rows.append({
                            'person_id': person_id,
                            'question_id': qid,
                            'parent_question_id': qid,
                            'block_name': block_name,
                            'question_type': 'MC',
                            'selector': selector,
                            'answer_type': answer_type,
                            'question_text': qtext.strip(),
                            'sub_item_text': '',
                            'full_question': qtext.strip(),
                            'answer_position': pos,
                            'answer_text': txt,
                            'options': options_str,
                            'num_options': num_options,
                        })

                # ==================================================
                # TE questions — text entry
                # ==================================================
                elif qtype == 'TE':
                    text_answers = answers.get('Text', [])

                    # TE can be a list of dicts (multi-field form) or a single value
                    if isinstance(text_answers, list):
                        # Multi-field: e.g., [{"Thought 1": "..."}, {"Thought 2": "..."}]
                        combined_text = []
                        for item in text_answers:
                            if isinstance(item, dict):
                                for k, v in item.items():
                                    if v:
                                        combined_text.append(f"{k}: {v}")
                            elif isinstance(item, str):
                                combined_text.append(item)
                        answer_text = ' | '.join(combined_text) if combined_text else ''
                    elif isinstance(text_answers, str):
                        answer_text = text_answers
                    else:
                        answer_text = str(text_answers)

                    rows.append({
                        'person_id': person_id,
                        'question_id': qid,
                        'parent_question_id': qid,
                        'block_name': block_name,
                        'question_type': 'TE',
                        'selector': selector,
                        'answer_type': 'text',
                        'question_text': qtext.strip(),
                        'sub_item_text': '',
                        'full_question': qtext.strip(),
                        'answer_position': None,
                        'answer_text': answer_text,
                        'options': '',
                        'num_options': 0,
                    })

                else:
                    # Unknown type — log but don't crash
                    errors.append(f"Unknown QType '{qtype}' for {qid} in {person_id}")

            except Exception as e:
                errors.append(f"Error processing {qid} for {person_id}: {str(e)}")

    if (json_files.index(fpath) + 1) % 50 == 0:
        print(f"  Processed {json_files.index(fpath) + 1}/{len(json_files)} personas...")

print(f"  Processed all {len(json_files)} personas")
print(f"  Total rows: {len(rows)}")
print(f"  Skipped DB blocks: {skipped_db}")
print(f"  Skipped empty answers: {skipped_empty}")
if errors:
    print(f"  Errors: {len(errors)}")
    for e in errors[:10]:
        print(f"    {e}")

# ============================================================
# [3/4] BUILD DATAFRAME AND VALIDATE
# ============================================================
print("\n" + "=" * 60)
print("[3/4] Building DataFrame and validating...")
print("=" * 60)

df = pd.DataFrame(rows)

print(f"\n  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Unique persons: {df['person_id'].nunique()}")
print(f"  Unique question_ids: {df['question_id'].nunique()}")
print(f"  Unique parent_question_ids: {df['parent_question_id'].nunique()}")

print(f"\n  Question types:")
for qt, count in df['question_type'].value_counts().items():
    print(f"    {qt}: {count} rows")

print(f"\n  Answer types:")
for at, count in df['answer_type'].value_counts().items():
    print(f"    {at}: {count} rows")

print(f"\n  Block names:")
for bn, count in df['block_name'].value_counts().items():
    print(f"    {bn}: {count} rows")

# Validate: check answer_position coverage for non-text questions
non_text = df[df['answer_type'] != 'text']
null_pos = non_text['answer_position'].isna().sum()
print(f"\n  Non-text rows with null answer_position: {null_pos} / {len(non_text)}")

# Check questions per person
qpp = df.groupby('person_id')['question_id'].count()
print(f"\n  Questions per person:")
print(f"    Min: {qpp.min()}, Max: {qpp.max()}, Mean: {qpp.mean():.0f}, Median: {qpp.median():.0f}")

# Check for consistency — do all persons have the same questions?
person_qsets = df.groupby('person_id')['question_id'].apply(set)
all_same = len(set(map(frozenset, person_qsets))) == 1
print(f"  All persons have identical question sets: {all_same}")
if not all_same:
    # Find the most common set size
    set_sizes = person_qsets.apply(len)
    print(f"  Question set sizes — Min: {set_sizes.min()}, Max: {set_sizes.max()}")

# ============================================================
# [4/4] SAVE
# ============================================================
print("\n" + "=" * 60)
print("[4/4] Saving master table...")
print("=" * 60)

output_path = os.path.join(OUTPUT_DIR, 'master_table.csv')
df.to_csv(output_path, index=False)
print(f"  Saved: {output_path}")
print(f"  File size: {os.path.getsize(output_path) / 1024**2:.1f} MB")

# Also save a quick reference of unique questions (for question embedding later)
unique_q = df.drop_duplicates(subset=['question_id'])[
    ['question_id', 'parent_question_id', 'block_name', 'question_type',
     'selector', 'answer_type', 'question_text', 'sub_item_text',
     'full_question', 'options', 'num_options']
].sort_values('question_id')
unique_q_path = os.path.join(OUTPUT_DIR, 'unique_questions.csv')
unique_q.to_csv(unique_q_path, index=False)
print(f"  Saved unique questions reference: {unique_q_path}")
print(f"  Total unique questions: {len(unique_q)}")

print("\n" + "=" * 60)
print("MASTER TABLE BUILD COMPLETE!")
print("=" * 60)
