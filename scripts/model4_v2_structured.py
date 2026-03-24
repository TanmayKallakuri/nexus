"""
Model 4 V2: Structured LLM predictor with confidence scores.
Returns JSON: {answer, confidence} per prediction.
Uses persona tags for richer prompts.
Supports parallel workers.

Usage:
  python scripts/model4_v2_structured.py <test.json> <output.json> [--workers 5]
"""

import json
import os
import re
import sys
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSONAS_DIR = PROJECT_ROOT / "personas_text"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TAGS_PATH = OUTPUT_DIR / "persona_tags.json"

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 50
TEMPERATURE = 0.2
DEFAULT_WORKERS = 5

client = anthropic.Anthropic()

# ============================================================
# LOAD DATA
# ============================================================
print("[1/4] Loading data...")

persona_texts = {}
for fpath in sorted(PERSONAS_DIR.glob("*_persona.txt")):
    pid = fpath.stem.replace("_persona", "")
    persona_texts[pid] = fpath.read_text(encoding="utf-8")

# Load persona tags if available
persona_tags = {}
if TAGS_PATH.exists():
    persona_tags = json.loads(TAGS_PATH.read_text())
    print(f"  Loaded persona tags for {len(persona_tags)} people")
else:
    print("  No persona tags found. Run extract_persona_tags.py first.")

print(f"  Loaded {len(persona_texts)} personas")

# ============================================================
# FEW-SHOT EXAMPLES
# ============================================================
import pandas as pd
master = pd.read_csv(OUTPUT_DIR / "master_table.csv")
ordinal = master[master["answer_type"] == "ordinal"].copy()
ordinal["answer_position"] = pd.to_numeric(ordinal["answer_position"], errors="coerce")

example_qid = "QID28_r1"
example_pids = ["00a1r", "sw55g", "k8gd7"]
example_q = ordinal[ordinal["question_id"] == example_qid]

few_shot_str = "\nExamples:\n"
for pid in example_pids:
    row = example_q[example_q["person_id"] == pid]
    if not row.empty:
        ans = int(row["answer_position"].values[0])
        few_shot_str += f'  Person {pid}: Question "I prefer to work without instructions" (1-7) -> {{"answer": {ans}, "confidence": 0.7}}\n'

# ============================================================
# PROMPT BUILDER
# ============================================================
def build_prompt(person_id, persona_text, question_text, context, options):
    # Build options string
    if isinstance(options, list):
        scale_str = ", ".join(str(o) for o in options)
        n_opts = len(options)
    elif isinstance(options, str) and "to" in options:
        scale_str = options
        parts = options.replace("to", " ").split()
        nums = [p for p in parts if p.lstrip("-").isdigit()]
        n_opts = int(nums[1]) - int(nums[0]) + 1 if len(nums) == 2 else 101
    else:
        scale_str = str(options)
        n_opts = 5

    # Build tags section
    tags_str = ""
    if person_id in persona_tags:
        tags = persona_tags[person_id]
        tag_lines = []
        for k, v in tags.items():
            if k != "person_id" and isinstance(v, (int, float)):
                tag_lines.append(f"  {k}: {v}")
        if tag_lines:
            tags_str = "\nStructured traits:\n" + "\n".join(tag_lines) + "\n"

    # Build context
    context_str = f"\nContext: {context}\n" if context and str(context).strip().lower() != "null" else ""

    return f"""You are predicting how one specific survey respondent would answer.

Respondent profile:
{persona_text[:2000]}
{tags_str}
{few_shot_str}
Now predict for this respondent:
{context_str}
Question: {question_text}
Answer options: {scale_str}

Instructions:
- Predict what THIS specific person would answer based on their profile.
- Do NOT default to middle options unless the profile strongly suggests neutrality.
- Consider their demographics, personality, political views, and behavioral patterns.
- Return ONLY valid JSON: {{"answer": <number>, "confidence": <0.0 to 1.0>}}
- answer must be a valid option number. confidence is how sure you are.
- No explanation. Just the JSON."""


def parse_response(text, options):
    text = text.strip()
    # Try JSON parse
    try:
        match = re.search(r'\{[^{}]+\}', text)
        if match:
            data = json.loads(match.group())
            answer = data.get("answer")
            confidence = data.get("confidence", 0.5)
            if answer is not None:
                answer = float(answer)
                confidence = float(confidence)
                # Clamp
                if isinstance(options, list):
                    answer = max(1, min(len(options), int(round(answer))))
                elif isinstance(options, str) and "to" in options:
                    parts = options.replace("to", " ").split()
                    nums = [int(p) for p in parts if p.lstrip("-").isdigit()]
                    if len(nums) == 2:
                        answer = max(nums[0], min(nums[1], int(round(answer))))
                confidence = max(0.0, min(1.0, confidence))
                return int(answer), confidence
    except:
        pass

    # Fallback: extract first number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        val = float(numbers[0])
        if isinstance(options, list):
            val = max(1, min(len(options), int(round(val))))
        return int(val), 0.3  # low confidence for fallback parse
    return None, 0.0


# ============================================================
# PREDICTION
# ============================================================
def predict_one(q):
    pid = q["person_id"]
    persona = persona_texts.get(pid, "")
    options = q.get("options", [])

    if not persona:
        mid = 3 if isinstance(options, list) else 50
        return {"answer": mid, "confidence": 0.0}

    prompt = build_prompt(pid, persona, q.get("question_text", ""),
                         q.get("context"), options)
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )
        answer, confidence = parse_response(resp.content[0].text, options)
        if answer is not None:
            return {"answer": answer, "confidence": confidence}
    except:
        pass

    # Fallback
    if isinstance(options, list):
        mid = int(round((1 + len(options)) / 2))
    else:
        mid = 50
    return {"answer": mid, "confidence": 0.0}


# ============================================================
# MAIN
# ============================================================
if len(sys.argv) >= 2:
    test_path = Path(sys.argv[1])
else:
    test_path = PROJECT_ROOT / "new_questions.json"

if len(sys.argv) >= 3:
    output_path = Path(sys.argv[2])
else:
    output_path = OUTPUT_DIR / "model4_v2_predictions.json"

n_workers = DEFAULT_WORKERS
for i, arg in enumerate(sys.argv):
    if arg == "--workers" and i + 1 < len(sys.argv):
        n_workers = int(sys.argv[i + 1])

with open(test_path, "r", encoding="utf-8") as f:
    test_questions = json.load(f)

print(f"[2/4] Loaded {len(test_questions)} questions")
print(f"[3/4] Running predictions with {n_workers} workers...")

start = time.time()
results = [None] * len(test_questions)

with ThreadPoolExecutor(max_workers=n_workers) as executor:
    future_to_idx = {executor.submit(predict_one, q): i for i, q in enumerate(test_questions)}
    done_count = 0
    for future in as_completed(future_to_idx):
        idx = future_to_idx[future]
        q = test_questions[idx]
        pred = future.result()
        result = dict(q)
        result["predicted_answer"] = pred["answer"]
        result["llm_confidence"] = pred["confidence"]
        results[idx] = result
        done_count += 1
        if done_count % 100 == 0 or done_count == len(test_questions):
            elapsed = time.time() - start
            rate = done_count / elapsed if elapsed > 0 else 0
            print(f"  [{done_count}/{len(test_questions)}] {rate:.1f}/s")

elapsed = time.time() - start
print(f"[4/4] Saving to {output_path}")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Stats
confidences = [r["llm_confidence"] for r in results if r]
print(f"  Done: {len(results)} in {elapsed:.0f}s")
print(f"  Confidence: mean={np.mean(confidences):.2f}, min={np.min(confidences):.2f}, max={np.max(confidences):.2f}")
print(f"  Rate: {len(results)/elapsed:.1f}/s")
