"""
Model 4 LLM Predictor — PARALLEL VERSION
Uses async workers for 5-10x faster prediction.

Usage:
  python scripts/model4_llm_parallel.py <test_input.json> <output.json> [--workers 10]
"""

import json
import os
import sys
import time
import re
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSONAS_DIR = PROJECT_ROOT / "personas_text"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 10
TEMPERATURE = 0.3
DEFAULT_WORKERS = 5

# ============================================================
# LOAD PERSONAS + BUILD SUMMARIES (same as model4_llm_predictor.py)
# ============================================================
print("[1/4] Loading personas and building summaries...")

persona_texts = {}
for fpath in sorted(PERSONAS_DIR.glob("*_persona.txt")):
    pid = fpath.stem.replace("_persona", "")
    persona_texts[pid] = fpath.read_text(encoding="utf-8")

def build_trait_summary(persona_text):
    lines = persona_text.split("\n")
    scores = {}
    for line in lines:
        if "=" in line and ("score" in line.lower() or "percentile" in line.lower()):
            match = re.search(r'(\w+)\s*=\s*([\d.\-]+)\s*\((\d+)(?:st|nd|rd|th)\s*percentile\)', line)
            if match:
                scores[match.group(1)] = {"value": float(match.group(2)), "percentile": int(match.group(3))}

    demo = {}
    for line in lines:
        for key in ["Gender:", "Age:", "Education level:", "Race:", "Religion:",
                     "Political affiliation:", "Income:", "Political views:",
                     "Employment status:", "Marital status:", "Geographic region:"]:
            if key in line:
                demo[key.replace(":", "")] = line.split(key)[-1].strip()

    parts = []
    if demo:
        parts.append("Demographics: " + ", ".join(f"{k}: {v}" for k, v in demo.items()))

    trait_map = {
        "score_extraversion": "Extraversion", "score_agreeableness": "Agreeableness",
        "score_openness": "Openness", "score_neuroticism": "Neuroticism",
        "score_needforcognition": "Need for cognition", "score_anxiety": "Anxiety",
        "score_depression": "Depression", "score_riskaversion": "Risk aversion",
        "score_selfmonitor": "Self-monitoring", "score_maximization": "Maximization",
        "score_needforclosure": "Need for closure", "score_SCC": "Self-concept clarity",
        "score_socialdesirability": "Social desirability", "score_minimalism": "Minimalism",
        "score_BES": "Empathy", "score_GREEN": "Environmentalism",
    }
    traits = []
    for sk, label in trait_map.items():
        if sk in scores:
            p = scores[sk]["percentile"]
            level = "high" if p >= 75 else ("low" if p <= 25 else "moderate")
            traits.append(f"{label}: {level} ({p}th pctl)")
    if traits:
        parts.append("Personality: " + ", ".join(traits))

    return "\n".join(parts)

trait_summaries = {pid: build_trait_summary(text) for pid, text in persona_texts.items()}
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

few_shot_examples = []
for pid in example_pids:
    row = example_q[example_q["person_id"] == pid]
    if not row.empty:
        few_shot_examples.append({
            "profile": trait_summaries.get(pid, "")[:500],
            "question": "I prefer to work without instructions from others",
            "scale": "1 (Definitely untrue) to 7 (Definitely true)",
            "answer": int(row["answer_position"].values[0]),
        })

# ============================================================
# PROMPT + PARSE
# ============================================================
def build_prompt(persona_text, trait_summary, question_text, context, options):
    if isinstance(options, list):
        scale_str = ", ".join(str(o) for o in options)
        allowed = [str(i+1) for i in range(len(options))]
        allowed_str = ", ".join(allowed)
    elif isinstance(options, str) and "to" in options:
        scale_str = options
        parts = options.replace("to", " ").split()
        nums = [p for p in parts if p.lstrip("-").isdigit()]
        allowed_str = f"integer from {nums[0]} to {nums[1]}"
    else:
        scale_str = str(options)
        allowed_str = "a single number"

    few_shot_str = ""
    if few_shot_examples:
        few_shot_str = "\nExamples of how different respondents answer survey questions:\n"
        for i, ex in enumerate(few_shot_examples, 1):
            few_shot_str += f"\nExample {i}:\nProfile: {ex['profile'][:300]}\nQuestion: {ex['question']}\nScale: {ex['scale']}\nAnswer: {ex['answer']}\n"

    context_str = f"\nContext: {context}\n" if context and str(context).strip().lower() != "null" else ""

    return f"""You are predicting how one specific survey respondent would answer.

Respondent trait summary:
{trait_summary}

Key profile details:
{persona_text[:2000]}
{few_shot_str}
Now predict for this respondent:
{context_str}
Question: {question_text}

Answer options: {scale_str}

Instructions:
- Use the profile to infer this respondent's likely position.
- Preserve individuality. Avoid defaulting to the average unless the profile strongly suggests neutrality.
- Do not predict what most people would say. Predict what THIS person would say.
- Do not explain your reasoning.
- Return ONLY {allowed_str}. Nothing else."""


def parse_response(text, options):
    numbers = re.findall(r'-?\d+\.?\d*', text.strip())
    if not numbers:
        return None
    val = float(numbers[0])
    if isinstance(options, list):
        return max(1, min(len(options), int(round(val))))
    elif isinstance(options, str) and "to" in options:
        parts = options.replace("to", " ").split()
        nums = [int(p) for p in parts if p.lstrip("-").isdigit()]
        if len(nums) == 2:
            return max(nums[0], min(nums[1], int(round(val))))
    return int(val)

# ============================================================
# PARALLEL PREDICTION
# ============================================================
import anthropic

client = anthropic.Anthropic()

def predict_one(q):
    pid = q["person_id"]
    persona = persona_texts.get(pid, "")
    summary = trait_summaries.get(pid, "No profile")
    options = q.get("options", [])

    if not persona:
        if isinstance(options, list):
            return int(round((1 + len(options)) / 2))
        return 50

    prompt = build_prompt(persona, summary, q.get("question_text", ""),
                         q.get("context"), options)
    try:
        response = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )
        result = parse_response(response.content[0].text, options)
        if result is not None:
            return result
    except Exception as e:
        pass

    if isinstance(options, list):
        return int(round((1 + len(options)) / 2))
    return 50


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
    output_path = OUTPUT_DIR / "model4_parallel_predictions.json"

n_workers = DEFAULT_WORKERS
for i, arg in enumerate(sys.argv):
    if arg == "--workers" and i + 1 < len(sys.argv):
        n_workers = int(sys.argv[i + 1])

with open(test_path, "r", encoding="utf-8") as f:
    test_questions = json.load(f)

print(f"[2/4] Loaded {len(test_questions)} questions from {test_path}")
print(f"[3/4] Running predictions with {n_workers} parallel workers...")

start = time.time()
results = [None] * len(test_questions)
success = 0
failed = 0

with ThreadPoolExecutor(max_workers=n_workers) as executor:
    future_to_idx = {executor.submit(predict_one, q): i for i, q in enumerate(test_questions)}
    done_count = 0
    for future in as_completed(future_to_idx):
        idx = future_to_idx[future]
        q = test_questions[idx]
        pred = future.result()
        result = dict(q)
        result["predicted_answer"] = pred
        results[idx] = result
        done_count += 1
        if done_count % 100 == 0 or done_count == len(test_questions):
            elapsed = time.time() - start
            rate = done_count / elapsed
            eta = (len(test_questions) - done_count) / rate if rate > 0 else 0
            print(f"  [{done_count}/{len(test_questions)}] {rate:.1f}/s ETA={eta:.0f}s")

elapsed = time.time() - start
print(f"\n[4/4] Saving to {output_path}")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

filled = sum(1 for r in results if r and r.get("predicted_answer") is not None)
print(f"  Done: {filled}/{len(test_questions)} in {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"  Rate: {len(test_questions)/elapsed:.1f} predictions/sec")
