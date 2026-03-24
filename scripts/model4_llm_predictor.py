"""
Model 4: LLM-based Respondent Simulator
Predicts survey answers by simulating each person using their profile.

Usage:
  python scripts/model4_llm_predictor.py <test_input.json> <output.json>

Requires: ANTHROPIC_API_KEY environment variable

Authors: Tanmay
"""

import json
import os
import sys
import time
import re
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSONAS_DIR = PROJECT_ROOT / "personas_text"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ============================================================
# CONFIGURATION
# ============================================================
MODEL = "claude-sonnet-4-20250514"  # fast + cheap for 233 x N calls
MAX_TOKENS = 10
TEMPERATURE = 0.3  # low temp = more deterministic
REQUESTS_PER_MINUTE = 50  # rate limit safety
DELAY_BETWEEN_CALLS = 60.0 / REQUESTS_PER_MINUTE

# ============================================================
# [1] LOAD PERSONA TEXT FILES
# ============================================================
print("[1/6] Loading persona text files...")

persona_texts = {}
for fpath in sorted(PERSONAS_DIR.glob("*_persona.txt")):
    pid = fpath.stem.replace("_persona", "")
    persona_texts[pid] = fpath.read_text(encoding="utf-8")

print(f"  Loaded {len(persona_texts)} personas")

# ============================================================
# [2] BUILD COMPRESSED TRAIT SUMMARIES
# ============================================================
print("[2/6] Building compressed trait summaries...")

def build_trait_summary(persona_text):
    """Extract key traits into a compressed summary."""
    lines = persona_text.split("\n")

    # Extract score lines
    scores = {}
    for line in lines:
        if "=" in line and ("score" in line.lower() or "percentile" in line.lower()):
            match = re.search(r'(\w+)\s*=\s*([\d.\-]+)\s*\((\d+)(?:st|nd|rd|th)\s*percentile\)', line)
            if match:
                name = match.group(1)
                value = float(match.group(2))
                pctl = int(match.group(3))
                scores[name] = {"value": value, "percentile": pctl}

    # Extract demographics
    demo = {}
    demo_keys = ["Gender:", "Age:", "Education level:", "Race:", "Religion:",
                 "Political affiliation:", "Income:", "Political views:",
                 "Employment status:", "Marital status:", "Geographic region:"]
    for line in lines:
        for key in demo_keys:
            if key in line:
                demo[key.replace(":", "")] = line.split(key)[-1].strip()

    # Build compressed summary
    summary_parts = []

    # Demographics
    demo_str = ", ".join(f"{k}: {v}" for k, v in demo.items())
    if demo_str:
        summary_parts.append(f"Demographics: {demo_str}")

    # Key personality traits (high/low classification)
    trait_map = {
        "score_extraversion": "Extraversion",
        "score_agreeableness": "Agreeableness",
        "score_openness": "Openness",
        "score_neuroticism": "Neuroticism",
        "score_needforcognition": "Need for cognition",
        "score_anxiety": "Anxiety",
        "score_depression": "Depression",
        "score_riskaversion": "Risk aversion",
        "score_selfmonitor": "Self-monitoring",
        "score_maximization": "Maximization",
        "score_needforclosure": "Need for closure",
        "score_SCC": "Self-concept clarity",
        "score_socialdesirability": "Social desirability",
        "score_minimalism": "Minimalism",
        "score_BES": "Empathy",
        "score_GREEN": "Environmentalism",
    }

    trait_strs = []
    for score_key, label in trait_map.items():
        if score_key in scores:
            pctl = scores[score_key]["percentile"]
            if pctl >= 75:
                trait_strs.append(f"{label}: high ({pctl}th pctl)")
            elif pctl <= 25:
                trait_strs.append(f"{label}: low ({pctl}th pctl)")
            else:
                trait_strs.append(f"{label}: moderate ({pctl}th pctl)")

    if trait_strs:
        summary_parts.append("Personality: " + ", ".join(trait_strs))

    # Cognitive
    cog_map = {
        "crt2_score": "CRT",
        "score_numeracy": "Numeracy",
        "score_finliteracy": "Financial literacy",
        "score_fluid": "Fluid intelligence",
        "score_crystallized": "Crystallized intelligence",
    }
    cog_strs = []
    for score_key, label in cog_map.items():
        if score_key in scores:
            pctl = scores[score_key]["percentile"]
            if pctl >= 75:
                cog_strs.append(f"{label}: high ({pctl}th pctl)")
            elif pctl <= 25:
                cog_strs.append(f"{label}: low ({pctl}th pctl)")
            else:
                cog_strs.append(f"{label}: moderate ({pctl}th pctl)")

    if cog_strs:
        summary_parts.append("Cognitive: " + ", ".join(cog_strs))

    # Economic behavior
    econ_map = {
        "score_dictator_sender": "Generosity (dictator game)",
        "score_trustgame_sender": "Trust (trust game sender)",
        "score_ultimatum_sender": "Fairness (ultimatum sender)",
    }
    econ_strs = []
    for score_key, label in econ_map.items():
        if score_key in scores:
            pctl = scores[score_key]["percentile"]
            if pctl >= 75:
                econ_strs.append(f"{label}: high ({pctl}th pctl)")
            elif pctl <= 25:
                econ_strs.append(f"{label}: low ({pctl}th pctl)")
            else:
                econ_strs.append(f"{label}: moderate ({pctl}th pctl)")

    if econ_strs:
        summary_parts.append("Economic behavior: " + ", ".join(econ_strs))

    # Qualitative self-concept (extract the 3 essays)
    for marker in ["They answered:", "they answered:"]:
        idx = persona_text.find(marker)
        if idx > 0:
            # Get the text after "They answered:" up to the next section
            snippet = persona_text[idx+len(marker):idx+len(marker)+200].strip().strip('"')
            if len(snippet) > 20:
                summary_parts.append(f"Self-description: \"{snippet[:150]}...\"")
                break

    return "\n".join(summary_parts)


trait_summaries = {}
for pid, text in persona_texts.items():
    trait_summaries[pid] = build_trait_summary(text)

# Show one example
example_pid = list(trait_summaries.keys())[0]
print(f"\n  Example summary for {example_pid}:")
print(f"  {trait_summaries[example_pid][:300]}...")

# ============================================================
# [3] BUILD FEW-SHOT EXAMPLES FROM TRAINING DATA
# ============================================================
print("\n[3/6] Building few-shot examples...")

# Load master table for real answers
master = pd.read_csv(OUTPUT_DIR / "master_table.csv")
ordinal = master[master["answer_type"] == "ordinal"].copy()
ordinal["answer_position"] = pd.to_numeric(ordinal["answer_position"], errors="coerce")

# Pick 3 diverse example people for few-shot
# Use people with different profiles
example_pids = ["00a1r", "sw55g", "k8gd7"]

# Pick a known question similar to sample format
# "I prefer to work without instructions from others" — Likert 1-7
example_qid = "QID28_r1"
example_q = ordinal[ordinal["question_id"] == example_qid]

few_shot_examples = []
for pid in example_pids:
    row = example_q[example_q["person_id"] == pid]
    if not row.empty:
        answer = int(row["answer_position"].values[0])
        summary = trait_summaries.get(pid, "No profile available")
        few_shot_examples.append({
            "profile": summary[:500],
            "question": "I prefer to work without instructions from others",
            "scale": "1 (Definitely untrue) to 7 (Definitely true)",
            "answer": answer,
        })

print(f"  Built {len(few_shot_examples)} few-shot examples")

# ============================================================
# [4] PROMPT BUILDER
# ============================================================
print("[4/6] Defining prompt builder...")

def build_prompt(persona_text, trait_summary, question_text, context, options, few_shot_examples):
    """Build the full prompt for Model 4."""

    # Parse options into a clean scale string
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

    # Build few-shot section
    few_shot_str = ""
    if few_shot_examples:
        few_shot_str = "\nExamples of how different respondents answer survey questions:\n"
        for i, ex in enumerate(few_shot_examples, 1):
            few_shot_str += f"\nExample {i}:\n"
            few_shot_str += f"Profile: {ex['profile'][:300]}\n"
            few_shot_str += f"Question: {ex['question']}\n"
            few_shot_str += f"Scale: {ex['scale']}\n"
            few_shot_str += f"Answer: {ex['answer']}\n"

    # Build context section
    context_str = ""
    if context and str(context).strip() and str(context).strip().lower() != "null":
        context_str = f"\nContext: {context}\n"

    prompt = f"""You are predicting how one specific survey respondent would answer.

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

    return prompt


def parse_llm_response(response_text, options):
    """Extract a numeric answer from LLM response."""
    text = response_text.strip()

    # Try to extract first number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if not numbers:
        return None

    val = float(numbers[0])

    # Clamp to valid range
    if isinstance(options, list):
        val = max(1, min(len(options), int(round(val))))
    elif isinstance(options, str) and "to" in options:
        parts = options.replace("to", " ").split()
        nums = [int(p) for p in parts if p.lstrip("-").isdigit()]
        if len(nums) == 2:
            val = max(nums[0], min(nums[1], int(round(val))))

    return int(val)

print("  Prompt builder defined.")

# ============================================================
# [5] LLM PREDICTION LOOP
# ============================================================
print("[5/6] Running LLM predictions...")

import anthropic

client = anthropic.Anthropic()

# Load test questions
if len(sys.argv) >= 2:
    test_path = Path(sys.argv[1])
else:
    test_path = PROJECT_ROOT / "sample_test_questions.json"

if len(sys.argv) >= 3:
    output_path = Path(sys.argv[2])
else:
    output_path = OUTPUT_DIR / "model4_predictions.json"

with open(test_path, "r", encoding="utf-8") as f:
    test_questions = json.load(f)

print(f"  Test file: {test_path}")
print(f"  Questions: {len(test_questions)}")

results = []
success = 0
failed = 0
start_time = time.time()

for i, q in enumerate(test_questions):
    pid = q["person_id"]
    qtext = q.get("question_text", "")
    context = q.get("context", None)
    options = q.get("options", [])

    # Get persona text and trait summary
    persona = persona_texts.get(pid, "")
    summary = trait_summaries.get(pid, "No profile available")

    if not persona:
        # Unknown person — can't predict
        result = dict(q)
        if isinstance(options, list):
            result["predicted_answer"] = int(round((1 + len(options)) / 2))
        else:
            result["predicted_answer"] = 50
        results.append(result)
        failed += 1
        continue

    # Build prompt
    prompt = build_prompt(persona, summary, qtext, context, options, few_shot_examples)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text
        predicted = parse_llm_response(response_text, options)

        if predicted is None:
            # Parse failed — midpoint fallback
            if isinstance(options, list):
                predicted = int(round((1 + len(options)) / 2))
            else:
                predicted = 50
            failed += 1
        else:
            success += 1

    except Exception as e:
        print(f"  ERROR on {pid}/{q.get('question_id','?')}: {e}")
        if isinstance(options, list):
            predicted = int(round((1 + len(options)) / 2))
        else:
            predicted = 50
        failed += 1

    result = dict(q)
    result["predicted_answer"] = predicted
    results.append(result)

    # Progress
    if (i + 1) % 50 == 0 or (i + 1) == len(test_questions):
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(test_questions) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(test_questions)}] success={success} failed={failed} rate={rate:.1f}/s ETA={eta:.0f}s")

    # Rate limiting
    time.sleep(DELAY_BETWEEN_CALLS)

# ============================================================
# [6] SAVE AND REPORT
# ============================================================
print(f"\n[6/6] Saving predictions to {output_path}...")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

elapsed = time.time() - start_time
print(f"  Saved: {output_path}")
print(f"  Total: {len(results)} predictions")
print(f"  Success: {success}, Failed: {failed}")
print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"\n  Sample predictions:")
for r in results[:5]:
    print(f"    {r['person_id']} | {r.get('question_id','?')} | pred={r['predicted_answer']}")
