"""
One-time LLM persona tag extraction.
Runs Claude once per person to extract stable structured traits.
Results cached to outputs/persona_tags.json.

Usage: python scripts/extract_persona_tags.py [--workers 5]
Cost: ~$1 for 233 people
"""

import json
import os
import re
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSONAS_DIR = PROJECT_ROOT / "personas_text"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "persona_tags.json"

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 200
TEMPERATURE = 0.0
DEFAULT_WORKERS = 5

client = anthropic.Anthropic()

TAG_PROMPT = """Read this respondent profile and extract structured tags.

Profile:
{persona_text}

Return ONLY valid JSON with these fields (each 1-5 scale, 1=very low, 5=very high):

{{
  "ideology_liberal_conservative": <1=very liberal, 3=moderate, 5=very conservative>,
  "media_trust": <1=very distrustful, 5=very trusting>,
  "institutional_trust": <1=very distrustful, 5=very trusting>,
  "economic_anxiety": <1=very secure, 5=very anxious>,
  "social_traditionalism": <1=very progressive, 5=very traditional>,
  "racial_progressivism": <1=low, 5=high>,
  "civic_engagement": <1=disengaged, 5=very engaged>,
  "political_certainty": <1=uncertain, 5=very certain>,
  "risk_tolerance": <1=very risk averse, 5=very risk seeking>,
  "social_conformity": <1=nonconformist, 5=highly conformist>,
  "openness_to_experience": <1=closed, 5=very open>,
  "financial_security": <1=insecure, 5=very secure>
}}

Return ONLY the JSON object. No explanation."""


def extract_tags(person_id, persona_text):
    prompt = TAG_PROMPT.format(persona_text=persona_text[:3000])
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text.strip()
        # Extract JSON from response
        match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if match:
            tags = json.loads(match.group())
            tags["person_id"] = person_id
            return tags
    except Exception as e:
        pass
    return {"person_id": person_id}


if __name__ == "__main__":
    # Check if already cached
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text())
        print(f"Cache exists: {len(existing)} people tagged")
        if len(existing) >= 233:
            print("All people tagged. Skipping.")
            sys.exit(0)

    n_workers = DEFAULT_WORKERS
    for i, arg in enumerate(sys.argv):
        if arg == "--workers" and i + 1 < len(sys.argv):
            n_workers = int(sys.argv[i + 1])

    # Load personas
    persona_texts = {}
    for fpath in sorted(PERSONAS_DIR.glob("*_persona.txt")):
        pid = fpath.stem.replace("_persona", "")
        persona_texts[pid] = fpath.read_text(encoding="utf-8")

    print(f"Extracting tags for {len(persona_texts)} people with {n_workers} workers...")
    start = time.time()

    results = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(extract_tags, pid, text): pid
                   for pid, text in persona_texts.items()}
        done = 0
        for future in as_completed(futures):
            pid = futures[future]
            tags = future.result()
            results[pid] = tags
            done += 1
            if done % 50 == 0 or done == len(persona_texts):
                elapsed = time.time() - start
                print(f"  [{done}/{len(persona_texts)}] {done/elapsed:.1f}/s")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Time: {elapsed:.0f}s. Tags extracted for {len(results)} people.")
