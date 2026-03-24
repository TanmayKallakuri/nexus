"""
Submission helper for the BlackBox Hackathon scoring API.

Recommended workflow:
1. Generate predictions with `python build_blackbox_submission.py`
2. Put your team token in `TEAM_TOKEN`
3. Run `python submit.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import requests


TEAM_TOKEN = "TEAM TOKEN"
API_URL = "https://blackboxhackathon-production.up.railway.app"
DEFAULT_PREDICTIONS_PATH = Path("artifacts/blackbox_submission_predictions.json")


def load_predictions(path: Path = DEFAULT_PREDICTIONS_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def submit_predictions(predictions):
    resp = requests.post(
        f"{API_URL}/score",
        headers={"Authorization": f"Bearer {TEAM_TOKEN}"},
        json=predictions,
        timeout=120,
    )

    if resp.status_code == 200:
        print(resp.text)
    elif resp.status_code == 401:
        print("ERROR: Invalid team token. Check your TEAM_TOKEN.")
    elif resp.status_code == 429:
        print("ERROR: Submission limit reached. Contact organizers.")
    else:
        print(f"ERROR ({resp.status_code}):")
        print(resp.text)


def check_submissions():
    resp = requests.get(
        f"{API_URL}/submissions",
        headers={"Authorization": f"Bearer {TEAM_TOKEN}"},
        timeout=120,
    )
    print(resp.text)


if __name__ == "__main__":
    predictions = load_predictions()
    print(f"Submitting {len(predictions)} predictions from {DEFAULT_PREDICTIONS_PATH}...\n")
    submit_predictions(predictions)
