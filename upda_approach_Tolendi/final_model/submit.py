from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests


API_URL = "https://blackboxhackathon-production.up.railway.app"
DEFAULT_PREDICTIONS_PATH = Path("artifacts/final_test_ml_predictions.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit prediction JSON to the BlackBox Hackathon scoring API.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--token", type=str, default=os.environ.get("BLACKBOX_TEAM_TOKEN", ""))
    parser.add_argument("--check-submissions", action="store_true")
    return parser.parse_args()


def load_predictions(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_token(token: str) -> str:
    token = token.strip()
    if not token:
        raise SystemExit(
            "Missing API token. Pass --token or set BLACKBOX_TEAM_TOKEN in the environment."
        )
    return token


def submit_predictions(predictions, token: str) -> None:
    resp = requests.post(
        f"{API_URL}/score",
        headers={"Authorization": f"Bearer {token}"},
        json=predictions,
        timeout=120,
    )
    print(resp.text)


def check_submissions(token: str) -> None:
    resp = requests.get(
        f"{API_URL}/submissions",
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    print(resp.text)


if __name__ == "__main__":
    args = parse_args()
    token = require_token(args.token)
    if args.check_submissions:
        check_submissions(token)
    else:
        predictions = load_predictions(args.predictions)
        print(f"Submitting {len(predictions)} predictions from {args.predictions}...\n")
        submit_predictions(predictions, token)
