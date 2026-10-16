"""Add a `broken_date` column (committer date of broken_hash) to annotated_cases.csv.

The Encoder uses it to split train/test chronologically. Usage:

    python -m data_processing.commit_dates annotated_cases.csv annotated_cases_dated.csv --token <GITHUB_TOKEN>

Dates are cached in <out>.cache.json, so an interrupted run can simply be restarted.
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.github.com/repos/{repo}/commits/{sha}"


def fetch_commit_date(repo, sha, token, session=requests):
    """Committer date (ISO 8601) of a commit, or None if GitHub no longer has it."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    for _ in range(5):
        response = session.get(API.format(repo=repo, sha=sha), headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()["commit"]["committer"]["date"]
        if response.status_code in (404, 409, 422):  # deleted repo / force-pushed away commit
            return None
        if response.status_code in (403, 429) and response.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            time.sleep(max(reset - time.time(), 1) + 1)
        else:
            time.sleep(5)
    raise RuntimeError(f"Could not fetch {repo}@{sha}: HTTP {response.status_code}")


def add_commit_dates(csv_in, csv_out, token, session=requests):
    df = pd.read_csv(csv_in, delimiter="|")
    cache_path = Path(str(csv_out) + ".cache.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    pairs = df[["repository_name", "broken_hash"]].drop_duplicates().itertuples(index=False)
    for i, (repo, sha) in enumerate(pairs, 1):
        key = f"{repo}@{sha}"
        if key not in cache:
            cache[key] = fetch_commit_date(repo, sha, token, session)
            cache_path.write_text(json.dumps(cache))
        if i % 100 == 0:
            print(f"{i} commits processed")

    df["broken_date"] = [cache[f"{r}@{s}"] for r, s in zip(df["repository_name"], df["broken_hash"])]
    missing = df["broken_date"].isna().sum()
    if missing:
        print(f"{missing} rows have no date (commit gone from GitHub); the Encoder drops them.")
    df.to_csv(csv_out, sep="|", index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_in")
    parser.add_argument("csv_out")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    args = parser.parse_args()
    add_commit_dates(args.csv_in, args.csv_out, args.token)
