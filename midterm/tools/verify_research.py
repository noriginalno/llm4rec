#!/usr/bin/env python3
"""Independent stdlib CSV/AP checks for completed round-two result folders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def ap(prediction, truth):
    actual, seen, hits, total = set(truth), set(), 0, 0.0
    for position, item in enumerate(prediction[:5], 1):
        if item in actual and item not in seen:
            seen.add(item)
            hits += 1
            total += hits / position
    return total / min(5, len(actual)) if actual else 0.0


def verify(folder: Path):
    metrics = json.loads((folder / "result.json").read_text())
    predictions = json.loads((folder / "predictions.json").read_text())
    users = json.loads((folder.parent / "template_users.json").read_text())
    candidates = set(json.loads((folder.parent / "candidate_ids.json").read_text()))
    labels = json.loads((folder.parent / "labels.json").read_text())
    if len(users) != len(set(users)) or set(predictions) != set(users):
        raise ValueError("Template/prediction identities differ")
    with (folder / "submission.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        header, *rows = reader
    if header != ["user_id", "tv_show_id"] or len(rows) != len(users):
        raise ValueError("Invalid submission shape/header")
    for row, user in zip(rows, users):
        if len(row) != 2 or row[0] != user or row[1] != " ".join(map(str, predictions[user])):
            raise ValueError("CSV order/content mismatch")
        values = predictions[user]
        if len(values) != 5 or len(set(values)) != 5 or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0 or v not in candidates for v in values
        ):
            raise ValueError("Invalid program identity or recommendation count")
    primary = [u for u in users if len(labels[u]) == 5]
    all_mean = sum(ap(predictions[u], labels[u]) for u in users) / len(users)
    strict_mean = sum(ap(predictions[u], labels[u]) for u in primary) / len(primary) if primary else None
    for key, expected in (("all_user_map_at_5", all_mean), ("map_at_5", strict_mean)):
        value = metrics[key]
        if (value is None) != (expected is None) or (value is not None and abs(value - expected) > 1e-12):
            raise ValueError(f"Independent AP disagrees: {folder}/{key}")
    for name, cohort in (("primary_metrics.json", primary), ("diagnostic_metrics.json", users)):
        payload = json.loads((folder / name).read_text())
        if [r["user_id"] for r in payload["per_user"]] != cohort:
            raise ValueError("Per-user metric order mismatch")
        for record in payload["per_user"]:
            user = record["user_id"]
            if abs(record["ap"] - ap(predictions[user], labels[user])) > 1e-12:
                raise ValueError("Per-user AP mismatch")
    digest = hashlib.sha256((folder / "submission.csv").read_bytes()).hexdigest()
    return {"path": str(folder), "users": len(users), "strict_five_users": len(primary),
            "empty_labels": sum(not labels[u] for u in users), "map_at_5": strict_mean,
            "all_user_map_at_5": all_mean, "csv_sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    files = sorted({p for root in args.roots for p in root.glob("*/*/result.json")})
    checked = [verify(p.parent) for p in files]
    if not checked:
        raise ValueError("No completed trials to verify")
    baseline = Path("results/research-v2/explore-priors-001/development/legacy_hybrid/predictions.json")
    archive = Path("results/official-run-001/development/predictions_hybrid_balanced.json")
    parity = None
    if baseline.is_file() and archive.is_file():
        parity = json.loads(baseline.read_text()) == json.loads(archive.read_text())
        if not parity:
            raise ValueError("Unchanged legacy hybrid did not reproduce archived predictions")
    result = {"independent_implementation": "stdlib, no sweettv/research_v2 imports",
              "trials_checked": len(checked), "legacy_development_prediction_parity": parity,
              "checks": checked, "new_kaggle_submissions": 0}
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"trials_checked": len(checked), "legacy_prediction_parity": parity,
                      "output": str(args.out)}))


if __name__ == "__main__":
    main()
