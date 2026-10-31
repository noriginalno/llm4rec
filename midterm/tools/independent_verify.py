#!/usr/bin/env python3
"""Independently check SWEET.TV CSV files and optionally recompute MAP@5.

This file uses only the Python standard library. It deliberately imports no
``sweettv`` modules, third-party packages, production parsers, or rankers.

Example (paths are relative to the midterm directory)::

    python3 -I tools/independent_verify.py \
        --submission outputs/submission.csv --template data/sample_submission.csv \
        --allowed-ids outputs/candidate_ids.json

``--allowed-ids`` is a JSON list of positive integer raw program IDs. Optional
``--predictions-json`` and ``--labels-json`` contain objects mapping string user
IDs to lists of positive integer raw program IDs. ``--users-json`` supplies the
explicit ordered evaluation cohort; all three optional arguments go together.
That evaluation can describe a separate validation cohort: it is not asserted to
be the same predictions as the submission. Missing prediction users are errors;
missing label users have empty labels and contribute zero, not a smaller cohort.

AP uses first-hit ranks, keeps duplicate recommendation slots occupied, and
divides by min(k, number of distinct relevant IDs). Precision divides by k even
for a short list. Exact rational arithmetic is used until JSON serialization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


HEADER = ["user_id", "tv_show_id"]
FIVE_IDS = re.compile(r"[0-9]+(?: [0-9]+){4}", flags=re.ASCII)


def _require_users(users: Any, name: str) -> list[str]:
    if not isinstance(users, list):
        raise ValueError(f"{name} must be a JSON/Python list of string user IDs")
    if any(not isinstance(user, str) or not user for user in users):
        raise ValueError(f"{name} must contain nonempty string user IDs")
    if len(set(users)) != len(users):
        raise ValueError(f"{name} contains duplicate users")
    return users


def _positive_ids(values: Any, name: str) -> list[int]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list of positive integer raw IDs")
    # bool is an int subclass; it is not a program ID. No float conversion occurs.
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integer raw IDs")
    return values


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_object)


def read_template_users(path: str | Path) -> list[str]:
    """Read exact user strings/order; the template's recommendation field is ignored."""
    users: list[str] = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream, strict=True)
        if next(rows, None) != HEADER:
            raise ValueError("Template header must be exactly user_id,tv_show_id")
        for line, row in enumerate(rows, start=2):
            if len(row) != 2:
                raise ValueError(f"Template record {line} must have exactly two fields")
            users.append(row[0])
    return _require_users(users, "Template users")


def inspect_submission(
    path: str | Path, template_users: list[str], allowed_ids: set[int]
) -> dict[str, Any]:
    """Validate a CSV without calling the production writer or validator.

    Program fields allow decimal digits separated by exactly one ASCII space.
    Positivity and distinctness are checked numerically, so padded spellings do
    not conceal zero or duplicate IDs. User strings are never stripped/coerced.
    """
    _require_users(template_users, "Template users")
    if not isinstance(allowed_ids, set) or any(
        type(item) is not int or item <= 0 for item in allowed_ids
    ):
        raise ValueError("Allowed IDs must be a set of positive integer raw IDs")
    seen: set[str] = set()
    recommended: set[int] = set()
    count = 0
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream, strict=True)
        if next(rows, None) != HEADER:
            raise ValueError("Submission header must be exactly user_id,tv_show_id")
        for position, row in enumerate(rows):
            if len(row) != 2:
                raise ValueError(f"Submission record {position + 2} must have two fields")
            user, field = row
            if not user or user in seen:
                raise ValueError(f"Empty or duplicate submission user at position {position}")
            seen.add(user)
            if position >= len(template_users):
                raise ValueError("Submission contains extra users")
            if user != template_users[position]:
                raise ValueError(f"Template user/order mismatch at position {position}")
            if FIVE_IDS.fullmatch(field) is None:
                raise ValueError(f"User {user!r} must have five single-space-separated IDs")
            items = [int(token) for token in field.split(" ")]
            if any(item <= 0 for item in items) or len(set(items)) != 5:
                raise ValueError(f"User {user!r} must have five distinct positive IDs")
            if not set(items).issubset(allowed_ids):
                raise ValueError(f"User {user!r} has noncandidate program IDs")
            recommended.update(items)
            count += 1
    if count != len(template_users):
        raise ValueError(f"Submission is missing users: expected {len(template_users)}, got {count}")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "valid": True,
        "user_count": count,
        "recommendations_per_user": 5,
        "unique_recommended_ids": len(recommended),
        "sha256": digest.hexdigest(),
    }


def verify_submission(
    submission: str | Path, template: str | Path, allowed_ids: str | Path
) -> dict[str, Any]:
    allowed = _positive_ids(load_json(allowed_ids), "Allowed IDs")
    return inspect_submission(submission, read_template_users(template), set(allowed))


def _require_mapping(value: Any, name: str) -> dict[str, list[int]]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object mapping user strings to ID lists")
    for user, items in value.items():
        if not isinstance(user, str) or not user:
            raise ValueError(f"{name} user IDs must be nonempty strings")
        _positive_ids(items, f"{name}[{user!r}]")
    return value


def recompute_metrics(
    predictions: dict[str, list[int]],
    labels: dict[str, list[int]],
    users: list[str],
    k: int = 5,
) -> dict[str, Any]:
    """Recompute macro metrics from distinct relevant first-hit rank positions."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    _require_mapping(predictions, "Predictions")
    _require_mapping(labels, "Labels")
    _require_users(users, "Evaluation users")
    totals = [Fraction(0), Fraction(0), Fraction(0)]
    records: list[dict[str, Any]] = []
    for user in users:
        if user not in predictions:
            raise ValueError(f"Missing predictions for evaluation user {user!r}")
        relevant = set(labels.get(user, []))
        first_rank: dict[int, int] = {}
        for rank, item in enumerate(predictions[user][:k], start=1):
            first_rank.setdefault(item, rank)
        hit_ranks = sorted(rank for item, rank in first_rank.items() if item in relevant)
        hits = len(hit_ranks)
        relevant_count = len(relevant)
        # The j-th distinct relevant first hit contributes j / its original rank.
        numerator = sum(
            (Fraction(j, rank) for j, rank in enumerate(hit_ranks, start=1)), Fraction(0)
        )
        ap = numerator / min(k, relevant_count) if relevant_count else Fraction(0)
        precision = Fraction(hits, k)
        recall = Fraction(hits, relevant_count) if relevant_count else Fraction(0)
        for index, metric in enumerate((ap, precision, recall)):
            totals[index] += metric
        records.append({
            "user_id": user,
            "ap": float(ap),
            "precision": float(precision),
            "recall": float(recall),
            "hits": hits,
            "relevant_count": relevant_count,
        })
    count = len(users)
    means = [float(total / count) if count else 0.0 for total in totals]
    return {
        f"map_at_{k}": means[0],
        f"precision_at_{k}": means[1],
        f"recall_at_{k}": means[2],
        "user_count": count,
        "per_user": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--allowed-ids", required=True, type=Path, help="JSON list of integer raw IDs")
    parser.add_argument("--predictions-json", type=Path)
    parser.add_argument("--labels-json", type=Path)
    parser.add_argument("--users-json", type=Path)
    args = parser.parse_args(argv)
    metric_paths = (args.predictions_json, args.labels_json, args.users_json)
    if any(path is not None for path in metric_paths) and not all(
        path is not None for path in metric_paths
    ):
        parser.error("--predictions-json, --labels-json, and --users-json must be supplied together")
    try:
        result: dict[str, Any] = {
            "submission": verify_submission(args.submission, args.template, args.allowed_ids)
        }
        if args.predictions_json is not None:
            result["evaluation"] = recompute_metrics(
                load_json(args.predictions_json),
                load_json(args.labels_json),
                load_json(args.users_json),
            )
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        print(json.dumps({"valid": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
