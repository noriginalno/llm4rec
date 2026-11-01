"""Strict Kaggle CSV serialization with lossless identifiers."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterable


def _programs(values: Iterable[int]) -> list[int]:
    result = []
    for value in values:
        if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
            raise ValueError("Program identifiers must be positive decimal integers")
        result.append(int(value))
    if len(result) != 5 or len(set(result)) != 5 or any(i <= 0 for i in result):
        raise ValueError("Exactly five distinct positive program IDs are required")
    return result


def _users(users: Iterable[str]) -> list[str]:
    users = list(users)
    if any(not isinstance(u, str) or not u or u.strip() != u for u in users):
        raise ValueError("User identifiers must be nonempty exact strings")
    if not users or len(set(users)) != len(users):
        raise ValueError("Template users must be nonempty and unique")
    return users


def write_submission(
    path: str | Path,
    template_users: list[str],
    predictions: dict[str, list[int]],
    *,
    force: bool = False,
) -> dict:
    path = Path(path)
    users = _users(template_users)
    if set(predictions) != set(users):
        raise ValueError("Prediction users must match the template exactly")
    parsed = {user: _programs(predictions[user]) for user in users}
    if path.exists() and not force:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=".submission-", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["user_id", "tv_show_id"])
            for user in users:
                writer.writerow([user, " ".join(map(str, parsed[user]))])
        if not force:
            # A hard link publishes the complete file atomically without replacing
            # a file created by another process after the initial existence check.
            os.link(temporary, path)
            temporary.unlink()
        else:
            temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(path), "rows": len(users), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_submission(
    path: str | Path,
    template_users: list[str],
    allowed_ids: Iterable[int],
) -> dict:
    path = Path(path)
    users = _users(template_users)
    allowed = {int(i) for i in allowed_ids}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["user_id", "tv_show_id"]:
            raise ValueError("Header must be user_id,tv_show_id")
        rows = list(reader)
    if any(len(row) != 2 for row in rows):
        raise ValueError("Submission must have exactly two columns")
    if [row[0] for row in rows] != users:
        raise ValueError("Submission users/order differ from the exact template")
    for user, text in rows:
        if not re.fullmatch(r"[0-9]+(?: [0-9]+){4}", text):
            raise ValueError(f"Invalid prediction spacing/format for {user}")
        programs = _programs(text.split(" "))
        if not set(programs).issubset(allowed):
            raise ValueError(f"Out-of-catalog prediction for {user}")
    return {"valid": True, "rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def copy_submission_exclusively(source: str | Path, destination: str | Path) -> None:
    """Publish exact source bytes atomically, never replacing a destination."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with source.open("rb") as reader, tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=".selected-", delete=False,
        ) as writer:
            temporary = Path(writer.name)
            shutil.copyfileobj(reader, writer)
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
