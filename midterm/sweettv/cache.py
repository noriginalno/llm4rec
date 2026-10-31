"""Versioned, non-executable Parquet/JSON snapshots of prepared observations.

``data_source_sha256`` is REQUIRED: the caller supplies the SHA-256 of the
preprocessor source (``sweettv/data.py``), and supplies the expected digest again
when loading. Both manifest names ``data_source_sha256`` and
``preprocessor_code_sha256`` record that exact caller-provided string. Neither
is a raw-data authentication claim. Original raw hashes/configuration remain in
the unchanged audit; this module does not reread raw CSVs or establish freshness
or authenticity of them. A forged manifest is not an authenticated source.

Normalized columns have explicit Arrow schemas, including empty tables. JSON
string columns, never inferred Arrow structs, preserve heterogeneous metadata.
An additional JSON payload for each complete original row preserves extra keys,
missing optional keys, and null versus absent values without changing semantics.
Rows are never sorted, deduplicated, or reprocessed by the cache.

Only four fixed payload basenames and manifest.json are accepted. The manifest
is written last; every payload's SHA-256 and byte count is checked before any
Parquet reader is opened. Hashes cover all payloads, not the manifest itself
(which cannot contain its own hash). Loading uses verified open file handles;
cache contents must not be concurrently modified in place during a load.

This avoids repeated CSV parsing/reconstruction, NOT materialization: PreparedData
lists, Arrow tables, and decoded Python rows still require memory. It is not an
out-of-core API. Failed saves may leave an incomplete directory, never a valid
manifest; that directory is not deleted or overwritten automatically.

Usage::

    code_sha = hashlib.sha256(Path("sweettv/data.py").read_bytes()).hexdigest()
    save_prepared(prepared, Path("cache/prepared-v1"), data_source_sha256=code_sha)
    prepared = load_prepared(Path("cache/prepared-v1"), data_source_sha256=code_sha)

The save directory must be NEW. There is no pickle, joblib, executable payload,
raw-file access, authentication, or implicit code-hash-validation bypass.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq

from .data import PreparedData


CACHE_VERSION = 1
_PAYLOADS = ("events.parquet", "schedules.parquet", "audit.json", "template_users.json")
_MANIFEST = "manifest.json"
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1

_EVENT_SCHEMA = pa.schema([
    pa.field("user_id", pa.string(), nullable=False),
    pa.field("tv_show_id", pa.int64(), nullable=False),
    pa.field("start", pa.int64(), nullable=False),
    pa.field("end", pa.int64(), nullable=False),
    pa.field("airing_id", pa.string(), nullable=False),
    pa.field("coverage", pa.float64(), nullable=True),
    pa.field("row_json", pa.string(), nullable=False),
])
_SCHEDULE_SCHEMA = pa.schema([
    pa.field("airing_id", pa.string(), nullable=False),
    pa.field("channel_id", pa.string(), nullable=False),
    pa.field("start", pa.int64(), nullable=False),
    pa.field("end", pa.int64(), nullable=False),
    pa.field("tv_show_id", pa.int64(), nullable=False),
    pa.field("metadata_json", pa.string(), nullable=False),
    pa.field("quarantined", pa.bool_(), nullable=False),
    pa.field("row_json", pa.string(), nullable=False),
])
_SCHEMAS = {"events.parquet": _EVENT_SCHEMA, "schedules.parquet": _SCHEDULE_SCHEMA}


class CacheValidationError(ValueError):
    """An incompatible, incomplete, corrupt, or malformed prepared-data cache."""


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256 string")
    return value


def _json_native(value: Any) -> None:
    """Reject lossy JSON coercions (e.g., integer object keys and tuples)."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if isinstance(value, list):
        for child in value:
            _json_native(child)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings; no coercion is performed")
            _json_native(child)
        return
    raise TypeError(f"Expected JSON-native data, not {type(value).__name__}")


def _json_text(value: Any) -> str:
    _json_native(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON number: {value}")


def _read_json(value: bytes | str, name: str) -> Any:
    try:
        result = json.loads(
            value, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        _json_native(result)  # Also rejects finite-syntax floats overflowing to inf.
        return result
    except (ValueError, TypeError, UnicodeError) as exc:
        raise CacheValidationError(f"Invalid JSON in {name}: {exc}") from exc


def _typed_row(row: dict, filename: str) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"{filename} rows must be dictionaries")
    strings = ("user_id", "airing_id") if filename == "events.parquet" else (
        "airing_id", "channel_id"
    )
    integers = ("tv_show_id", "start", "end")
    missing = set((*strings, *integers)) - row.keys()
    if missing:
        raise ValueError(f"{filename} row is missing required fields: {sorted(missing)}")
    result = {}
    for key in strings:
        if not isinstance(row[key], str):
            raise ValueError(f"{filename} {key} must already be a string")
        result[key] = row[key]
    for key in integers:
        value = row[key]
        if type(value) is not int or not _INT64_MIN <= value <= _INT64_MAX:
            raise ValueError(f"{filename} {key} must be an int64 integer, not bool/float")
        result[key] = value
    if filename == "events.parquet":
        coverage = row.get("coverage")
        if coverage is not None and (
            type(coverage) not in (int, float) or not math.isfinite(coverage)
        ):
            raise ValueError("Event coverage must be finite numeric data or null")
        result["coverage"] = float(coverage) if coverage is not None else None
    else:
        metadata = row.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Schedule metadata must be a JSON dictionary or null")
        result["metadata_json"] = _json_text(metadata)  # None -> JSON null, NOT "None".
        quarantined = row.get("quarantined", False)
        if type(quarantined) is not bool:
            raise ValueError("Schedule quarantined must be boolean when present")
        result["quarantined"] = quarantined
    return result


def _table(rows: list[dict], filename: str) -> pa.Table:
    if not isinstance(rows, list):
        raise TypeError(f"{filename} requires a list of prepared rows")
    encoded = []
    for row in rows:
        typed = _typed_row(row, filename)
        typed["row_json"] = _json_text(row)
        encoded.append(typed)
    return pa.Table.from_pylist(encoded, schema=_SCHEMAS[filename])


def _open_regular(path: Path) -> BinaryIO:
    """Disallow symlinks/devices; keep one descriptor for verification and use."""
    if not stat.S_ISREG(path.lstat().st_mode):
        raise CacheValidationError(f"Cache entry must be a regular, nonsymlink file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CacheValidationError(f"Cache entry is not a regular file: {path.name}")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _file_info(stream: BinaryIO) -> dict:
    stream.seek(0)
    hasher, size = hashlib.sha256(), 0
    while block := stream.read(1_048_576):
        hasher.update(block)
        size += len(block)
    stream.seek(0)
    return {"sha256": hasher.hexdigest(), "bytes": size}


def _write_json(path: Path, value: Any) -> None:
    payload = (_json_text(value) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)


def _template_users(value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(user, str) for user in value):
        raise ValueError("template_users must be a JSON list of string user IDs")
    return value


def save_prepared(
    prepared: PreparedData, directory: Path, *, data_source_sha256: str,
) -> dict:
    """Create a NEW cache directory, returning the exact JSON-native manifest.

    No destination is overwritten, including an existing empty directory. The
    caller must hash the preprocessor version used to prepare this snapshot;
    this utility deliberately does not inspect a concurrently changing data.py.
    """
    source_sha = _require_sha256(data_source_sha256, "data_source_sha256")
    if not isinstance(prepared, PreparedData):
        raise TypeError("prepared must be PreparedData")
    _template_users(prepared.template_users)
    if not isinstance(prepared.audit, dict):
        raise TypeError("audit must be a JSON-native dictionary")
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    for filename, rows in (
        ("events.parquet", prepared.events), ("schedules.parquet", prepared.schedules)
    ):
        table = _table(rows, filename)
        with (directory / filename).open("xb") as stream:
            pq.write_table(table, stream, compression="zstd")
        del table
    _write_json(directory / "audit.json", prepared.audit)
    _write_json(directory / "template_users.json", prepared.template_users)
    files = {}
    for filename in _PAYLOADS:
        with _open_regular(directory / filename) as stream:
            files[filename] = _file_info(stream)
    manifest = {
        "version": CACHE_VERSION,
        "data_source_sha256": source_sha,
        "preprocessor_code_sha256": source_sha,
        "files": files,
    }
    _write_json(directory / _MANIFEST, manifest)  # Commit marker, always LAST.
    return manifest


def _validate_manifest(manifest: Any, expected_source_sha: str) -> dict:
    if not isinstance(manifest, dict):
        raise CacheValidationError("Cache manifest must be a JSON object")
    if type(manifest.get("version")) is not int or manifest["version"] != CACHE_VERSION:
        raise CacheValidationError("Unsupported cache manifest version; expected version 1")
    for key in ("data_source_sha256", "preprocessor_code_sha256"):
        try:
            actual = _require_sha256(manifest.get(key), key)
        except ValueError as exc:
            raise CacheValidationError(f"Invalid preprocessor code SHA-256: {exc}") from exc
        if actual.lower() != expected_source_sha.lower():
            raise CacheValidationError("Preprocessor code SHA-256 does not match the expected value")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(_PAYLOADS):
        raise CacheValidationError("Manifest filenames must exactly match the cache whitelist")
    for filename in _PAYLOADS:
        record = files[filename]
        if not isinstance(record, dict) or set(record) != {"sha256", "bytes"}:
            raise CacheValidationError(f"Invalid checksum/bytes record for {filename}")
        if type(record["bytes"]) is not int or record["bytes"] < 0:
            raise CacheValidationError(f"Invalid byte count for {filename}")
        try:
            _require_sha256(record["sha256"], f"{filename} checksum")
        except ValueError as exc:
            raise CacheValidationError(str(exc)) from exc
    return files


def _restore(stream: BinaryIO, filename: str) -> list[dict]:
    # Single-file reader: never infer partition columns from parent directories.
    parquet = pq.ParquetFile(stream)
    if not parquet.schema_arrow.equals(_SCHEMAS[filename], check_metadata=True):
        raise CacheValidationError(f"Unexpected typed Parquet schema in {filename}")
    table = parquet.read()
    restored = []
    for stored in table.to_pylist():
        row = _read_json(stored["row_json"], f"{filename} row_json")
        try:
            expected = _typed_row(row, filename)
        except (ValueError, TypeError) as exc:
            raise CacheValidationError(f"Invalid prepared row in {filename}: {exc}") from exc
        if any(stored[key] != value for key, value in expected.items()):
            raise CacheValidationError(f"Typed columns disagree with row_json in {filename}")
        restored.append(row)
    return restored


def load_prepared(directory: Path, *, data_source_sha256: str) -> PreparedData:
    """Verify version/code SHA/all payload hashes and bytes, then deserialize data.

    This reads only fixed local JSON/Parquet files, never raw CSVs or code from a
    cache. Checksums detect corruption relative to the manifest, not forgery or
    authenticated raw-source identity. No validation argument has a default.
    """
    expected_sha = _require_sha256(data_source_sha256, "data_source_sha256")
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise CacheValidationError("Cache directory must be an existing nonsymlink directory")
    if {entry.name for entry in directory.iterdir()} != {*_PAYLOADS, _MANIFEST}:
        raise CacheValidationError("Cache directory has missing or nonwhitelisted files")
    with _open_regular(directory / _MANIFEST) as stream:
        manifest = _read_json(stream.read(), _MANIFEST)
    expected_files = _validate_manifest(manifest, expected_sha)
    with ExitStack() as stack:
        streams = {
            filename: stack.enter_context(_open_regular(directory / filename))
            for filename in _PAYLOADS
        }
        # Finish verification for EVERY file before reading JSON payloads or
        # opening either Arrow reader. The same verified descriptors are reused.
        for filename in _PAYLOADS:
            actual = _file_info(streams[filename])
            expected = expected_files[filename]
            if (actual["bytes"] != expected["bytes"]
                    or actual["sha256"] != expected["sha256"].lower()):
                raise CacheValidationError(f"Cache checksum/bytes mismatch for {filename}")
        audit = _read_json(streams["audit.json"].read(), "audit.json")
        users = _read_json(streams["template_users.json"].read(), "template_users.json")
        if not isinstance(audit, dict):
            raise CacheValidationError("Cached audit must be a JSON dictionary")
        try:
            _template_users(users)
        except ValueError as exc:
            raise CacheValidationError(str(exc)) from exc
        events = _restore(streams["events.parquet"], "events.parquet")
        schedules = _restore(streams["schedules.parquet"], "schedules.parquet")
    return PreparedData(template_users=users, schedules=schedules, events=events, audit=audit)
