"""Auditable CSV ingestion, broadcast reconstruction, and temporal folds.

``RawSchema`` retains the official documentation's endpoint defaults rather than
silently changing them to fit a particular export. Bounded official-source
samples now show log stop_time/duraton (source typo) and day-first EPG datetimes;
configure those exact columns and formats explicitly. Samples do not establish
full-source consistency, reconstruction fidelity, or the recorded time zone.
Only BOM/outer whitespace is removed from headers; configure exact names if the
source differs. Defaults:

* logs: user_id, vsetv_id, start_time, end_time, optional duration;
* EPG: channel_id, tv_show_id, start_time, duration (seconds); end is derived;
* template: user_id (other columns are ignored, never parsed as behavior).

No EPG end column is documented, so ``schedule_end`` defaults to None. Alternate
headers, including legacy start/end names, require an explicit ``RawSchema``.
Legacy logical filename constants are unchanged. Pass ``file_names`` to map
these roles to different source basenames explicitly; no files are discovered,
renamed, or rewritten. Per-file parsing options remain keyed by logical names.

Numeric timestamps are auto-detected only if exactly one of seconds/milliseconds
lands in [2000-01-01, 2100-01-01) UTC. This is a detection guard, not a statement
about dataset dates. Other epochs require an explicit ``timestamp_unit``.
Subsecond values are rejected rather than rounded. Non-ISO formats require an
explicit ``timestamp_format``/``timestamp_formats`` and use strptime, never
format guessing. Naive datetimes in either path require an explicit IANA
``assumed_timezone``; ambiguous/nonexistent local times fail. An explicitly
chosen UTC anchor is not evidence that the source clock was recorded in UTC.

Coverage is endpoint-based. Raw log duration is audited as a possible number of
seconds, without rescaling or using it to repair timestamps. Genuine same-channel
EPG overlaps quarantine the entire connected overlap component for labels;
metadata-only duplicates do not. Quarantined schedules remain in the known
schedule catalog, flagged, so candidate selection never depends on labels.

The initial API materializes schedules and events as lists. CSV logs stream into
per-user/channel interval lists, so memory is O(log rows + schedules + events),
not an out-of-core promise. No Cartesian log/EPG join or dense user/item matrix
is built. See ``reviews/data-contract.md`` for the assumptions and limits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from numbers import Integral
from pathlib import Path, PureWindowsPath
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from scipy.sparse import coo_matrix

from .contracts import TrainingData


LOG_FILE = "dataset11-30week.csv"
SCHEDULE_FILES = (
    "export_arh_10-19.csv",
    "export_arh_20-30.csv",
    "export_arh_31-42.csv",
)
TEMPLATE_FILE = "submission.csv"
SNIFF_BYTES = 65_536
NULL_TOKENS = frozenset({"", "null", "none", "nan", "na", "n/a", "\\n"})
_DECIMAL_ID = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+)?\Z")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_AUTO_MIN = 946_684_800  # 2000-01-01 UTC
_AUTO_MAX = 4_102_444_800  # 2100-01-01 UTC
_MAX_ITEM = 2**63 - 1


@dataclass
class PreparedData:
    template_users: list[str]
    schedules: list[dict]
    events: list[dict]
    audit: dict


@dataclass(frozen=True)
class RawSchema:
    """Documented source defaults by normalized role; no aliases are guessed.

    The documented EPG is duration-only: end is derived, with no end column
    configured by default. Set ``schedule_end`` to an exact name to opt into
    checking an additional endpoint. Set optional ``log_duration`` or
    ``schedule_end`` to None to ignore a column. EPG end may substitute for
    duration only by explicitly setting
    ``schedule_duration=None``. Log endpoints are always required because the
    source log-duration unit is unknown. Metadata keeps all unused EPG columns.
    """

    log_user: str = "user_id"
    log_channel: str = "vsetv_id"
    log_start: str = "start_time"
    log_end: str = "end_time"
    log_duration: str | None = "duration"
    schedule_channel: str = "channel_id"
    schedule_item: str = "tv_show_id"
    schedule_start: str = "start_time"
    schedule_duration: str | None = "duration"
    schedule_end: str | None = None
    template_user: str = "user_id"


class DataSchemaError(ValueError):
    """Fail-closed source error with the partial, nonbehavioral parsing audit."""

    def __init__(self, message: str, audit: dict | None = None):
        super().__init__(message)
        self.audit = audit if audit is not None else {}


def _missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in NULL_TOKENS
    )


def _decimal(value: Any, name: str) -> Decimal:
    # An already-coerced float cannot establish 64-bit identity or exact time.
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{name}: binary floats/bools are not accepted")
    if isinstance(value, Integral):
        return Decimal(int(value))
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, str) and _DECIMAL_ID.fullmatch(value.strip()):
        result = Decimal(value.strip())
    else:
        raise ValueError(f"{name}: expected an exact decimal number")
    if not result.is_finite():
        raise ValueError(f"{name}: expected a finite number")
    return result


def parse_identifier(value: Any, *, preserve_digits: bool = False) -> str | None:
    """Parse integral decimal text without float coercion; null tokens -> None.

    Plain unsigned digit user IDs retain leading zeros. Decimal-style IDs such
    as '9007199254740993.0' are normalized exactly, not through binary floats.
    Scientific notation, fractions and pre-coerced floats are rejected.
    """

    if _missing(value):
        return None
    number = _decimal(value, "identifier")
    if number != number.to_integral_value():
        raise ValueError("identifier: nonintegral decimal")
    if preserve_digits and isinstance(value, str) and value.strip().isascii():
        if value.strip().isdigit():
            return value.strip()
    return str(int(number))


def _item_id(value: Any) -> int | None:
    text = parse_identifier(value)
    if text is None:
        return None
    result = int(text)
    if result > _MAX_ITEM:
        raise ValueError("tv_show_id exceeds the int64 model interface")
    return result


def _localize_strict(value: datetime, zone: ZoneInfo) -> datetime:
    choices = set()
    for fold in (0, 1):
        candidate = value.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
        if candidate.astimezone(zone).replace(tzinfo=None) == value:
            choices.add(candidate)
    if not choices:
        raise ValueError("nonexistent local datetime (DST gap)")
    if len(choices) != 1:
        raise ValueError("ambiguous local datetime (DST fold); supply an offset")
    return choices.pop()


def _normalize_datetime(
    parsed: datetime, *, assumed_timezone: str | None, parser_kind: str, audit: dict | None,
) -> int:
    """Apply the same exactness and strict local-time policy to both parsers."""

    offset = parsed.utcoffset()
    if parsed.microsecond or (offset is not None and offset.microseconds):
        raise ValueError("subsecond timestamp cannot be represented as int seconds")
    was_naive = parsed.tzinfo is None
    if was_naive:
        if assumed_timezone is None:
            raise ValueError("timezone-naive datetime requires explicit assumed_timezone")
        try:
            parsed = _localize_strict(parsed, ZoneInfo(assumed_timezone))
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown assumed_timezone: {assumed_timezone}") from exc
    delta = parsed.astimezone(timezone.utc) - _EPOCH
    if delta.microseconds:
        raise ValueError("subsecond timestamp cannot be represented as int seconds")
    if audit is not None:
        suffix = "naive_explicit_zone" if was_naive else "with_offset"
        key = f"{parser_kind}_{suffix}"
        audit[key] = audit.get(key, 0) + 1
    return delta.days * 86_400 + delta.seconds


def _validate_timestamp_format(value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("timestamp_format must be a nonempty strptime format string or None")


def parse_timestamp(
    value: Any,
    *,
    assumed_timezone: str | None = None,
    timestamp_unit: str | None = None,
    timestamp_format: str | None = None,
    audit: dict | None = None,
) -> int:
    """Return integer seconds from ISO/UNIX input or an explicit strptime format.

    A supplied format is authoritative: only strptime is tried, without ISO or
    numeric fallback. Naive parsed clocks still need an explicit zone/anchor;
    their normalized seconds do not prove the source's recorded time zone.
    """

    if timestamp_unit not in (None, "seconds", "milliseconds"):
        raise ValueError("timestamp_unit must be seconds, milliseconds, or None")
    _validate_timestamp_format(timestamp_format)
    if timestamp_format is not None:
        if not isinstance(value, str) or _missing(value):
            raise ValueError("explicit timestamp_format requires nonempty text input")
        try:
            parsed = datetime.strptime(value.strip(), timestamp_format)
        except ValueError as exc:
            raise ValueError(f"timestamp does not match explicit timestamp_format {timestamp_format!r}") from exc
        return _normalize_datetime(parsed, assumed_timezone=assumed_timezone,
                                   parser_kind="explicit_format", audit=audit)
    numeric = isinstance(value, (Integral, Decimal, float)) or (
        isinstance(value, str) and _DECIMAL_ID.fullmatch(value.strip()) is not None
    )
    if numeric:
        number = _decimal(value, "timestamp")
        unit = timestamp_unit
        if unit is None:
            plausible = []
            for candidate, divisor in (("seconds", 1), ("milliseconds", 1000)):
                if _AUTO_MIN * divisor <= number < _AUTO_MAX * divisor:
                    plausible.append(candidate)
            if len(plausible) != 1:
                raise ValueError(
                    "ambiguous/unsupported UNIX scale; explicitly set timestamp_unit"
                )
            unit = plausible[0]
        if number != number.to_integral_value():
            raise ValueError("subsecond timestamp cannot be represented as int seconds")
        # Integer division avoids Decimal's context-dependent rounding for large
        # explicit epochs. No precision is discarded, even outside the auto guard.
        result, remainder = divmod(int(number), 1000 if unit == "milliseconds" else 1)
        if remainder:
            raise ValueError("subsecond timestamp cannot be represented as int seconds")
        if audit is not None:
            key = f"unix_{unit}"
            audit[key] = audit.get(key, 0) + 1
        return result
    if not isinstance(value, str) or _missing(value):
        raise ValueError("missing/unsupported timestamp")
    text = value.strip()
    if not re.search(r"[Tt ]\d{2}:?\d{2}", text):
        raise ValueError("expected an ISO datetime, not a date or unknown format")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ValueError("invalid ISO datetime") from exc
    # Only validated ISO text reaches this check: day-first date dots must not
    # be mistaken for fractional seconds. Inspect the original text because
    # fromisoformat can truncate precision beyond microseconds before returning.
    if any(any(digit != "0" for digit in part)
           for part in re.findall(r"[.,]([0-9]+)", text)):
        raise ValueError("subsecond timestamp cannot be represented as int seconds")
    return _normalize_datetime(parsed, assumed_timezone=assumed_timezone,
                               parser_kind="iso", audit=audit)


def _seconds(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must already be normalized integer epoch seconds")
    return int(value)


def _select_option(option: Any, filename: str) -> Any:
    if isinstance(option, Mapping):
        return option.get(filename)
    return option


def _resolve_file_names(file_names: Mapping[str, str] | None) -> dict[str, str]:
    """Validate an explicit logical-name -> source-basename adapter, before I/O."""

    expected = (LOG_FILE, *SCHEDULE_FILES, TEMPLATE_FILE)
    if file_names is not None and not isinstance(file_names, Mapping):
        raise DataSchemaError("file_names must be a mapping of logical names to source basenames")
    supplied = {} if file_names is None else dict(file_names)
    unknown = sorted(repr(key) for key in supplied if key not in expected)
    if unknown:
        raise DataSchemaError(f"file_names has unknown logical keys: {unknown}; expected {list(expected)}")
    resolved = {logical: supplied.get(logical, logical) for logical in expected}
    for logical, source in resolved.items():
        if (not isinstance(source, str) or not source.strip() or source in (".", "..")
                or any(char in source for char in ("/", "\\", "\0"))
                or PureWindowsPath(source).drive):
            raise DataSchemaError(
                f"file_names[{logical!r}] must be a nonempty relative basename "
                f"without paths or NUL characters; found {source!r}"
            )
    if len(set(resolved.values())) != len(expected):
        raise DataSchemaError("file_names must resolve to five distinct source basenames")
    return resolved


def _csv_rows(
    path: Path,
    file_audit: dict,
    *,
    encoding: str = "utf-8-sig",
    delimiter: str | None = None,
    required: Iterable[str] = (),
) -> Iterable[tuple[int, dict[str, str]]]:
    """Stream strict text rows, auditing exact headers/bytes and bounded sniffing."""

    if encoding not in ("utf-8", "utf-8-sig", "cp1251"):
        raise DataSchemaError("encoding must be utf-8, utf-8-sig, or explicit cp1251")
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as raw:
        while block := raw.read(1_048_576):
            hasher.update(block)
            size += len(block)
    file_audit.update(encoding=encoding, size_bytes=size, sha256=hasher.hexdigest(), rows=0)
    try:
        with path.open("r", encoding=encoding, errors="strict", newline="") as source:
            # Decode at most this many characters, without cutting a UTF-8 codepoint.
            sample = source.read(SNIFF_BYTES)
            source.seek(0)
            if not sample:
                raise DataSchemaError(f"{path.name}: empty CSV")
            if delimiter is None:
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                    delimiter = dialect.delimiter
                    mode = "csv.Sniffer"
                except csv.Error as exc:
                    raise DataSchemaError(
                        f"{path.name}: delimiter not unambiguous in bounded sample; "
                        "provide delimiter explicitly"
                    ) from exc
            else:
                mode = "explicit"
                if len(delimiter) != 1 or delimiter in "\r\n\0":
                    raise DataSchemaError("delimiter must be one non-newline character")
            file_audit.update(delimiter=delimiter, delimiter_source=mode,
                              sniff_sample_characters=min(len(sample), SNIFF_BYTES))
            reader = csv.reader(source, delimiter=delimiter, strict=True)
            headers = next(reader)
            normalized = [name.strip().lstrip("\ufeff").strip() for name in headers]
            if not normalized or any(not name for name in normalized):
                raise DataSchemaError(f"{path.name}: empty column name")
            if len(set(normalized)) != len(normalized):
                raise DataSchemaError(f"{path.name}: duplicate normalized column names")
            file_audit.update(raw_columns=headers, columns=normalized)
            _require_columns(dict.fromkeys(normalized), required, path.name)
            for row in reader:
                if not row:  # Ignore physical blank lines, not malformed data rows.
                    file_audit["blank_lines"] = file_audit.get("blank_lines", 0) + 1
                    continue
                if len(row) != len(normalized):
                    raise DataSchemaError(
                        f"{path.name}:{reader.line_num}: expected {len(normalized)} fields, "
                        f"found {len(row)}"
                    )
                file_audit["rows"] += 1
                yield reader.line_num, dict(zip(normalized, row))
    except UnicodeDecodeError as exc:
        raise DataSchemaError(
            f"{path.name}: strict {encoding} decoding failed; if justified by source "
            "inspection, choose encoding='cp1251' explicitly (no replacement used)"
        ) from exc
    except csv.Error as exc:
        raise DataSchemaError(f"{path.name}: malformed CSV: {exc}") from exc


def _require_columns(row: Mapping, required: Iterable[str], filename: str) -> None:
    missing = sorted(set(required) - row.keys())
    if missing:
        raise DataSchemaError(
            f"{filename}: missing exact columns {missing}; observed {sorted(row)}. "
            "Set RawSchema to the inspected source names; no fuzzy renaming is used."
        )


def _metadata_key(metadata: Mapping) -> str:
    return json.dumps(dict(metadata), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _airing_id(channel: str, start: int, end: int, item: int) -> str:
    payload = json.dumps([channel, start, end, item], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prepare_schedules(schedules: Iterable[Mapping]) -> tuple[list[dict], dict]:
    unique: dict[tuple, dict] = {}
    multiplicity: Counter = Counter()
    metadata_occurrences: dict[tuple, Counter] = defaultdict(Counter)
    seen_ids: dict[str, tuple] = {}
    audit = {
        "input_schedule_rows": 0, "missing_program_rows": 0, "zero_program_rows": 0,
        "negative_program_rows": 0, "null_channel_rows": 0, "invalid_interval_rows": 0,
        "duplicate_airing_rows": 0, "metadata_variant_duplicate_rows": 0,
        "conflict_components": 0, "conflict_unique_airings": 0,
        "conflict_affected_schedule_rows": 0,
        "conflict_policy": "quarantine all airings in each positive-overlap component",
        "duplicate_identity": "channel_id,start,end,tv_show_id (metadata excluded)",
        "metadata_variant_count_policy": "rows differing from the final canonical metadata row",
    }
    for source in schedules:
        audit["input_schedule_rows"] += 1
        item = _item_id(source.get("tv_show_id"))
        if item is None or item <= 0:
            key = "missing_program_rows" if item is None else (
                "zero_program_rows" if item == 0 else "negative_program_rows"
            )
            audit[key] += 1
            continue
        channel = source.get("channel_id")
        if _missing(channel):
            audit["null_channel_rows"] += 1
            continue
        if not isinstance(channel, str):
            raise ValueError("normalized channel_id must be a string")
        start, end = (_seconds(source[k], k) for k in ("start", "end"))
        if end <= start:
            audit["invalid_interval_rows"] += 1
            continue
        identity = (channel, start, end, item)
        supplied_id = source.get("airing_id")
        if supplied_id is not None and (not isinstance(supplied_id, str) or not supplied_id):
            raise ValueError("airing_id must be a nonempty string")
        airing = supplied_id or _airing_id(*identity)
        if airing in seen_ids and seen_ids[airing] != identity:
            raise ValueError("one airing_id refers to multiple broadcast identities")
        seen_ids[airing] = identity
        metadata = dict(source.get("metadata") or {})
        row = dict(channel_id=channel, start=start, end=end, tv_show_id=item,
                   airing_id=airing, metadata=metadata, quarantined=False)
        multiplicity[identity] += 1
        metadata_occurrences[identity][_metadata_key(metadata)] += 1
        previous = unique.get(identity)
        if previous is not None:
            audit["duplicate_airing_rows"] += 1
            # Canonical choices are independent of input ordering and viewing counts.
            previous["airing_id"] = min(previous["airing_id"], airing)
            if _metadata_key(metadata) < _metadata_key(previous["metadata"]):
                previous["metadata"] = metadata
        else:
            unique[identity] = row
    audit["metadata_variant_duplicate_rows"] = sum(
        multiplicity[identity] - metadata_occurrences[identity][_metadata_key(row["metadata"])]
        for identity, row in unique.items()
    )
    ordered = sorted(unique.values(), key=lambda s: (
        s["channel_id"], s["start"], s["end"], s["tv_show_id"], s["airing_id"]
    ))
    component: list[dict] = []
    component_end = 0

    def finish() -> None:
        if len(component) < 2:
            return
        audit["conflict_components"] += 1
        for row in component:
            row["quarantined"] = True
            audit["conflict_unique_airings"] += 1
            identity = (row["channel_id"], row["start"], row["end"], row["tv_show_id"])
            audit["conflict_affected_schedule_rows"] += multiplicity[identity]

    for row in ordered:
        if component and (row["channel_id"] != component[0]["channel_id"]
                          or row["start"] >= component_end):
            finish()
            component = []
        component.append(row)
        component_end = max(component_end, row["end"]) if len(component) > 1 else row["end"]
    finish()
    audit["unique_airings"] = len(ordered)
    audit["matchable_airings"] = len(ordered) - audit["conflict_unique_airings"]
    return ordered, audit


def _union(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def _duration_audit(value: Any, elapsed: int, audit: dict) -> None:
    if _missing(value):
        audit["raw_duration_missing_rows"] += 1
        return
    try:
        duration = _decimal(value, "log duration")
    except (ValueError, InvalidOperation):
        audit["raw_duration_invalid_rows"] += 1
        return
    if duration <= 0:
        audit["raw_duration_invalid_rows"] += 1
    if duration != elapsed:
        audit["raw_duration_mismatch_rows"] += 1


def _reconstruct(
    logs: Iterable[Mapping], schedules: list[dict], schedule_audit: dict, cutoff: int | None,
) -> tuple[list[dict], dict]:
    if cutoff is not None:
        cutoff = _seconds(cutoff, "cutoff")
    audit = dict(schedule_audit)
    audit.update({
        "input_log_rows": 0, "null_user_rows": 0, "null_log_channel_rows": 0,
        "invalid_log_interval_rows": 0, "raw_duration_missing_rows": 0,
        "raw_duration_invalid_rows": 0, "raw_duration_mismatch_rows": 0,
        "logs_clipped_at_cutoff": 0, "logs_at_or_after_cutoff": 0,
        "conflict_affected_log_rows": 0, "conflict_affected_users": 0,
        "log_rows_without_channel_schedule": 0,
        "log_rows_without_matchable_channel_schedule": 0, "merged_viewing_intervals": 0,
        "matched_user_airings": 0, "qualifying_events": 0,
        "qualifying_cutoff_crossing_airings": 0,
        "raw_duration_comparison": "numeric seconds hypothesis; endpoints authoritative; no rescaling",
        "coverage_policy": "half-open interval union; exact 5*covered >= 4*scheduled_duration",
        "cutoff": cutoff,
        "cutoff_policy": "clip viewing to before cutoff; keep full airing denominator and timestamps; "
                         "training must additionally require event.end <= cutoff",
    })
    by_channel: dict[str, list[dict]] = defaultdict(list)
    conflicts: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in schedules:
        if row["quarantined"]:
            conflicts[row["channel_id"]].append((row["start"], row["end"]))
        else:
            by_channel[row["channel_id"]].append(row)
    ends = {channel: [s["end"] for s in rows] for channel, rows in by_channel.items()}
    conflict_index = {}
    for channel, intervals in conflicts.items():
        merged = _union(intervals)
        conflict_index[channel] = (merged, [end for _, end in merged])
    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    affected_users = set()
    for row in logs:
        audit["input_log_rows"] += 1
        user, channel = row.get("user_id"), row.get("channel_id")
        if _missing(user):
            audit["null_user_rows"] += 1
            continue
        if not isinstance(user, str):
            raise ValueError("normalized user_id must be a string (no float ID conversion)")
        if _missing(channel):
            audit["null_log_channel_rows"] += 1
            continue
        if not isinstance(channel, str):
            raise ValueError("normalized channel_id must be a string")
        start, end = (_seconds(row[k], k) for k in ("start", "end"))
        _duration_audit(row.get("duration_raw", row.get("duration")), end - start, audit)
        if end <= start:
            audit["invalid_log_interval_rows"] += 1
            continue
        if cutoff is not None:
            if start >= cutoff:
                audit["logs_at_or_after_cutoff"] += 1
                continue
            if end > cutoff:
                audit["logs_clipped_at_cutoff"] += 1
                end = cutoff
        if channel in conflict_index:
            intervals, conflict_ends = conflict_index[channel]
            i = bisect_right(conflict_ends, start)
            if i < len(intervals) and intervals[i][0] < end:
                audit["conflict_affected_log_rows"] += 1
                affected_users.add(user)
        if channel not in by_channel:
            audit["log_rows_without_matchable_channel_schedule"] += 1
            if channel not in conflicts:
                audit["log_rows_without_channel_schedule"] += 1
            continue
        groups[(user, channel)].append((start, end))
    events = []
    for (user, channel), fragments in groups.items():
        merged = _union(fragments)
        audit["merged_viewing_intervals"] += len(merged)
        rows = by_channel[channel]
        covered: Counter = Counter()
        for start, end in merged:
            i = bisect_right(ends[channel], start)
            while i < len(rows) and rows[i]["start"] < end:
                airing = rows[i]
                overlap = min(end, airing["end"]) - max(start, airing["start"])
                if overlap > 0:
                    covered[i] += overlap
                i += 1
        audit["matched_user_airings"] += len(covered)
        for index, seconds in covered.items():
            airing = rows[index]
            duration = airing["end"] - airing["start"]
            if 5 * seconds >= 4 * duration:
                events.append(dict(user_id=user, tv_show_id=airing["tv_show_id"],
                                   start=airing["start"], end=airing["end"],
                                   airing_id=airing["airing_id"], coverage=seconds / duration))
                if cutoff is not None and airing["end"] > cutoff:
                    audit["qualifying_cutoff_crossing_airings"] += 1
    events.sort(key=lambda e: (e["user_id"], e["start"], e["end"], e["tv_show_id"], e["airing_id"]))
    audit["conflict_affected_users"] = len(affected_users)
    audit["qualifying_events"] = len(events)
    audit["user_channel_groups"] = len(groups)
    return events, audit


def reconstruct_events(
    logs: Iterable[Mapping], schedules: Iterable[Mapping], cutoff: int | None = None,
) -> tuple[list[dict], dict]:
    """Count each >=80%-covered airing once per user, never pool repeat airings.

    Inputs already have UTC integer seconds and string user/channel IDs. With a
    cutoff, future viewing is clipped, not the scheduled denominator. Crossing
    airings can still qualify; ``build_fold`` excludes them from training by end.
    """

    normalized, audit = _prepare_schedules(schedules)
    return _reconstruct(logs, normalized, audit, cutoff)


def load_raw(
    raw_dir: Path,
    *,
    schema: RawSchema | None = None,
    assumed_timezone: str | None = None,
    timestamp_unit: str | Mapping[str, str] | None = None,
    encoding: str | Mapping[str, str] = "utf-8-sig",
    delimiter: str | Mapping[str, str] | None = None,
    file_names: Mapping[str, str] | None = None,
    timestamp_formats: str | Mapping[str, str] | None = None,
) -> PreparedData:
    """Read the five expected files using an explicit, audited parsing contract.

    ``file_names`` maps the existing LOG_FILE/SCHEDULE_FILES/TEMPLATE_FILE values
    to explicit, distinct source basenames in raw_dir. Omitted keys retain their
    logical default names; unknown keys and paths are errors. Nothing is renamed
    or heuristically discovered. Audit files remain keyed by logical name and
    record the actual ``source_filename``; hashes cover the actual opened bytes.

    Encoding/delimiter/unit may be supplied globally or by exact *logical* name,
    even when the source basename is aliased. Actual aliases are not option keys.
    Missing per-file encoding defaults to UTF-8; missing delimiter/unit is auto.
    ``timestamp_formats`` supplies an authoritative strptime format globally or
    by logical filename; omitted/None roles keep the existing ISO/UNIX parser.
    Formats are never guessed. Configured patterns, their caller-supplied basis,
    successful parse counts, and the assumed naive-clock anchor are audited.
    This function never authorizes, downloads, modifies raw files, or installs.
    """

    schema = schema or RawSchema()
    raw_dir = Path(raw_dir)
    resolved_names = _resolve_file_names(file_names)
    if isinstance(timestamp_formats, Mapping):
        unknown = sorted(repr(key) for key in timestamp_formats if key not in resolved_names)
        if unknown:
            raise DataSchemaError(f"timestamp_formats has unknown logical keys: {unknown}")
        configured_formats = dict(timestamp_formats)
        format_values = configured_formats.values()
    else:
        configured_formats = timestamp_formats
        format_values = (timestamp_formats,)
    try:
        for pattern in format_values:
            _validate_timestamp_format(pattern)
    except ValueError as exc:
        raise DataSchemaError(f"invalid timestamp_formats: {exc}") from exc
    missing = [source for source in resolved_names.values() if not (raw_dir / source).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw CSV files: {missing}")
    if schema.schedule_duration is None and schema.schedule_end is None:
        raise DataSchemaError("configure schedule_duration or schedule_end")
    audit: dict[str, Any] = {
        "schema": asdict(schema), "schema_verified_on_actual_source": False,
        "file_names": dict(resolved_names),
        "file_name_policy": "explicit logical-name to source-basename map; omitted roles unchanged; "
                            "files/options keyed by logical names; no discovery or raw-file changes",
        "schema_default_basis": "official competition documented log start_time/end_time and "
                                "EPG start_time/duration, confirmed in the author-response request; "
                                "not actual CSV-byte verification or a new source retrieval",
        "files": {}, "timestamp_formats": {}, "identifier_formats": {},
        "timestamp_format_config": configured_formats,
        "timestamp_format_policy": "caller-supplied strptime formats only; missing/None logical roles "
                                   "use ISO/UNIX parsing; no format guessing or fallback on mismatch",
        "assumed_timezone": assumed_timezone,
        "assumed_timezone_basis": "explicit caller-chosen naive-clock zone/anchor; not evidence of "
                                  "the source's recorded timezone" if assumed_timezone is not None else
                                  "none; timezone-naive datetimes are rejected",
        "timestamp_unit": dict(timestamp_unit) if isinstance(timestamp_unit, Mapping) else timestamp_unit,
        "null_tokens": sorted(NULL_TOKENS),
        "log_parse": {"null_user_rows": 0, "null_channel_rows": 0},
        "schedule_parse": {"missing_program_rows": 0, "zero_program_rows": 0,
                           "negative_program_rows": 0, "null_channel_rows": 0,
                           "invalid_duration_rows": 0, "duration_end_mismatch_rows": 0},
        "unknowns": [
            "Configured parsing and bounded source samples do not certify full-source header/format "
            "consistency, dates/week coverage, row counts, recorded timezone or all log-duration units.",
            "Organizer fragment, duplicate, conflict, DST and airing-boundary policies are unknown; "
            "the documented reconstruction is an engineering assumption, not organizer fidelity.",
            "Input IDs already rounded by an upstream float export cannot be recovered or detected reliably.",
        ],
        "memory_policy": "in-memory schedules/events and grouped log intervals; no out-of-core guarantee",
        "row_count_policy": "file rows are raw; reconstruction rows follow source ID/duration exclusions",
    }

    def rows(name: str, required: Iterable[str]) -> Iterable[tuple[int, dict]]:
        nonlocal location
        source_name = resolved_names[name]
        location = source_name
        pattern = _select_option(configured_formats, name)
        fa: dict = {
            "logical_filename": name, "source_filename": source_name,
            "timestamp_format": pattern,
            "timestamp_format_basis": "explicit caller-supplied strptime format" if pattern is not None
                                      else "existing ISO/UNIX parser; no explicit strptime format",
        }
        audit["files"][name] = fa
        yield from _csv_rows(raw_dir / source_name, fa,
                             encoding=_select_option(encoding, name) or "utf-8-sig",
                             delimiter=_select_option(delimiter, name), required=required)

    def timestamp(value: Any, name: str) -> int:
        formats = audit["files"][name].setdefault("timestamp_formats", {})
        result = parse_timestamp(value, assumed_timezone=assumed_timezone,
                                 timestamp_unit=_select_option(timestamp_unit, name),
                                 timestamp_format=_select_option(configured_formats, name), audit=formats)
        return result

    def identifier(value: Any, role: str, *, preserve_digits: bool = False) -> str | None:
        result = parse_identifier(value, preserve_digits=preserve_digits)
        formats = audit["identifier_formats"].setdefault(role, {
            "missing": 0, "plain_integer": 0, "decimal_style": 0,
            "canonicalized": 0, "leading_zero_digits_preserved": 0,
        })
        if result is None:
            formats["missing"] += 1
        else:
            text = str(value).strip()
            formats["decimal_style" if "." in text else "plain_integer"] += 1
            formats["canonicalized"] += int(text != result)
            formats["leading_zero_digits_preserved"] += int(
                preserve_digits and len(result) > 1 and result.startswith("0")
            )
        return result

    def observe_interval(name: str, start: int, end: int) -> None:
        fa = audit["files"][name]
        fa["normalized_interval_min_start"] = min(fa.get("normalized_interval_min_start", start), start)
        fa["normalized_interval_max_end"] = max(fa.get("normalized_interval_max_end", end), end)

    location = "source"
    try:
        template_users = []
        template_seen = set()
        for line, row in rows(TEMPLATE_FILE, [schema.template_user]):
            location = f"{resolved_names[TEMPLATE_FILE]}:{line}"
            user = identifier(row[schema.template_user], "template_user", preserve_digits=True)
            if user is None:
                raise DataSchemaError("template contains a null user ID")
            if user in template_seen:
                raise DataSchemaError("template contains a duplicate normalized user ID")
            template_seen.add(user)
            template_users.append(user)
        if not template_users:
            raise DataSchemaError("template has no users")

        schedules = []
        required = [schema.schedule_channel, schema.schedule_item, schema.schedule_start]
        if schema.schedule_duration is not None:
            required.append(schema.schedule_duration)
        elif schema.schedule_end is not None:
            required.append(schema.schedule_end)
        for name in SCHEDULE_FILES:
            for line, row in rows(name, required):
                location = f"{resolved_names[name]}:{line}"
                item = _item_id(identifier(row[schema.schedule_item], "schedule_item"))
                if item is None or item <= 0:
                    key = "missing_program_rows" if item is None else (
                        "zero_program_rows" if item == 0 else "negative_program_rows"
                    )
                    audit["schedule_parse"][key] += 1
                    continue
                channel = identifier(row[schema.schedule_channel], "schedule_channel")
                if channel is None:
                    audit["schedule_parse"]["null_channel_rows"] += 1
                    continue
                start = timestamp(row[schema.schedule_start], name)
                duration = None
                if schema.schedule_duration is not None:
                    try:
                        parsed_duration = _decimal(row[schema.schedule_duration], "EPG duration")
                        if parsed_duration != parsed_duration.to_integral_value() or parsed_duration <= 0:
                            raise ValueError("EPG duration must be positive whole seconds")
                        duration = int(parsed_duration)
                    except ValueError:
                        audit["schedule_parse"]["invalid_duration_rows"] += 1
                        continue
                explicit_end = None
                if schema.schedule_end is not None and schema.schedule_end in row:
                    if not _missing(row[schema.schedule_end]):
                        explicit_end = timestamp(row[schema.schedule_end], name)
                if duration is not None:
                    end = start + duration
                    if explicit_end is not None and explicit_end != end:
                        audit["schedule_parse"]["duration_end_mismatch_rows"] += 1
                        continue
                elif explicit_end is not None:
                    end = explicit_end
                else:
                    audit["schedule_parse"]["invalid_duration_rows"] += 1
                    continue
                observe_interval(name, start, end)
                core = {schema.schedule_channel, schema.schedule_item, schema.schedule_start,
                        schema.schedule_duration, schema.schedule_end}
                metadata = {key: None if _missing(value) else value
                            for key, value in row.items() if key not in core}
                schedules.append(dict(airing_id=_airing_id(channel, start, end, item),
                                      channel_id=channel, start=start, end=end,
                                      tv_show_id=item, metadata=metadata))
        normalized, schedule_audit = _prepare_schedules(schedules)

        def logs() -> Iterable[dict]:
            nonlocal location
            for line, row in rows(LOG_FILE, [schema.log_user, schema.log_channel,
                                             schema.log_start, schema.log_end]):
                location = f"{resolved_names[LOG_FILE]}:{line}"
                user = identifier(row[schema.log_user], "log_user", preserve_digits=True)
                if user is None:
                    audit["log_parse"]["null_user_rows"] += 1
                    continue
                channel = identifier(row[schema.log_channel], "log_channel")
                if channel is None:
                    audit["log_parse"]["null_channel_rows"] += 1
                    continue
                start = timestamp(row[schema.log_start], LOG_FILE)
                end = timestamp(row[schema.log_end], LOG_FILE)
                observe_interval(LOG_FILE, start, end)
                yield dict(user_id=user, channel_id=channel, start=start, end=end,
                           duration_raw=row.get(schema.log_duration) if schema.log_duration else None)

        events, event_audit = _reconstruct(logs(), normalized, schedule_audit, None)
        audit["reconstruction"] = event_audit
        for fa in audit["files"].values():
            for key, count in fa.get("timestamp_formats", {}).items():
                audit["timestamp_formats"][key] = audit["timestamp_formats"].get(key, 0) + count
        audit["template_users"] = len(template_users)
        return PreparedData(template_users, normalized, events, audit)
    except (ValueError, csv.Error) as exc:
        raise DataSchemaError(f"{location}: {exc}", audit) from exc


def build_fold(
    prepared: PreparedData, train_end: int, forecast_start: int, forecast_end: int,
) -> tuple[TrainingData, np.ndarray, dict[str, list[int]], dict]:
    """Create a count CSR, candidate *indices*, top-five labels and split audit.

    Train events must end <= train_end. Labels use whole airings contained in
    [forecast_start, forecast_end); candidates use all known positive-ID airings
    starting in that window, including crossing airings, seen IDs and cold items.
    The catalog is the union of historical/forecast schedules starting before
    forecast_end, never later schedules. Labels cover the fixed matrix cohort:
    all training users plus every template user (including empty labels).
    """

    train_end = _seconds(train_end, "train_end")
    forecast_start = _seconds(forecast_start, "forecast_start")
    forecast_end = _seconds(forecast_end, "forecast_end")
    if not train_end <= forecast_start < forecast_end:
        raise ValueError("require train_end <= forecast_start < forecast_end")
    if any(not isinstance(u, str) or not u for u in prepared.template_users):
        raise ValueError("template user IDs must be nonempty strings")
    if len(set(prepared.template_users)) != len(prepared.template_users):
        raise ValueError("duplicate template users")
    schedules, schedule_audit = _prepare_schedules(prepared.schedules)
    known = [s for s in schedules if s["start"] < forecast_end]
    item_ids = np.array(sorted({s["tv_show_id"] for s in known}), dtype=np.int64)
    item_index = {int(item): i for i, item in enumerate(item_ids)}
    candidate_ids = {s["tv_show_id"] for s in known if s["start"] >= forecast_start}
    candidates = np.array([item_index[item] for item in sorted(candidate_ids)], dtype=np.int64)
    canonical = {}
    for row in sorted(known, key=lambda s: (
        s["start"], s["end"], s["channel_id"], s["airing_id"], _metadata_key(s["metadata"])
    )):
        canonical.setdefault(row["tv_show_id"], dict(row["metadata"]))
    train_counts: Counter = Counter()
    heldout_counts: dict[str, Counter] = defaultdict(Counter)
    train_events = heldout_events = boundary_events = duplicates = 0
    seen_events = {}
    for event in prepared.events:
        user = event["user_id"]
        if not isinstance(user, str) or not user:
            raise ValueError("event user_id must be a nonempty string")
        start, end = (_seconds(event[k], k) for k in ("start", "end"))
        item = _item_id(event["tv_show_id"])
        if item is None or item <= 0 or end <= start:
            raise ValueError("events must have positive IDs and nonempty intervals")
        identity = (user, event["airing_id"])
        signature = (item, start, end)
        if identity in seen_events:
            if seen_events[identity] != signature:
                raise ValueError("conflicting duplicate user/airing event")
            duplicates += 1
            continue
        seen_events[identity] = signature
        if end <= train_end:
            if item not in item_index:
                raise ValueError("training event missing from known schedule catalog")
            train_counts[(user, item)] += 1
            train_events += 1
        elif start >= forecast_start and end <= forecast_end:
            if item not in candidate_ids:
                raise ValueError("held-out event missing from forecast schedule candidates")
            heldout_counts[user][item] += 1
            heldout_events += 1
        elif (start < train_end < end or start < forecast_start < end
              or start < forecast_end < end):
            boundary_events += 1
    user_ids = sorted({user for user, _ in train_counts} | set(prepared.template_users))
    user_index = {user: i for i, user in enumerate(user_ids)}
    row_indices, col_indices, counts = [], [], []
    for (user, item), count in sorted(train_counts.items()):
        if count > 2**24:
            raise ValueError("broadcast count exceeds exact float32 integer representation")
        row_indices.append(user_index[user])
        col_indices.append(item_index[item])
        counts.append(count)
    interactions = coo_matrix((np.asarray(counts, dtype=np.float32),
                               (np.asarray(row_indices, dtype=np.int64),
                                np.asarray(col_indices, dtype=np.int64))),
                              shape=(len(user_ids), len(item_ids))).tocsr()
    provenance = {
        "train_end": train_end, "forecast_start": forecast_start, "forecast_end": forecast_end,
        "training_policy": "qualifying event.end <= train_end; full scheduled timestamps",
        "label_policy": "forecast_start <= event.start and event.end <= forecast_end",
        "candidate_policy": "known positive-ID schedule starts in [forecast_start,forecast_end); "
                            "seen/cold IDs eligible; no behavior-based filtering",
        "catalog_policy": "historical and forecast schedules with start < forecast_end",
        "metadata_policy": "earliest scheduled row per ID; deterministic channel/airing/metadata tie-break",
        "user_policy": "sorted string union of training and template users; no held-out-only additions",
        "train_event_count": train_events,
        "behavior_max_end": max((end for _, (_, _, end) in seen_events.items()
                                 if end <= train_end), default=None),
    }
    data = TrainingData(user_ids, item_ids, interactions,
                        [canonical[int(item)] for item in item_ids], provenance)
    labels = {}
    tie_users = boundary_tie_users = cohort_heldout_events = 0
    for user in user_ids:
        frequencies = heldout_counts.get(user, Counter())
        ranked = sorted(frequencies, key=lambda item: (-frequencies[item], item))
        labels[user] = ranked[:5]
        cohort_heldout_events += sum(frequencies.values())
        if len(set(frequencies.values())) < len(frequencies):
            tie_users += 1
        if len(ranked) > 5 and frequencies[ranked[4]] == frequencies[ranked[5]]:
            boundary_tie_users += 1
    audit = {
        "provenance": dict(provenance), "users": len(user_ids), "items": len(item_ids),
        "candidates": len(candidates), "train_event_count": train_events,
        "heldout_event_count": heldout_events, "cohort_heldout_event_count": cohort_heldout_events,
        "heldout_users_outside_cohort": len(set(heldout_counts) - set(user_ids)),
        "heldout_events_outside_cohort": heldout_events - cohort_heldout_events,
        "excluded_boundary_event_count": boundary_events, "duplicate_events_ignored": duplicates,
        "heldout_frequency_tie_users": tie_users, "heldout_top5_boundary_tie_users": boundary_tie_users,
        "users_with_fewer_than_five_labels": sum(len(v) < 5 for v in labels.values()),
        "users_with_empty_labels": sum(not v for v in labels.values()),
        "template_users_with_fewer_than_five_labels": sum(
            len(labels[u]) < 5 for u in prepared.template_users),
        "seen_candidate_items": len(candidate_ids & {item for _, item in train_counts}),
        "cold_candidate_items": len(candidate_ids - {item for _, item in train_counts}),
        "known_quarantined_airings": sum(s["quarantined"] for s in known),
        "schedule_audit": schedule_audit,
        "label_tie_break": "qualifying airing count descending, numeric raw tv_show_id ascending",
    }
    return data, candidates, labels, audit
