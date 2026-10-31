"""Synthetic contracts; no assertion below claims observed competition data."""

from __future__ import annotations

import csv
import hashlib
import random
from copy import deepcopy
from decimal import Decimal
from itertools import permutations

import numpy as np
import pytest

from sweettv.data import (
    LOG_FILE,
    SCHEDULE_FILES,
    TEMPLATE_FILE,
    DataSchemaError,
    PreparedData,
    RawSchema,
    build_fold,
    load_raw,
    parse_identifier,
    parse_timestamp,
    reconstruct_events,
)


# Source filenames were listed by the authenticated official CLI (user-carried
# provenance). Fixture contents below are still synthetic, not downloaded data.
ACTUAL_FILE_NAMES = {
    "dataset11-30week.csv": "dataset11-30.csv",
    "export_arh_10-19.csv": "export_arh_11-20-final.csv",
    "export_arh_20-30.csv": "export_arh_21-30-final.csv",
    "export_arh_31-42.csv": "export_arh_31-42-final.csv",
    "submission.csv": "submission.csv",
}
DAY_FIRST_FORMAT = "%d.%m.%Y %H:%M:%S"
SOURCE_UNSIGNED_USER = "17226860011138219284"
SOURCE_UNSIGNED_USER_2 = "18417531283109304442"


def airing(start=0, end=100, item=1, channel="7", **extra):
    row = dict(airing_id=f"{channel}:{start}:{end}:{item}", channel_id=channel,
               start=start, end=end, tv_show_id=item, metadata={})
    row.update(extra)
    return row


def log(start=0, end=100, user="u", channel="7", **extra):
    return dict(user_id=user, channel_id=channel, start=start, end=end, **extra)


def event(schedule, user="u", coverage=1.0):
    return {**{key: schedule[key] for key in ("start", "end", "airing_id", "tv_show_id")},
            "user_id": user, "coverage": coverage}


def test_fragment_union_exact_threshold_and_scheduled_timestamps():
    schedule = airing()
    # Union is [0,60) U [80,100): exactly 80 seconds, not 95 or 60.
    logs = [log(0, 40), log(20, 60), log(20, 60), log(80, 100)]
    events, audit = reconstruct_events(logs, [schedule])
    assert events == [event(schedule, coverage=0.8)]
    assert audit["matched_user_airings"] == audit["qualifying_events"] == 1
    assert audit["merged_viewing_intervals"] == 2
    assert reconstruct_events([log(0, 79)], [schedule])[0] == []


def test_no_rounding_and_no_pooling_across_repeat_broadcasts():
    schedules = [airing(0, 101), airing(101, 202), airing(202, 303)]
    # 80/101 < .8, while 81/101 >= .8. Each broadcast is an independent event.
    events, _ = reconstruct_events([log(0, 80), log(101, 182), log(202, 283)], schedules)
    assert [e["start"] for e in events] == [101, 202]
    assert len({e["airing_id"] for e in events}) == 2
    assert reconstruct_events([log(0, 60), log(101, 161)], schedules)[0] == []


def test_half_open_adjacency_channel_identity_and_invalid_intervals():
    schedules = [airing(0, 10), airing(10, 20, item=2), airing(10, 20, item=3, channel="8")]
    events, audit = reconstruct_events([log(10, 20), log(20, 30), log(10, 10)], schedules)
    assert [e["tv_show_id"] for e in events] == [2]
    assert audit["conflict_unique_airings"] == 0
    assert audit["invalid_log_interval_rows"] == 1


def test_adjacent_viewing_fragments_union_without_a_gap():
    schedule = airing(0, 5)
    events, _ = reconstruct_events([log(0, 2), log(2, 4), log(5, 6)], [schedule])
    assert events == [event(schedule, coverage=0.8)]


def test_normalized_input_requires_string_ids_and_integer_seconds():
    with pytest.raises(ValueError, match="user_id must be a string"):
        reconstruct_events([log(user=9223372036854775807)], [airing()])
    with pytest.raises(ValueError, match="integer epoch seconds"):
        reconstruct_events([log(start=0.0)], [airing()])
    with pytest.raises(ValueError, match="multiple broadcast identities"):
        reconstruct_events([log()], [airing(airing_id="same"), airing(item=2, airing_id="same")])


def test_metadata_only_duplicates_are_not_conflicts_and_audit_is_order_invariant():
    base = airing()
    variants = [base, {**base, "metadata": {"title": "B"}},
                {**base, "metadata": {"title": "B"}}, {**base, "metadata": {"title": "A"}}]
    outputs = [reconstruct_events([log()], order) for order in permutations(variants)]
    assert all(output == outputs[0] for output in outputs)
    events, audit = outputs[0]
    assert len(events) == 1
    assert audit["duplicate_airing_rows"] == 3
    assert audit["metadata_variant_duplicate_rows"] == 3
    assert audit["conflict_unique_airings"] == 0


def test_full_transitive_conflict_quarantine_and_affected_counts():
    a, b, c = airing(0, 100), airing(90, 110, item=2), airing(105, 200, item=3)
    safe = airing(200, 210, item=4)
    schedules = [a, a, b, c, safe]
    logs = [log(0, 80, user="a"), log(95, 105, user="b"), log(200, 210, user="c")]
    events, audit = reconstruct_events(logs, schedules)
    assert events == [event(safe, user="c")]
    assert audit["conflict_components"] == 1
    assert audit["conflict_unique_airings"] == 3
    assert audit["conflict_affected_schedule_rows"] == 4
    assert audit["conflict_affected_log_rows"] == 2
    assert audit["conflict_affected_users"] == 2
    # Identical timestamps but different IDs are genuine conflicts as well.
    assert reconstruct_events([log()], [airing(), airing(item=2)])[0] == []


def test_missing_zero_negative_program_ids_and_null_channels_are_counted():
    schedules = [airing(item=0), airing(item=None), airing(item=-2), airing(channel=None), airing()]
    events, audit = reconstruct_events([log(), log(channel=None), log(user=None)], schedules)
    assert len(events) == 1
    assert audit["zero_program_rows"] == audit["missing_program_rows"] == 1
    assert audit["negative_program_rows"] == audit["null_channel_rows"] == 1
    assert audit["null_log_channel_rows"] == audit["null_user_rows"] == 1


def test_cutoff_never_uses_future_coverage_or_shortens_airing_denominator():
    schedule = airing()
    assert reconstruct_events([log()], [schedule], cutoff=79)[0] == []
    events, audit = reconstruct_events([log(), log(90, 100)], [schedule], cutoff=80)
    assert events == [event(schedule, coverage=0.8)]
    assert audit["qualifying_cutoff_crossing_airings"] == 1
    assert audit["logs_at_or_after_cutoff"] == 1
    assert audit["logs_clipped_at_cutoff"] == 1
    prepared = PreparedData(["u"], [schedule], events, {})
    data, _, labels, fold_audit = build_fold(prepared, 80, 80, 200)
    assert data.interactions.nnz == 0
    assert labels == {"u": []}
    assert fold_audit["excluded_boundary_event_count"] == 1


def test_cutoff_and_full_reconstruction_give_the_same_ended_training_events():
    schedules = [airing(0, 10), airing(10, 20), airing(20, 30)]
    logs = [log(0, 30), log(17, 28)]
    full_events, _ = reconstruct_events(logs, schedules)
    clipped_events, _ = reconstruct_events(logs, schedules, cutoff=18)
    assert [e for e in full_events if e["end"] <= 18] == [e for e in clipped_events if e["end"] <= 18]


def test_raw_log_duration_is_audited_without_rescaling_or_repair():
    events, audit = reconstruct_events(
        [log(duration_raw="100000"), log(user="bad", duration_raw="nonsense"),
         log(user="zero", duration_raw="0"), log(user="good", duration_raw="100.0")],
        [airing()],
    )
    assert len(events) == 4
    assert audit["raw_duration_mismatch_rows"] == 2
    assert audit["raw_duration_invalid_rows"] == 2


@pytest.mark.parametrize("text, expected", [
    ("1700000000", 1_700_000_000), ("1700000000000", 1_700_000_000),
    ("1700000000.000", 1_700_000_000), ("2023-11-14T22:13:20Z", 1_700_000_000),
    ("2023-11-15T00:13:20+02:00", 1_700_000_000),
])
def test_timestamp_seconds_milliseconds_and_iso(text, expected):
    assert parse_timestamp(text) == expected


@pytest.mark.parametrize("text", ["1000", "1700000000000000", "1700000000001", "1700000000.5"])
def test_ambiguous_or_subsecond_timestamps_fail_closed(text):
    with pytest.raises(ValueError):
        parse_timestamp(text)


@pytest.mark.parametrize("text", [
    "2023-11-14T22:13:20.0000001Z", "2023-11-14T22:13:20,0000001Z",
    "2023-11-14T22:13:20+00:00:00.0000001",
])
def test_iso_fractional_precision_is_not_silently_truncated(text):
    with pytest.raises(ValueError, match="subsecond"):
        parse_timestamp(text)
    assert parse_timestamp("2023-11-14T22:13:20.0000000Z") == 1_700_000_000


def test_explicit_timestamp_units_are_exact_even_outside_auto_window():
    assert parse_timestamp("1000", timestamp_unit="seconds") == 1000
    assert parse_timestamp("1000", timestamp_unit="milliseconds") == 1
    huge = 10**40 + 123
    assert parse_timestamp(str(huge), timestamp_unit="seconds") == huge
    assert parse_timestamp(str(huge * 1000), timestamp_unit="milliseconds") == huge
    with pytest.raises(ValueError):
        parse_timestamp(1_700_000_000.0)


def test_local_datetime_requires_explicit_zone_and_dst_is_strict():
    with pytest.raises(ValueError, match="timezone-naive"):
        parse_timestamp("2023-11-14 22:13:20")
    audit = {}
    assert parse_timestamp("2023-11-14 22:13:20", assumed_timezone="UTC", audit=audit) == 1_700_000_000
    assert audit == {"iso_naive_explicit_zone": 1}
    with pytest.raises(ValueError, match="ambiguous"):
        parse_timestamp("2024-11-03 01:30:00", assumed_timezone="America/New_York")
    with pytest.raises(ValueError, match="nonexistent"):
        parse_timestamp("2024-03-10 02:30:00", assumed_timezone="America/New_York")
    with pytest.raises(ValueError, match="ISO datetime"):
        parse_timestamp("2024-03-10")


@pytest.mark.parametrize("text, expected", [
    ("09.03.2020 06:00:00", 1_583_733_600),
    ("09.03.2020 07:40:00", 1_583_739_600),
    ("18.05.2020 06:30:00", 1_589_783_400),
    ("27.07.2020 06:30:00", 1_595_831_400),
])
def test_explicit_day_first_source_clocks_use_only_the_chosen_anchor(text, expected):
    audit = {}
    assert parse_timestamp(text, timestamp_format=DAY_FIRST_FORMAT,
                           assumed_timezone="UTC", audit=audit) == expected
    assert audit == {"explicit_format_naive_explicit_zone": 1}
    with pytest.raises(ValueError, match="timezone-naive"):
        parse_timestamp(text, timestamp_format=DAY_FIRST_FORMAT)
    # No day-first format is auto-detected, even with a clock anchor selected.
    with pytest.raises(ValueError, match="ISO datetime"):
        parse_timestamp(text, assumed_timezone="UTC")


@pytest.mark.parametrize("text, pattern", [
    ("09/03/2020 06:00:00", DAY_FIRST_FORMAT),
    ("09.03.2020 06:00:00 trailing", DAY_FIRST_FORMAT),
    ("31.02.2020 06:00:00", DAY_FIRST_FORMAT),
    ("18.05.2020 06:30:00", "%m.%d.%Y %H:%M:%S"),
    ("2020-03-09 06:00:00", DAY_FIRST_FORMAT),
    ("1583733600", DAY_FIRST_FORMAT),
    ("09.03.2020 06:00:00", "%Q"),
])
def test_explicit_format_mismatches_never_fall_back_to_iso_or_unix(text, pattern):
    with pytest.raises(ValueError, match="does not match explicit timestamp_format"):
        parse_timestamp(text, timestamp_format=pattern, assumed_timezone="UTC")


def test_explicit_timestamp_format_requires_nonempty_format_and_text_input():
    for pattern in ("", " ", 42):
        with pytest.raises(ValueError, match="nonempty strptime format"):
            parse_timestamp("09.03.2020 06:00:00", timestamp_format=pattern, assumed_timezone="UTC")
    with pytest.raises(ValueError, match="requires nonempty text"):
        parse_timestamp(1_583_733_600, timestamp_format=DAY_FIRST_FORMAT, assumed_timezone="UTC")


def test_explicit_datetime_parser_retains_dst_gap_and_fold_rejection():
    for text, message in (("10.03.2024 02:30:00", "nonexistent"),
                          ("03.11.2024 01:30:00", "ambiguous")):
        with pytest.raises(ValueError, match=message):
            parse_timestamp(text, timestamp_format=DAY_FIRST_FORMAT,
                            assumed_timezone="America/New_York")


def test_explicit_datetime_offsets_and_zero_fraction_are_exact():
    audit = {}
    assert parse_timestamp("09.03.2020 08:00:00+02:00", timestamp_format=DAY_FIRST_FORMAT + "%z",
                           audit=audit) == 1_583_733_600
    assert audit == {"explicit_format_with_offset": 1}
    assert parse_timestamp("09.03.2020 06:00:00.000000", timestamp_format=DAY_FIRST_FORMAT + ".%f",
                           assumed_timezone="UTC") == 1_583_733_600


@pytest.mark.parametrize("text, pattern", [
    ("09.03.2020 06:00:00.000001", DAY_FIRST_FORMAT + ".%f"),
    ("09.03.2020 06:00:00.0000001", DAY_FIRST_FORMAT + ".%f"),
    ("09.03.2020 06:00:00+00:00:00.000001", DAY_FIRST_FORMAT + "%z"),
])
def test_explicit_datetime_fractional_seconds_are_not_truncated(text, pattern):
    with pytest.raises(ValueError):
        parse_timestamp(text, timestamp_format=pattern, assumed_timezone="UTC")


def test_strict_identifier_parsing_preserves_large_values_and_digit_user_spelling():
    assert parse_identifier("9223372036854775807") == "9223372036854775807"
    assert parse_identifier("9007199254740993.0") == "9007199254740993"
    assert parse_identifier("0007", preserve_digits=True) == "0007"
    assert parse_identifier("0007") == "7"
    assert parse_identifier("NULL") is None
    assert parse_identifier(Decimal("42.0")) == "42"
    for value in (9007199254740993.0, "1.2", "1e3", True, "unexpected"):
        with pytest.raises(ValueError):
            parse_identifier(value)


def write_csv(path, headers, rows, *, delimiter=",", encoding="utf-8"):
    with path.open("w", newline="", encoding=encoding) as output:
        writer = csv.writer(output, delimiter=delimiter)
        writer.writerow(headers)
        writer.writerows(rows)


def raw_fixture(tmp_path, *, delimiter=",", encoding="utf-8", naive=False,
                legacy_headers=False, file_names=None):
    # Literal documented spellings, not values read from production defaults.
    # The legacy variant is exercised only with an explicit exact-name adapter.
    log_start, log_end, schedule_start = (
        ("start", "end", "start") if legacy_headers else ("start_time", "end_time", "start_time")
    )
    base = "2023-11-14T22:13:20" if naive else "1700000000"
    end = "2023-11-14T22:15:00" if naive else "1700000100"

    def source_path(logical):
        return tmp_path / (file_names or {}).get(logical, logical)

    write_csv(source_path(TEMPLATE_FILE), ["user_id", "tv_show_id"],
              [["9223372036854775807", ""], ["0009", ""]], delimiter=delimiter, encoding=encoding)
    write_csv(source_path(LOG_FILE), [" user_id ", "\ufeffvsetv_id ", log_start, log_end, "duration"],
              [["9223372036854775807.0", "7.0", base, end, "100000"],
               ["0009", "7", base, end, "100"]], delimiter=delimiter, encoding=encoding)
    for name in SCHEDULE_FILES:
        write_csv(source_path(name),
                  ["channel_id", "tv_show_id", schedule_start, "duration", "tv_show_title"],
                  [["7", "41.0", base, "100", "News"]], delimiter=delimiter, encoding=encoding)


def sampled_format_fixture(tmp_path, *, formatted_logs=False):
    """Copy bounded sample log spellings/values into a tiny *synthetic* catalog.

    The log examples are source-sample values; matching schedule IDs/times below
    are deliberately constructed test airings, not observed real completions.
    """

    users = ["3398461054087191302", SOURCE_UNSIGNED_USER, SOURCE_UNSIGNED_USER_2]
    write_csv(tmp_path / ACTUAL_FILE_NAMES[TEMPLATE_FILE], ["user_id", "tv_show_id"],
              [[user, "0 0 0 0 0"] for user in users])
    if formatted_logs:
        starts = ["09.03.2020 00:00:02", "09.03.2020 00:00:02", "09.03.2020 00:00:05"]
        stops = ["09.03.2020 00:01:53", "09.03.2020 00:03:26", "09.03.2020 00:01:26"]
    else:
        starts = ["2020-03-09 00:00:02", "2020-03-09 00:00:02", "2020-03-09 00:00:05"]
        stops = ["2020-03-09 00:01:53", "2020-03-09 00:03:26", "2020-03-09 00:01:26"]
    write_csv(tmp_path / ACTUAL_FILE_NAMES[LOG_FILE],
              ["user_id", "vsetv_id", "start_time", "stop_time", "duraton"],
              [[user, channel, start, stop, duration] for user, channel, start, stop, duration in
               zip(users, [6, 353, 332], starts, stops, [111, 204, 81])])
    headers = ["channel_id", "start_time", "duration", "tv_show_id", "tv_show_title"]
    write_csv(tmp_path / ACTUAL_FILE_NAMES[SCHEDULE_FILES[0]], headers, [
        [6, "09.03.2020 00:00:02", 111, 101, "Synthetic A"],
        [353, "09.03.2020 00:00:02", 204, 202, "Synthetic B"],
        [332, "09.03.2020 00:00:05", 81, 303, "Synthetic C"],
    ])
    write_csv(tmp_path / ACTUAL_FILE_NAMES[SCHEDULE_FILES[1]], headers,
              [[3, "18.05.2020 06:30:00", 1800, 404, "Synthetic future A"]])
    write_csv(tmp_path / ACTUAL_FILE_NAMES[SCHEDULE_FILES[2]], headers,
              [[3, "27.07.2020 06:30:00", 1800, 505, "Synthetic future B"]])


def test_sampled_format_adapter_preserves_unsigned_ids_source_typo_and_clock_basis(tmp_path):
    sampled_format_fixture(tmp_path)
    formats = {logical: DAY_FIRST_FORMAT for logical in SCHEDULE_FILES}
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = load_raw(tmp_path, file_names=ACTUAL_FILE_NAMES,
                      schema=RawSchema(log_end="stop_time", log_duration="duraton"),
                      timestamp_formats=formats, assumed_timezone="UTC")
    assert result.template_users == ["3398461054087191302", SOURCE_UNSIGNED_USER, SOURCE_UNSIGNED_USER_2]
    assert int(SOURCE_UNSIGNED_USER) > 2**63
    assert int(SOURCE_UNSIGNED_USER_2) > 2**63
    assert {(e["user_id"], e["tv_show_id"], e["start"], e["end"], e["coverage"])
            for e in result.events} == {
        ("3398461054087191302", 101, 1_583_712_002, 1_583_712_113, 1.0),
        (SOURCE_UNSIGNED_USER, 202, 1_583_712_002, 1_583_712_206, 1.0),
        (SOURCE_UNSIGNED_USER_2, 303, 1_583_712_005, 1_583_712_086, 1.0),
    }
    assert result.audit["schema"]["log_end"] == "stop_time"
    assert result.audit["schema"]["log_duration"] == "duraton"
    assert RawSchema().log_end == "end_time" and RawSchema().log_duration == "duration"
    fa = result.audit["files"][LOG_FILE]
    assert fa["raw_columns"] == fa["columns"] == [
        "user_id", "vsetv_id", "start_time", "stop_time", "duraton"
    ]
    assert fa["timestamp_format"] is None
    assert fa["timestamp_formats"] == {"iso_naive_explicit_zone": 6}
    for logical in SCHEDULE_FILES:
        assert result.audit["files"][logical]["source_filename"] == ACTUAL_FILE_NAMES[logical]
        assert result.audit["files"][logical]["timestamp_format"] == DAY_FIRST_FORMAT
        assert result.audit["files"][logical]["timestamp_format_basis"] == "explicit caller-supplied strptime format"
    assert result.audit["timestamp_format_config"] == formats
    assert result.audit["timestamp_formats"] == {
        "explicit_format_naive_explicit_zone": 5, "iso_naive_explicit_zone": 6,
    }
    assert result.audit["assumed_timezone"] == "UTC"
    assert "caller-chosen naive-clock" in result.audit["assumed_timezone_basis"]
    assert "not evidence" in result.audit["assumed_timezone_basis"]
    assert result.audit["reconstruction"]["raw_duration_mismatch_rows"] == 0
    assert result.audit["reconstruction"]["raw_duration_missing_rows"] == 0
    assert result.audit["schema_verified_on_actual_source"] is False
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    data, _, _, _ = build_fold(result, 1_583_712_300, 1_589_760_000, 1_595_808_000)
    assert data.user_ids == sorted(result.template_users)
    assert data.interactions.sum() == 3
    assert data.item_ids.tolist() == [101, 202, 303, 404]


def test_global_explicit_timestamp_format_applies_to_every_datetime_file(tmp_path):
    sampled_format_fixture(tmp_path, formatted_logs=True)
    result = load_raw(tmp_path, file_names=ACTUAL_FILE_NAMES,
                      schema=RawSchema(log_end="stop_time", log_duration="duraton"),
                      timestamp_formats=DAY_FIRST_FORMAT, assumed_timezone="UTC")
    assert len(result.events) == 3
    assert result.audit["timestamp_format_config"] == DAY_FIRST_FORMAT
    assert result.audit["timestamp_formats"] == {"explicit_format_naive_explicit_zone": 11}


def test_explicit_per_file_format_requires_a_zone_and_has_no_mismatch_fallback(tmp_path):
    sampled_format_fixture(tmp_path)
    formats = {logical: DAY_FIRST_FORMAT for logical in SCHEDULE_FILES}
    schema = RawSchema(log_end="stop_time", log_duration="duraton")
    with pytest.raises(DataSchemaError, match="timezone-naive") as error:
        load_raw(tmp_path, file_names=ACTUAL_FILE_NAMES, schema=schema, timestamp_formats=formats)
    assert error.value.audit["files"][SCHEDULE_FILES[0]]["timestamp_format"] == DAY_FIRST_FORMAT
    assert error.value.audit["assumed_timezone"] is None
    # A global day-first pattern also governs the ISO logs: do not silently fall
    # back when a caller has explicitly chosen an incompatible format for them.
    with pytest.raises(DataSchemaError, match="does not match explicit timestamp_format") as error:
        load_raw(tmp_path, file_names=ACTUAL_FILE_NAMES, schema=schema,
                 timestamp_formats=DAY_FIRST_FORMAT, assumed_timezone="UTC")
    assert ACTUAL_FILE_NAMES[LOG_FILE] in str(error.value)
    assert error.value.audit["files"][LOG_FILE]["timestamp_format"] == DAY_FIRST_FORMAT


@pytest.mark.parametrize("formats", ["", " ", 42, [], {LOG_FILE: ""}, {LOG_FILE: 42}])
def test_invalid_explicit_timestamp_format_configuration_fails_before_io(tmp_path, formats):
    with pytest.raises(DataSchemaError, match="invalid timestamp_formats"):
        load_raw(tmp_path, timestamp_formats=formats)


def test_timestamp_format_mapping_rejects_actual_alias_keys_instead_of_guessing(tmp_path):
    with pytest.raises(DataSchemaError, match="timestamp_formats has unknown logical keys"):
        load_raw(tmp_path, file_names=ACTUAL_FILE_NAMES,
                 timestamp_formats={ACTUAL_FILE_NAMES[SCHEDULE_FILES[0]]: DAY_FIRST_FORMAT})


def test_documented_default_schema_sniffs_delimiters_and_preserves_exact_template_order(tmp_path):
    raw_fixture(tmp_path, delimiter=";")
    result = load_raw(tmp_path)
    assert result.template_users == ["9223372036854775807", "0009"]
    assert {e["user_id"] for e in result.events} == {"9223372036854775807", "0009"}
    assert len(result.schedules) == 1
    assert {e["tv_show_id"] for e in result.events} == {41}
    assert result.audit["reconstruction"]["duplicate_airing_rows"] == 2
    assert result.audit["reconstruction"]["raw_duration_mismatch_rows"] == 1
    assert result.audit["assumed_timezone"] is None
    assert result.audit["schema"]["log_start"] == "start_time"
    assert result.audit["schema"]["log_end"] == "end_time"
    assert result.audit["schema"]["schedule_start"] == "start_time"
    assert result.audit["schema"]["schedule_end"] is None
    assert result.audit["schema_verified_on_actual_source"] is False
    identity_names = {name: name for name in (LOG_FILE, *SCHEDULE_FILES, TEMPLATE_FILE)}
    assert result.audit["file_names"] == identity_names
    assert result.schedules[0]["end"] == result.schedules[0]["start"] + 100
    assert result.audit["identifier_formats"]["log_user"]["decimal_style"] == 1
    assert result.audit["identifier_formats"]["log_user"]["leading_zero_digits_preserved"] == 1
    assert result.audit["files"][LOG_FILE]["normalized_interval_min_start"] == 1_700_000_000
    assert result.audit["files"][LOG_FILE]["normalized_interval_max_end"] == 1_700_000_100
    for logical, file_audit in result.audit["files"].items():
        assert file_audit["logical_filename"] == file_audit["source_filename"] == logical
        assert file_audit["delimiter"] == ";"
        assert file_audit["delimiter_source"] == "csv.Sniffer"
        assert len(file_audit["sha256"]) == 64


def test_explicit_actual_filename_map_reads_actual_bytes_and_preserves_sources(tmp_path):
    source_names = dict(ACTUAL_FILE_NAMES)
    before_mapping = dict(source_names)
    raw_fixture(tmp_path, file_names=source_names)
    before_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert set(before_files) == set(source_names.values())
    # The default remains the old logical names; no automatic discovery occurs.
    with pytest.raises(FileNotFoundError, match=LOG_FILE):
        load_raw(tmp_path)
    result = load_raw(tmp_path, file_names=source_names)
    assert source_names == before_mapping
    assert result.template_users == ["9223372036854775807", "0009"]
    assert len(result.events) == 2
    assert {(e["user_id"], e["tv_show_id"], e["start"], e["end"], e["coverage"])
            for e in result.events} == {
        ("9223372036854775807", 41, 1_700_000_000, 1_700_000_100, 1.0),
        ("0009", 41, 1_700_000_000, 1_700_000_100, 1.0),
    }
    assert result.audit["file_names"] == source_names
    assert result.audit["schema"]["log_start"] == "start_time"
    assert result.audit["schema_verified_on_actual_source"] is False
    assert set(result.audit["files"]) == set(source_names)
    for logical, actual in source_names.items():
        fa = result.audit["files"][logical]
        assert fa["logical_filename"] == logical
        assert fa["source_filename"] == actual
        assert fa["size_bytes"] == len(before_files[actual])
        assert fa["sha256"] == hashlib.sha256(before_files[actual]).hexdigest()
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before_files


def test_partial_filename_map_can_alias_template_and_preserves_omitted_defaults(tmp_path):
    file_names = {TEMPLATE_FILE: "explicit-template.csv"}
    raw_fixture(tmp_path, file_names=file_names)
    result = load_raw(tmp_path, file_names=file_names)
    expected = {name: name for name in (LOG_FILE, *SCHEDULE_FILES, TEMPLATE_FILE)}
    expected.update(file_names)
    assert result.audit["file_names"] == expected
    assert result.audit["files"][TEMPLATE_FILE]["source_filename"] == "explicit-template.csv"
    assert len(result.events) == 2


def test_empty_filename_map_is_the_identity_mapping(tmp_path):
    raw_fixture(tmp_path)
    result = load_raw(tmp_path, file_names={})
    assert result.audit["file_names"] == {
        name: name for name in (LOG_FILE, *SCHEDULE_FILES, TEMPLATE_FILE)
    }
    assert len(result.events) == 2


@pytest.mark.parametrize("bad_name", [
    "", "   ", ".", "..", "../outside.csv", "./inside.csv", "nested/data.csv",
    "nested\\data.csv", "/absolute.csv", "\\absolute.csv", "C:\\data.csv", "C:data.csv",
    "bad\0.csv", None, 42, True,
])
def test_filename_alias_must_be_a_nonempty_relative_basename_before_io(tmp_path, bad_name):
    # No raw files exist: invalid mapping must fail before filesystem discovery.
    with pytest.raises(DataSchemaError, match="nonempty relative basename"):
        load_raw(tmp_path, file_names={LOG_FILE: bad_name})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mapping", [{"unknown.csv": "anything.csv"}, {42: "anything.csv"}])
def test_filename_map_rejects_unknown_logical_keys(tmp_path, mapping):
    with pytest.raises(DataSchemaError, match="unknown logical keys"):
        load_raw(tmp_path, file_names=mapping)


@pytest.mark.parametrize("mapping", [[], "not-a-mapping"])
def test_filename_map_requires_a_mapping(tmp_path, mapping):
    with pytest.raises(DataSchemaError, match="file_names must be a mapping"):
        load_raw(tmp_path, file_names=mapping)


@pytest.mark.parametrize("mapping", [
    {LOG_FILE: SCHEDULE_FILES[0]},  # Collision with an omitted, defaulted role.
    {LOG_FILE: "same.csv", SCHEDULE_FILES[0]: "same.csv"},
])
def test_filename_map_must_resolve_to_five_distinct_filenames(tmp_path, mapping):
    with pytest.raises(DataSchemaError, match="five distinct source basenames"):
        load_raw(tmp_path, file_names=mapping)


def test_missing_files_error_lists_actual_mapped_names(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        load_raw(tmp_path, file_names=ACTUAL_FILE_NAMES)
    message = str(error.value)
    for actual in ACTUAL_FILE_NAMES.values():
        assert actual in message
    for logical, actual in ACTUAL_FILE_NAMES.items():
        if logical != actual:
            assert logical not in message


def test_explicit_filename_alias_never_falls_back_to_existing_logical_file(tmp_path):
    raw_fixture(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileNotFoundError) as error:
        load_raw(tmp_path, file_names=ACTUAL_FILE_NAMES)
    assert ACTUAL_FILE_NAMES[LOG_FILE] in str(error.value)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_alias_parse_error_and_partial_audit_identify_actual_source_filename(tmp_path):
    raw_fixture(tmp_path, file_names=ACTUAL_FILE_NAMES)
    write_csv(tmp_path / ACTUAL_FILE_NAMES[LOG_FILE], ["user_id", "vsetv_id", "wrong", "end_time"],
              [["0009", 7, 1_700_000_000, 1_700_000_100]])
    with pytest.raises(DataSchemaError, match="missing exact columns") as error:
        load_raw(tmp_path, file_names=ACTUAL_FILE_NAMES)
    assert ACTUAL_FILE_NAMES[LOG_FILE] in str(error.value)
    assert LOG_FILE not in str(error.value)
    assert error.value.audit["file_names"] == ACTUAL_FILE_NAMES
    assert error.value.audit["files"][LOG_FILE]["source_filename"] == ACTUAL_FILE_NAMES[LOG_FILE]


def test_alias_parsing_options_remain_keyed_by_logical_names(tmp_path):
    raw_fixture(tmp_path, file_names=ACTUAL_FILE_NAMES, delimiter="^")
    # '^' is not in the automatic sniffing set, epoch 0 is outside automatic
    # scale detection, and Cyrillic CP1251 bytes are not UTF-8. Wrong option-key
    # dispatch therefore cannot accidentally pass via the automatic defaults.
    write_csv(tmp_path / ACTUAL_FILE_NAMES[LOG_FILE],
              ["user_id", "vsetv_id", "start_time", "end_time"],
              [["0009", 7, 0, 100]], delimiter="^")
    for logical in SCHEDULE_FILES:
        write_csv(tmp_path / ACTUAL_FILE_NAMES[logical],
                  ["channel_id", "tv_show_id", "start_time", "duration", "tv_show_title"],
                  [[7, 41, 0, 100, "Новости"]], delimiter="^", encoding="cp1251")
    result = load_raw(
        tmp_path, file_names=ACTUAL_FILE_NAMES,
        delimiter={logical: "^" for logical in ACTUAL_FILE_NAMES},
        timestamp_unit={logical: "seconds" for logical in ACTUAL_FILE_NAMES},
        encoding={logical: "cp1251" for logical in SCHEDULE_FILES},
    )
    assert [(e["user_id"], e["tv_show_id"], e["start"], e["end"])
            for e in result.events] == [("0009", 41, 0, 100)]
    assert result.schedules[0]["metadata"]["tv_show_title"] == "Новости"
    for logical, fa in result.audit["files"].items():
        assert fa["source_filename"] == ACTUAL_FILE_NAMES[logical]
        assert fa["delimiter"] == "^"
        assert fa["delimiter_source"] == "explicit"
    assert result.audit["files"][SCHEDULE_FILES[0]]["encoding"] == "cp1251"


def test_legacy_start_end_headers_require_explicit_exact_adapter(tmp_path):
    raw_fixture(tmp_path, legacy_headers=True)
    with pytest.raises(DataSchemaError, match="missing exact columns.*start_time"):
        load_raw(tmp_path)
    schema = RawSchema(log_start="start", log_end="end", schedule_start="start", schedule_end="end")
    result = load_raw(tmp_path, schema=schema)
    assert result.template_users == ["9223372036854775807", "0009"]
    assert len(result.events) == 2
    assert {(e["user_id"], e["tv_show_id"], e["start"], e["end"], e["coverage"])
            for e in result.events} == {
        ("9223372036854775807", 41, 1_700_000_000, 1_700_000_100, 1.0),
        ("0009", 41, 1_700_000_000, 1_700_000_100, 1.0),
    }
    assert result.audit["schema"]["schedule_start"] == "start"
    assert result.audit["schema"]["schedule_end"] == "end"
    assert result.audit["schema_verified_on_actual_source"] is False


def test_load_raw_timezone_choice_is_required_and_recorded(tmp_path):
    raw_fixture(tmp_path, naive=True)
    with pytest.raises(DataSchemaError, match="timezone-naive") as error:
        load_raw(tmp_path)
    assert error.value.audit["assumed_timezone"] is None
    result = load_raw(tmp_path, assumed_timezone="UTC")
    assert result.audit["assumed_timezone"] == "UTC"
    assert result.events[0]["start"] == 1_700_000_000


def test_cp1251_requires_explicit_choice_and_never_replaces(tmp_path):
    raw_fixture(tmp_path)
    for name in SCHEDULE_FILES:
        write_csv(tmp_path / name,
                  ["channel_id", "tv_show_id", "start_time", "duration", "tv_show_title"],
                  [[7, 41, 1_700_000_000, 100, "Новости"]], encoding="cp1251")
    with pytest.raises(DataSchemaError, match="strict utf-8-sig decoding failed"):
        load_raw(tmp_path)
    result = load_raw(tmp_path, encoding={name: "cp1251" for name in SCHEDULE_FILES})
    assert result.schedules[0]["metadata"]["tv_show_title"] == "Новости"
    assert result.audit["files"][SCHEDULE_FILES[0]]["encoding"] == "cp1251"


def test_unknown_columns_fail_with_observed_schema_and_explicit_adapter_works(tmp_path):
    raw_fixture(tmp_path)
    write_csv(tmp_path / LOG_FILE, ["uid", "station", "began", "finished"],
              [["0009", "7", "1700000000", "1700000100"]])
    with pytest.raises(DataSchemaError, match="missing exact columns.*observed"):
        load_raw(tmp_path)
    schema = RawSchema(log_user="uid", log_channel="station", log_start="began",
                       log_end="finished", log_duration=None)
    assert load_raw(tmp_path, schema=schema).events[0]["user_id"] == "0009"


def test_load_mixed_timestamp_scales_are_explicit_in_file_audit(tmp_path):
    raw_fixture(tmp_path)
    write_csv(tmp_path / LOG_FILE, ["user_id", "vsetv_id", "start_time", "end_time"],
              [["0009", 7, 1_700_000_000_000, 1_700_000_100_000]])
    result = load_raw(tmp_path)
    assert result.events[0]["start"] == 1_700_000_000
    assert result.audit["files"][LOG_FILE]["timestamp_formats"] == {"unix_milliseconds": 2}
    assert result.audit["timestamp_formats"] == {"unix_seconds": 3, "unix_milliseconds": 2}


def test_fractional_source_identifiers_are_not_silently_truncated(tmp_path):
    raw_fixture(tmp_path)
    write_csv(tmp_path / LOG_FILE, ["user_id", "vsetv_id", "start_time", "end_time"],
              [["9007199254740993.25", 7, 1_700_000_000, 1_700_000_100]])
    with pytest.raises(DataSchemaError, match="nonintegral decimal") as error:
        load_raw(tmp_path)
    assert LOG_FILE in str(error.value)
    assert error.value.audit["files"][LOG_FILE]["rows"] == 1


def test_single_column_template_requires_explicit_delimiter_when_sniffer_cannot_infer(tmp_path):
    raw_fixture(tmp_path)
    write_csv(tmp_path / TEMPLATE_FILE, ["user_id"], [["0009"]])
    with pytest.raises(DataSchemaError, match="provide delimiter explicitly"):
        load_raw(tmp_path)
    assert load_raw(tmp_path, delimiter={TEMPLATE_FILE: ","}).template_users == ["0009"]


def test_empty_csvs_still_validate_headers(tmp_path):
    raw_fixture(tmp_path)
    write_csv(tmp_path / SCHEDULE_FILES[0], ["wrong", "columns"], [])
    with pytest.raises(DataSchemaError, match="missing exact columns"):
        load_raw(tmp_path)


def test_epg_duration_and_program_id_exclusions_are_audited(tmp_path):
    raw_fixture(tmp_path)
    write_csv(tmp_path / SCHEDULE_FILES[0],
              ["channel_id", "tv_show_id", "start_time", "duration", "end_time"],
              [[7, 0, 1_700_000_000, 100, ""], [7, "", 1_700_000_000, 100, ""],
               ["", 41, 1_700_000_000, 100, ""], [7, 41, 1_700_000_000, -1, ""],
               [7, 41, 1_700_000_000, 100, 1_700_000_099]])
    result = load_raw(tmp_path, schema=RawSchema(schedule_end="end_time"))
    audit = result.audit["schedule_parse"]
    assert audit["zero_program_rows"] == audit["missing_program_rows"] == 1
    assert audit["null_channel_rows"] == audit["invalid_duration_rows"] == 1
    assert audit["duration_end_mismatch_rows"] == 1


def test_malformed_csv_duplicate_header_and_missing_files_fail(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing raw CSV"):
        load_raw(tmp_path)
    raw_fixture(tmp_path)
    write_csv(tmp_path / LOG_FILE, ["user_id", " user_id "], [["1", "1"]])
    with pytest.raises(DataSchemaError, match="duplicate normalized"):
        load_raw(tmp_path)
    (tmp_path / LOG_FILE).write_text("user_id,vsetv_id,start_time,end_time\n1,7,1700000000\n")
    with pytest.raises(DataSchemaError, match="expected 4 fields"):
        load_raw(tmp_path, delimiter=",")


def test_repeat_counts_and_forecast_catalog_include_seen_and_cold_items():
    schedules = [airing(0, 10, item=5), airing(10, 20, item=5),
                 airing(20, 30, item=1), airing(30, 40, item=5),
                 airing(40, 50, item=7), airing(50, 60, item=9)]
    events, _ = reconstruct_events([log(0, 60)], schedules)
    prepared = PreparedData(["cold", "u"], schedules, events, {})
    data, candidates, labels, audit = build_fold(prepared, 20, 20, 50)
    assert data.user_ids == ["cold", "u"]
    assert data.item_ids.tolist() == [1, 5, 7]
    assert data.item_ids[candidates].tolist() == [1, 5, 7]
    assert data.interactions.toarray().tolist() == [[0, 0, 0], [0, 2, 0]]
    assert labels == {"cold": [], "u": [1, 5, 7]}
    assert audit["train_event_count"] == 2
    assert audit["heldout_event_count"] == 3
    assert audit["seen_candidate_items"] == 1
    assert audit["cold_candidate_items"] == 2
    assert data.provenance["behavior_max_end"] == 20


def test_labels_top5_frequency_then_numeric_raw_id_and_ties():
    schedules = [airing(10 * i, 10 * (i + 1), item=item)
                 for i, item in enumerate([100, 10, 2, 1, 8, 4, 7, 100])]
    events, _ = reconstruct_events([log(0, 80)], schedules)
    prepared = PreparedData(["u", "empty"], schedules, events, {})
    _, _, labels, audit = build_fold(prepared, 0, 0, 80)
    assert labels == {"empty": [], "u": [100, 1, 2, 4, 7]}
    assert audit["heldout_frequency_tie_users"] == 1
    assert audit["heldout_top5_boundary_tie_users"] == 1
    assert audit["users_with_fewer_than_five_labels"] == 1


def test_fold_fully_contained_labels_but_crossing_schedule_still_candidate():
    schedules = [airing(0, 10), airing(10, 20, item=2), airing(20, 35, item=3),
                 airing(35, 45, item=4)]
    events, _ = reconstruct_events([log(0, 45)], schedules)
    prepared = PreparedData(["u"], schedules, events, {})
    data, candidates, labels, audit = build_fold(prepared, 10, 15, 40)
    assert data.item_ids[candidates].tolist() == [3, 4]
    assert labels == {"u": [3]}
    assert audit["train_event_count"] == 1
    assert audit["excluded_boundary_event_count"] == 2


def test_fold_metadata_and_train_counts_are_independent_of_future_behavior():
    schedules = [airing(0, 10, metadata={"tv_show_title": "History"}),
                 airing(10, 20, item=2, metadata={"tv_show_title": "Forecast"}),
                 airing(20, 30, item=99, metadata={"tv_show_title": "Too late"})]
    events, _ = reconstruct_events([log(0, 20)], schedules)
    prepared = PreparedData(["u", "cold"], schedules, events, {})
    first = build_fold(prepared, 10, 10, 20)
    changed = deepcopy(prepared)
    changed.events = [e for e in events if e["end"] <= 10]
    changed.events += [event(schedules[1], user="cold"), event(schedules[1], user="not-in-cohort")]
    second = build_fold(changed, 10, 10, 20)
    a, b = first[0], second[0]
    assert a.user_ids == b.user_ids == ["cold", "u"]
    np.testing.assert_array_equal(a.item_ids, b.item_ids)
    np.testing.assert_array_equal(a.interactions.toarray(), b.interactions.toarray())
    assert a.metadata == b.metadata
    assert a.provenance == b.provenance
    assert a.item_ids.tolist() == [1, 2]
    assert second[3]["heldout_users_outside_cohort"] == 1
    assert second[3]["heldout_events_outside_cohort"] == 1


def test_fold_is_deterministic_under_input_permutation_and_duplicate_events():
    schedules = [airing(0, 10, metadata={"tv_show_title": "B"}),
                 airing(0, 10, metadata={"tv_show_title": "A"}), airing(10, 20, item=2)]
    events, _ = reconstruct_events([log(0, 20)], schedules)
    prepared = PreparedData(["u"], schedules, events + [events[0]], {})
    a = build_fold(prepared, 10, 10, 20)
    b = build_fold(PreparedData(["u"], schedules[::-1], prepared.events[::-1], {}), 10, 10, 20)
    assert a[0].metadata == b[0].metadata == [{"tv_show_title": "A"}, {}]
    assert a[0].provenance == b[0].provenance
    assert a[3] == b[3]
    assert a[3]["duplicate_events_ignored"] == 1


def test_conflicted_schedules_still_supply_nonbehavioral_candidates():
    schedules = [airing(0, 10), airing(0, 10, item=2)]
    events, audit = reconstruct_events([log(0, 10)], schedules)
    data, candidates, labels, fold_audit = build_fold(
        PreparedData(["u"], schedules, events, audit), 0, 0, 10)
    assert data.item_ids[candidates].tolist() == [1, 2]
    assert labels == {"u": []}
    assert fold_audit["known_quarantined_airings"] == 2


def test_empty_fold_keeps_template_users_and_valid_sparse_shape():
    data, candidates, labels, audit = build_fold(PreparedData(["9", "10"], [], [], {}), 0, 0, 10)
    assert data.user_ids == ["10", "9"]
    assert data.interactions.shape == (2, 0)
    assert candidates.size == 0
    assert labels == {"10": [], "9": []}
    assert audit["train_event_count"] == audit["heldout_event_count"] == 0
    with pytest.raises(ValueError, match="train_end"):
        build_fold(PreparedData([], [], [], {}), 10, 0, 20)


def test_randomized_reconstruction_matches_independent_discrete_time_oracle():
    # Oracle enumerates seconds as a set, not the implementation's union/join.
    rng = random.Random(178)
    schedules = [airing(i * 10, i * 10 + 10, item=i % 3 + 1, channel=channel)
                 for channel in ("7", "8") for i in range(8)]
    for _ in range(40):
        logs = []
        for user in ("u", "v"):
            for channel in ("7", "8"):
                for _ in range(rng.randrange(1, 15)):
                    start = rng.randrange(-5, 85)
                    logs.append(log(start, start + rng.randrange(1, 30), user=user, channel=channel))
        cutoff = rng.choice([None, 17, 40, 79])
        expected = []
        for user in ("u", "v"):
            for schedule in schedules:
                watched = set()
                for row in logs:
                    if row["user_id"] == user and row["channel_id"] == schedule["channel_id"]:
                        stop = min(row["end"], cutoff) if cutoff is not None else row["end"]
                        watched.update(range(max(row["start"], schedule["start"]),
                                             min(stop, schedule["end"])))
                covered = len(watched)
                duration = schedule["end"] - schedule["start"]
                if covered * 5 >= duration * 4:
                    expected.append((user, schedule["airing_id"], covered / duration))
        actual, _ = reconstruct_events(logs, schedules, cutoff=cutoff)
        assert sorted((e["user_id"], e["airing_id"], e["coverage"]) for e in actual) == sorted(expected)
