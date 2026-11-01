"""Independent synthetic challenges; no actual Kaggle CSV has been inspected.

Expected count matrices, label lists, catalog IDs, and qualifying broadcasts are
written below before calling production. Helpers only assemble literal records
or compare outputs: they never call a production reconstruction/counting helper
to manufacture an expected value. Integer seconds here are synthetic, not an
inference about the competition's timestamp units, dates, or time zone.

The documented raw column spellings are source facts supplied in the independent
review request: logs start_time/end_time and EPG start_time. The default-schema
tests deliberately require compatibility with those documented names. Explicit
adapter controls and fail-closed tests are separate, not xfailed/hidden.
"""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sweettv.data import (
    DataSchemaError,
    PreparedData,
    RawSchema,
    build_fold,
    load_raw,
    reconstruct_events,
)


BASE = 1_700_000_000
WIDE_ID = 9_007_199_254_740_993
MAX_ID = 9_223_372_036_854_775_807
LOG_NAME = "dataset11-30week.csv"
EPG_NAMES = ("export_arh_10-19.csv", "export_arh_20-30.csv", "export_arh_31-42.csv")
DOCUMENTED_ADAPTER = {
    "log_start": "start_time", "log_end": "end_time", "schedule_start": "start_time",
}


def _airing(start, end, item, channel="7", metadata=None, airing_id=None):
    return {
        "start": start, "end": end, "tv_show_id": item, "channel_id": channel,
        "airing_id": airing_id or f"fixture:{channel}:{start}:{end}:{item}",
        "metadata": {} if metadata is None else metadata,
    }


def _event(schedule, user):
    return {
        "user_id": user, "tv_show_id": schedule["tv_show_id"],
        "start": schedule["start"], "end": schedule["end"],
        "airing_id": schedule["airing_id"], "coverage": 1.0,
    }


def _log(start, end, user="u", channel="7", **extra):
    return {"start": start, "end": end, "user_id": user, "channel_id": channel, **extra}


def _assert_fold(result, *, users, items, counts, candidates, labels, metadata=None):
    data, candidate_indices, actual_labels, _ = result
    assert data.user_ids == users
    assert data.item_ids.tolist() == items
    assert data.interactions.format == "csr"
    np.testing.assert_array_equal(data.interactions.toarray(), np.asarray(counts))
    assert candidate_indices.dtype.kind in "iu"
    assert data.item_ids[candidate_indices].tolist() == candidates
    assert actual_labels == labels
    assert list(actual_labels) == users
    if metadata is not None:
        assert data.metadata == metadata


def _assert_same_training(left, right):
    a, a_candidates, _, _ = left
    b, b_candidates, _, _ = right
    assert a.user_ids == b.user_ids
    np.testing.assert_array_equal(a.item_ids, b.item_ids)
    np.testing.assert_array_equal(a_candidates, b_candidates)
    for field in ("indptr", "indices", "data"):
        np.testing.assert_array_equal(getattr(a.interactions, field), getattr(b.interactions, field))
    assert a.metadata == b.metadata
    assert a.provenance == b.provenance


def test_all_heldout_event_mutations_leave_grounded_training_state_unchanged():
    schedules = [
        _airing(0, 10, 101, metadata={"tv_show_title": "History"}),
        _airing(10, 20, 101, metadata={"tv_show_title": "Later history"}),
        _airing(0, 10, 202, channel="8", metadata={"tv_show_title": "Other history"}),
        _airing(20, 30, 101, metadata={"tv_show_title": "Forecast repeat"}),
        _airing(30, 40, 303, metadata={"tv_show_title": "Cold forecast"}),
        _airing(40, 50, 404, metadata={"year": None}),
        _airing(50, 70, 505, metadata={"tv_show_title": "Crossing forecast"}),
        _airing(60, 70, 909, channel="8", metadata={"tv_show_title": "Too late"}),
    ]
    template = ["template-only", "0007", str(WIDE_ID)]
    training = [_event(schedules[0], "0007"), _event(schedules[1], "0007"),
                _event(schedules[2], str(WIDE_ID))]
    original_future = [_event(schedules[3], "0007"), _event(schedules[4], "0007"),
                       _event(schedules[5], str(WIDE_ID)), _event(schedules[4], "future-only")]
    changed_future = [_event(schedules[3], "template-only"), _event(schedules[4], str(WIDE_ID)),
                      _event(schedules[5], "different-future-only"),
                      _event(schedules[7], "post-horizon-only")]
    expected = {
        "users": ["0007", str(WIDE_ID), "template-only"],
        "items": [101, 202, 303, 404, 505],
        "counts": [[2, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
        "candidates": [101, 303, 404, 505],
        "metadata": [{"tv_show_title": "History"}, {"tv_show_title": "Other history"},
                     {"tv_show_title": "Cold forecast"}, {"year": None},
                     {"tv_show_title": "Crossing forecast"}],
    }
    expected_labels = [
        {"0007": [101, 303], str(WIDE_ID): [404], "template-only": []},
        {"0007": [], str(WIDE_ID): [303], "template-only": [101]},
        {"0007": [], str(WIDE_ID): [], "template-only": []},
    ]
    results = []
    for future, labels in zip([original_future, changed_future, []], expected_labels):
        prepared = PreparedData(template[:], deepcopy(schedules), deepcopy(training + future), {})
        before = deepcopy(prepared)
        result = build_fold(prepared, 20, 20, 60)
        _assert_fold(result, **expected, labels=labels)
        assert prepared == before, "Fold construction must not modify the supplied observations/catalog"
        assert result[0].provenance["train_event_count"] == 3
        assert result[0].provenance["behavior_max_end"] == 20
        results.append(result)
    _assert_same_training(results[0], results[1])
    _assert_same_training(results[0], results[2])
    assert results[1][3]["heldout_users_outside_cohort"] == 1
    assert results[2][3]["heldout_event_count"] == 0


def test_mutating_raw_future_seconds_of_a_cross_boundary_log_cannot_complete_training():
    schedules = [_airing(0, 100, 11), _airing(100, 200, 22)]
    original = [_log(0, 200), _log(100, 200, user="future-only")]
    changed = [_log(0, 100), _log(100, 180, user="cold"),
               _log(100, 200, user="different-future-only"), _log(0, 200, user="unreachable", channel="999")]
    expected = {"users": ["cold", "u"], "items": [11, 22], "counts": [[0, 0], [1, 0]],
                "candidates": [22]}
    labels = [{"cold": [], "u": [22]}, {"cold": [22], "u": []}]
    results = []
    for logs, expected_labels in zip([original, changed], labels):
        events, _ = reconstruct_events(logs, schedules)
        result = build_fold(PreparedData(["cold", "u"], schedules, events, {}), 100, 100, 200)
        _assert_fold(result, **expected, labels=expected_labels)
        results.append(result)
    _assert_same_training(*results)


def test_schedule_only_catalog_contains_cold_nodes_and_metadata_without_behavior():
    schedules = [_airing(0, 10, 90, metadata={"tv_show_title": "History"}),
                 _airing(10, 20, 2, metadata={"tv_show_title": "Unwatched"}),
                 _airing(20, 35, 10, metadata={"year": None}),
                 _airing(30, 40, 1, metadata={"tv_show_title": "After horizon"})]
    expected = {"users": ["cold"], "items": [2, 10, 90], "counts": [[0, 0, 0]],
                "candidates": [2, 10], "labels": {"cold": []},
                "metadata": [{"tv_show_title": "Unwatched"}, {"year": None}, {"tv_show_title": "History"}]}
    result = build_fold(PreparedData(["cold"], schedules, [], {}), 10, 10, 30)
    _assert_fold(result, **expected)
    assert result[3]["cold_candidate_items"] == 2


@pytest.mark.parametrize("include_unobserved_in_template", [False, True])
def test_unreachable_and_future_only_users_enter_mapping_only_via_template(include_unobserved_in_template):
    schedules = [_airing(0, 10, 11), _airing(10, 20, 22)]
    logs = [_log(0, 10, user="1"), _log(10, 20, user="9"), _log(0, 20, user="2", channel="999")]
    template = ["9", "001", "2"] if include_unobserved_in_template else ["001"]
    if include_unobserved_in_template:
        expected = {"users": ["001", "1", "2", "9"], "counts": [[0, 0], [1, 0], [0, 0], [0, 0]],
                    "labels": {"001": [], "1": [], "2": [], "9": [22]}}
    else:
        expected = {"users": ["001", "1"], "counts": [[0, 0], [1, 0]], "labels": {"001": [], "1": []}}
    events, audit = reconstruct_events(logs, schedules)
    result = build_fold(PreparedData(template[:], schedules, events, audit), 10, 10, 20)
    _assert_fold(result, **expected, items=[11, 22], candidates=[22])
    assert result[3]["heldout_users_outside_cohort"] == (0 if include_unobserved_in_template else 1)


@pytest.mark.parametrize("bounds,expected", [
    ((20, 20, 50), {"counts": [[1, 1, 0, 0, 0]], "candidates": [30, 40, 50],
                    "labels": {"u": [30, 40, 50]}, "boundary": 0}),
    ((15, 25, 45), {"counts": [[1, 0, 0, 0, 0]], "candidates": [40, 50],
                    "labels": {"u": [40]}, "boundary": 3}),
])
def test_exact_endpoints_and_crossing_airings_have_distinct_training_label_candidate_rules(bounds, expected):
    schedules = [_airing(start, start + 10, start + 10) for start in (0, 10, 20, 30, 40, 50)]
    # Expected matrix/labels above are fixed before reconstructing [0,60).
    events, _ = reconstruct_events([_log(0, 60)], schedules)
    result = build_fold(PreparedData(["u"], schedules, events, {}), *bounds)
    _assert_fold(result, users=["u"], items=[10, 20, 30, 40, 50], counts=expected["counts"],
                 candidates=expected["candidates"], labels=expected["labels"])
    assert result[3]["excluded_boundary_event_count"] == expected["boundary"]


def test_eighty_percent_before_cutoff_does_not_make_an_unfinished_airing_training_history():
    schedules = [_airing(0, 100, 11), _airing(100, 200, 22)]
    expected = {"users": ["u"], "items": [11, 22], "counts": [[0, 0]],
                "candidates": [22], "labels": {"u": []}}
    events, audit = reconstruct_events([_log(0, 200)], schedules, cutoff=80)
    assert [(row["tv_show_id"], row["start"], row["end"], row["coverage"]) for row in events] == [(11, 0, 100, .8)]
    result = build_fold(PreparedData(["u"], schedules, events, audit), 80, 80, 200)
    _assert_fold(result, **expected)
    assert result[3]["excluded_boundary_event_count"] == 1


def test_fragment_union_is_per_user_channel_broadcast_not_program_lifetime():
    schedules = [_airing(0, 100, 11), _airing(100, 200, 11),
                 _airing(200, 300, 11), _airing(300, 400, 11), _airing(0, 100, 22, channel="8")]
    logs = [_log(0, 40), _log(20, 60), _log(20, 60), _log(80, 100),
            _log(100, 160), _log(180, 190), _log(200, 240), _log(260, 300), _log(300, 340),
            _log(100, 180, user="v"), _log(0, 79, channel="8"), _log(0, 80, user="v", channel="8"),
            _log(0, 60, user="pool"), _log(100, 160, user="pool")]
    # Manual occupied-second totals for u/ch7's four airings: 80,70,80,40.
    # v has 80s of the second airing and 80s of ch8. "pool" has two 60s
    # broadcasts: neither qualifies despite 120 total seconds of program 11.
    expected_events = sorted([("u", schedules[0]["airing_id"], .8), ("u", schedules[2]["airing_id"], .8),
                              ("v", schedules[1]["airing_id"], .8), ("v", schedules[4]["airing_id"], .8)])
    expected_fold = {"users": ["cold", "pool", "u", "v"], "items": [11, 22],
                     "counts": [[0, 0], [0, 0], [1, 0], [1, 1]], "candidates": [11],
                     "labels": {"cold": [], "pool": [], "u": [11], "v": []}}
    events, audit = reconstruct_events(logs, schedules)
    assert sorted((row["user_id"], row["airing_id"], row["coverage"]) for row in events) == expected_events
    _assert_fold(build_fold(PreparedData(["cold", "pool", "u", "v"], schedules, events, audit), 200, 200, 400), **expected_fold)


@pytest.mark.parametrize("covered,qualifies", [(79, False), (80, False), (81, True)])
def test_coverage_threshold_uses_exact_seconds_not_rounded_percent(covered, qualifies):
    # A 101-second airing requires 81 seconds. 80/101 rounds to 0.8 at one
    # decimal place, but 5*80=400 < 4*101=404.
    schedule = _airing(0, 101, 11)
    expected = [(11, covered / 101)] if qualifies else []
    events, _ = reconstruct_events([_log(0, covered)], [schedule])
    assert [(row["tv_show_id"], row["coverage"]) for row in events] == expected


def test_metadata_duplicate_identity_cannot_multiply_counts_or_create_false_conflicts():
    a = _airing(0, 10, 11, metadata={"tv_show_title": "A"})
    z = {**a, "metadata": {"tv_show_title": "Z"}}
    future = _airing(10, 20, 11, metadata={"tv_show_title": "Future"})
    expected = {"users": ["u"], "items": [11], "counts": [[1]], "candidates": [11],
                "labels": {"u": [11]}, "metadata": [{"tv_show_title": "A"}]}
    results = []
    for schedules in ([z, a, a, future], [future, a, a, z], [a, z, future, a]):
        events, audit = reconstruct_events([_log(0, 20)], schedules)
        assert len(events) == 2
        assert audit["duplicate_airing_rows"] == 2
        assert audit["metadata_variant_duplicate_rows"] == 1
        assert audit["conflict_components"] == 0
        result = build_fold(PreparedData(["u"], schedules, events, audit), 10, 10, 20)
        _assert_fold(result, **expected)
        results.append(result)
    _assert_same_training(results[0], results[1])
    _assert_same_training(results[0], results[2])


def test_distinct_channels_are_distinct_broadcast_identities_even_for_same_item_and_time():
    schedules = [_airing(0, 10, 11), _airing(0, 10, 11, channel="8"), _airing(10, 20, 11)]
    expected = {"users": ["u"], "items": [11], "counts": [[2]], "candidates": [11], "labels": {"u": []}}
    events, audit = reconstruct_events([_log(0, 10), _log(0, 10, channel="8")], schedules)
    assert len(events) == 2
    assert audit["conflict_components"] == 0
    _assert_fold(build_fold(PreparedData(["u"], schedules, events, audit), 10, 10, 20), **expected)


def test_conservative_transitive_quarantine_loses_safe_coverage_but_preserves_candidate_nodes():
    schedules = [_airing(0, 100, 11), _airing(90, 110, 22), _airing(105, 200, 33),
                 _airing(200, 300, 44), _airing(300, 400, 55), _airing(390, 410, 66), _airing(410, 510, 77)]
    logs = [_log(0, 80), _log(200, 280), _log(300, 380), _log(410, 490), _log(90, 110, user="unreachable")]
    # Full-component quarantine intentionally discards u's otherwise qualifying
    # [0,80) and [300,380), although neither viewing span touches an overlap.
    expected = {"users": ["empty", "u"], "items": [11, 22, 33, 44, 55, 66, 77],
                "counts": [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
                "candidates": [55, 66, 77], "labels": {"empty": [], "u": [77]}}
    events, audit = reconstruct_events(logs, schedules)
    assert [(row["user_id"], row["tv_show_id"]) for row in events] == [("u", 44), ("u", 77)]
    assert audit["conflict_components"] == 2
    assert audit["conflict_unique_airings"] == 5
    assert audit["conflict_affected_log_rows"] == 3
    assert audit["conflict_affected_users"] == 2
    result = build_fold(PreparedData(["u", "empty"], schedules, events, audit), 300, 300, 510)
    _assert_fold(result, **expected)
    assert result[3]["known_quarantined_airings"] == 5
    assert result[3]["cold_candidate_items"] == 3


def test_labels_use_broadcast_frequency_not_viewing_seconds_then_numeric_top_five():
    schedules = [_airing(0, 10, 8)]
    for index, item in enumerate([100, 100, 3, 3, 10, 2, 4, 5, 6, 7]):
        schedules.append(_airing(10 + 10 * index, 20 + 10 * index, item))
    schedules.append(_airing(110, 210, 999))
    expected = {"users": ["u"], "items": [2, 3, 4, 5, 6, 7, 8, 10, 100, 999],
                "counts": [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
                "candidates": [2, 3, 4, 5, 6, 7, 10, 100, 999], "labels": {"u": [3, 100, 2, 4, 5]}}
    # Program 999 contributes one 80s-completed airing, not eight 10s votes.
    events, audit = reconstruct_events([_log(0, 110), _log(110, 190)], schedules)
    result = build_fold(PreparedData(["u"], schedules, events, audit), 10, 10, 210)
    _assert_fold(result, **expected)
    assert result[3]["heldout_top5_boundary_tie_users"] == 1


def _write_csv(path: Path, headers, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def _documented_raw_fixture(directory: Path, *, year="NaN"):
    # These are synthetic rows using the documented column names, not exports.
    _write_csv(directory / "submission.csv", ["user_id", "tv_show_id"],
               [["0007", "ignored"], [str(MAX_ID), "0"], [str(WIDE_ID), "NaN"], ["0000", ""]])
    _write_csv(directory / LOG_NAME, ["user_id", "vsetv_id", "start_time", "end_time", "duration", "tv_show_id"], [
        ["0007", "0007", BASE, BASE + 100, "100000", "999"],
        [str(MAX_ID), "7.0", BASE, BASE + 100, "100", "999"],
        [f"{WIDE_ID}.0", "7", BASE, BASE + 100, "100", "999"],
        ["0007", "7", BASE + 100, BASE + 200, "100", "999"],
        ["8", "7", BASE + 100, BASE + 200, "100", "999"],
    ])
    for name, item, start, title, raw_year in zip(
        EPG_NAMES, [WIDE_ID, MAX_ID, 77], [BASE, BASE + 100, BASE + 200],
        ["History", "Forecast", "After horizon"], [year, "NULL", "2026"],
    ):
        _write_csv(directory / name,
                   ["channel_id", "tv_show_id", "start_time", "duration", "tv_show_title", "year", "tv_show_genre_1"],
                   [["7", str(item), start, "100", title, raw_year, "None"]])


def _assert_documented_raw_fold(prepared, *, year=None):
    assert prepared.template_users == ["0007", str(MAX_ID), str(WIDE_ID), "0000"]
    assert prepared.audit["schema_verified_on_actual_source"] is False
    assert len(prepared.events) == 5
    # Decoy log/template tv_show_id values (999 / zero / text) are never labels.
    expected = {"users": ["0000", "0007", str(WIDE_ID), str(MAX_ID)], "items": [WIDE_ID, MAX_ID],
                "counts": [[0, 0], [1, 0], [1, 0], [1, 0]], "candidates": [MAX_ID],
                "labels": {"0000": [], "0007": [MAX_ID], str(WIDE_ID): [], str(MAX_ID): []},
                "metadata": [{"tv_show_title": "History", "year": year, "tv_show_genre_1": None},
                             {"tv_show_title": "Forecast", "year": None, "tv_show_genre_1": None}]}
    result = build_fold(prepared, BASE + 100, BASE + 100, BASE + 200)
    _assert_fold(result, **expected)
    assert result[3]["heldout_users_outside_cohort"] == 1
    assert prepared.audit["reconstruction"]["raw_duration_mismatch_rows"] == 1


def test_documented_raw_csv_headers_work_with_default_schema(tmp_path):
    _documented_raw_fixture(tmp_path)
    # Do not substitute observed defaults in this test: compatibility with the
    # documented start_time/end_time source interface is what is challenged.
    prepared = load_raw(tmp_path, delimiter=",", timestamp_unit="seconds")
    _assert_documented_raw_fold(prepared)


def test_documented_log_headers_work_when_only_epg_adapter_is_configured(tmp_path):
    _documented_raw_fixture(tmp_path)
    prepared = load_raw(tmp_path, schema=RawSchema(schedule_start="start_time"),
                        delimiter=",", timestamp_unit="seconds")
    _assert_documented_raw_fold(prepared)


def test_documented_columns_are_parseable_with_explicit_exact_adapter(tmp_path):
    _documented_raw_fixture(tmp_path)
    prepared = load_raw(tmp_path, schema=RawSchema(**DOCUMENTED_ADAPTER), delimiter=",", timestamp_unit="seconds")
    _assert_documented_raw_fold(prepared)


@pytest.mark.parametrize("year,expected", [("", None), ("NaN", None), ("NULL", None), ("N/A", None),
                                           ("None", None), ("2026", "2026"), ("2026.0", "2026.0"), ("unknown", "unknown")])
def test_null_or_nonintegral_year_metadata_does_not_trigger_integer_cast(tmp_path, year, expected):
    _documented_raw_fixture(tmp_path, year=year)
    prepared = load_raw(tmp_path, schema=RawSchema(**DOCUMENTED_ADAPTER), delimiter=",", timestamp_unit="seconds")
    _assert_documented_raw_fold(prepared, year=expected)


@pytest.mark.parametrize("raw_item,counter", [("NaN", "missing_program_rows"), ("NULL", "missing_program_rows"),
                                             ("", "missing_program_rows"), ("0", "zero_program_rows"),
                                             ("-1", "negative_program_rows")])
def test_source_null_and_nonpositive_program_ids_are_dropped_not_cast_into_programs(tmp_path, raw_item, counter):
    _documented_raw_fixture(tmp_path)
    _write_csv(tmp_path / EPG_NAMES[2], ["channel_id", "tv_show_id", "start_time", "duration"],
               [["7", raw_item, BASE + 200, "100"]])
    prepared = load_raw(tmp_path, schema=RawSchema(**DOCUMENTED_ADAPTER), delimiter=",", timestamp_unit="seconds")
    _assert_documented_raw_fold(prepared)
    assert prepared.audit["schedule_parse"][counter] == 1
    assert [row["tv_show_id"] for row in prepared.schedules] == [WIDE_ID, MAX_ID]


@pytest.mark.parametrize("raw_item", [str(MAX_ID + 1), f"{WIDE_ID}.25", "1e3"])
def test_unrepresentable_or_inexact_program_ids_fail_closed_instead_of_silent_cast(tmp_path, raw_item):
    _documented_raw_fixture(tmp_path)
    _write_csv(tmp_path / EPG_NAMES[2], ["channel_id", "tv_show_id", "start_time", "duration"],
               [["7", raw_item, BASE + 200, "100"]])
    with pytest.raises(DataSchemaError) as error:
        load_raw(tmp_path, schema=RawSchema(**DOCUMENTED_ADAPTER), delimiter=",", timestamp_unit="seconds")
    assert EPG_NAMES[2] in str(error.value)
    assert error.value.audit["schema_verified_on_actual_source"] is False


@pytest.mark.parametrize("missing_role", ["user_id", "vsetv_id", "start_time", "end_time"])
def test_missing_documented_log_role_is_a_schema_error_even_with_other_plausible_columns(tmp_path, missing_role):
    _documented_raw_fixture(tmp_path)
    headers = [name for name in ["user_id", "vsetv_id", "start_time", "end_time"] if name != missing_role]
    row = {"user_id": "0007", "vsetv_id": "7", "start_time": str(BASE), "end_time": str(BASE + 100)}
    _write_csv(tmp_path / LOG_NAME, headers, [[row[name] for name in headers]])
    with pytest.raises(DataSchemaError, match="missing exact columns") as error:
        load_raw(tmp_path, schema=RawSchema(**DOCUMENTED_ADAPTER), delimiter=",", timestamp_unit="seconds")
    assert missing_role in str(error.value)


def test_incorrect_empty_epg_headers_are_not_mistaken_for_an_empty_candidate_catalog(tmp_path):
    _documented_raw_fixture(tmp_path)
    _write_csv(tmp_path / EPG_NAMES[1], ["channel_id", "tv_show_id", "not_start_time", "duration"], [])
    with pytest.raises(DataSchemaError, match="missing exact columns"):
        load_raw(tmp_path, schema=RawSchema(**DOCUMENTED_ADAPTER), delimiter=",", timestamp_unit="seconds")


def test_precoerced_nan_and_float_identifiers_fail_closed_at_normalized_boundary():
    # CSV null tokens are handled above. These are invalid *already normalized*
    # objects: rejecting them is deliberate, not proof of a raw-CSV null bug.
    for item in (float("nan"), float(WIDE_ID), True):
        with pytest.raises(ValueError):
            reconstruct_events([_log(0, 10)], [_airing(0, 10, item)])
