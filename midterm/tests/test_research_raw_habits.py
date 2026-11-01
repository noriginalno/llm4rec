"""Offline fixtures with hand arithmetic and an independent per-second oracle."""

from __future__ import annotations

import builtins
from copy import deepcopy
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.sparse import csr_matrix, issparse

from research_v2.contracts import Airing, TemporalContext
from research_v2.raw_habits import RawScheduleModel, build_cache
from sweettv.contracts import TrainingData


DAY = 86400
HOUR = 3600
MONDAY = 1583712000  # naive numeric anchor, NOT a claimed physical timezone
EPOCH = datetime(1970, 1, 1)
USER = "18417531283109304442"  # exceeds signed int64; never a numeric matrix ID


def cache(tmp_path, rows, *, name="fixture"):
    directory = tmp_path / name
    directory.mkdir()
    source = directory / "logs.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_id", "vsetv_id", "start_time", "stop_time", "duraton"])
        for row in rows:
            user, channel, start, end, *reported = row
            writer.writerow([user, channel, (EPOCH + timedelta(seconds=start)).isoformat(sep=" "),
                             (EPOCH + timedelta(seconds=end)).isoformat(sep=" "),
                             reported[0] if reported else end - start])
    manifest = directory / "source-manifest.json"
    manifest.write_text(json.dumps({"files": {source.name: {
        "bytes": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }}}), encoding="utf-8")
    output = directory / "union.parquet"
    result = build_cache(source, output, manifest)
    return output, source, manifest, result


def context(forecast=(), *, user_ids=(USER, "001", "cold"), n_items=6,
            history_start=MONDAY, train_end=MONDAY + 7 * DAY, forecast_end=None,
            candidates=None, counts=None):
    data = TrainingData(
        list(user_ids), np.arange(901, 901 + n_items, dtype=np.int64),
        csr_matrix((len(user_ids), n_items)) if counts is None else counts,
        [{"tv_show_title": f"metadata only {i}"} for i in range(n_items)],
    )
    empty = np.array([], dtype=np.int64)
    return TemporalContext(
        data, np.arange(n_items) if candidates is None else candidates,
        history_start, train_end, train_end,
        train_end + 8 * DAY if forecast_end is None else forecast_end,
        empty, empty, empty, empty, np.array([], dtype=str), (), tuple(forecast),
    )


def model(output, ctx, **kwargs):
    return RawScheduleModel(output, half_life_days=None, **kwargs).fit(ctx)


def oracle(rows, ctx, user, slot_minutes, half_life=None, power=1, mode="week"):
    """Brute-force seconds, datetime.weekday and literal set union; no model helpers.

    Tiny windows only. Future schedule means are evaluated one second at a time;
    numerical expectations do not reuse the production splitter/matrix path.
    """
    watched = {}
    for raw_user, channel, start, end, *_ in rows:
        if raw_user == user:
            watched.setdefault(channel, set()).update(range(max(start, ctx.history_start), min(end, ctx.train_end)))

    def bin_and_weight(second):
        dt = EPOCH + timedelta(seconds=second)
        minute_of_day = dt.hour * 60 + dt.minute
        slot = minute_of_day // slot_minutes
        if mode == "week":
            slot += dt.weekday() * (1440 // slot_minutes)
        midpoint = dt.replace(hour=0, minute=0, second=0) + timedelta(
            minutes=(minute_of_day // slot_minutes + 0.5) * slot_minutes)
        age_days = max(0, (EPOCH + timedelta(seconds=ctx.train_end) - midpoint).total_seconds() / DAY)
        return slot, 1 if half_life is None else 2 ** (-age_days / half_life)

    denominator, numerators = {}, {}
    for second in range(ctx.history_start, ctx.train_end):
        slot, weight = bin_and_weight(second)
        denominator[slot] = denominator.get(slot, 0) + weight
        for channel, seconds in watched.items():
            if second in seconds:
                key = (channel, slot)
                numerators[key] = numerators.get(key, 0) + weight
    output = np.zeros(len(ctx.data.item_ids))
    unique = {(a.item_index, a.channel_id, a.start, a.end) for a in ctx.forecast_schedules if not a.quarantined}
    for item, channel, start, end in unique:
        total = 0.0
        for second in range(start, end):
            slot, _ = bin_and_weight(second)
            total += numerators.get((channel, slot), 0) / denominator.get(slot, 1)
        output[item] += (total / (end - start)) ** power
    output[output <= 0] = np.nan
    return output


def test_cache_unions_nested_duplicate_fragmented_devices_and_preserves_opaque_ids(tmp_path):
    rows = [
        (USER, "006", MONDAY, MONDAY + 100, 999),
        (USER, "006", MONDAY + 5, MONDAY + 10),
        (USER, "006", MONDAY + 20, MONDAY + 25),  # LAG(end) would split this wrongly.
        (USER, "006", MONDAY, MONDAY + 100),
        (USER, "006", MONDAY + 100, MONDAY + 120),
        (USER, "006", MONDAY + 121, MONDAY + 130),
        (USER, "6", MONDAY, MONDAY + 10),
        ("001", "006", MONDAY, MONDAY + 10),
        ("1", "006", MONDAY, MONDAY + 10),
    ]
    output, source, manifest, result = cache(tmp_path, rows)
    table = pq.read_table(output)
    assert table.schema.equals(pa.schema([
        ("user_id", pa.string()), ("channel_id", pa.string()),
        ("start", pa.int64()), ("end", pa.int64()),
    ]))
    expected = [
        {"user_id": "001", "channel_id": "006", "start": MONDAY, "end": MONDAY + 10},
        {"user_id": "1", "channel_id": "006", "start": MONDAY, "end": MONDAY + 10},
        {"user_id": USER, "channel_id": "006", "start": MONDAY, "end": MONDAY + 120},
        {"user_id": USER, "channel_id": "006", "start": MONDAY + 121, "end": MONDAY + 130},
        {"user_id": USER, "channel_id": "6", "start": MONDAY, "end": MONDAY + 10},
    ]
    assert table.to_pylist() == expected
    assert result["audit"]["raw_rows"] == 9 and result["audit"]["union_rows"] == 5
    assert result["audit"]["union_seconds"] == 159
    assert result["audit"]["duration_mismatch_rows"] == 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == result["source"]["sha256"]
    source_stat, output_stat = source.stat(), output.stat()
    reused = build_cache(source, output, manifest)
    assert reused["reused"]
    assert source.stat() == source_stat and output.stat() == output_stat


@pytest.mark.parametrize("power", [1, 2])
@pytest.mark.parametrize("mode", ["week", "day"])
def test_independent_seconds_oracle_splits_hour_midnight_and_week_boundary(tmp_path, power, mode):
    start = MONDAY - HOUR  # Sunday 23:00 -> Monday 01:00.
    end = MONDAY + HOUR
    rows = [
        (USER, "c", start + 20 * 60, MONDAY + 10 * 60),
        (USER, "c", start + 25 * 60, MONDAY + 5 * 60),
        (USER, "c", MONDAY + 35 * 60, end + 100),  # clipped at train cutoff
        (USER, "d", start + 45 * 60, MONDAY + 35 * 60),
        ("001", "c", start, end),
    ]
    output, *_ = cache(tmp_path, rows)
    shift = 7 * DAY
    forecast = [
        Airing(0, "c", start + shift + 25 * 60, MONDAY + shift + 40 * 60),
        Airing(1, "d", start + shift + 45 * 60, MONDAY + shift + 10 * 60),
        Airing(2, "unknown", MONDAY + shift, MONDAY + shift + 10 * 60),
        Airing(0, "c", MONDAY + shift + 45 * 60, end + shift),
    ]
    ctx = context(forecast, history_start=start, train_end=end)
    fitted = RawScheduleModel(output, half_life_days=0.1, slot_minutes=30,
                              coverage_power=power, slot_mode=mode, score_block_size=2).fit(ctx)
    expected = oracle(rows, ctx, USER, 30, half_life=0.1, power=power, mode=mode)
    assert_allclose(fitted.score(0, np.arange(6)), expected, rtol=2e-12, atol=2e-12, equal_nan=True)
    assert fitted.diagnostics["maximum_occupancy"] <= 1
    assert fitted.diagnostics["maximum_behavior_end"] == end
    assert fitted.diagnostics["source"]["sha256"]
    assert fitted.diagnostics["source"]["cache_sha256"]


def test_global_union_then_clip_is_invariant_to_future_viewing_mutations(tmp_path):
    cutoff = MONDAY + HOUR
    past = [(USER, "c", MONDAY, MONDAY + 20 * 60),
            (USER, "c", cutoff - 20 * 60, cutoff + 10)]
    future = [(USER, "c", cutoff, cutoff + 50 * DAY),
              (USER, "future-only", cutoff + 1, cutoff + DAY),
              ("001", "c", cutoff, cutoff + 4 * DAY)]
    cache_a, *_, meta_a = cache(tmp_path, past, name="before")
    cache_b, *_, meta_b = cache(tmp_path, past + future, name="mutated")
    assert meta_a["source"]["sha256"] != meta_b["source"]["sha256"]
    ctx = context([Airing(0, "c", MONDAY + 7 * DAY, MONDAY + 7 * DAY + HOUR)], train_end=cutoff)
    a = RawScheduleModel(cache_a, half_life_days=0.01).fit(ctx)
    b = RawScheduleModel(cache_b, half_life_days=0.01).fit(ctx)
    assert a.channel_ids_ == b.channel_ids_ == ("c",)
    assert_allclose(a.occupancy_.toarray(), b.occupancy_.toarray(), atol=1e-15)
    assert_allclose(a.score(0, [0]), b.score(0, [0]), atol=1e-15)
    assert a.diagnostics["training_feature_sha256"] == b.diagnostics["training_feature_sha256"]
    assert b.diagnostics["maximum_behavior_end"] == cutoff
    assert b.diagnostics["history_window"]["source_cutoff_exclusive"] == cutoff


@pytest.mark.parametrize("power", [1, 2, 0.5])
def test_duration_normalization_airing_aggregation_and_no_watched_filter(tmp_path, power):
    rows = [(USER, "c", MONDAY, MONDAY + 15 * 60),
            (USER, "c", MONDAY + 30 * 60, MONDAY + 45 * 60)]
    output, *_ = cache(tmp_path, rows)
    future = MONDAY + 7 * DAY
    airings = [Airing(0, "c", future, future + 10 * 60),
               Airing(1, "c", future, future + HOUR),
               Airing(2, "c", future + 10 * 60, future + 20 * 60),
               Airing(2, "c", future + 20 * 60, future + 30 * 60)]
    counts = csr_matrix(([17], ([0], [0])), shape=(3, 6))
    ctx = context(airings, train_end=MONDAY + HOUR, counts=counts, candidates=np.array([5]))
    fitted = model(output, ctx, coverage_power=power, score_block_size=1)
    base = 0.5 ** power
    assert_allclose(fitted.score(0, [0, 1, 2]), [base, base, 2 * base], atol=1e-15)
    assert fitted.diagnostics["behavior_source"].startswith("all raw")
    assert fitted.diagnostics["active_users"] == 1  # raw evidence, not count CSR


def test_non_linear_power_is_applied_after_mean_per_airing_not_per_bin_or_program(tmp_path):
    # Occupancy [1, .5]; program0 has two airings with respective means 1 and .5.
    rows = [(USER, "c", MONDAY, MONDAY + 30 * 60),
            (USER, "c", MONDAY + 30 * 60, MONDAY + 45 * 60)]
    output, *_ = cache(tmp_path, rows)
    future = MONDAY + 7 * DAY
    ctx = context([Airing(0, "c", future, future + 30 * 60),
                   Airing(0, "c", future + 30 * 60, future + HOUR),
                   Airing(1, "c", future, future + HOUR)], train_end=MONDAY + HOUR)
    fitted = model(output, ctx, coverage_power=2)
    assert_allclose(fitted.score(0, [0, 1]), [1.25, 0.75 ** 2], atol=1e-15)


def test_identical_bin_midpoint_weights_keep_full_coverage_one_at_partial_edges(tmp_path):
    start, cutoff = MONDAY + 11 * 60 + 3, MONDAY + HOUR + 7 * 60 + 9
    rows = [(USER, "c", start - HOUR, cutoff + HOUR)]
    output, *_ = cache(tmp_path, rows)
    ctx = context([Airing(0, "c", MONDAY + 7 * DAY, MONDAY + 7 * DAY + HOUR)],
                  history_start=start, train_end=cutoff)
    fitted = RawScheduleModel(output, half_life_days=0.01, lookback_days=40 / 1440).fit(ctx)
    assert_allclose(fitted.occupancy_.data, 1, atol=1e-15)
    assert fitted.diagnostics["minimum_behavior_start"] == cutoff - 40 * 60
    assert fitted.diagnostics["maximum_behavior_end"] == cutoff
    assert fitted.diagnostics["clipped_union_seconds"] == 40 * 60


def test_repeated_week_slots_have_consistent_midpoint_weighted_denominator(tmp_path):
    output, *_ = cache(tmp_path, [(USER, "c", MONDAY, MONDAY + HOUR)])
    cutoff = MONDAY + 14 * DAY
    ctx = context([Airing(0, "c", cutoff, cutoff + HOUR)], train_end=cutoff)
    fitted = RawScheduleModel(output, half_life_days=7, slot_minutes=60).fit(ctx)
    assert fitted.score(0, [0])[0] == pytest.approx(1 / 3, abs=1e-15)
    no_decay = model(output, ctx, slot_minutes=60)
    assert no_decay.score(0, [0])[0] == pytest.approx(1 / 2)
    recent_only = RawScheduleModel(output, half_life_days=7, lookback_days=7).fit(ctx)
    assert np.isnan(recent_only.score(0, [0])[0])


@pytest.mark.parametrize("slot_minutes", [15, 30, 60])
def test_second_resolution_forecast_overlap_is_linear_not_start_slot_assignment(tmp_path, slot_minutes):
    width = slot_minutes * 60
    # Each of two bins has a different occupancy. A future airing has only
    # seven seconds in the first, eleven in the second: exact fractional mean.
    rows = [(USER, "c", MONDAY, MONDAY + width),
            (USER, "c", MONDAY + width, MONDAY + width + width // 4)]
    output, *_ = cache(tmp_path, rows)
    future = MONDAY + 7 * DAY
    ctx = context([Airing(0, "c", future + width - 7, future + width + 11)],
                  train_end=MONDAY + 2 * width)
    fitted = model(output, ctx, slot_minutes=slot_minutes)
    assert fitted.score(0, [0])[0] == pytest.approx((7 + 11 / 4) / 18, abs=1e-15)


def test_empty_user_item_dimensions_and_subsecond_lookback_are_safe(tmp_path):
    output, *_ = cache(tmp_path, [(USER, "c", MONDAY, MONDAY + HOUR)])
    fitted = model(output, context(user_ids=(), n_items=0, train_end=MONDAY + HOUR))
    assert fitted.occupancy_.shape == (0, 0)
    assert fitted.score(0, np.array([], dtype=np.int64)).shape == (0,)
    future = MONDAY + 7 * DAY
    ctx = context([Airing(0, "c", future, future + HOUR)], train_end=MONDAY + HOUR)
    fitted = model(output, ctx, lookback_days=0.1 / DAY)
    assert fitted.diagnostics["history_window"]["effective_start"] == ctx.train_end
    assert np.isnan(fitted.score(0, [0])[0])


def test_day_mode_pools_weekdays_but_week_mode_does_not_invent_cross_weekday_evidence(tmp_path):
    output, *_ = cache(tmp_path, [(USER, "c", MONDAY, MONDAY + HOUR)])
    future_tuesday = MONDAY + 8 * DAY
    ctx = context([Airing(0, "c", future_tuesday, future_tuesday + HOUR)],
                  train_end=MONDAY + 2 * DAY)
    daily = model(output, ctx, slot_mode="day")
    weekly = model(output, ctx, slot_mode="week")
    assert daily.score(0, [0])[0] == pytest.approx(1 / 2)
    assert np.isnan(weekly.score(0, [0])[0])


def test_forecast_metadata_can_change_count_but_completed_event_arrays_and_metadata_do_not(tmp_path):
    output, *_ = cache(tmp_path, [(USER, "c", MONDAY, MONDAY + HOUR)])
    future = MONDAY + 7 * DAY
    one = Airing(0, "c", future, future + HOUR)
    two = Airing(0, "c", future + 7 * DAY, future + 7 * DAY + HOUR)
    ctx = context([one], train_end=MONDAY + HOUR)
    first = model(output, ctx)
    changed = context([one, two], train_end=MONDAY + HOUR,
                      forecast_end=future + 8 * DAY,
                      counts=csr_matrix(np.ones((3, 6))))
    changed.data.metadata[:] = [{"tv_show_title": "changed unrelated supplied text"}] * 6
    second = model(output, changed)
    assert_allclose(first.occupancy_.toarray(), second.occupancy_.toarray())
    assert second.score(0, [0])[0] == pytest.approx(2 * first.score(0, [0])[0])
    assert first.diagnostics["training_feature_sha256"] == second.diagnostics["training_feature_sha256"]


@pytest.mark.parametrize("power", [1, 2])
def test_candidate_order_duplicates_unknowns_and_opaque_user_index_separation(tmp_path, power):
    rows = [(USER, "006", MONDAY, MONDAY + HOUR),
            ("001", "006", MONDAY, MONDAY + HOUR)]
    output, *_ = cache(tmp_path, rows)
    future = MONDAY + 7 * DAY
    ctx = context([Airing(0, "006", future, future + HOUR),
                   Airing(1, "6", future, future + HOUR)], train_end=MONDAY + HOUR)
    fitted = model(output, ctx, coverage_power=power)
    indices = np.array([1, 0, -1, 6, 0, 5, 1])
    expected = np.array([np.nan, 1, np.nan, np.nan, 1, np.nan, np.nan])
    assert_allclose(fitted.score(0, indices), expected, equal_nan=True)
    assert_allclose(fitted.score(1, indices), expected, equal_nan=True)
    assert_allclose(fitted.score(0, indices[::-1]), expected[::-1], equal_nan=True)
    assert_array_equal(indices, [1, 0, -1, 6, 0, 5, 1])
    assert np.isnan(fitted.score(0, np.array([2**64 - 1], dtype=np.uint64))).all()
    assert np.isnan(fitted.score(0, ctx.data.item_ids)).all()
    for user in (-1, 2, 3, 999):
        assert np.isnan(fitted.score(user, indices)).all()
    assert fitted.score(np.int64(0), np.array([], dtype=np.int64)).shape == (0,)


@pytest.mark.parametrize("kind", ["no_raw", "other_user", "old_only", "future_only", "no_forecast"])
def test_absent_evidence_has_no_internal_fallback(tmp_path, kind):
    row = (USER, "c", MONDAY, MONDAY + HOUR)
    if kind == "no_raw":
        rows = []
    elif kind == "other_user":
        rows = [("absent", *row[1:])]
    elif kind == "old_only":
        rows = [(USER, "c", MONDAY - HOUR, MONDAY)]
    elif kind == "future_only":
        rows = [(USER, "c", MONDAY + HOUR, MONDAY + DAY)]
    else:
        rows = [row]
    output, *_ = cache(tmp_path, rows)
    future = MONDAY + 7 * DAY
    forecast = [] if kind == "no_forecast" else [Airing(0, "c", future, future + HOUR)]
    fitted = model(output, context(forecast, train_end=MONDAY + HOUR))
    assert np.isnan(fitted.score(0, [0, 1, -1])).all()


def test_forecast_duplicates_quarantine_or_policy_and_refit_reset(tmp_path):
    output, *_ = cache(tmp_path, [(USER, "c", MONDAY, MONDAY + HOUR)])
    future = MONDAY + 7 * DAY
    ordinary = Airing(0, "c", future, future + HOUR)
    flagged = Airing(0, "c", future, future + HOUR, True)
    ctx = context([ordinary, flagged, ordinary], train_end=MONDAY + HOUR)
    excluded = model(output, ctx)
    assert np.isnan(excluded.score(0, [0])[0])
    included = model(output, ctx, quarantined_future_policy="include")
    assert included.score(0, [0])[0] == pytest.approx(1)
    assert included.diagnostics["forecast_duplicates_ignored"] == 2
    assert included.diagnostics["quarantined_forecast_airings"] == 1
    new = context([], user_ids=("new",), n_items=2, train_end=MONDAY + HOUR)
    assert included.fit(new) is included
    assert included.occupancy_.shape == (1, 0) and included.occupancy_.nnz == 0
    assert included.forecast_features_.shape == (0, 2)
    assert included.channel_ids_ == () and included.user_ids_ == ("new",)
    assert np.isnan(included.score(0, [0, 1])).all()


def test_fit_does_not_mutate_inputs_and_later_input_parameter_edits_do_not_change_snapshot(tmp_path):
    output, *_ = cache(tmp_path, [(USER, "c", MONDAY, MONDAY + 15 * 60)])
    future = MONDAY + 7 * DAY
    ctx = context([Airing(0, "c", future, future + 30 * 60)], train_end=MONDAY + HOUR)
    original_ids, original_metadata = ctx.data.item_ids.copy(), deepcopy(ctx.data.metadata)
    original_users = list(ctx.data.user_ids)
    original_counts = ctx.data.interactions.copy()
    original_schedules = ctx.forecast_schedules
    fitted = model(output, ctx)
    assert_allclose(fitted.score(0, [0]), [0.5])
    assert_array_equal(ctx.data.item_ids, original_ids)
    assert ctx.data.metadata == original_metadata and ctx.data.user_ids == original_users
    assert ctx.forecast_schedules == original_schedules
    for field in ("data", "indices", "indptr"):
        assert_array_equal(getattr(ctx.data.interactions, field), getattr(original_counts, field))
    for field in ("candidate_indices", "event_users", "event_items", "event_starts", "event_ends", "event_channels"):
        assert not getattr(ctx, field).flags.writeable
    fitted.coverage_power = 2
    fitted.slot_mode = "day"
    fitted.slot_minutes = 60
    fitted.score_block_size = 1
    ctx.data.item_ids[:] += 100
    ctx.data.user_ids[0] = "mutated-user"
    assert_allclose(fitted.score(0, [0]), [0.5])
    assert_array_equal(fitted.item_ids_, original_ids)
    ctx.data.user_ids[0] = USER
    fitted.fit(ctx)
    assert_allclose(fitted.score(0, [0]), [0.25 ** 2])


def test_sparse_fit_and_bounded_scoring_need_no_network_or_dense_users_times_airings(tmp_path, monkeypatch):
    output, *_ = cache(tmp_path, [(USER, "c", MONDAY, MONDAY + HOUR)])
    future = MONDAY + 7 * DAY
    ctx = context([Airing(i, "c", future, future + HOUR) for i in range(6)], train_end=MONDAY + HOUR)

    def forbidden(*args, **kwargs):
        raise AssertionError("No network or sparse densification during fit")

    with monkeypatch.context() as patch:
        patch.setattr(socket, "socket", forbidden)
        patch.setattr(csr_matrix, "toarray", forbidden)
        fitted = model(output, ctx, coverage_power=2, score_block_size=2)
    assert issparse(fitted.occupancy_) and issparse(fitted.forecast_features_)
    original = csr_matrix.toarray
    shapes = []

    def bounded(self, *args, **kwargs):
        shapes.append(self.shape)
        assert self.shape[0] == 1 and self.shape[1] <= 2
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(socket, "socket", forbidden)
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(csr_matrix, "toarray", bounded)
        assert_allclose(fitted.score(0, np.arange(6)), np.ones(6))
    assert shapes == [(1, 2)] * 3


@pytest.mark.parametrize("target", ["source", "manifest", "cache", "missing_sidecar"])
def test_source_bound_cache_rejects_changes_or_missing_provenance(tmp_path, target):
    output, source, manifest, _ = cache(tmp_path, [(USER, "c", MONDAY, MONDAY + HOUR)])
    if target == "source":
        with source.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    elif target == "manifest":
        manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    elif target == "cache":
        with output.open("ab") as stream:
            stream.write(b"altered")
    else:
        output.with_suffix(".parquet.meta.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        model(output, context())
    with pytest.raises((ValueError, FileNotFoundError)):
        build_cache(source, output, manifest)


def test_build_refuses_wrong_manifest_sha_without_modifying_raw(tmp_path):
    output, source, manifest, _ = cache(tmp_path, [(USER, "c", MONDAY, MONDAY + HOUR)])
    source_bytes = source.read_bytes()
    value = json.loads(manifest.read_text())
    value["files"][source.name]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        build_cache(source, output.with_name("new.parquet"), manifest)
    assert source.read_bytes() == source_bytes
    assert not output.with_name("new.parquet").exists()


def test_cache_schema_rejects_numeric_user_ids_even_if_checksum_metadata_is_updated(tmp_path):
    output, *_ = cache(tmp_path, [("7", "c", MONDAY, MONDAY + HOUR)])
    table = pq.read_table(output)
    table = table.set_column(0, "user_id", pa.array([7], type=pa.int64()))
    pq.write_table(table, output)
    sidecar = output.with_suffix(".parquet.meta.json")
    meta = json.loads(sidecar.read_text())
    meta["cache"]["guard"] = {"bytes": output.stat().st_size, "mtime_ns": output.stat().st_mtime_ns}
    meta["cache"]["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="string IDs"):
        model(output, context(user_ids=("7",)))


@pytest.mark.parametrize("change", ["timezone", "subsecond", "backward", "empty_user"])
def test_build_rejects_invalid_raw_clocks_or_ids_instead_of_coercing(tmp_path, change):
    directory = tmp_path / "invalid"
    directory.mkdir()
    source = directory / "logs.csv"
    user, start, stop = USER, "2020-03-09 00:00:00", "2020-03-09 01:00:00"
    if change == "timezone":
        start += "+03:00"
    elif change == "subsecond":
        start += ".100"
    elif change == "backward":
        stop = "2020-03-08 23:59:59"
    else:
        user = ""
    source.write_text(f"user_id,vsetv_id,start_time,stop_time,duraton\n{user},6,{start},{stop},0\n", encoding="utf-8")
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"files": {source.name: {
        "bytes": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }}}), encoding="utf-8")
    output = directory / "union.parquet"
    with pytest.raises(ValueError, match="Invalid raw"):
        build_cache(source, output, manifest)
    assert not output.exists() and not output.with_suffix(".parquet.meta.json").exists()


@pytest.mark.parametrize("kwargs", [
    {"half_life_days": 0}, {"half_life_days": np.inf}, {"half_life_days": True},
    {"lookback_days": -1}, {"lookback_days": np.nan}, {"lookback_days": "7"},
    {"slot_minutes": 0}, {"slot_minutes": 7}, {"slot_minutes": 30.0}, {"slot_minutes": True},
    {"coverage_power": 0}, {"coverage_power": np.inf}, {"coverage_power": True},
    {"slot_mode": "physical_UTC"}, {"score_block_size": 0}, {"score_block_size": 3.0},
    {"quarantined_future_policy": "drop_program"},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        RawScheduleModel(**kwargs)


@pytest.mark.parametrize("candidates", [[0.0], [True], ["0"], [[0, 1]]])
def test_invalid_candidates(tmp_path, candidates):
    output, *_ = cache(tmp_path, [])
    fitted = model(output, context())
    with pytest.raises(ValueError):
        fitted.score(0, np.array(candidates))


@pytest.mark.parametrize("user", [True, 0.0, USER])
def test_invalid_user_matrix_indices(tmp_path, user):
    output, *_ = cache(tmp_path, [])
    fitted = model(output, context())
    with pytest.raises(ValueError):
        fitted.score(user, [0])


def test_fit_required_and_failed_refit_cannot_serve_stale_scores(tmp_path):
    fitted = RawScheduleModel()
    with pytest.raises(RuntimeError, match="Fit"):
        fitted.score(0, [0])
    output, *_ = cache(tmp_path, [(USER, "c", MONDAY, MONDAY + HOUR)])
    fitted.raw_cache_path = output
    fitted.fit(context(train_end=MONDAY + HOUR))
    fitted.coverage_power = -1
    with pytest.raises(ValueError):
        fitted.fit(context())
    with pytest.raises(RuntimeError, match="Fit"):
        fitted.score(0, [0])
