"""Independent raw-occupancy challenges; only synthetic temporary input/cache files.

No author's fixtures, production splitters, labels, predictions, or real logs are
used as numerical oracles. Literal second sets, an endpoint-depth sweep, naive
datetime calendar arithmetic, and hand-derived fractions provide expectations.
"""

from __future__ import annotations

import builtins
from collections import defaultdict
from copy import deepcopy
import csv
from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import socket

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pyarrow.parquet as pq
import pytest
from scipy.sparse import csr_matrix, issparse

from research_v2.contracts import Airing, TemporalContext
import research_v2.raw_habits as raw
from research_v2.raw_habits import RawScheduleModel, build_cache
from sweettv.contracts import TrainingData


DAY = 86400
EPOCH = datetime(1970, 1, 1)
USER = "18446744073709551615"  # uint64 maximum, always an opaque string.
OTHER_WIDE = "18446744073709551614"


def seconds(text):
    return int((datetime.fromisoformat(text) - EPOCH).total_seconds())


MONDAY = seconds("2020-03-30 00:00:00")


def artifact_state(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_cache(tmp_path, rows, name="synthetic"):
    directory = tmp_path / name
    directory.mkdir()
    source = directory / "opaque-raw.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_id", "vsetv_id", "start_time", "stop_time", "duraton"])
        for user, channel, start, end, *reported in rows:
            writer.writerow([user, channel, (EPOCH + timedelta(seconds=start)).isoformat(sep="T"),
                             (EPOCH + timedelta(seconds=end)).isoformat(sep=" "),
                             reported[0] if reported else end - start])
    manifest = directory / "source-manifest.json"
    manifest.write_text(json.dumps({"files": {source.name: {
        "bytes": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }}}), encoding="utf-8")
    source_before = artifact_state(source)
    cache = directory / "union.parquet"
    meta = build_cache(source, cache, manifest, temp_dir=directory / "spill")
    assert artifact_state(source) == source_before
    return cache, source, manifest, meta


def context(airings=(), *, users=(USER, "cold"), items=8, lower=MONDAY,
            cutoff=MONDAY + 3600, forecast_end=None, candidates=None, counts=None):
    raw_ids = np.array([9007199254740993, 707, 11, 5, 999, 37, 222, 444][:items]
                       + list(range(10000, 10000 + max(0, items - 8))), dtype=np.int64)
    data = TrainingData(list(users), raw_ids,
                        csr_matrix((len(users), items)) if counts is None else counts,
                        [{"channel_title": "not authoritative", "tv_show_title": "unused"} for _ in range(items)])
    empty = np.array([], dtype=np.int64)
    return TemporalContext(data, np.arange(items) if candidates is None else candidates,
                           lower, cutoff, cutoff,
                           cutoff + 30 * DAY if forecast_end is None else forecast_end,
                           empty, empty, empty, empty, np.array([], dtype=str), (), tuple(airings))


def endpoint_union(intervals):
    """Independent endpoint-depth sweep, not the builder's running-max islands."""
    changes = defaultdict(int)
    for start, end in intervals:
        changes[start] += 1
        changes[end] -= 1
    output, depth, previous = [], 0, None
    for point in sorted(changes):
        if previous is not None and previous < point and depth:
            if output and output[-1][1] == previous:
                output[-1] = (output[-1][0], point)
            else:
                output.append((previous, point))
        depth += changes[point]
        previous = point
    assert depth == 0
    return tuple(output)


def independently_clipped_union(rows, lower, cutoff):
    grouped = defaultdict(list)
    for user, channel, start, end, *_ in rows:
        clipped = max(start, lower), min(end, cutoff)
        if clipped[0] < clipped[1]:
            grouped[user, channel].append(clipped)
    return {key: endpoint_union(values) for key, values in grouped.items()}


def second_oracle(rows, ctx, *, slot_minutes, mode, half_life, power):
    """Small-window literal coverage sets; calendar slots use datetime.weekday."""
    assert ctx.train_end - ctx.history_start <= 600
    watched = defaultdict(set)
    for user, channel, start, end, *_ in rows:
        if user == ctx.data.user_ids[0]:
            watched[channel].update(range(max(start, ctx.history_start), min(end, ctx.train_end)))

    def slot_weight(second):
        dt = EPOCH + timedelta(seconds=second)
        minute = dt.hour * 60 + dt.minute
        local_slot = minute // slot_minutes
        key = (dt.weekday() if mode == "week" else 0, local_slot)
        midnight = dt.replace(hour=0, minute=0, second=0)
        midpoint = midnight + timedelta(minutes=(local_slot + 0.5) * slot_minutes)
        age = max(0, (EPOCH + timedelta(seconds=ctx.train_end) - midpoint).total_seconds())
        return key, 1.0 if half_life is None else math.exp2(-age / (DAY * half_life))

    denominator, numerator = defaultdict(float), defaultdict(float)
    for second in range(ctx.history_start, ctx.train_end):
        slot, weight = slot_weight(second)
        denominator[slot] += weight
        for channel, covered in watched.items():
            if second in covered:
                numerator[channel, slot] += weight
    result = np.zeros(len(ctx.data.item_ids), dtype=np.float64)
    for airing in ctx.forecast_schedules:
        assert not airing.quarantined  # These particular fixtures contain no duplicates/flags.
        total = 0.0
        for second in range(airing.start, airing.end):
            slot, _ = slot_weight(second)
            if denominator[slot]:
                total += numerator[airing.channel_id, slot] / denominator[slot]
        result[airing.item_index] += (total / (airing.end - airing.start)) ** power
    result[result == 0] = np.nan
    return result


def test_running_max_union_matches_independent_endpoint_sweep_and_duration_is_audit_only(tmp_path):
    start = MONDAY
    rows = [
        (USER, "006", start, start + 180, 999),
        (USER, "006", start + 5, start + 8),
        (USER, "006", start + 20, start + 22),
        (USER, "006", start + 90, start + 95),
        (USER, "006", start, start + 180),
        (USER, "006", start + 180, start + 240),
        (USER, "006", start + 241, start + 250, "NaN"),
        (USER, "6", start, start + 30),
        (OTHER_WIDE, "006", start, start + 60),
        ("raw-only-not-a-context-user", "006", start, start + 17),
    ]
    cache, _, _, meta = fixture_cache(tmp_path, rows)
    grouped = defaultdict(list)
    for user, channel, s, e, *_ in rows:
        grouped[user, channel].append((s, e))
    expected = sorted((u, c, s, e) for (u, c), intervals in grouped.items()
                      for s, e in endpoint_union(intervals))
    actual = [(r["user_id"], r["channel_id"], r["start"], r["end"])
              for r in pq.read_table(cache).to_pylist()]
    assert actual == expected
    assert (USER, "006", start, start + 240) in actual
    assert meta["audit"]["union_seconds"] == sum(e - s for _, _, s, e in expected)
    assert meta["audit"]["duration_mismatch_rows"] == 1
    assert meta["audit"]["duration_invalid_rows"] == 1
    assert meta["audit"]["users"] == 3  # Builder must not restrict to a template cohort.


def test_uint64_neighbor_ids_leading_zero_scientific_and_nullish_ids_stay_distinct(tmp_path):
    durations = {USER: 3600, OTHER_WIDE: 1800, "0009": 900, "9": 450,
                 "1e3": 300, "1000": 150, "NA": 60, "null": 30}
    rows = [(u, "006", MONDAY, MONDAY + d) for u, d in durations.items()]
    rows += [(USER, "6", MONDAY, MONDAY + 60),
             ("raw-only-not-a-context-user", "006", MONDAY, MONDAY + 120)]
    cache, _, _, meta = fixture_cache(tmp_path, rows)
    future = MONDAY + 7 * DAY
    users = tuple(durations) + ("cold",)
    ctx = context([Airing(0, "006", future, future + 60), Airing(1, "6", future, future + 60)], users=users)
    fitted = RawScheduleModel(cache, half_life_days=None, slot_minutes=60).fit(ctx)
    assert fitted.user_ids_ == users
    for user_index, duration in enumerate(durations.values()):
        assert fitted.score(user_index, [0])[0] == pytest.approx(duration / 3600, abs=1e-15)
    assert fitted.score(0, [1])[0] == pytest.approx(1 / 60)
    assert np.isnan(fitted.score(1, [1])[0])
    assert np.isnan(fitted.score(len(users) - 1, [0, 1])).all()
    values = pq.read_table(cache, columns=["user_id"]).column(0).to_pylist()
    assert "raw-only-not-a-context-user" in values and "cold" not in values
    assert meta["audit"]["users"] == 9


@pytest.mark.parametrize("mode", ["week", "day"])
def test_union_clip_commutation_and_future_mutation_are_exact_before_feature_estimation(tmp_path, mode):
    lower = seconds("2020-03-31 23:59:30")
    cutoff = lower + 120
    base = [(USER, "c", lower - 40, lower + 20), (USER, "c", lower + 10, lower + 40),
            (USER, "c", lower + 60, cutoff + 10), (USER, "c", lower + 70, lower + 75),
            (USER, "old-only-channel", lower - 100, lower),
            ("not-in-context", "c", lower, cutoff)]
    variants = [base,
                [(u, c, s, cutoff + 300 * DAY if e == cutoff + 10 else e) for u, c, s, e in base]
                + [(USER, "c", cutoff, cutoff + 400 * DAY),
                   (USER, "future-only-channel", cutoff, cutoff + DAY),
                   ("cold", "future-only-channel", cutoff, cutoff + DAY)],
                base + [(USER, "c", lower - 100 * DAY, lower),
                        (USER, "old-only-channel", lower - 1000, lower)]]
    expected_union = independently_clipped_union(base, lower, cutoff)
    future = lower + 7 * DAY
    ctx = context([Airing(0, "c", future, future + 120)], lower=lower, cutoff=cutoff)
    models, hashes = [], []
    for index, rows in enumerate(variants):
        assert independently_clipped_union(rows, lower, cutoff) == expected_union
        cache, _, _, meta = fixture_cache(tmp_path, rows, name=f"variant-{index}")
        cached_rows = [(r["user_id"], r["channel_id"], r["start"], r["end"])
                       for r in pq.read_table(cache).to_pylist()]
        assert independently_clipped_union(cached_rows, lower, cutoff) == expected_union
        fitted = RawScheduleModel(cache, half_life_days=0.0005, slot_minutes=1,
                                  slot_mode=mode).fit(ctx)
        expected = second_oracle(base, ctx, slot_minutes=1, mode=mode, half_life=0.0005, power=1)
        assert_allclose(fitted.score(0, np.arange(8)), expected, rtol=2e-13, atol=2e-13)
        assert fitted.channel_ids_ == ("c",)
        assert fitted.diagnostics["minimum_behavior_start"] == lower
        assert fitted.diagnostics["maximum_behavior_end"] == cutoff
        assert fitted.diagnostics["clipped_union_seconds"] == 100
        models.append(fitted)
        hashes.append(meta["source"]["sha256"])
    assert len(set(hashes)) == 3
    for fitted in models[1:]:
        assert_array_equal(fitted.occupancy_.data, models[0].occupancy_.data)
        assert_array_equal(fitted.occupancy_.indices, models[0].occupancy_.indices)
        assert_array_equal(fitted.occupancy_.indptr, models[0].occupancy_.indptr)
        assert_array_equal(fitted.observable_seconds_, models[0].observable_seconds_)
        assert_array_equal(fitted.score(0, [0, 0, 1]), models[0].score(0, [0, 0, 1]))
        assert fitted.diagnostics["training_feature_sha256"] == models[0].diagnostics["training_feature_sha256"]


@pytest.mark.parametrize("edge", ["2020-02-29 23:59:30", "2020-03-31 23:59:30",
                                 "2020-12-31 23:59:30", "2020-04-05 23:59:30"])
@pytest.mark.parametrize("mode,power", [("week", 1), ("week", 2), ("day", 0.5)])
def test_month_year_leap_day_and_sunday_crossings_use_real_source_calendar(tmp_path, edge, mode, power):
    lower = seconds(edge)
    cutoff = lower + 180
    rows = [(USER, "c", lower - 20, lower + 60), (USER, "c", lower + 40, lower + 90),
            (USER, "c", lower + 120, cutoff + 20),
            (USER, "d", lower + 20, lower + 150)]
    cache, *_ = fixture_cache(tmp_path, rows)
    future = lower + 7 * DAY
    ctx = context([Airing(0, "c", future, future + 180), Airing(1, "d", future + 10, future + 160),
                   Airing(2, "unknown", future, future + 10)], lower=lower, cutoff=cutoff)
    fitted = RawScheduleModel(cache, half_life_days=0.002, slot_minutes=1,
                              slot_mode=mode, coverage_power=power).fit(ctx)
    expected = second_oracle(rows, ctx, slot_minutes=1, mode=mode, half_life=0.002, power=power)
    assert_allclose(fitted.score(0, np.arange(8)), expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("lookback", [None, 10.0])
def test_partial_repeated_week_bins_match_numerator_and_denominator_midpoint_recency(tmp_path, lookback):
    lower = seconds("2020-03-31 00:50:00")
    cutoff = seconds("2020-04-14 00:20:00")
    first_midnight = seconds("2020-03-31 00:00:00")
    middle_midnight = seconds("2020-04-07 00:00:00")
    last_midnight = seconds("2020-04-14 00:00:00")
    rows = [(USER, "c", lower - 20, lower + 300),
            (USER, "c", middle_midnight, middle_midnight + 900),
            (USER, "c", last_midnight, last_midnight + 600)]
    cache, *_ = fixture_cache(tmp_path, rows)
    future = seconds("2020-04-21 00:00:00")
    ctx = context([Airing(0, "c", future, future + 2400)], lower=lower, cutoff=cutoff)
    fitted = RawScheduleModel(cache, half_life_days=7, lookback_days=lookback, slot_minutes=60).fit(ctx)
    w_old = math.exp2(-(cutoff - first_midnight - 1800) / (7 * DAY))
    w_middle = math.exp2(-(cutoff - middle_midnight - 1800) / (7 * DAY))
    numerator = 900 * w_middle + 600
    denominator = 3600 * w_middle + 1200
    if lookback is None:
        numerator += 300 * w_old
        denominator += 600 * w_old
    expected = numerator / denominator
    assert fitted.score(0, [0])[0] == pytest.approx(expected, abs=2e-15)
    # Tuesday's 00:00 hour is index 24; the partial current bin midpoint lies
    # AFTER cutoff, so the documented age floor makes its common weight exactly 1.
    assert fitted.observable_seconds_[24] == pytest.approx(denominator, abs=2e-12)
    if lookback is not None:
        assert fitted.diagnostics["history_window"]["effective_start"] == cutoff - 10 * DAY


@pytest.mark.parametrize("power", [1.0, 2.0, 0.5])
def test_full_airing_duration_normalization_precedes_power_and_program_sum(tmp_path, power):
    rows = [(USER, "c", MONDAY, MONDAY + 1800), (USER, "c", MONDAY + 1800, MONDAY + 2400)]
    cache, *_ = fixture_cache(tmp_path, rows)
    future = MONDAY + 7 * DAY
    airings = [Airing(0, "c", future, future + 3600), Airing(1, "c", future, future + 600),
               Airing(2, "c", future, future + 900), Airing(2, "c", future + 1800, future + 2700),
               Airing(3, "c", future, future + 2700),
               Airing(4, "c", future + 3600, future + 3660), Airing(5, "unknown", future, future + 60)]
    counts = csr_matrix(([17.0], ([0], [0])), shape=(2, 8))
    ctx = context(airings, candidates=np.array([7]), counts=counts)
    fitted = RawScheduleModel(cache, half_life_days=None, coverage_power=power, score_block_size=1).fit(ctx)
    # Occupancy [1,1/3]. Whole-hour mean=2/3; 45-minute mean=7/9.
    expected = [(2 / 3) ** power, 1, 1 + (1 / 3) ** power, (7 / 9) ** power, np.nan, np.nan]
    assert_allclose(fitted.score(0, np.arange(6)), expected, rtol=0, atol=2e-15)
    assert np.isfinite(fitted.score(0, [0, 2])).all()  # Seen and brand-new programs both eligible.
    assert ctx.data.interactions[0, 0] == 17


@pytest.mark.parametrize("mode", ["week", "day"])
@pytest.mark.parametrize("power", [1.0, 2.0])
def test_one_airing_can_cross_and_repeat_a_whole_calendar_period(tmp_path, mode, power):
    cache, *_ = fixture_cache(tmp_path, [(USER, "c", MONDAY, MONDAY + 3600)])
    future = MONDAY + 14 * DAY
    airings = [Airing(0, "c", future, future + 8 * DAY),
               Airing(1, "c", future, future + 7 * DAY),
               Airing(2, "c", future, future + 3600),
               Airing(2, "c", future + 7 * DAY, future + 7 * DAY + 3600)]
    ctx = context(airings, cutoff=MONDAY + 7 * DAY)
    fitted = RawScheduleModel(cache, half_life_days=None, slot_mode=mode, coverage_power=power).fit(ctx)
    # Weekly: Monday 00:00 occupancy=1; an eight-day airing includes TWO such
    # hours, so its mean is 2/(8*24)=1/96. Daily: occupancy at 00:00 is 1/7;
    # eight pooled hours in eight days give 1/168. A full week gives 1/168
    # under either policy. Two separate hour-long airings must sum AFTER power.
    long_mean = 1 / 96 if mode == "week" else 1 / 168
    hour_mean = 1 if mode == "week" else 1 / 7
    expected = [long_mean ** power, (1 / 168) ** power, 2 * hour_mean ** power]
    assert_allclose(fitted.score(0, [0, 1, 2]), expected, rtol=2e-14, atol=1e-16)


def test_completion_arrays_counts_and_descriptions_cannot_inject_raw_evidence(tmp_path):
    cache, *_ = fixture_cache(tmp_path, [(USER, "c", MONDAY, MONDAY + 1800)])
    future = MONDAY + 7 * DAY
    ctx = context([Airing(0, "c", future, future + 1800)])
    first = RawScheduleModel(cache, half_life_days=None).fit(ctx)
    changed = context(ctx.forecast_schedules, counts=csr_matrix(np.full((2, 8), 700.0)))
    changed.data.metadata[:] = [{"channel_title": "future supplied prose", "director": "changed"}] * 8
    changed = replace(changed, event_users=np.array([1]), event_items=np.array([7]),
                      event_starts=np.array([MONDAY]), event_ends=np.array([MONDAY + 60]),
                      event_channels=np.array(["not-raw"]))
    before = deepcopy(changed.data.metadata)
    second = RawScheduleModel(cache, half_life_days=None).fit(changed)
    assert_array_equal(first.occupancy_.data, second.occupancy_.data)
    assert_array_equal(first.score(0, [0, 1]), second.score(0, [0, 1]))
    assert first.diagnostics["training_feature_sha256"] == second.diagnostics["training_feature_sha256"]
    assert np.isnan(second.score(1, [0, 7])).all()
    assert changed.data.metadata == before


@pytest.mark.parametrize("power", [1.0, 2.0])
def test_nan_support_candidate_bounds_order_duplicates_and_uint64_indices(tmp_path, power):
    cache, *_ = fixture_cache(tmp_path, [(USER, "c", MONDAY, MONDAY + 900),
                                       (USER, "future-only", MONDAY + 3600, MONDAY + DAY)])
    future = MONDAY + 7 * DAY
    ctx = context([Airing(0, "c", future, future + 1800), Airing(1, "future-only", future, future + 1800)])
    fitted = RawScheduleModel(cache, half_life_days=None, coverage_power=power).fit(ctx)
    indices = np.array([1, 0, 0, 7, -1, 8, 2])
    expected = [np.nan, 0.5 ** power, 0.5 ** power, np.nan, np.nan, np.nan, np.nan]
    snapshot = indices.copy()
    assert_allclose(fitted.score(0, indices), expected, rtol=0, atol=0)
    assert_allclose(fitted.score(0, indices[::-1]), expected[::-1], rtol=0, atol=0)
    assert_array_equal(indices, snapshot)
    assert np.isnan(fitted.score(0, np.array([2**64 - 1], dtype=np.uint64))).all()
    assert np.isnan(fitted.score(0, ctx.data.item_ids[:3])).all()
    for user in [-1, 1, 2, 1000, np.uint64(2**64 - 1)]:
        assert np.isnan(fitted.score(user, indices)).all()
    assert fitted.score(0, []).shape == (0,)
    with pytest.raises(ValueError):
        fitted.score(True, indices)
    with pytest.raises(ValueError):
        fitted.score(0, [0.0])


@pytest.mark.parametrize("power", [1.0, 2.0])
def test_cpu_sparse_fit_and_block_bounded_scores_with_no_score_io(tmp_path, monkeypatch, power):
    configurations = []
    original_connect = raw.duckdb.connect

    def audited_connect(*args, **kwargs):
        configurations.append(kwargs["config"].copy())
        assert kwargs["config"]["threads"] == 4
        assert kwargs["config"]["memory_limit"] == "4GB"
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(raw.duckdb, "connect", audited_connect)
    cache, *_ = fixture_cache(tmp_path, [(USER, "c", MONDAY, MONDAY + 900)])
    future = MONDAY + 7 * DAY
    airings = [Airing(i, "c", future + week * 7 * DAY, future + week * 7 * DAY + 600)
               for i in range(73) for week in range(3)]
    ctx = context(airings, users=(USER,) + tuple(f"unobserved-{i}" for i in range(1023)), items=73)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected network, score I/O, or dense training matrix")

    original_dense = csr_matrix.toarray
    with monkeypatch.context() as patch:
        patch.setattr(socket, "socket", forbidden)
        patch.setattr(csr_matrix, "toarray", forbidden)
        fitted = RawScheduleModel(cache, half_life_days=None, coverage_power=power, score_block_size=7).fit(ctx)
    assert issparse(fitted.occupancy_) and issparse(fitted.forecast_features_)
    assert fitted.occupancy_.shape == (1024, 336) and fitted.occupancy_.nnz == 1
    shapes = []

    def guarded_dense(matrix, *args, **kwargs):
        shapes.append(matrix.shape)
        assert matrix.shape[0] == 1 and matrix.shape[1] <= 7
        return original_dense(matrix, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(socket, "socket", forbidden)
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(csr_matrix, "toarray", guarded_dense)
        result = fitted.score(0, np.r_[np.arange(73), 0, 72])
        assert_allclose(result, 3 * 0.5 ** power, rtol=0, atol=1e-15)
        assert np.isnan(fitted.score(1023, np.arange(73))).all()
    assert shapes and configurations


def test_existing_fixture_cache_is_reused_without_mutation_and_records_source_code_hash(tmp_path):
    cache, source, manifest, meta = fixture_cache(tmp_path, [(USER, "c", MONDAY, MONDAY + 60)])
    sidecar = cache.with_suffix(".parquet.meta.json")
    before = {p: artifact_state(p) for p in (cache, source, manifest, sidecar)}
    reused = build_cache(source, cache, manifest)
    assert reused["reused"]
    assert {p: artifact_state(p) for p in before} == before
    assert meta["construction"]["implementation_sha256"] == hashlib.sha256(Path(raw.__file__).read_bytes()).hexdigest()


@pytest.mark.parametrize("target", ["source_stat", "manifest", "cache_same_stat", "different_source_binding"])
def test_synthetic_integrity_negative_controls_reject_changed_bound_artifacts(tmp_path, target):
    cache, source, manifest, meta = fixture_cache(tmp_path, [(USER, "c", MONDAY, MONDAY + 60)])
    if target == "source_stat":
        source.write_bytes(source.read_bytes() + b"\n")
    elif target == "manifest":
        manifest.write_bytes(manifest.read_bytes() + b"\n")
    elif target == "cache_same_stat":
        stat = cache.stat()
        data = bytearray(cache.read_bytes())
        data[len(data) // 2] ^= 1
        cache.write_bytes(data)
        os.utime(cache, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert cache.stat().st_size == meta["cache"]["guard"]["bytes"]
        assert cache.stat().st_mtime_ns == meta["cache"]["guard"]["mtime_ns"]
    else:
        other_cache, other_source, other_manifest, _ = fixture_cache(
            tmp_path, [(OTHER_WIDE, "c", MONDAY, MONDAY + 60)], name="different")
        assert other_cache.exists()
        with pytest.raises(ValueError, match="different source/manifest"):
            build_cache(other_source, cache, other_manifest)
        return
    with pytest.raises(ValueError):
        RawScheduleModel(cache, half_life_days=None).fit(context())


def test_build_rejects_a_wrong_source_manifest_sha_without_publishing_fixture_cache(tmp_path):
    _, source, manifest, _ = fixture_cache(tmp_path, [(USER, "c", MONDAY, MONDAY + 60)])
    value = json.loads(manifest.read_text())
    value["files"][source.name]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(value), encoding="utf-8")
    before = source.read_bytes()
    new_cache = tmp_path / "must-not-publish.parquet"
    with pytest.raises(ValueError, match="SHA-256"):
        build_cache(source, new_cache, manifest, temp_dir=tmp_path / "failed-build-spill")
    assert source.read_bytes() == before
    assert not new_cache.exists() and not new_cache.with_suffix(".parquet.meta.json").exists()
