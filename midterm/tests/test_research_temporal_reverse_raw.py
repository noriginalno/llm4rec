"""Raw-habits author's independent, frozen-source temporal-model challenge.

Only synthetic contexts, scalar arithmetic and numerical quadrature are used.
No author fixtures, production feature/weight helpers, cache, labels, evaluator,
prediction files or parameter runner are imported as an oracle.
"""

from __future__ import annotations

import builtins
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import math
from pathlib import Path
import socket

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import coo_matrix, csr_matrix, issparse

from research_v2.contracts import Airing, TemporalContext
from research_v2.temporal import ScheduleAdjustedRepeatModel, TemporalRepeatModel
from sweettv.contracts import TrainingData


DAY = 86400
T = 1585526400  # 2020-03-30 00:00:00, the common naive source-clock anchor.
ROOT = Path(__file__).resolve().parents[1]
USERS = ("cold-first", "18446744073709551615", "00007", "cold-last")
ITEMS = (501, 9007199254740993, 77, 55, 99, 8888)


@pytest.fixture(scope="module", autouse=True)
def frozen_source_hash_guard():
    paths = [ROOT / relative for relative in (
        "research_v2/temporal.py", "research_v2/contracts.py",
        "research_v2/raw_habits.py", "sweettv/contracts.py",
    )]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    yield
    assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths} == before


def past(item, age_seconds=0, *, channel="c", duration=60):
    end = T - age_seconds
    return Airing(item, channel, end - duration, end)


def future(item, number=0, *, channel="c", gap_days=0, duration=60, flagged=False):
    start = T + gap_days * DAY + 120 * number
    return Airing(item, channel, start, start + duration, flagged)


def toy(events=(), historical=(), forecast=(), *, history_days=4, horizon_days=2,
        gap_days=0, candidates=(0, 1, 2, 3, 4, 5), users=USERS, item_ids=ITEMS):
    events = tuple(events)
    u = np.array([user for user, _ in events], dtype=np.int64)
    i = np.array([airing.item_index for _, airing in events], dtype=np.int64)
    data = TrainingData(
        list(users), np.array(item_ids, dtype=np.int64),
        coo_matrix((np.ones(len(events)), (u, i)), shape=(len(users), len(item_ids))).tocsr(),
        [{"tv_show_title": f"metadata {raw_id}"} for raw_id in item_ids],
    )
    return TemporalContext(
        data=data, candidate_indices=np.array(candidates, dtype=np.int64),
        history_start=T - int(history_days * DAY), train_end=T,
        forecast_start=T + gap_days * DAY,
        forecast_end=T + gap_days * DAY + int(horizon_days * DAY),
        event_users=u, event_items=i,
        event_starts=np.array([a.start for _, a in events], dtype=np.int64),
        event_ends=np.array([a.end for _, a in events], dtype=np.int64),
        event_channels=np.array([a.channel_id for _, a in events], dtype=str),
        historical_schedules=tuple(historical), forecast_schedules=tuple(forecast),
    )


def integral_by_simpson(half_life_days, duration_days):
    """Integrate exp2(-age/h) directly; not the implementation's antiderivative."""
    n = 2048
    step = duration_days / n
    total = 1 + math.exp2(-duration_days / half_life_days)
    for j in range(1, n):
        total += (4 if j % 2 else 2) * math.exp2(-(j * step) / half_life_days)
    return total * step / 3


@pytest.mark.parametrize("horizon_days,expected_ratio", [(1, 2.0), (2, 5 / 3), (4, 5 / 4)])
def test_prior_units_are_forecast_airings_and_history_is_projected_to_actual_horizon(horizon_days, expected_ratio):
    # H=2, D=4 days, F=3, s=2 airings. B=H*Q/D=Q/2.
    h0, h1 = past(0), past(0, DAY)
    ctx = toy([(1, h0)], [h0, h1], [future(0, j) for j in range(3)], horizon_days=horizon_days)
    fitted = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1,
                                         smoothing_airings=2, ratio_cap=100).fit(ctx)
    assert fitted.historical_counts_[0] == 2 and fitted.forecast_counts_[0] == 3
    assert fitted.availability_ratios_[0] == pytest.approx(expected_ratio, abs=2e-15)
    assert fitted.score(1, [0])[0] == pytest.approx(expected_ratio, abs=2e-15)
    assert fitted.diagnostics["availability_ratios"]["forecast_days"] == horizon_days
    assert fitted.diagnostics["availability_ratios"]["smoothing_units"] == "scheduled airings per forecast horizon"


def test_gap_after_training_cutoff_does_not_inflate_forecast_horizon():
    h = past(0)
    outputs = []
    for gap in (0, 9):
        ctx = toy([(1, h)], [h], [future(0, gap_days=gap)],
                  history_days=4, horizon_days=2, gap_days=gap)
        fitted = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1,
                                             smoothing_airings=1).fit(ctx)
        outputs.append(fitted.score(1, [0])[0])
    # Q=2, not 11 days: B=1/2, ratio=2/(1+1/2)=4/3.
    assert_allclose(outputs, [4 / 3, 4 / 3], rtol=0, atol=2e-15)


def test_rare_and_frequent_stationary_schedule_intensities_are_both_neutral():
    historical = [past(0, j * 3600) for j in range(5)]
    historical += [past(1, j * 3600) for j in range(20)]
    forecast = [future(0)] + [future(1, j) for j in range(4)]
    events = [(1, historical[0]), (1, historical[1]),
              (1, historical[5]), (1, historical[6]), (1, historical[7])]
    ctx = toy(events, historical, forecast, history_days=10, horizon_days=2)
    fitted = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1,
                                         smoothing_airings=3).fit(ctx)
    # lambda0=.5/day, lambda1=2/day; H*Q/D equals F=1/4 respectively.
    assert_allclose(fitted.availability_ratios_[:2], [1, 1], rtol=0, atol=1e-15)
    assert_allclose(fitted.score(1, [1, 0, 1]), [3, 2, 3], rtol=0, atol=2e-15)


@pytest.mark.parametrize("history_days,lookback,expected_span", [(6, None, 6), (6, 3, 3), (3, 6, 3)])
def test_effective_history_kernel_is_independently_integrated_in_days(history_days, lookback, expected_span):
    h0, h1 = past(0), past(0, DAY)
    ctx = toy([(1, h0), (1, h1)], [h0, h1], [future(0, 0), future(0, 1)],
              history_days=history_days, horizon_days=2)
    fitted = ScheduleAdjustedRepeatModel(half_life_days=3, lookback_days=lookback,
                                         schedule_power=0.5, smoothing_airings=2).fit(ctx)
    effective = integral_by_simpson(3, expected_span)
    weighted_count = 1 + math.exp2(-1 / 3)
    projected = weighted_count * 2 / effective
    expected = weighted_count * math.sqrt((2 + 2) / (projected + 2))
    assert fitted.diagnostics["availability_ratios"]["effective_weighted_history_days"] == pytest.approx(effective, rel=2e-13)
    assert fitted.diagnostics["availability_ratios"]["history_window_days"] == expected_span
    assert fitted.repeat_counts_[1, 0] == pytest.approx(weighted_count, abs=1e-15)
    assert fitted.historical_counts_[0] == pytest.approx(weighted_count, abs=1e-15)
    assert fitted.score(1, [0])[0] == pytest.approx(expected, rel=2e-13)


def test_window_matched_end_weights_include_lookback_equality_and_deduplicate_only_schedule_rows():
    newest = past(0)
    inclusive = past(0, 2 * DAY, duration=3 * DAY)  # start outside history; end is authoritative
    expired = past(0, 2 * DAY + 1)
    boundary = past(0, 4 * DAY)
    older = past(0, 5 * DAY)
    events = [(1, a) for a in (newest, inclusive, expired, boundary, older)]
    historical = [newest, inclusive, inclusive, expired, boundary, older]
    ctx = toy(events, historical, [future(0), future(0, 1), future(0, 1)], history_days=4, horizon_days=3)
    fitted = ScheduleAdjustedRepeatModel(half_life_days=2, lookback_days=2,
                                         schedule_power=1, smoothing_airings=1).fit(ctx)
    # Retain end ages 0 and exactly 2 days -> R=H=1+.5, F=2.
    # E = integral_0^2 2^(-a/2) da = 1/ln2 days; B = 4.5 ln2.
    expected = 1.5 * 3 / (4.5 * math.log(2) + 1)
    assert fitted.repeat_counts_[1, 0] == fitted.historical_counts_[0] == 1.5
    assert fitted.forecast_counts_[0] == 2
    assert fitted.score(1, [0])[0] == pytest.approx(expected, abs=2e-15)
    assert ctx.data.interactions[1, 0] == 5  # frozen original counts, not our feature window
    assert len(ctx.event_ends) == 5 and len(ctx.historical_schedules) == 6


def test_full_airing_identity_retains_same_program_same_time_on_distinct_channels():
    c, d = past(0, channel="c"), past(0, channel="d")
    fc, fd = future(0, channel="c"), future(0, channel="d")
    ctx = toy([(1, c)], [c, d, c, d], [fc, fd, fc, fd], history_days=4, horizon_days=2)
    fitted = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1,
                                         smoothing_airings=1).fit(ctx)
    # H=2 distinct channel-airings; B=2*2/4=1; F=2; ratio=(2+1)/(1+1).
    assert fitted.historical_counts_[0] == fitted.forecast_counts_[0] == 2
    assert fitted.score(1, [0])[0] == pytest.approx(1.5, abs=2e-15)


def test_t1_strict_history_boundary_fix_is_independently_reconfirmed_without_count_mutation():
    old, boundary, recent = past(0, 3 * DAY), past(0, 2 * DAY), past(0, DAY)
    ctx = toy([(1, a) for a in (old, boundary, recent)], [old, boundary, recent],
              [future(0)], history_days=2, horizon_days=1)
    original = {field: getattr(ctx.data.interactions, field).copy() for field in ("data", "indices", "indptr")}
    fitted = ScheduleAdjustedRepeatModel(half_life_days=None, lookback_days=2,
                                         schedule_power=1, smoothing_airings=1).fit(ctx)
    assert fitted.repeat_counts_[1, 0] == fitted.historical_counts_[0] == 1
    assert fitted.score(1, [0])[0] == pytest.approx(4 / 3, abs=2e-15)
    assert fitted.diagnostics["events_before_history_start_excluded"] == 2
    assert ctx.data.interactions[1, 0] == 3
    for field, values in original.items():
        assert_array_equal(getattr(ctx.data.interactions, field), values)


def test_source_wall_clock_midnight_and_one_second_lookback_use_86400_seconds_not_timezone():
    origin = datetime(1970, 1, 1)
    naive_seconds = lambda text: (datetime.fromisoformat(text) - origin) // timedelta(seconds=1)
    # The wall-clock interval spanning the 2020-03-29 DST date is still 2*86400
    # on this expressly naive anchor; the test makes no physical-time claim.
    assert naive_seconds("2020-03-30 00:00:00") == T
    assert T - naive_seconds("2020-03-28 00:00:00") == 2 * DAY
    newest, inclusive, expired = past(0), past(0, DAY), past(0, DAY + 1)
    ctx = toy([(1, a) for a in (newest, inclusive, expired)], [newest, inclusive, expired],
              [future(0)], history_days=2, horizon_days=1)
    fitted = TemporalRepeatModel(half_life_days=1, lookback_days=1).fit(ctx)
    assert fitted.score(1, [0])[0] == 1.5  # 1 + exactly one half-life, not 24h+1s


@pytest.mark.parametrize("power", [0, 0.5, 1])
def test_ratio_cap_bounds_both_tails_with_exact_zero_power_ablation(power):
    historical = [past(0, j * 3600) for j in range(50)] + [past(1)]
    events = [(1, historical[0]), (1, historical[1]), (1, historical[-1])]
    forecast = [future(0)] + [future(1, j) for j in range(50)]
    ctx = toy(events, historical, forecast, history_days=4, horizon_days=8)
    fitted = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=power,
                                         smoothing_airings=1, ratio_cap=3).fit(ctx)
    # Before clipping: ratios 2/101 and 51/3; after clipping: 1/3 and 3.
    assert_allclose(fitted.availability_ratios_[:2], [1 / 3, 3], rtol=0, atol=1e-15)
    assert_allclose(fitted.score(1, [0, 1]), [2 * (1 / 3) ** power, 3 ** power], rtol=0, atol=2e-15)


@pytest.mark.parametrize("half_life,lookback", [(None, None), (2, None), (2, 1)])
def test_zero_history_zero_forecast_and_power_zero_have_exact_supported_nan_locations(half_life, lookback):
    observed, missing_forecast = past(0), past(1, DAY)
    ctx = toy([(1, observed), (1, missing_forecast)], [observed, missing_forecast],
              [future(0), future(2)], candidates=(1, 0, 2, 1, 3))
    repeat = TemporalRepeatModel(half_life_days=half_life, lookback_days=lookback).fit(ctx)
    zero = ScheduleAdjustedRepeatModel(half_life_days=half_life, lookback_days=lookback, schedule_power=0).fit(ctx)
    positive = ScheduleAdjustedRepeatModel(half_life_days=half_life, lookback_days=lookback, schedule_power=1).fit(ctx)
    query = np.array([1, 0, 2, 1, -1, 6, 3, 0])
    for user in (0, 1, 2, 3, -1, 4):
        a, b = repeat.score(user, query), zero.score(user, query)
        assert_array_equal(a, b)
        assert np.array_equal(a[np.isfinite(a)].view(np.uint64), b[np.isfinite(b)].view(np.uint64))
    weighted_old = 1 if half_life is None else math.exp2(-1 / half_life)
    assert_allclose(zero.score(1, query), [weighted_old, 1, np.nan, weighted_old, np.nan, np.nan, np.nan, 1],
                    rtol=0, atol=0, equal_nan=True)
    assert np.isnan(positive.score(1, [1, 2, 3])).all()  # F=0 OR R=0; no prior-created evidence
    assert np.isfinite(positive.score(1, [0])).all()
    assert positive.diagnostics["candidate_coverage"]["unique_candidates"] == 4
    assert positive.diagnostics["candidate_coverage"]["eligible_candidates"] == 2


def test_missing_historical_opportunities_are_prior_regularized_not_invented_completions():
    observed = past(0)
    ctx = toy([(1, observed)], [], [future(0), future(0, 1), future(1)])
    fitted = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=0.5,
                                         smoothing_airings=2).fit(ctx)
    assert fitted.historical_counts_[0] == 0
    assert fitted.score(1, [0])[0] == pytest.approx(math.sqrt(2), abs=1e-15)
    assert np.isnan(fitted.score(1, [1])[0])  # a schedule prior cannot create R_ui


def test_full_containment_counts_airings_not_duration_and_leaves_boundary_only_candidate_in_place():
    historical = [past(i) for i in range(3)]
    contained = Airing(0, "c", T, T + DAY)
    one_second = Airing(1, "c", T + DAY - 1, T + DAY)
    ctx = toy([(1, a) for a in historical], historical, [contained, contained, one_second],
              horizon_days=1, candidates=(2, 1, 0, 2))
    fitted = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1,
                                         smoothing_airings=1).fit(ctx)
    assert_array_equal(fitted.forecast_counts_[:3], [1, 1, 0])
    assert_allclose(fitted.score(1, [2, 1, 0, 2]), [np.nan, 1.6, 1.6, np.nan], rtol=0, atol=2e-15)
    assert_array_equal(ctx.candidate_indices, [2, 1, 0, 2])
    # Partial/out-of-horizon airings and future behavior are a context error,
    # never legal hidden inputs to a model or evidence for changing candidates.
    with pytest.raises(ValueError, match="fully contained"):
        toy(forecast=[Airing(2, "c", T + DAY - 1, T + DAY + 1)], horizon_days=1)
    with pytest.raises(ValueError, match="Future or invalid behavior"):
        toy(events=[(1, Airing(0, "c", T, T + 1))])


def test_quarantined_forecast_metadata_is_intentionally_included_not_a_raw_policy_bug():
    h = past(0)
    scores = []
    for flagged in (False, True):
        ctx = toy([(1, h)], [h], [future(0, flagged=flagged)], history_days=4, horizon_days=2)
        fitted = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1,
                                             smoothing_airings=1).fit(ctx)
        scores.append(fitted.score(1, [0])[0])
        assert fitted.diagnostics["schedule_counts"]["forecast_quarantined_rows"] == int(flagged)
    assert_allclose(scores, [4 / 3, 4 / 3], rtol=0, atol=2e-15)


def test_cold_index_alignment_and_unique_candidate_coverage_denominator_are_not_reindexed():
    h0, h1, h2 = past(0), past(1), past(2)
    ctx = toy([(1, h0), (1, h1), (2, h0)], [h0, h1, h2], [future(1), future(2)],
              history_days=2, horizon_days=1, candidates=(2, 1, 1, 0))
    fitted = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1,
                                         smoothing_airings=1).fit(ctx)
    assert fitted.user_ids_ == USERS
    assert_array_equal(fitted.item_ids_, ITEMS)
    query = np.array([1, 0, 1, 2, -1, 6])
    expected = np.array([4 / 3, np.nan, 4 / 3, np.nan, np.nan, np.nan])
    assert_allclose(fitted.score(1, query), expected, rtol=0, atol=2e-15)
    assert_allclose(fitted.score(1, query[::-1]), expected[::-1], rtol=0, atol=2e-15)
    for user in (0, 2, 3, -1, 4):
        assert np.isnan(fitted.score(user, query)).all()
    assert np.isnan(fitted.score(1, np.array(ITEMS, dtype=np.int64))).all()
    assert np.isnan(fitted.score(1, np.array([2**64 - 1], dtype=np.uint64))).all()
    assert fitted.diagnostics["candidate_coverage"] == {
        "unique_candidates": 3, "eligible_candidates": 2, "supported_candidates": 1,
        "supported_users": 1, "supported_user_candidate_pairs": 1,
        "user_candidate_pair_coverage": 1 / 12,
    }
    assert_array_equal(query, [1, 0, 1, 2, -1, 6])


def test_trimmed_fit_is_sparse_in_memory_and_does_not_mutate_shared_inputs(monkeypatch):
    old, current = past(0, 5 * DAY), past(0)
    ctx = toy([(1, old), (1, current)], [old, current], [future(0)], history_days=2)
    matrix = ctx.data.interactions.copy()
    metadata = deepcopy(ctx.data.metadata)
    arrays = {name: getattr(ctx, name).copy() for name in (
        "candidate_indices", "event_users", "event_items", "event_starts", "event_ends", "event_channels",
    )}

    def forbidden(*args, **kwargs):
        raise AssertionError("Temporal fit must use only in-memory sparse training context")

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", forbidden)
        guard.setattr(Path, "open", forbidden)
        guard.setattr(socket, "socket", forbidden)
        guard.setattr(csr_matrix, "toarray", forbidden)
        fitted = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1).fit(ctx)
    assert issparse(fitted.repeat_counts_)
    assert fitted.repeat_counts_[1, 0] == 1 and ctx.data.interactions[1, 0] == 2
    for name, original in arrays.items():
        assert_array_equal(getattr(ctx, name), original)
        assert not getattr(ctx, name).flags.writeable
    for name in ("data", "indices", "indptr"):
        assert_array_equal(getattr(ctx.data.interactions, name), getattr(matrix, name))
    assert ctx.data.metadata == metadata
