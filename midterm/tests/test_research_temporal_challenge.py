"""Independent closed-loop recheck of T1 and the temporal author's three reviews.

Synthetic inputs and scalar/Fraction oracles only: no other author's fixtures,
production weighting helpers, source logs, labels, evaluator, or grid runner.
The production modules are read-only, including during the parent's experiment.
"""

from __future__ import annotations

import builtins
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import socket

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import coo_matrix, csr_matrix

from research_v2.contracts import Airing, TemporalContext
from research_v2.schedule import ChannelScheduleModel
from research_v2.temporal import ScheduleAdjustedRepeatModel, TemporalRepeatModel
from sweettv.contracts import TrainingData


DAY = 86400
MONDAY = 1583712000
CUTOFF = MONDAY + 14 * DAY
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def read_only_dependency_guard():
    """A source edit by any session invalidates this frozen-snapshot recheck."""
    paths = [ROOT / relative for relative in (
        "research_v2/temporal.py", "research_v2/schedule.py", "research_v2/contracts.py",
        "tests/test_research_temporal.py", "tests/test_research_schedule.py",
    )]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    yield
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert after == before, "A frozen source/dependency changed during independent recheck"


def ending(item, end, channel="c"):
    return Airing(item, channel, end - 60, end)


def past(item, age_days, channel="c"):
    return ending(item, CUTOFF - int(round(age_days * DAY)), channel)


def future(item, number=0, channel="c", *, quarantined=False):
    start = CUTOFF + 60 + 120 * number
    return Airing(item, channel, start, start + 60, quarantined)


def fixture(historical=(), events=(), forecast=(), *, history_start=MONDAY,
            forecast_days=7, candidates=None):
    events = tuple(events)  # (user matrix index, full qualifying airing).
    users = np.array([u for u, _ in events], dtype=np.int64)
    items = np.array([a.item_index for _, a in events], dtype=np.int64)
    matrix = coo_matrix((np.ones(len(events)), (users, items)), shape=(3, 6)).tocsr()
    data = TrainingData(
        user_ids=["0007", "18446744073709551615", "cold"],
        item_ids=np.array([9007199254740993, 707, 902, 37, 999, 55], dtype=np.int64),
        interactions=matrix,
        metadata=[{"tv_show_title": f"permitted forecast descriptor {i}"} for i in range(6)],
    )
    return TemporalContext(
        data=data, candidate_indices=np.arange(6) if candidates is None else candidates,
        history_start=history_start, train_end=CUTOFF,
        forecast_start=CUTOFF, forecast_end=CUTOFF + forecast_days * DAY,
        event_users=users, event_items=items,
        event_starts=np.array([a.start for _, a in events], dtype=np.int64),
        event_ends=np.array([a.end for _, a in events], dtype=np.int64),
        event_channels=np.array([a.channel_id for _, a in events], dtype=str),
        historical_schedules=tuple(historical), forecast_schedules=tuple(forecast),
    )


def snapshot(context):
    arrays = {name: getattr(context, name).copy() for name in (
        "candidate_indices", "event_users", "event_items", "event_starts", "event_ends", "event_channels",
    )}
    return arrays, context.data.interactions.copy(), context.data.item_ids.copy(), deepcopy(context.data.metadata)


def assert_unchanged(context, original):
    arrays, matrix, item_ids, metadata = original
    for name, expected in arrays.items():
        assert_array_equal(getattr(context, name), expected)
        assert not getattr(context, name).flags.writeable
    for name in ("data", "indices", "indptr"):
        assert_array_equal(getattr(context.data.interactions, name), getattr(matrix, name))
    assert_array_equal(context.data.item_ids, item_ids)
    assert context.data.metadata == metadata


def scalar_oracle(context, *, half_life, lookback, smoothing=1.0, cap=10.0, power=1.0):
    """Direct scan with scalar math; never call temporal_weights/matrix or fit."""
    rows, columns = context.data.interactions.shape
    repeats = np.zeros((rows, columns), dtype=np.float64)  # Tiny, test-only oracle.
    history = np.zeros(columns, dtype=np.float64)
    forecast = np.zeros(columns, dtype=np.float64)

    def weight(end):
        if not context.history_start < end <= context.train_end:
            return 0.0
        age = (context.train_end - end) / DAY
        if lookback is not None and age > lookback:
            return 0.0
        return 1.0 if half_life is None else math.exp2(-age / half_life)

    for user, item, end in zip(context.event_users, context.event_items, context.event_ends):
        repeats[int(user), int(item)] += weight(int(end))
    identities = set()
    for airing in context.historical_schedules:
        identity = (airing.item_index, airing.channel_id, airing.start, airing.end)
        if identity not in identities:
            history[airing.item_index] += weight(airing.end)
            identities.add(identity)
    identities = set()
    for airing in context.forecast_schedules:
        identity = (airing.item_index, airing.channel_id, airing.start, airing.end)
        if identity not in identities:
            forecast[airing.item_index] += 1.0  # Temporal expert explicitly includes flags.
            identities.add(identity)

    days = (context.train_end - context.history_start) / DAY
    if lookback is not None:
        days = min(days, lookback)
    effective = (days if half_life is None else
                 half_life / math.log(2) * (1 - math.exp2(-days / half_life)))
    horizon = (context.forecast_end - context.forecast_start) / DAY
    factors = np.array([
        min(cap, max(1 / cap, (f + smoothing) / (h * horizon / effective + smoothing))) ** power
        for h, f in zip(history, forecast)
    ])
    scores = repeats * factors
    scores[repeats == 0] = np.nan
    if power > 0:
        scores[:, forecast == 0] = np.nan
    return repeats, history, forecast, effective, factors, scores


def test_t1_original_shortened_history_oracle_is_now_four_thirds_not_eight_thirds():
    old, recent = past(0, 3), past(0, 1)
    context = fixture([recent], [(0, old), (0, recent)], [future(0)],
                      history_start=CUTOFF - 2 * DAY, forecast_days=1)
    original = snapshot(context)
    model = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1,
                                        smoothing_airings=1).fit(context)
    expected = scalar_oracle(context, half_life=None, lookback=None)
    # R=H=F=1; E=2 days; Q=1 day -> factor=(1+1)/(1/2+1)=4/3.
    assert model.repeat_counts_[0, 0] == model.historical_counts_[0] == model.forecast_counts_[0] == 1
    assert model.diagnostics["availability_ratios"]["effective_weighted_history_days"] == 2
    assert model.score(0, [0])[0] == pytest.approx(4 / 3, abs=1e-15)
    assert abs(model.score(0, [0])[0] - 8 / 3) > 1
    assert_allclose(model.score(0, np.arange(6)), expected[-1][0], rtol=0, atol=1e-15)
    assert model.diagnostics["events_before_history_start_excluded"] == 1
    assert context.data.interactions[0, 0] == 2
    assert_unchanged(context, original)


@pytest.mark.parametrize("lookback", [None, 2.0])
def test_t1_history_start_remains_strict_even_at_inclusive_lookback_boundary(lookback):
    old, boundary, recent = past(0, 3), past(0, 2), past(0, 1)
    context = fixture([old, boundary, recent], [(0, a) for a in (old, boundary, recent)],
                      [future(0)], history_start=CUTOFF - 2 * DAY, forecast_days=1)
    original = snapshot(context)
    model = ScheduleAdjustedRepeatModel(half_life_days=None, lookback_days=lookback,
                                        schedule_power=1, smoothing_airings=1).fit(context)
    expected = scalar_oracle(context, half_life=None, lookback=lookback)
    assert model.repeat_counts_[0, 0] == model.historical_counts_[0] == 1
    assert model.score(0, [0])[0] == pytest.approx(4 / 3, abs=1e-15)
    assert_allclose(model.score(0, np.arange(6)), expected[-1][0], rtol=0, atol=1e-15)
    assert model.diagnostics["events_before_history_start_excluded"] == 2
    assert model.diagnostics["schedule_counts"]["historical_airings_before_history_start_excluded"] == 2
    assert context.data.interactions[0, 0] == 3
    assert_unchanged(context, original)


@pytest.mark.parametrize("half_life,lookback", [(None, 1.0), (2.0, None), (2.0, 1.0)])
def test_t1_decay_and_one_second_lookback_boundaries_match_independent_scan(half_life, lookback):
    ends = [CUTOFF - 5 * DAY, CUTOFF - 4 * DAY, CUTOFF - 2 * DAY,
            CUTOFF - DAY - 1, CUTOFF - DAY, CUTOFF]
    historical = [ending(0, end) for end in ends]
    context = fixture(historical + [historical[-1]], [(0, a) for a in historical],
                      [future(0, 0), future(0, 1), future(1)],
                      history_start=CUTOFF - 4 * DAY, forecast_days=2)
    model = ScheduleAdjustedRepeatModel(half_life_days=half_life, lookback_days=lookback,
                                        schedule_power=1, smoothing_airings=1).fit(context)
    repeats, history, forecast, effective, factors, scores = scalar_oracle(
        context, half_life=half_life, lookback=lookback)
    assert_allclose(model.repeat_counts_.toarray(), repeats, rtol=0, atol=2e-15)
    assert_allclose(model.historical_counts_, history, rtol=0, atol=2e-15)
    assert_array_equal(model.forecast_counts_, forecast)
    assert_allclose(model.schedule_factors_, factors, rtol=0, atol=2e-15)
    assert model.diagnostics["availability_ratios"]["effective_weighted_history_days"] == pytest.approx(effective)
    assert_allclose(model.score(0, np.arange(6)), scores[0], rtol=0, atol=4e-15)
    assert context.data.interactions[0, 0] == 6
    if lookback == 1:
        # Exactly one day stays; one second older is gone on BOTH sides.
        expected_count = 2.0 if half_life is None else 1 + math.sqrt(0.5)
        assert repeats[0, 0] == history[0] == pytest.approx(expected_count)


@pytest.mark.parametrize("age", [2, 3])
def test_t1_all_pre_history_or_exact_boundary_events_stay_cold_without_mutating_counts(age):
    old = past(0, age)
    context = fixture([old], [(0, old)], [future(0)], history_start=CUTOFF - 2 * DAY)
    original = snapshot(context)
    models = [TemporalRepeatModel(half_life_days=None),
              ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=0),
              ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1)]
    for model in models:
        model.fit(context)
        assert model.repeat_counts_.nnz == 0
        assert np.isnan(model.score(0, [0, 0, 1])).all()
        assert model.diagnostics["events_before_history_start_excluded"] == 1
        assert model.diagnostics["active_users"] == 0
        if isinstance(model, ScheduleAdjustedRepeatModel):
            assert model.historical_counts_[0] == 0
    assert context.data.interactions[0, 0] == 1
    assert_unchanged(context, original)


def test_t1_zero_power_preserves_exact_repeat_including_no_forecast_programme_and_duplicates():
    old, recent, no_forecast = past(0, 3), past(0, 1), past(1, 0.5)
    context = fixture([old, recent, no_forecast], [(0, a) for a in (old, recent, no_forecast)],
                      [future(0)], history_start=CUTOFF - 2 * DAY)
    repeat = TemporalRepeatModel(half_life_days=None).fit(context)
    zero = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=0).fit(context)
    positive = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1).fit(context)
    candidates = np.array([1, 0, 1, 4, -1, 0])
    assert_array_equal(repeat.score(0, candidates), [1, 1, 1, np.nan, np.nan, 1])
    for user in (0, 1, 2, -1):
        assert_array_equal(zero.score(user, candidates), repeat.score(user, candidates))
    assert np.isnan(positive.score(0, [1]))[0]
    assert_array_equal(candidates, [1, 0, 1, 4, -1, 0])


def test_t1_refit_with_shortened_start_rebuilds_both_sides_without_touching_context():
    old, recent = past(0, 3), past(0, 1)
    initial = fixture([old, recent], [(0, old), (0, recent)], [future(0)], forecast_days=1)
    shortened = replace(initial, history_start=CUTOFF - 2 * DAY)
    original = snapshot(shortened)
    model = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1, smoothing_airings=1).fit(initial)
    assert model.repeat_counts_[0, 0] == model.historical_counts_[0] == 2
    assert model.fit(shortened) is model
    assert model.repeat_counts_[0, 0] == model.historical_counts_[0] == 1
    assert model.score(0, [0])[0] == pytest.approx(4 / 3, abs=1e-15)
    assert_unchanged(shortened, original)
    assert initial.data is shortened.data
    assert initial.data.interactions[0, 0] == 2


def test_t1_trimmed_fit_requires_no_files_network_or_sparse_densification(monkeypatch):
    old, recent = past(0, 3), past(0, 1)
    context = fixture([old, recent], [(0, old), (0, recent)], [future(0)],
                      history_start=CUTOFF - 2 * DAY, forecast_days=1)
    original = snapshot(context)

    def forbidden(*args, **kwargs):
        raise AssertionError("trimmed fit must remain in-memory and sparse")

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", forbidden)
        guard.setattr(Path, "open", forbidden)
        guard.setattr(socket, "socket", forbidden)
        guard.setattr(csr_matrix, "toarray", forbidden)
        repeat = TemporalRepeatModel(half_life_days=None).fit(context)
        adjusted = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1,
                                               smoothing_airings=1).fit(context)
    assert repeat.score(0, [0])[0] == 1
    assert adjusted.score(0, [0])[0] == pytest.approx(4 / 3, abs=1e-15)
    assert_unchanged(context, original)


def test_response_r1_cold_channel_transfer_order_duplicates_and_absent_evidence_reproduced():
    c0, d1, c2 = past(0, 1, "c"), past(1, 2, "d"), past(2, 3, "c")
    historical = [c0, d1, c2]
    forecast = [future(3, 0, "c"), future(3, 1, "c"), future(4, 0, "d"), future(5, 0, "unseen")]
    context = fixture(historical, [(0, c0), (0, d1)], forecast)
    original = snapshot(context)
    candidates = np.array([5, 4, 3, 4, -1, 6])
    # Independent counts: channel c is 1/(2+2), channel d is 1/(1+2).
    c_rate, d_rate = Fraction(1, 4), Fraction(1, 3)
    expected = np.array([np.nan, float(d_rate), float(2 * c_rate), float(d_rate), np.nan, np.nan])
    model = ChannelScheduleModel(half_life_days=None, slot_mode="channel",
                                 prior_strength=2, programme_blend=0).fit(context)
    assert_allclose(model.score(0, candidates), expected, rtol=0, atol=1e-15)
    duplicate = replace(context, historical_schedules=tuple(historical + [c0]),
                        forecast_schedules=tuple(forecast + [forecast[0]]))
    duplicate_model = ChannelScheduleModel(half_life_days=None, slot_mode="channel",
                                           prior_strength=2, programme_blend=0).fit(duplicate)
    assert_allclose(duplicate_model.score(0, candidates[::-1]), expected[::-1], rtol=0, atol=1e-15)
    assert duplicate_model.diagnostics["historical_duplicates_ignored"] == 1
    assert duplicate_model.diagnostics["forecast_duplicates_ignored"] == 1
    for user in (-1, 1, 2, 3):
        assert np.isnan(model.score(user, candidates)).all()
    assert np.isnan(model.score(0, context.data.item_ids)).all()
    assert model.programme_counts_[0, 3] == model.programme_counts_[0, 4] == 0
    assert_unchanged(context, original)


@pytest.mark.parametrize("mode", ["channel", "hour", "weekhour"])
def test_response_r2_exact_fraction_hierarchy_and_unseen_weekday_reproduced(mode):
    h = Airing(0, "c", CUTOFF - 7 * DAY + 60, CUTOFF - 7 * DAY + 120)
    same = Airing(3, "c", CUTOFF + 60, CUTOFF + 120)
    next_day = Airing(4, "c", CUTOFF + DAY + 60, CUTOFF + DAY + 120)
    context = fixture([h], [(0, h)], [same, next_day])
    a, activity = Fraction(2), Fraction(1, 3)
    channel = Fraction(1, 3)
    hour = (1 + a * channel * activity) / 3
    weekhour = (1 + a * hour * activity) / 3
    assert hour == Fraction(11, 27) and weekhour == Fraction(103, 243)
    expected = {"channel": channel, "hour": hour, "weekhour": weekhour}[mode]
    model = ChannelScheduleModel(half_life_days=None, slot_mode=mode,
                                 prior_strength=2, programme_blend=0).fit(context)
    actual = model.score(0, [3, 4, 3])
    next_score = np.nan if mode == "weekhour" else float(expected)
    assert_allclose(actual, [float(expected), next_score, float(expected)], rtol=0, atol=1e-15)
    assert model.diagnostics["last_query"]["max_group_rate"] <= 1


def test_response_r3_quarantine_support_change_and_unchanged_catalog_reproduced():
    h = past(0, 0)
    context = fixture([h], [(0, h)], [future(3, quarantined=True)], candidates=np.array([3, 4]))
    original = snapshot(context)
    parameters = {"half_life_days": None, "slot_mode": "channel", "prior_strength": 2, "programme_blend": 0}
    exclude = ChannelScheduleModel(**parameters, quarantined_future_policy="exclude").fit(context)
    include = ChannelScheduleModel(**parameters, quarantined_future_policy="include").fit(context)
    assert np.isnan(exclude.score(0, [3, 3, 4])).all()
    assert_allclose(include.score(0, [3, 3, 4]), [1 / 3, 1 / 3, np.nan], rtol=0, atol=1e-15)
    for model in (exclude, include):
        assert not model.diagnostics["quarantine_score_irrelevance_proven"]
        assert model.diagnostics["quarantined_airings_with_any_training_support"] == 1
        assert model.diagnostics["candidate_ids_removed"] == 0
    assert exclude.forecast_item_opportunities_[3] == 0
    assert include.forecast_item_opportunities_[3] == 1
    assert_array_equal(context.candidate_indices, [3, 4])
    assert_unchanged(context, original)
