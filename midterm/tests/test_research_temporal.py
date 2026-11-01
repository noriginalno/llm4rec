"""Independent arithmetic fixtures for the research-only temporal experts.

Expected counts/ratios are hand-derived here, never obtained from the shared
time-weight helper, temporal_matrix, production ranker or evaluator.
"""

from __future__ import annotations

import builtins
from dataclasses import replace
import json
import math
from pathlib import Path
import socket

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import coo_matrix, csr_matrix, isspmatrix_csr

from research_v2.contracts import Airing, TemporalContext
from research_v2.temporal import ScheduleAdjustedRepeatModel, TemporalRepeatModel
from sweettv.contracts import TrainingData


DAY = 86400
CUTOFF = 100 * DAY
MODELS = (TemporalRepeatModel, ScheduleAdjustedRepeatModel)


def context(events=(), *, users=3, items=6, history_days=84, forecast_days=14,
            historical=(), forecast=(), candidates=None, gap_days=0):
    """events are (user matrix index, item matrix index, end-age in days)."""
    event_users = np.array([u for u, _, _ in events], dtype=np.int64)
    event_items = np.array([i for _, i, _ in events], dtype=np.int64)
    ends = np.array([CUTOFF - int(age * DAY) for _, _, age in events], dtype=np.int64)
    matrix = coo_matrix((np.ones(len(events)), (event_users, event_items)), shape=(users, items)).tocsr()
    data = TrainingData(
        user_ids=["18446744073709551615", "0007", "user-é"][:users]
        + [f"user-{u}" for u in range(3, users)],
        item_ids=np.array([707, 11, 9007199254740993, 5, 999, 37][:items]
                          + list(range(1000, 1000 + max(0, items - 6))), dtype=np.int64),
        interactions=matrix, metadata=[{} for _ in range(items)],
    )
    return TemporalContext(
        data=data, candidate_indices=np.arange(items) if candidates is None else candidates,
        history_start=CUTOFF - int(history_days * DAY), train_end=CUTOFF,
        forecast_start=CUTOFF + int(gap_days * DAY),
        forecast_end=CUTOFF + int((gap_days + forecast_days) * DAY),
        event_users=event_users, event_items=event_items, event_starts=ends - 60,
        event_ends=ends, event_channels=np.array(["channel"] * len(events)),
        historical_schedules=tuple(historical), forecast_schedules=tuple(forecast),
    )


def past(item, age, *, channel="channel", duration=60):
    end = CUTOFF - int(age * DAY)
    return Airing(item, channel, end - duration, end)


def future(item, number=0, *, channel="channel", quarantined=False):
    start = CUTOFF + 60 + number * 120
    return Airing(item, channel, start, start + 60, quarantined)


def test_half_lives_zero_one_two_are_1_half_quarter_and_repeats_aggregate():
    ctx = context([(0, 0, 0), (0, 0, 28), (0, 0, 56), (0, 1, 28), (1, 2, 56)])
    model = TemporalRepeatModel().fit(ctx)
    assert model.half_life_days == 28.0
    assert model.lookback_days is None
    assert_allclose(model.score(0, [1, 0, 2]), [0.5, 1.75, np.nan], rtol=0, atol=0)
    assert_allclose(model.score(1, [2, 0]), [0.25, np.nan], rtol=0, atol=0)
    assert isspmatrix_csr(model.repeat_counts_)
    assert model.repeat_counts_.dtype == np.float64
    assert model.diagnostics["training_events"] == 5
    assert model.diagnostics["temporal_user_item_pairs"] == 3
    assert model.diagnostics["total_temporal_count"] == 2.5


@pytest.mark.parametrize("half_life,expected", [(None, 2.0), (28.0, 1.5)])
def test_lookback_uses_end_not_start_and_includes_exact_boundary(half_life, expected):
    # The second event STARTS outside the window but ENDS exactly on its boundary.
    ctx = context([(0, 0, 0), (0, 0, 28), (0, 0, 28 + 1 / DAY)])
    model = TemporalRepeatModel(half_life_days=half_life, lookback_days=28).fit(ctx)
    assert model.score(0, [0])[0] == expected
    assert model.diagnostics["events_in_lookback"] == 2


def test_shortened_history_excludes_older_events_without_lookback_and_keeps_zero_power_exact():
    ctx = context([(0, 0, 3), (0, 0, 1)], history_days=2, forecast_days=1,
                  historical=[past(0, 1)], forecast=[future(0)])
    repeat = TemporalRepeatModel(half_life_days=None).fit(ctx)
    adjusted = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1,
                                           smoothing_airings=1).fit(ctx)
    zero = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=0).fit(ctx)
    # One event lies before declared history_start. Retained R=H=F=1;
    # E=2 days, Q=1 day, so ratio=(1+1)/(1/2+1)=4/3, not old score 8/3.
    assert repeat.score(0, [0])[0] == 1
    assert adjusted.score(0, [0])[0] == pytest.approx(4 / 3)
    assert_array_equal(zero.score(0, [0, 1]), repeat.score(0, [0, 1]))
    assert ctx.data.interactions[0, 0] == 2  # Original production counts unchanged.


def test_declared_history_start_is_strict_for_both_events_and_opportunities():
    ctx = context([(0, 0, 3), (0, 0, 2), (0, 0, 1)], history_days=2, forecast_days=1,
                  historical=[past(0, 3), past(0, 2), past(0, 1)], forecast=[future(0)])
    model = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1,
                                       smoothing_airings=1).fit(ctx)
    assert model.repeat_counts_[0, 0] == model.historical_counts_[0] == 1
    assert model.score(0, [0])[0] == pytest.approx(4 / 3)
    assert model.diagnostics["events_before_history_start_excluded"] == 2
    assert model.diagnostics["schedule_counts"]["historical_airings_before_history_start_excluded"] == 2
    assert ctx.data.interactions[0, 0] == 3


@pytest.mark.parametrize("cls", MODELS)
def test_all_events_before_history_start_are_feature_cold_not_original_count_mutations(cls):
    ctx = context([(0, 0, 3)], history_days=2,
                  historical=[past(0, 3)], forecast=[future(0)])
    model = cls().fit(ctx)
    assert model.repeat_counts_.nnz == 0
    assert np.isnan(model.score(0, [0, 1])).all()
    assert model.diagnostics["training_events"] == 1
    assert model.diagnostics["events_before_history_start_excluded"] == 1
    assert model.diagnostics["active_users"] == 0
    assert ctx.data.interactions[0, 0] == 1


def test_none_half_life_preserves_actual_counts_and_ids_not_matrix_indices():
    ctx = context([(0, 0, 0), (0, 0, 28), (0, 0, 56), (0, 1, 0)])
    model = TemporalRepeatModel(half_life_days=None).fit(ctx)
    assert_allclose(model.score(0, [1, 0, 2]), [1, 3, np.nan], rtol=0, atol=0)
    assert_array_equal(model.item_ids_, ctx.data.item_ids)
    assert model.user_ids_ == ("18446744073709551615", "0007", "user-é")
    assert np.isnan(model.score(0, ctx.data.item_ids[:3])).all()


@pytest.mark.parametrize("cls", MODELS)
def test_candidate_order_duplicates_unknowns_and_seen_eligibility(cls):
    ctx = context([(0, 0, 0), (0, 1, 28)], forecast=[future(0), future(1)])
    model = cls().fit(ctx)
    candidates = np.array([1, 0, 1, 2, -1, 99])
    original = candidates.copy()
    actual = model.score(0, candidates)
    assert np.isfinite(actual[:3]).all()  # Already watched, not excluded.
    assert actual[0] == actual[2]
    assert np.isnan(actual[3:]).all()  # No personal evidence, never popularity.
    permutation = np.array([5, 3, 1, 4, 0, 2])
    assert_allclose(model.score(0, candidates[permutation]), actual[permutation], rtol=0, atol=0)
    assert_array_equal(candidates, original)
    assert np.isnan(model.score(0, np.array([2**64 - 1], dtype=np.uint64))).all()


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize("user", [-1, 3, 99, np.int64(-1), np.uint64(2**64 - 1)])
def test_unknown_user_bounds_never_wrap_to_last_sparse_row(cls, user):
    model = cls().fit(context([(2, 0, 0)], forecast=[future(0)]))
    assert np.isnan(model.score(user, [0, 1])).all()


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize("user", [True, np.bool_(False), 0.0, "0", None])
def test_invalid_user_types_fail_even_for_empty_candidates(cls, user):
    model = cls().fit(context())
    with pytest.raises(ValueError, match="integer"):
        model.score(user, [])


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize("candidates", [[[0]], [0.0], [True], ["0"], np.array(0)])
def test_candidate_shape_and_integer_validation(cls, candidates):
    model = cls().fit(context())
    with pytest.raises(ValueError, match="one-dimensional integer"):
        model.score(0, candidates)


@pytest.mark.parametrize("cls", MODELS)
def test_unfitted_empty_cold_and_empty_catalog(cls):
    with pytest.raises(RuntimeError, match="fitted"):
        cls().score(0, [])
    model = cls().fit(context())
    assert np.isnan(model.score(0, [0, 1])).all()
    assert model.score(0, []).shape == (0,)
    empty = cls().fit(context(users=0, items=0))
    assert empty.repeat_counts_.shape == (0, 0)
    assert np.isnan(empty.score(0, [0, -1])).all()
    assert empty.score(-1, []).dtype == np.float64
    json.dumps(empty.diagnostics, allow_nan=False)


@pytest.mark.parametrize("cls", MODELS)
@pytest.mark.parametrize("name", ["half_life_days", "lookback_days"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf, True, "28"])
def test_decay_parameters_must_be_none_or_positive_finite_float(cls, name, value):
    with pytest.raises(ValueError, match=name):
        cls(**{name: value})


@pytest.mark.parametrize("name,value", [
    ("schedule_power", -0.1), ("schedule_power", 1.1), ("schedule_power", np.inf),
    ("schedule_power", np.nan), ("schedule_power", True),
    ("smoothing_airings", 0), ("smoothing_airings", -1), ("smoothing_airings", np.nan),
    ("smoothing_airings", np.inf), ("smoothing_airings", "10"),
    ("ratio_cap", 0.9), ("ratio_cap", np.inf), ("ratio_cap", np.nan), ("ratio_cap", True),
])
def test_schedule_parameters_are_bounded_and_finite(name, value):
    with pytest.raises(ValueError, match=name):
        ScheduleAdjustedRepeatModel(**{name: value})


def test_schedule_formula_same_end_weights_window_and_explicit_smoothing_units():
    # History weights for item0: 1 + 1/2 + 1/4, with the 2h boundary retained.
    # Item1: 1/2, and its older occurrence is outside the 56-day lookback.
    historical = [past(0, 0), past(0, 28), past(0, 56), past(1, 28), past(1, 56 + 1 / DAY)]
    forecast = [future(0, n) for n in range(3)] + [future(1)]
    ctx = context([(0, 0, 0), (0, 0, 28), (0, 1, 28)], history_days=84,
                  historical=historical, forecast=forecast, forecast_days=14)
    model = ScheduleAdjustedRepeatModel(lookback_days=56, schedule_power=1).fit(ctx)
    # Integral from 0 to 56 of 2**(-age/28): 28/ln2 * (1 - 1/4).
    effective_days = 21 / math.log(2)
    expected_ratios = [(3 + 10) / (1.75 * 14 / effective_days + 10),
                       (1 + 10) / (0.5 * 14 / effective_days + 10)]
    assert_allclose(model.historical_counts_[:2], [1.75, 0.5], rtol=0, atol=0)
    assert_allclose(model.forecast_counts_[:2], [3, 1], rtol=0, atol=0)
    assert_allclose(model.score(0, [0, 1]), [1.5 * expected_ratios[0], 0.5 * expected_ratios[1]], rtol=1e-14)
    assert model.diagnostics["availability_ratios"]["effective_weighted_history_days"] == pytest.approx(effective_days)
    assert model.diagnostics["availability_ratios"]["smoothing_units"] == "scheduled airings per forecast horizon"


def test_unchanged_schedule_rate_gives_one_for_rare_and_frequent_programs():
    # 28 historical days, 14 forecast days: rates 2/28 == 1/14 and 20/28 == 10/14.
    hist = [past(0, n) for n in range(2)] + [past(1, n) for n in range(20)]
    fc = [future(0)] + [future(1, n) for n in range(10)]
    ctx = context([(0, 0, 0), (0, 1, 0)], history_days=28, historical=hist, forecast=fc)
    model = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1).fit(ctx)
    assert_allclose(model.availability_ratios_[:2], [1, 1], rtol=0, atol=1e-15)
    assert_allclose(model.score(0, [0, 1]), [1, 1], rtol=0, atol=1e-15)


def test_forecast_change_moves_adjusted_scores_but_not_temporal_counts():
    hist = [past(0, n) for n in range(4)] + [past(1, n) for n in range(4)]
    first = context([(0, 0, 0), (0, 1, 0)], history_days=28, historical=hist,
                    forecast=[future(0), future(1)])
    changed = replace(first, forecast_schedules=tuple([future(0, n) for n in range(8)] + [future(1)]))
    original_model = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1).fit(first)
    changed_model = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1).fit(changed)
    # Historical count projected to 14-day forecast = 4 * 14 / 28 = 2.
    assert_allclose(original_model.score(0, [0, 1]), [11 / 12, 11 / 12], rtol=1e-14)
    assert_allclose(changed_model.score(0, [0, 1]), [18 / 12, 11 / 12], rtol=1e-14)
    assert_array_equal(original_model.repeat_counts_.data, changed_model.repeat_counts_.data)
    assert_array_equal(TemporalRepeatModel().fit(first).score(0, [0, 1]),
                       TemporalRepeatModel().fit(changed).score(0, [0, 1]))


def test_no_personal_history_is_nan_even_when_forecast_is_abundant():
    ctx = context([(0, 0, 0)], historical=[], forecast=[future(i) for i in range(3)])
    model = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1).fit(ctx)
    assert model.score(0, [0])[0] == pytest.approx(11 / 10)
    assert np.isnan(model.score(0, [1, 2])).all()
    assert np.isnan(model.score(1, [0, 1, 2])).all()
    assert model.diagnostics["schedule_counts"]["forecast_programs_without_weighted_history"] == 3


def test_no_forecast_occurrence_and_boundary_only_candidate_stay_nan_in_place():
    # Candidate 1 represents an airing starting before forecast_end but ending
    # after it: retained by the broad shared catalog, absent from contained EPG.
    ctx = context([(0, 0, 0), (0, 1, 0)], candidates=[1, 0, 2], forecast=[future(0)])
    model = ScheduleAdjustedRepeatModel().fit(ctx)
    actual = model.score(0, [1, 0, 1, 2])
    assert actual.shape == (4,)
    assert np.isnan(actual[[0, 2, 3]]).all()
    assert np.isfinite(actual[1])
    assert model.diagnostics["candidate_coverage"]["unique_candidates"] == 3
    assert model.diagnostics["candidate_coverage"]["eligible_candidates"] == 1
    assert_array_equal(ctx.candidate_indices, [1, 0, 2])


@pytest.mark.parametrize("half_life,lookback", [(None, None), (28, None), (28, 28), (None, 14)])
def test_zero_schedule_power_is_bitwise_repeat_even_for_boundary_and_unknowns(half_life, lookback):
    ctx = context([(0, 0, 0), (0, 0, 28), (0, 1, 56), (1, 2, 0)], forecast=[])
    repeat = TemporalRepeatModel(half_life, lookback).fit(ctx)
    adjusted = ScheduleAdjustedRepeatModel(half_life, lookback, schedule_power=0).fit(ctx)
    for user in [0, 1, 2, -1, 3]:
        candidates = np.array([1, 0, 0, 2, -1, 88])
        assert_array_equal(adjusted.score(user, candidates), repeat.score(user, candidates))
    assert adjusted.diagnostics["candidate_coverage"] == repeat.diagnostics["candidate_coverage"]


def test_cap_smoothing_and_fractional_power_control_rare_history_bursts():
    ctx = context([(0, 0, 0), (0, 1, 0)], history_days=14,
                  historical=[past(0, 0)] + [past(1, n / 100) for n in range(1000)],
                  forecast=[future(0, n) for n in range(500)] + [future(1)])
    model = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=0.5, ratio_cap=10).fit(ctx)
    assert_allclose(model.availability_ratios_[:2], [10, 0.1], rtol=0, atol=0)
    assert_allclose(model.score(0, [0, 1]), [math.sqrt(10), math.sqrt(0.1)], rtol=1e-14)
    assert model.diagnostics["availability_ratios"]["upper_clipped_programs"] == 1
    assert model.diagnostics["availability_ratios"]["lower_clipped_programs"] == 1
    shrink = ScheduleAdjustedRepeatModel(half_life_days=None, schedule_power=1,
                                       smoothing_airings=1000, ratio_cap=100).fit(ctx)
    assert shrink.score(0, [0])[0] == pytest.approx(1500 / 1001)


def test_ratio_cap_one_neutralizes_available_items_but_not_availability_support():
    ctx = context([(0, 0, 0), (0, 1, 0)], forecast=[future(0)])
    model = ScheduleAdjustedRepeatModel(ratio_cap=1).fit(ctx)
    assert model.score(0, [0])[0] == 1
    assert np.isnan(model.score(0, [1])[0])


def test_duplicate_schedule_rows_count_once_but_distinct_channels_are_opportunities():
    historical = [past(0, 0), past(0, 0), past(0, 0, channel="other")]
    forecast = [future(0), future(0), future(0, channel="other", quarantined=True)]
    ctx = context([(0, 0, 0)], historical=historical, forecast=forecast)
    model = ScheduleAdjustedRepeatModel().fit(ctx)
    assert model.historical_counts_[0] == 2
    assert model.forecast_counts_[0] == 2
    diag = model.diagnostics["schedule_counts"]
    assert diag["historical_rows"] == diag["forecast_rows"] == 3
    assert diag["unique_historical_airings"] == diag["unique_forecast_airings"] == 2
    assert diag["forecast_quarantined_rows"] == 1


def test_common_horizon_uses_forecast_duration_not_gap_since_cutoff():
    first = context([(0, 0, 0)], historical=[past(0, 0)], forecast=[future(0)])
    shifted_forecast = tuple(replace(a, start=a.start + 7 * DAY, end=a.end + 7 * DAY)
                             for a in first.forecast_schedules)
    later = replace(first, forecast_start=first.forecast_start + 7 * DAY,
                    forecast_end=first.forecast_end + 7 * DAY, forecast_schedules=shifted_forecast)
    original = ScheduleAdjustedRepeatModel().fit(first)
    shifted = ScheduleAdjustedRepeatModel().fit(later)
    assert_array_equal(original.score(0, [0]), shifted.score(0, [0]))


@pytest.mark.parametrize("cls", MODELS)
def test_no_mutation_or_aliasing_of_data_context_arrays_or_sparse_counts(cls):
    ctx = context([(0, 0, 0), (0, 1, 28)], historical=[past(0, 0)], forecast=[future(0), future(1)])
    before = ctx.data.interactions.copy()
    arrays = {name: getattr(ctx, name).copy() for name in (
        "candidate_indices", "event_users", "event_items", "event_starts", "event_ends", "event_channels",
    )}
    ids = ctx.data.item_ids.copy()
    user_ids = ctx.data.user_ids.copy()
    model = cls().fit(ctx)
    model.score(0, [1, 0, 1])
    assert_array_equal(ctx.data.interactions.data, before.data)
    assert_array_equal(ctx.data.interactions.indices, before.indices)
    assert_array_equal(ctx.data.interactions.indptr, before.indptr)
    assert ctx.data.interactions.dtype == np.float32
    for name, snapshot in arrays.items():
        assert_array_equal(getattr(ctx, name), snapshot)
        assert not getattr(ctx, name).flags.writeable
    assert_array_equal(ctx.data.item_ids, ids)
    assert ctx.data.user_ids == user_ids
    assert not np.shares_memory(model.repeat_counts_.data, ctx.data.interactions.data)
    expected = model.score(0, [0, 1]).copy()
    ctx.data.interactions.data[:] = 99
    assert_array_equal(model.score(0, [0, 1]), expected)


@pytest.mark.parametrize("cls", MODELS)
def test_fit_never_opens_files_network_or_densifies_sparse_training_matrix(cls, monkeypatch):
    ctx = context([(0, 0, 0), (0, 0, 3)], history_days=2,
                  historical=[past(0, 0), past(0, 3)], forecast=[future(0)])

    def forbidden(*args, **kwargs):
        raise AssertionError("fit must use only supplied in-memory sparse context")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(socket, "socket", forbidden)
        patch.setattr(csr_matrix, "toarray", forbidden)
        model = cls().fit(ctx)
    assert isspmatrix_csr(model.repeat_counts_)
    assert np.isfinite(model.score(0, [0])[0])


@pytest.mark.parametrize("cls", MODELS)
def test_refit_replaces_old_shape_ids_sparse_evidence_and_schedule_state(cls):
    model = cls().fit(context([(0, 0, 0), (1, 1, 0)], forecast=[future(0), future(1)]))
    second = context([(0, 2, 28)], users=1, items=3, forecast=[future(2)])
    assert model.fit(second) is model
    assert model.repeat_counts_.shape == (1, 3)
    assert np.isnan(model.score(0, [0, 1])).all()
    assert np.isfinite(model.score(0, [2])[0])
    assert np.isnan(model.score(1, [2])[0])
    assert model.diagnostics["training_events"] == 1
    assert_array_equal(model.item_ids_, second.data.item_ids)
    if isinstance(model, ScheduleAdjustedRepeatModel):
        assert_array_equal(model.forecast_counts_, [0, 0, 1])


@pytest.mark.parametrize("cls", MODELS)
def test_fit_requires_context_and_revalidates_mutated_parameters_atomically(cls):
    ctx = context([(0, 0, 0)], forecast=[future(0)])
    model = cls().fit(ctx)
    before = model.score(0, [0])
    with pytest.raises(TypeError, match="TemporalContext"):
        model.fit(ctx.data)
    assert_array_equal(model.score(0, [0]), before)
    model.half_life_days = -1
    with pytest.raises(ValueError, match="half_life_days"):
        model.fit(ctx)
    assert_array_equal(model.score(0, [0]), before)


def test_failed_schedule_refit_cannot_mix_new_repeat_with_old_schedule_state():
    ctx = context([(0, 0, 0)], forecast=[future(0)])
    model = ScheduleAdjustedRepeatModel().fit(ctx)
    before = model.score(0, [0, 1])
    invalid_schedule = replace(ctx, event_items=np.array([1]),
                               forecast_schedules=(future(88),))
    with pytest.raises(ValueError, match="Schedule item index"):
        model.fit(invalid_schedule)
    assert_array_equal(model.score(0, [0, 1]), before)


@pytest.mark.parametrize("airing,error", [
    (Airing(0, "", CUTOFF - 60, CUTOFF), "channels"),
    (Airing(0, "c", CUTOFF - 60.5, CUTOFF), "integer"),
    (Airing(0, "c", CUTOFF, CUTOFF), "positive duration"),
])
def test_schedule_input_validation_does_not_silently_change_clock_units(airing, error):
    ctx = context([(0, 0, 0)], historical=[airing], forecast=[future(0)])
    with pytest.raises(ValueError, match=error):
        ScheduleAdjustedRepeatModel().fit(ctx)


def test_future_behavior_and_noncontained_forecast_cannot_enter_context():
    ctx = context([(0, 0, 0)])
    with pytest.raises(ValueError, match="Future"):
        replace(ctx, event_ends=np.array([CUTOFF + 1]))
    with pytest.raises(ValueError, match="fully contained"):
        replace(ctx, forecast_schedules=(Airing(0, "c", ctx.forecast_end - 1, ctx.forecast_end + 1),))


@pytest.mark.parametrize("half_life", [np.nextafter(0.0, 1.0), 1e-100, 1e100, np.finfo(np.float64).max])
def test_valid_extreme_half_lives_have_finite_bounded_schedule_adjustments(half_life):
    ctx = context([(0, 0, 0)], historical=[past(0, 0)], forecast=[future(0)])
    with np.errstate(over="ignore", under="ignore"):
        model = ScheduleAdjustedRepeatModel(half_life_days=half_life,
                                           smoothing_airings=np.nextafter(0.0, 1.0)).fit(ctx)
    assert np.isfinite(model.score(0, [0])[0])
    assert 0.1 <= model.availability_ratios_[0] <= 10
    json.dumps(model.diagnostics, allow_nan=False)


def test_effective_history_duration_handles_valid_tiny_window_and_huge_half_life():
    ctx = context([(0, 0, 0)], historical=[past(0, 0)], forecast=[future(0)])
    model = ScheduleAdjustedRepeatModel(half_life_days=np.finfo(np.float64).max,
                                       lookback_days=1e-300).fit(ctx)
    assert model.diagnostics["availability_ratios"]["effective_weighted_history_days"] == 1e-300
    assert np.isfinite(model.score(0, [0])[0])
    json.dumps(model.diagnostics, allow_nan=False)


def test_unrepresentable_extreme_schedule_scale_is_rejected_before_fit_commit():
    ctx = context([(0, 0, 0), (0, 0, 0)], historical=[], forecast=[future(0)])
    model = ScheduleAdjustedRepeatModel().fit(ctx)
    before = model.score(0, [0, 1])
    model.schedule_power = 1
    model.ratio_cap = 1e308
    model.smoothing_airings = np.nextafter(0.0, 1.0)
    # R=2, no H, tiny prior: capped ratio=1e308 would make R*ratio infinite.
    with pytest.raises(ValueError, match="overflow"):
        model.fit(ctx)
    assert_array_equal(model.score(0, [0, 1]), before)


def test_diagnostics_report_actual_counts_window_coverage_and_json_safe_ratios():
    ctx = context([(0, 0, 0), (0, 1, 28), (1, 1, 56)],
                  candidates=[0, 1, 1, 2], historical=[past(0, 0), past(1, 56)], forecast=[future(0)])
    model = ScheduleAdjustedRepeatModel(lookback_days=28).fit(ctx)
    diag = model.diagnostics
    assert diag["users"] == 3 and diag["items"] == 6
    assert diag["training_events"] == 3
    assert diag["events_in_lookback"] == 2
    assert diag["positive_weight_events"] == 2
    assert diag["active_users"] == 1
    assert diag["history_window"]["window_span_days"] == 28
    assert diag["schedule_counts"]["positive_weight_historical_airings"] == 1
    assert diag["candidate_coverage"] == {
        "unique_candidates": 3, "eligible_candidates": 1, "supported_candidates": 1,
        "supported_users": 1, "supported_user_candidate_pairs": 1, "user_candidate_pair_coverage": 1 / 9,
    }
    assert diag["availability_ratios"]["clipped_ratio_available"]["count"] == 1
    assert diag["parameters"] == {"half_life_days": 28, "lookback_days": 28,
                                  "schedule_power": 0.5, "smoothing_airings": 10, "ratio_cap": 10}
    json.dumps(diag, allow_nan=False)


# Role reversal: independently challenge the channel/time author's production
# module without executing or importing that author's test fixtures/oracles.
def test_channel_challenge_cold_program_rates_duplicate_identity_and_candidate_order():
    from research_v2.schedule import ChannelScheduleModel

    hist = [past(0, 1, channel="c"), past(1, 2, channel="d"), past(2, 3, channel="c")]
    fc = [future(3, 0, channel="c"), future(3, 1, channel="c"),
          future(4, 0, channel="d"), future(5, 0, channel="unseen")]
    ctx = replace(context([(0, 0, 1), (0, 1, 2)], historical=hist, forecast=fc),
                  event_channels=np.array(["c", "d"]))
    model = ChannelScheduleModel(half_life_days=None, slot_mode="channel",
                                 prior_strength=2, programme_blend=0).fit(ctx)
    # Channel c: one completed / (two scheduled + two prior) = 1/4.
    # Channel d: one completed / (one scheduled + two prior) = 1/3.
    # Cold item3 has two future c airings: 2*(1/4) = 1/2.
    candidates = np.array([5, 4, 3, 4, -1, 6])
    expected = [np.nan, 1 / 3, 1 / 2, 1 / 3, np.nan, np.nan]
    assert_allclose(model.score(0, candidates), expected, rtol=0, atol=1e-15)
    duplicates = replace(ctx, historical_schedules=tuple(hist + [hist[0]]),
                         forecast_schedules=tuple(fc + [fc[0]]))
    repeated = ChannelScheduleModel(half_life_days=None, slot_mode="channel",
                                    prior_strength=2, programme_blend=0).fit(duplicates)
    assert_allclose(repeated.score(0, candidates[::-1]), expected[::-1], rtol=0, atol=1e-15)
    assert np.isnan(model.score(-1, candidates)).all()
    assert np.isnan(model.score(1, candidates)).all()


def test_channel_challenge_weekhour_arithmetic_and_zero_preserving_new_day():
    from research_v2.schedule import ChannelScheduleModel

    h = Airing(0, "c", CUTOFF - 7 * DAY + 60, CUTOFF - 7 * DAY + 120)
    f_same = Airing(3, "c", CUTOFF + 60, CUTOFF + 120)
    f_next_day = Airing(4, "c", CUTOFF + DAY + 60, CUTOFF + DAY + 120)
    ctx = replace(context([(0, 0, 7)], historical=[h], forecast=[f_same, f_next_day]),
                  event_channels=np.array(["c"]), event_starts=np.array([h.start]),
                  event_ends=np.array([h.end]))
    model = ChannelScheduleModel(half_life_days=None, slot_mode="weekhour",
                                 prior_strength=2, programme_blend=0).fit(ctx)
    # r_channel=1/3; activity_hour=1/3; r_hour=(1+2/9)/3=11/27.
    # activity_weekhour=1/3; r_weekhour=(1+22/81)/3=103/243.
    assert_allclose(model.score(0, [3, 4, 3]), [103 / 243, np.nan, 103 / 243], rtol=0, atol=1e-15)
    assert model.diagnostics["last_query"]["max_group_rate"] <= 1


def test_channel_challenge_quarantine_is_explicit_support_not_candidate_removal():
    from research_v2.schedule import ChannelScheduleModel

    ctx = context([(0, 0, 0)], historical=[past(0, 0)],
                  forecast=[future(3, quarantined=True)], candidates=[3, 4])
    exclude = ChannelScheduleModel(half_life_days=None, slot_mode="channel",
                                   prior_strength=2, programme_blend=0).fit(ctx)
    include = ChannelScheduleModel(half_life_days=None, slot_mode="channel",
                                   prior_strength=2, programme_blend=0,
                                   quarantined_future_policy="include").fit(ctx)
    assert np.isnan(exclude.score(0, [3, 3, 4])).all()
    assert_allclose(include.score(0, [3, 3, 4]), [1 / 3, 1 / 3, np.nan], rtol=0, atol=1e-15)
    assert not include.diagnostics["quarantine_score_irrelevance_proven"]
    assert_array_equal(ctx.candidate_indices, [3, 4])
