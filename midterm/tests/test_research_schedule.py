"""Independent synthetic fixtures; expected rates are hand-derived, not fitted.

No source logs, held-out viewing, or evaluation labels are read. Monday's numeric
anchor is the same naive source-clock representation as TemporalContext.
"""

from __future__ import annotations

import builtins
from copy import deepcopy
from pathlib import Path
import socket

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import coo_matrix, csr_matrix, issparse

from research_v2.contracts import Airing, TemporalContext
from research_v2.schedule import ChannelScheduleModel
from sweettv.contracts import TrainingData


DAY = 86400
HOUR = 3600
MONDAY = 1583712000  # 2020-03-09 00:00:00 on the explicitly naive anchor.


def airing(item, channel, day, hour=20, *, minute=0, duration=1800, quarantined=False):
    start = MONDAY + day * DAY + hour * HOUR + minute * 60
    return Airing(item, channel, start, start + duration, quarantined)


def context(historical=(), completed=(), forecast=(), *, n_users=3, n_items=5,
            item_ids=None, history_start=MONDAY, train_end=MONDAY + 28 * DAY,
            forecast_end=MONDAY + 35 * DAY, candidates=None):
    """completed contains (user matrix index, full scheduled airing) pairs."""
    completed = list(completed)
    counts = coo_matrix(
        (np.ones(len(completed)),
         ([u for u, _ in completed], [a.item_index for _, a in completed])),
        shape=(n_users, n_items),
    ).tocsr()
    data = TrainingData(
        [f"000-user-{i}" for i in range(n_users)],
        np.arange(901, 901 + n_items) if item_ids is None else item_ids,
        counts,
        [{"channel_title": "not authoritative", "tv_show_title": f"future metadata {i}"}
         for i in range(n_items)],
    )
    return TemporalContext(
        data=data,
        candidate_indices=np.arange(n_items) if candidates is None else candidates,
        history_start=history_start, train_end=train_end,
        forecast_start=train_end, forecast_end=forecast_end,
        event_users=np.array([u for u, _ in completed], dtype=np.int64),
        event_items=np.array([a.item_index for _, a in completed], dtype=np.int64),
        event_starts=np.array([a.start for _, a in completed], dtype=np.int64),
        event_ends=np.array([a.end for _, a in completed], dtype=np.int64),
        event_channels=np.array([a.channel_id for _, a in completed], dtype=str),
        historical_schedules=tuple(historical), forecast_schedules=tuple(forecast),
    )


@pytest.mark.parametrize("mode", ["channel", "hour", "weekhour"])
def test_cold_future_programme_on_preferred_channel_beats_unrelated_channel(mode):
    preferred = [airing(0, "favorite", d) for d in range(8)]
    unrelated = [airing(1, "other", d) for d in range(8)]
    ctx = context(preferred + unrelated,
                  [(0, a) for a in preferred] + [(0, unrelated[0])],
                  [airing(2, "favorite", 28), airing(3, "other", 28)],
                  # The model must not implement its own candidate filter.
                  candidates=np.array([0, 1]))
    model = ChannelScheduleModel(half_life_days=None, slot_mode=mode,
                                 prior_strength=2, programme_blend=0).fit(ctx)
    scores = model.score(0, np.array([2, 3]))
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1] > 0
    assert model.programme_counts_[0, 2] == model.programme_counts_[0, 3] == 0
    if mode == "channel":
        assert_allclose(scores, [8 / 10, 1 / 10], atol=1e-15)
    assert model.diagnostics["last_query"]["max_group_rate"] <= 1


@pytest.mark.parametrize("mode", ["hour", "weekhour"])
def test_observed_slot_not_beaten_by_prior_in_never_active_hour(mode):
    # Numerous opportunities depress the supported hour's observed rate. A naive
    # parent-only prior divided by a tiny new-slot denominator could rank 03:00
    # higher than 20:00; zero-preserving activity gates must not do that.
    historical = [airing(0, "favorite", d % 28, minute=(d // 28) * 30)
                  for d in range(40)]
    historical.append(airing(1, "favorite", 0, 3))
    ctx = context(historical, [(0, historical[0])],
                  [airing(2, "favorite", 28, 20), airing(3, "favorite", 28, 3)])
    model = ChannelScheduleModel(half_life_days=None, slot_mode=mode,
                                 prior_strength=20, programme_blend=0).fit(ctx)
    scores = model.score(0, np.array([2, 3]))
    assert scores[0] > 0
    assert np.isnan(scores[1])  # No completion/activity evidence, not a zero vote.


def test_weekday_is_source_clock_weekday_not_just_hour():
    historical = [airing(0, "favorite", d, 20) for d in (0, 7, 14, 21)]
    ctx = context(historical, [(0, a) for a in historical],
                  [airing(2, "favorite", 28, 20), airing(3, "favorite", 29, 20)])
    hour = ChannelScheduleModel(half_life_days=None, slot_mode="hour", programme_blend=0).fit(ctx)
    week = ChannelScheduleModel(half_life_days=None, slot_mode="weekhour", programme_blend=0).fit(ctx)
    # a=4: r_c=4/8, A_20=4/8, r_c20=(4+4*(1/2)*(1/2))/8=5/8.
    assert_allclose(hour.score(0, np.array([2, 3])), [5 / 8, 5 / 8])
    scores = week.score(0, np.array([2, 3]))
    assert scores[0] > 0 and np.isnan(scores[1])
    assert week.weekhour_counts_[0, 20] == 4  # Monday=0, not Unix Thursday=0.
    assert week.diagnostics["source_clock"].startswith("naive source-clock")


@pytest.mark.parametrize("mode,expected", [
    ("channel", 1 / 3), ("hour", 4 / 15), ("weekhour", 53 / 180),
])
def test_exact_hierarchical_rate_arithmetic(mode, expected):
    c_mon8 = airing(0, "c", 0, 8)
    c_next_mon8 = airing(0, "c", 7, 8)
    c_mon9 = airing(1, "c", 0, 9)
    c_tue8 = airing(1, "c", 1, 8)
    d_tue8 = airing(2, "d", 1, 8)
    ctx = context([c_mon8, c_next_mon8, c_mon9, c_tue8, d_tue8],
                  [(0, c_mon8), (0, c_mon9), (0, d_tue8)], [airing(3, "c", 28, 8)])
    # a=2: r_c=2/(4+2)=1/3; A_8=2/(2+2)=1/2;
    # r_c8=(1+2*(1/3)*(1/2))/(3+2)=4/15;
    # A_Mon8=1/3; r_cMon8=(1+2*(4/15)*(1/3))/(2+2)=53/180.
    model = ChannelScheduleModel(half_life_days=None, slot_mode=mode,
                                 prior_strength=2, programme_blend=0.5).fit(ctx)
    assert model.score(0, np.array([3]))[0] == pytest.approx(expected, abs=1e-15)
    for level in model.diagnostics["hierarchy"].values():
        assert 0 < level["min_observed_rate"] <= level["max_observed_rate"] <= 1


@pytest.mark.parametrize("mode,expected", [("hour", 1 / 9), ("weekhour", 1 / 27)])
def test_backoff_transfers_only_when_both_channel_and_activity_have_evidence(mode, expected):
    c9, d8 = airing(0, "c", 0, 9), airing(1, "d", 0, 8)
    ctx = context([c9, d8], [(0, c9), (0, d8)],
                  [airing(2, "c", 28, 8), airing(3, "unseen", 28, 8)])
    model = ChannelScheduleModel(half_life_days=None, slot_mode=mode,
                                 prior_strength=2, programme_blend=0).fit(ctx)
    scores = model.score(0, np.array([2, 3]))
    assert scores[0] == pytest.approx(expected)
    assert np.isnan(scores[1])


@pytest.mark.parametrize("blend,cap,expected", [(0, 3, 6 / 5), (0.5, 3, 51 / 40), (0.5, 0.25, 41 / 40)])
def test_exact_programme_repeat_blend_and_capped_arithmetic(blend, cap, expected):
    a0, a1, b = airing(0, "c", 0, 8), airing(0, "c", 1, 8), airing(1, "c", 2, 8)
    future = [airing(0, "c", d, 8) for d in (28, 29, 30)]
    future += [airing(2, "c", d, 9) for d in (28, 29)]
    model = ChannelScheduleModel(half_life_days=None, slot_mode="channel", prior_strength=2,
                                 programme_blend=blend, max_repeat_multiplier=cap).fit(
                                     context([a0, a1, b], [(0, a0), (0, b)], future))
    # r_c=2/5, B_0=6/5, H_0=2, C_0=1, F_0=3.
    # Uncapped blend .5: .75*(6/5)+.5*(3/4)=51/40.
    # Cap .25: .75*(6/5)+.5*(1/4)=41/40.
    scores = model.score(0, np.array([0, 2]))
    assert_allclose(scores, [expected, 4 / 5], atol=1e-15)
    assert model.diagnostics["last_query"]["repeat_caps_applied"] == int(cap == 0.25 and blend > 0)


@pytest.mark.parametrize("future_count", [20, 100, 1000])
def test_one_view_overscheduled_repeat_has_bounded_personal_uplift(future_count):
    a = airing(0, "known", 0)
    # New channel has no channel habit; this isolates the programme-following
    # term and makes the hand-derived cap independent of the forecast base rate.
    forecast = [airing(0, "new-channel", 28, 0, minute=i, duration=30)
                for i in range(future_count)]
    model = ChannelScheduleModel(half_life_days=None, slot_mode="channel", prior_strength=4,
                                 programme_blend=1, max_repeat_multiplier=2).fit(
                                     context([a], [(0, a)], forecast))
    assert model.score(0, np.array([0]))[0] == pytest.approx(2)
    assert model.diagnostics["last_query"]["repeat_caps_applied"] == 1
    assert model.diagnostics["forecast_items_repeat_multiplier_capped"] == 1


def test_end_decay_and_inclusive_window_are_identical_for_counts_and_opportunities():
    cutoff = MONDAY + 28 * DAY
    ends = [cutoff, cutoff - 2 * DAY, cutoff - 2 * DAY - 1, cutoff - 3 * DAY]
    historical = [Airing(0, "c", end - 1800, end) for end in ends]
    ctx = context(historical, [(0, a) for a in historical], [airing(2, "c", 28)],
                  history_start=cutoff - 3 * DAY)
    model = ChannelScheduleModel(half_life_days=2, lookback_days=2,
                                 slot_mode="channel", prior_strength=2, programme_blend=0).fit(ctx)
    # Ages 0 and exactly 2 days give 1 + 1/2. One second older is excluded;
    # end exactly at history_start is also excluded. No forecast views enter.
    assert_allclose(model.historical_item_opportunities_, [1.5, 0, 0, 0, 0])
    assert model.programme_counts_[0, 0] == pytest.approx(1.5)
    assert model.channel_opportunities_[0] == pytest.approx(1.5)
    assert model.channel_counts_[0, 0] == pytest.approx(1.5)
    assert model.score(0, np.array([2]))[0] == pytest.approx(3 / 7)
    assert model.diagnostics["completion_events_outside_weight_window"] == 2
    assert model.diagnostics["unmatched_completion_events_excluded"] == 0


def test_start_slot_is_not_end_slot_at_hour_and_day_boundary():
    a = airing(0, "c", 0, 23, minute=50, duration=1200)  # Monday -> Tuesday.
    ctx = context([a], [(0, a)],
                  [airing(2, "c", 28, 23), airing(3, "c", 29, 0)])
    model = ChannelScheduleModel(half_life_days=None, programme_blend=0).fit(ctx)
    assert model.weekhour_counts_[0, 23] == 1
    assert model.weekhour_counts_[0, 24] == 0
    scores = model.score(0, np.array([2, 3]))
    assert scores[0] > 0 and np.isnan(scores[1])


def test_duplicate_airings_and_completed_events_are_counted_once():
    a, f = airing(0, "c", 0), airing(2, "c", 28)
    model = ChannelScheduleModel(half_life_days=None, slot_mode="channel",
                                 prior_strength=1, programme_blend=0).fit(
                                     context([a, a], [(0, a), (0, a)], [f, f]))
    assert model.score(0, np.array([2]))[0] == pytest.approx(1 / 2)
    assert model.diagnostics["historical_duplicates_ignored"] == 1
    assert model.diagnostics["duplicate_completion_events_ignored"] == 1
    assert model.diagnostics["forecast_duplicates_ignored"] == 1


def test_candidate_order_duplicates_unknown_indices_and_raw_id_separation():
    a, b = airing(0, "c", 0), airing(1, "d", 0)
    ids = np.array([9007199254740993, 707, 902, 55, 8888], dtype=np.int64)
    ctx = context([a, b], [(0, a), (0, b)],
                  [airing(2, "c", 28), airing(3, "d", 28)], item_ids=ids)
    model = ChannelScheduleModel(half_life_days=None, slot_mode="channel", programme_blend=0).fit(ctx)
    indices = np.array([3, 2, 3, -1, 5, 4, 2])
    expected = np.array([1 / 5, 1 / 5, 1 / 5, np.nan, np.nan, np.nan, 1 / 5])
    assert_allclose(model.score(0, indices), expected, equal_nan=True)
    permutation = np.array([6, 4, 1, 3, 0, 5, 2])
    assert_allclose(model.score(0, indices[permutation]), expected[permutation], equal_nan=True)
    assert np.isnan(model.score(0, ids)).all()
    assert_array_equal(model.item_ids_, ids)
    assert np.isnan(model.score(0, np.array([2**64 - 1], dtype=np.uint64))).all()


def test_no_internal_popularity_for_cold_or_unknown_users():
    a = airing(0, "c", 0)
    model = ChannelScheduleModel().fit(context([a], [(0, a)], [airing(2, "c", 28)]))
    for user in (1, 2, -1, 3, 999):
        assert np.isnan(model.score(user, np.array([0, 2, -1, 5]))).all()
    assert model.score(np.int64(0), np.array([2]))[0] > 0


@pytest.mark.parametrize("kind", ["no_events", "no_forecast", "no_historical_schedule", "everything_empty"])
def test_empty_histories_forecasts_and_missing_opportunities_are_safe(kind):
    a = airing(0, "c", 0)
    historical = [] if kind in {"no_historical_schedule", "everything_empty"} else [a]
    completed = [] if kind in {"no_events", "everything_empty"} else [(0, a)]
    forecast = [] if kind in {"no_forecast", "everything_empty"} else [airing(2, "c", 28)]
    model = ChannelScheduleModel().fit(context(historical, completed, forecast))
    assert np.isnan(model.score(0, np.array([0, 1, 2, 3, 4]))).all()
    if kind == "no_historical_schedule":
        assert model.diagnostics["unmatched_completion_events_excluded"] == 1
    assert model.score(0, np.array([], dtype=np.int64)).shape == (0,)


def test_empty_catalog_and_user_axis_are_safe():
    model = ChannelScheduleModel().fit(context(n_users=0, n_items=0))
    assert model.forecast_group_items_.shape == (0, 0)
    assert model.score(0, np.array([], dtype=np.int64)).shape == (0,)
    assert np.isnan(model.score(-1, np.array([0, -1]))).all()


def test_refit_discards_old_channels_counts_ids_and_coverage():
    a = airing(0, "old", 0)
    model = ChannelScheduleModel().fit(context([a], [(0, a)], [airing(2, "old", 28)]))
    assert model.score(0, np.array([2]))[0] > 0
    new = context(forecast=[airing(1, "new", 28)], n_users=1, n_items=2, item_ids=[8, 7])
    assert model.fit(new) is model
    assert model.channel_ids_ == ("new",)
    assert_array_equal(model.item_ids_, [8, 7])
    assert np.isnan(model.score(0, np.array([0, 1, 2]))).all()
    assert model.programme_counts_.shape == (1, 2) and model.programme_counts_.nnz == 0
    assert model.diagnostics["active_users"] == 0
    assert "last_query" in model.diagnostics


def test_parameter_edits_only_take_effect_on_refit():
    a = airing(0, "known", 0)
    ctx = context([a], [(0, a)], [airing(0, "known", 28), airing(2, "known", 28)])
    model = ChannelScheduleModel(half_life_days=None).fit(ctx)
    original = model.score(0, np.array([0, 2]))
    model.slot_mode = "channel"
    model.prior_strength = 2
    model.programme_blend = 0
    model.max_repeat_multiplier = 0.1
    assert_allclose(model.score(0, np.array([0, 2])), original)
    model.fit(ctx)
    assert_allclose(model.score(0, np.array([0, 2])), [1 / 3, 1 / 3])
    assert model.diagnostics["parameters"]["slot_mode"] == "channel"


def test_history_start_clips_behavior_and_opportunities_even_without_lookback():
    old, recent = airing(0, "known", 0), airing(0, "known", 3)
    ctx = context([recent], [(0, old), (0, recent)], [airing(2, "known", 4)],
                  history_start=MONDAY + 2 * DAY, train_end=MONDAY + 4 * DAY,
                  forecast_end=MONDAY + 5 * DAY)
    model = ChannelScheduleModel(half_life_days=None, slot_mode="channel", prior_strength=1,
                                 programme_blend=0).fit(ctx)
    assert model.programme_counts_[0, 0] == 1
    assert model.historical_item_opportunities_[0] == 1
    assert model.score(0, np.array([2]))[0] == pytest.approx(1 / 2)
    assert model.diagnostics["completion_events_outside_weight_window"] == 1
    assert model.diagnostics["unmatched_completion_events_excluded"] == 0


def test_eight_synthetic_future_flags_are_score_irrelevant_without_global_item_removal():
    a = airing(0, "known", 0)
    flags = [airing(2, "never-observed", 28, i, quarantined=True) for i in range(8)]
    # Same raw program also has an ordinary airing on the preferred channel.
    # Removing its ID globally would incorrectly lose this cold-program evidence.
    future = flags + [airing(2, "known", 28), airing(3, "known", 29)]
    ctx = context([a], [(0, a)], future)
    excluded = ChannelScheduleModel(half_life_days=None, slot_mode="channel").fit(ctx)
    included = ChannelScheduleModel(half_life_days=None, slot_mode="channel",
                                    quarantined_future_policy="include").fit(ctx)
    assert_allclose(excluded.score(0, np.arange(5)), included.score(0, np.arange(5)), equal_nan=True)
    assert excluded.score(0, np.array([2]))[0] > 0
    assert excluded.forecast_item_opportunities_[2] == 1
    assert included.forecast_item_opportunities_[2] == 9
    for model in (excluded, included):
        assert model.diagnostics["quarantined_forecast_airings"] == 8
        assert model.diagnostics["quarantine_changes_opportunity_counts"]
        assert model.diagnostics["quarantine_score_irrelevance_proven"]
        assert model.diagnostics["candidate_ids_removed"] == 0
    assert excluded.diagnostics["quarantined_airings_excluded"] == 8
    assert included.diagnostics["quarantined_airings_excluded"] == 0


def test_flagged_airing_with_training_support_is_not_claimed_irrelevant():
    a = airing(0, "known", 0)
    ctx = context([a], [(0, a)], [airing(2, "known", 28, quarantined=True)])
    excluded = ChannelScheduleModel().fit(ctx)
    included = ChannelScheduleModel(quarantined_future_policy="include").fit(ctx)
    assert np.isnan(excluded.score(0, np.array([2]))[0])
    assert included.score(0, np.array([2]))[0] > 0
    assert not excluded.diagnostics["quarantine_score_irrelevance_proven"]
    assert excluded.diagnostics["quarantined_airings_with_any_training_support"] == 1


def test_fit_and_score_do_not_mutate_shared_arrays_csr_metadata_or_schedules():
    a = airing(0, "known", 0)
    ctx = context([a], [(0, a)], [airing(2, "known", 28)])
    arrays = {name: getattr(ctx, name).copy() for name in (
        "candidate_indices", "event_users", "event_items", "event_starts", "event_ends", "event_channels")}
    original_data = ctx.data.interactions.copy()
    original_ids, metadata = ctx.data.item_ids.copy(), deepcopy(ctx.data.metadata)
    schedules = (ctx.historical_schedules, ctx.forecast_schedules)
    model = ChannelScheduleModel().fit(ctx)
    candidates = np.array([2, 0, 2, -1])
    model.score(0, candidates)
    for name, original in arrays.items():
        assert_array_equal(getattr(ctx, name), original)
        assert not getattr(ctx, name).flags.writeable
    assert_array_equal(ctx.data.item_ids, original_ids)
    for field in ("data", "indices", "indptr"):
        assert_array_equal(getattr(ctx.data.interactions, field), getattr(original_data, field))
    assert ctx.data.metadata == metadata
    assert (ctx.historical_schedules, ctx.forecast_schedules) == schedules
    assert_array_equal(candidates, [2, 0, 2, -1])


def test_fit_uses_sparse_counts_and_no_files_network_or_dense_user_item_arrays(monkeypatch):
    a = airing(0, "known", 0)
    ctx = context([a], [(0, a)], [airing(2, "known", 28)])

    def forbidden(*args, **kwargs):
        raise AssertionError("fit must not read files/network or densify a sparse matrix")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(socket, "socket", forbidden)
        patch.setattr(csr_matrix, "toarray", forbidden)
        model = ChannelScheduleModel().fit(ctx)
    for name in ("channel_counts_", "hour_counts_", "weekhour_counts_", "channel_hour_counts_",
                 "channel_weekhour_counts_", "programme_counts_", "forecast_group_items_"):
        assert issparse(getattr(model, name))
    assert model.forecast_group_items_.shape == (168, 5)
    assert model.forecast_group_items_.nnz == 1
    assert model.score(0, np.array([2]))[0] > 0


@pytest.mark.parametrize("kwargs", [
    {"half_life_days": 0}, {"half_life_days": np.inf}, {"half_life_days": True},
    {"lookback_days": -1}, {"lookback_days": np.nan}, {"lookback_days": "7"},
    {"prior_strength": 0}, {"prior_strength": np.inf}, {"prior_strength": 1j},
    {"max_repeat_multiplier": 0}, {"max_repeat_multiplier": np.inf},
    {"programme_blend": -0.1}, {"programme_blend": 1.1}, {"programme_blend": np.nan},
    {"programme_blend": True}, {"slot_mode": "local-time"},
    {"quarantined_future_policy": "remove-items"},
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ChannelScheduleModel(**kwargs)


@pytest.mark.parametrize("candidates", [[0.0], [True], ["0"], [[0, 1]]])
def test_malformed_candidate_indices_are_rejected(candidates):
    model = ChannelScheduleModel().fit(context())
    with pytest.raises(ValueError):
        model.score(0, np.array(candidates))


@pytest.mark.parametrize("user", [True, 0.0, "0"])
def test_malformed_users_are_rejected(user):
    model = ChannelScheduleModel().fit(context())
    with pytest.raises(ValueError):
        model.score(user, np.array([0]))


def test_score_requires_fit_and_mutated_parameters_are_revalidated():
    model = ChannelScheduleModel()
    with pytest.raises(RuntimeError, match="Fit"):
        model.score(0, np.array([0]))
    model.prior_strength = 0
    with pytest.raises(ValueError):
        model.fit(context())
    with pytest.raises(RuntimeError):
        model.score(0, np.array([0]))


def test_future_behavior_is_rejected_at_the_context_boundary():
    future = airing(0, "c", 28)
    with pytest.raises(ValueError, match="Future or invalid behavior"):
        context([], [(0, future)], [future])
