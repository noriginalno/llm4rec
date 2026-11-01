"""Independent temporal-context checks; all fixtures are synthetic.

Strict xfails document reproduced API limitations, not failures observed on the
normalized official cache. Remove each marker when the parent-owned contract is
fixed. No model, external source, credentials, or prepared-cache writes are used.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from research_v2.contracts import (
    Airing,
    TemporalContext,
    candidate_array,
    make_context,
    temporal_matrix,
    temporal_weights,
    valid_user,
)
from sweettv.data import PreparedData, build_fold


DAY = 86_400
HISTORY_START = 1_583_712_000
TRAIN_END = HISTORY_START + 10 * DAY
FORECAST_END = TRAIN_END + 4 * DAY
BIG_USER = "18446744073709551615"


def schedule(name, start, end, item, channel="0007", **extra):
    return {
        "airing_id": name,
        "channel_id": channel,
        "start": start,
        "end": end,
        "tv_show_id": item,
        "metadata": {"tv_show_title": name},
        "quarantined": False,
        **extra,
    }


def event(row, user=BIG_USER):
    return {
        **{key: row[key] for key in ("airing_id", "start", "end", "tv_show_id")},
        "user_id": user,
        "coverage": 0.8,
    }


@pytest.fixture
def prepared():
    past = [
        schedule("old", TRAIN_END - 3 * DAY - 100, TRAIN_END - 3 * DAY, 101),
        schedule("middle", TRAIN_END - DAY - 100, TRAIN_END - DAY, 101),
        schedule("at-cutoff", TRAIN_END - 100, TRAIN_END, 303),
    ]
    future = [
        schedule("forecast-repeat", TRAIN_END, TRAIN_END + 100, 101),
        schedule("forecast-cold", TRAIN_END + 100, TRAIN_END + 200, 90_000_001),
        schedule("forecast-other", TRAIN_END + 200, TRAIN_END + 300, 303),
        schedule("crossing-end", FORECAST_END - 50, FORECAST_END + 50, 70_009),
        schedule("too-late", FORECAST_END, FORECAST_END + 100, 99_999),
    ]
    return PreparedData(
        ["cold", BIG_USER, "0007"],
        past + future,
        [*(event(row) for row in past), event(past[1], "0007"),
         event(future[0]), event(future[1]), event(future[2], "cold")],
        {"fixture": "synthetic, not competition data"},
    )


def context_for(prepared, **kwargs):
    return make_context(
        prepared, TRAIN_END, TRAIN_END, FORECAST_END,
        history_start=kwargs.pop("history_start", HISTORY_START), **kwargs,
    )


def assert_same_fit_inputs(left, right):
    """Compare every exposed input, not just a production fingerprint helper."""
    assert left.data.user_ids == right.data.user_ids
    assert_array_equal(left.data.item_ids, right.data.item_ids)
    assert_array_equal(left.data.interactions.toarray(), right.data.interactions.toarray())
    assert left.data.metadata == right.data.metadata
    assert left.data.provenance == right.data.provenance
    for field in fields(TemporalContext):
        if field.name == "data":
            continue
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, np.ndarray):
            assert_array_equal(a, b)
        else:
            assert a == b


@pytest.mark.parametrize("mutation", ["delete", "reassign", "heldout-only-user"])
def test_future_label_mutations_do_not_change_any_fit_input(prepared, mutation):
    original = deepcopy(prepared)
    first, first_labels, _ = context_for(prepared)
    changed = deepcopy(prepared)
    changed.events = [e for e in changed.events if e["end"] <= TRAIN_END]
    if mutation == "reassign":
        changed.events += [event(row, "cold") for row in changed.schedules[3:5]]
    elif mutation == "heldout-only-user":
        changed.events += [event(row, "never-a-training-user") for row in changed.schedules[3:6]]
    second, second_labels, second_audit = context_for(changed)
    assert first_labels != second_labels  # Mutation really reaches evaluation labels.
    assert_same_fit_inputs(first, second)
    assert "never-a-training-user" not in second.data.user_ids
    if mutation == "heldout-only-user":
        assert second_audit["heldout_users_outside_cohort"] == 1
    assert prepared == original


def test_future_coverage_and_event_order_are_not_fit_features(prepared):
    first, _, _ = context_for(prepared)
    changed = deepcopy(prepared)
    for row in changed.events:
        if row["start"] >= TRAIN_END:
            row["coverage"] = 0.999
    # Keep the training order fixed: fit-input event order itself is not canonical.
    past = [row for row in changed.events if row["end"] <= TRAIN_END]
    future = [row for row in changed.events if row["start"] >= TRAIN_END]
    changed.events = past + future[::-1]
    second, _, _ = context_for(changed)
    assert_same_fit_inputs(first, second)


def test_cutoff_crossing_behavior_is_not_training_even_when_airing_starts_before_cutoff(prepared):
    crossing = schedule("crossing-cutoff", TRAIN_END - 50, TRAIN_END + 50, 505, "other")
    prepared.schedules.append(crossing)
    without, _, _ = context_for(prepared)
    prepared.events.append(event(crossing, "never-a-training-user"))
    with_crossing, _, audit = context_for(prepared)
    assert_same_fit_inputs(without, with_crossing)
    assert len(with_crossing.event_users) == 4
    assert audit["excluded_boundary_event_count"] == 1


def test_metadata_starting_after_forecast_end_cannot_expand_this_fit_catalog(prepared):
    first, _, _ = context_for(prepared)
    changed = deepcopy(prepared)
    late = schedule("unknown-this-fold", FORECAST_END + DAY, FORECAST_END + DAY + 100,
                    123_456_789, "other")
    changed.schedules.append(late)
    changed.events.append(event(late, "never-a-training-user"))
    second, _, _ = context_for(changed)
    assert_same_fit_inputs(first, second)


def test_training_and_forecast_metadata_are_allowed_but_labels_are_separate(prepared):
    context, labels, audit = context_for(prepared)
    names = {field.name for field in fields(context)}
    assert not names & {"labels", "heldout_events", "prepared", "audit"}
    assert context.data.user_ids == ["0007", BIG_USER, "cold"]
    assert int(BIG_USER) > 2**63 - 1
    assert context.data.metadata[context.data.item_ids.tolist().index(101)] == {
        "tv_show_title": "old",
    }
    assert context.data.metadata[context.data.item_ids.tolist().index(90_000_001)] == {
        "tv_show_title": "forecast-cold",
    }
    assert 99_999 not in context.data.item_ids
    assert labels[BIG_USER] == [101, 90_000_001]
    assert audit["train_event_count"] == 4


def test_fixed_forecast_metadata_may_change_features_without_behavior_leakage(prepared):
    first, first_labels, _ = context_for(prepared)
    changed = deepcopy(prepared)
    changed.schedules[4]["metadata"]["tv_show_title"] = "permitted revised static title"
    second, second_labels, _ = context_for(changed)
    assert first_labels == second_labels
    assert_array_equal(first.data.interactions.toarray(), second.data.interactions.toarray())
    assert first.data.metadata != second.data.metadata


def test_raw_counts_match_independent_event_counter_and_shared_fold(prepared):
    context, labels, audit = context_for(prepared)
    data, candidates, shared_labels, shared_audit = build_fold(
        prepared, TRAIN_END, TRAIN_END, FORECAST_END,
    )
    counted = Counter(
        (e["user_id"], e["tv_show_id"]) for e in prepared.events if e["end"] <= TRAIN_END
    )
    expected = np.zeros(context.data.interactions.shape, dtype=float)
    for (user, item), count in counted.items():
        expected[context.data.user_ids.index(user), context.data.item_ids.tolist().index(item)] = count
    assert_array_equal(context.data.interactions.toarray(), expected)
    assert_array_equal(temporal_matrix(context).toarray(), expected)
    assert_array_equal(data.interactions.toarray(), expected)
    assert_array_equal(context.candidate_indices, candidates)
    assert labels == shared_labels and audit == shared_audit
    assert len(context.event_users) == int(expected.sum()) == 4
    assert context.data.interactions.dtype == np.float32  # Preserve raw-count contract.


def test_temporal_weights_use_airing_end_and_seconds_not_start_or_milliseconds(prepared):
    context, _, _ = context_for(prepared)
    assert_allclose(temporal_weights(context, half_life_days=1), [0.125, 0.5, 1, 0.5])
    assert_array_equal(temporal_weights(context), [1, 1, 1, 1])
    assert context.event_ends[2] == TRAIN_END


def test_lookback_includes_exact_end_boundary_and_decay_does_not_rescale_counts(prepared):
    context, _, _ = context_for(prepared)
    assert_allclose(temporal_weights(context, 1, 1), [0, 0.5, 1, 0.5])
    assert_array_equal(temporal_weights(context, lookback_days=1), [0, 1, 1, 1])
    expected = np.zeros(context.data.interactions.shape)
    item = context.data.item_ids.tolist().index(101)
    expected[context.data.user_ids.index(BIG_USER), item] = 0.625
    expected[context.data.user_ids.index("0007"), item] = 0.5
    expected[context.data.user_ids.index(BIG_USER), context.data.item_ids.tolist().index(303)] = 1
    assert_allclose(temporal_matrix(context, half_life_days=1).toarray(), expected)
    assert context.data.interactions.sum() == 4


@pytest.mark.parametrize("argument", ["half_life_days", "lookback_days"])
@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_temporal_hyperparameters_fail_closed(prepared, argument, bad):
    context, _, _ = context_for(prepared)
    with pytest.raises(ValueError, match="finite and positive"):
        temporal_weights(context, **{argument: bad})


def test_empty_behavior_preserves_cohort_candidates_and_sparse_shape(prepared):
    prepared.events = []
    context, labels, audit = context_for(prepared)
    assert len(context.event_users) == len(context.event_channels) == 0
    assert temporal_weights(context).shape == (0,)
    assert temporal_matrix(context, 1, 1).shape == context.data.interactions.shape
    assert temporal_matrix(context).nnz == 0
    assert set(labels) == set(prepared.template_users)
    assert audit["train_event_count"] == 0


def test_context_array_snapshots_are_read_only_and_do_not_alias_callers(prepared):
    context, _, _ = context_for(prepared)
    for name in ("candidate_indices", "event_users", "event_items", "event_starts",
                 "event_ends", "event_channels"):
        value = getattr(context, name)
        assert value.ndim == 1 and not value.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            value[0] = value[0]
        supplied = value.copy()
        copied = replace(context, **{name: supplied})
        assert not np.shares_memory(supplied, getattr(copied, name))
        supplied[0] = "changed" if name == "event_channels" else int(supplied[0]) + 1
        assert_array_equal(getattr(copied, name), value)
    assert context.event_channels.tolist() == ["0007"] * 4
    with pytest.raises(FrozenInstanceError):
        context.train_end = TRAIN_END + 1


def test_airing_records_are_frozen_and_use_matrix_indices(prepared):
    context, _, _ = context_for(prepared)
    first = context.historical_schedules[0]
    assert first.item_index == context.data.item_ids.tolist().index(101)
    assert first.item_index != 101
    with pytest.raises(FrozenInstanceError):
        first.item_index = -1
    assert isinstance(context.historical_schedules, tuple)
    assert isinstance(context.forecast_schedules, tuple)


@pytest.mark.parametrize("name", ["candidate_indices", "event_users", "event_items",
                                 "event_starts", "event_ends"])
def test_numeric_context_arrays_reject_scalar_or_matrix_shapes(prepared, name):
    context, _, _ = context_for(prepared)
    for value in (np.array(0), np.array([[0]])):
        with pytest.raises(ValueError, match="one-dimensional"):
            replace(context, **{name: value})


def test_context_rejects_future_event_ends_and_out_of_range_indices(prepared):
    context, _, _ = context_for(prepared)
    ends = context.event_ends.copy()
    ends[0] = TRAIN_END + 1
    with pytest.raises(ValueError, match="Future or invalid behavior"):
        replace(context, event_ends=ends)
    for name, limit in (("event_users", len(context.data.user_ids)),
                        ("event_items", len(context.data.item_ids)),
                        ("candidate_indices", len(context.data.item_ids))):
        for bad in (-1, limit):
            value = getattr(context, name).copy()
            value[0] = bad
            with pytest.raises(ValueError, match="index out of range"):
                replace(context, **{name: value})


def test_context_rejects_invalid_chronology_and_schedule_end_boundaries(prepared):
    context, _, _ = context_for(prepared)
    with pytest.raises(ValueError, match="Invalid context time boundaries"):
        replace(context, history_start=TRAIN_END)
    with pytest.raises(ValueError, match="Historical opportunities"):
        replace(context, historical_schedules=(Airing(0, "c", TRAIN_END - 1, TRAIN_END + 1),))
    with pytest.raises(ValueError, match="Historical opportunities"):
        replace(context, historical_schedules=(Airing(0, "c", TRAIN_END - 2, TRAIN_END - 1, True),))
    with pytest.raises(ValueError, match="Forecast opportunities"):
        replace(context, forecast_schedules=(Airing(0, "c", FORECAST_END - 1, FORECAST_END + 1),))


def test_crossing_candidate_policy_is_wider_than_contained_forecast_features(prepared):
    context, _, _ = context_for(prepared)
    crossing = context.data.item_ids.tolist().index(70_009)
    assert crossing in context.candidate_indices
    assert crossing not in {a.item_index for a in context.forecast_schedules}
    # Characterization, not an asserted bug: labels also require containment.


def test_history_start_does_not_apply_behavioral_lookback_implicitly(prepared):
    full, _, _ = context_for(prepared)
    recent, _, _ = context_for(prepared, history_start=TRAIN_END - DAY)
    assert_array_equal(full.data.interactions.toarray(), recent.data.interactions.toarray())
    assert_array_equal(full.event_ends, recent.event_ends)
    assert len(full.historical_schedules) == 3 and len(recent.historical_schedules) == 1
    # Consumers must align effective weighted history with their opportunity denominator.


def test_candidate_query_keeps_read_only_permutations_duplicates_and_empty_vectors():
    supplied = np.array([2, 0, 2], dtype=np.int64)
    supplied.setflags(write=False)
    assert_array_equal(candidate_array(supplied), [2, 0, 2])
    assert not supplied.flags.writeable
    assert candidate_array(np.array([], dtype=float)).shape == (0,)


@pytest.mark.parametrize("bad", [np.array(2), [[1]], [1.0], [True], ["1"]])
def test_candidate_query_rejects_noninteger_or_nonscalar_vector_representations(bad):
    with pytest.raises(ValueError, match="one-dimensional integer"):
        candidate_array(bad)


@pytest.mark.parametrize("value, expected", [(0, True), (np.int64(1), True), (-1, False),
                                           (3, False), (2**100, False)])
def test_scalar_user_indices_do_not_alias_unknown_users(value, expected):
    assert valid_user(value, 3) == expected


@pytest.mark.parametrize("bad", [True, np.bool_(False), 0.0, "0", None, [0], np.array(0)])
def test_noninteger_scalar_user_indices_fail_explicitly(bad):
    with pytest.raises(ValueError, match="integer matrix index"):
        valid_user(bad, 3)


def test_duplicate_schedule_rows_should_not_inflate_airing_opportunities(prepared):
    first, _, _ = context_for(prepared)
    prepared.schedules += [deepcopy(prepared.schedules[0]), deepcopy(prepared.schedules[3])]
    duplicated, _, audit = context_for(prepared)
    assert audit["schedule_audit"]["duplicate_airing_rows"] == 2
    assert (len(duplicated.historical_schedules), len(duplicated.forecast_schedules)) == (
        len(first.historical_schedules), len(first.forecast_schedules),
    )


def test_duplicate_training_event_should_match_shared_fold_deduplication(prepared):
    prepared.events.append(deepcopy(prepared.events[0]))
    shared, _, _, audit = build_fold(prepared, TRAIN_END, TRAIN_END, FORECAST_END)
    assert shared.interactions.sum() == 4 and audit["duplicate_events_ignored"] == 1
    context, _, _ = context_for(prepared)
    assert_array_equal(temporal_matrix(context).toarray(), shared.interactions.toarray())


def test_event_channel_matrix_should_not_be_accepted_as_one_channel_per_event(prepared):
    context, _, _ = context_for(prepared)
    with pytest.raises(ValueError, match="one-dimensional"):
        replace(context, event_channels=context.event_channels[:, None])


def test_direct_context_should_reject_fractional_candidate_indices(prepared):
    context, _, _ = context_for(prepared)
    with pytest.raises(ValueError):
        replace(context, candidate_indices=np.array([0.9]))
