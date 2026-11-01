"""Independent EASE challenge: tiny synthetic matrices, no quality search/I/O.

The rational oracle uses Fraction Gauss-Jordan inversion, not the author's
SciPy solve or test helpers. Transformed features use separate target-wise
ridge regressions. Strict xfails record reproduced boundary inconsistencies;
they are not assertions that the frozen-cache experiment was corrupted.
"""

from __future__ import annotations

import builtins
from collections import Counter
from copy import deepcopy
from dataclasses import fields, replace
from fractions import Fraction
import io
from itertools import combinations
import json
import math
import os
import socket

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import csr_matrix

from research_v2.collaborative import EASEModel
from research_v2.contracts import Airing, TemporalContext, make_context
from research_v2.evaluate import label_frequencies, tie_marginalized_ap
from sweettv.contracts import TrainingData
from sweettv.data import PreparedData
from sweettv.evaluation import rank_candidates


DAY = 86_400
CUTOFF = 1_600_000_000


def context_from_counts(values, *, ids=None, ages=None, candidates=None):
    """Expand small integer counts into event coordinates, without model helpers."""
    counts = np.asarray(values, dtype=np.int64)
    n_users, n_items = counts.shape
    assert np.all(counts >= 0)
    ids = np.array(ids if ids is not None else [9001 + 11 * j for j in range(n_items)])
    users, items, starts, ends = [], [], [], []
    for u in range(n_users):
        for j in range(n_items):
            for repeat in range(int(counts[u, j])):
                age_days = 0 if ages is None else ages(u, j, repeat)
                end = CUTOFF - round(age_days * DAY)
                users.append(u)
                items.append(j)
                starts.append(end - 60)
                ends.append(end)
    data = TrainingData(
        ["18446744073709551615" if u == 0 else f"user-{u}" for u in range(n_users)],
        ids, csr_matrix(counts), [{"title": f"metadata-{j}"} for j in range(n_items)],
        {"train_end": CUTOFF, "behavior_max_end": max(ends, default=None)},
    )
    return TemporalContext(
        data, np.arange(n_items) if candidates is None else np.array(candidates),
        CUTOFF - 100 * DAY, CUTOFF, CUTOFF, CUTOFF + 4 * DAY,
        np.array(users, dtype=np.int64), np.array(items, dtype=np.int64),
        np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64),
        np.array(["0007"] * len(users)),
        tuple(Airing(j, f"channel-{j}", CUTOFF - 1000, CUTOFF - 900) for j in range(n_items)),
        tuple(Airing(j, f"channel-{j}", CUTOFF + 120 * j, CUTOFF + 120 * j + 60)
              for j in range(n_items)),
    )


def fraction_inverse_ease(values, regularization):
    """Exact inverse, B, and reconstructed scores, without NumPy/SciPy algebra."""
    x = [[Fraction(int(v)) for v in row] for row in values]
    n = len(x[0])
    ridge = Fraction(str(regularization))
    augmented = [
        [sum(row[i] * row[j] for row in x) + (ridge if i == j else 0)
         for j in range(n)] + [Fraction(i == j) for j in range(n)]
        for i in range(n)
    ]
    for column in range(n):
        pivot = next(row for row in range(column, n) if augmented[row][column])
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row != column:
                multiple = augmented[row][column]
                augmented[row] = [a - multiple * b
                                  for a, b in zip(augmented[row], augmented[column])]
    precision = [row[n:] for row in augmented]
    coefficients = [[Fraction(0) if i == j else -precision[i][j] / precision[j][j]
                     for j in range(n)] for i in range(n)]
    scores = [[sum(row[i] * coefficients[i][j] for i in range(n))
               for j in range(n)] for row in x]
    return (np.array(precision, dtype=float), np.array(coefficients, dtype=float),
            np.array(scores, dtype=float))


def independently_transformed_features(context, *, transform="raw", half_life=None,
                                     idf=False, row_l2=False, cap=4096):
    shape = context.data.interactions.shape
    cells = {}
    for u, j, end in zip(context.event_users, context.event_items, context.event_ends):
        age = (context.train_end - int(end)) / DAY
        cells.setdefault((int(u), int(j)), []).append(
            1.0 if half_life is None else 2.0 ** (-age / half_life)
        )
    source = np.zeros(shape, dtype=float)
    for (u, j), values in cells.items():
        source[u, j] = math.fsum(values)
    raw_mass = context.data.interactions.sum(axis=0).A1
    support = [sum(source[u, j] > 0 for u in range(shape[0])) for j in range(shape[1])]
    effective_users = sum(any(v > 0 for v in row) for row in source)
    eligible = [j for j, count in enumerate(support) if count]
    active_items = sorted(sorted(
        eligible, key=lambda j: (-float(raw_mass[j]), int(context.data.item_ids[j])),
    )[:cap])
    active_users = [u for u in range(shape[0]) if any(source[u, j] > 0 for j in active_items)]
    weights = [1 + math.log((1 + effective_users) / (1 + support[j])) if idf else 1
               for j in active_items]
    features = []
    for u in active_users:
        row = []
        for position, j in enumerate(active_items):
            value = source[u, j]
            if transform == "log":
                value = math.log1p(value)
            elif transform == "binary":
                value = float(value > 0)
            row.append(value * weights[position])
        norm = math.hypot(*row) if row_l2 else 1
        features.append([value / norm for value in row])
    return (np.array(active_users), np.array(active_items),
            np.array(features).reshape(len(active_users), len(active_items)), np.array(weights))


def separate_target_ridge(features, regularization):
    """Solve the constrained objective as separate regressions, not normalized P."""
    n_items = features.shape[1]
    coefficients = np.zeros((n_items, n_items))
    for target in range(n_items):
        predictors = [j for j in range(n_items) if j != target]
        design = features[:, predictors]
        coefficients[predictors, target] = np.linalg.solve(
            design.T @ design + regularization * np.eye(len(predictors)),
            design.T @ features[:, target],
        )
    return coefficients


def evidence_scores(features, coefficients):
    values = features @ coefficients
    for u, row in enumerate(features):
        for target in range(coefficients.shape[1]):
            if not any(value > 0 and coefficients[j, target] != 0
                       for j, value in enumerate(row)):
                values[u, target] = np.nan
    return values


def input_state(context):
    def array_state(value):
        return value.shape, value.dtype.str, value.flags.writeable, value.tobytes()

    arrays = tuple((field.name, array_state(getattr(context, field.name)))
                   for field in fields(context) if isinstance(getattr(context, field.name), np.ndarray))
    return (
        tuple(context.data.user_ids), array_state(context.data.item_ids),
        tuple(array_state(getattr(context.data.interactions, key))
              for key in ("data", "indices", "indptr")),
        context.data.interactions.shape,
        json.dumps([context.data.metadata, context.data.provenance], sort_keys=True),
        arrays, context.history_start, context.train_end, context.forecast_start,
        context.forecast_end, context.historical_schedules, context.forecast_schedules,
    )


@pytest.mark.parametrize("ridge", [0.5, 2, 13])
def test_exact_fraction_inverse_column_orientation_and_scores(ridge):
    counts = [[3, 1, 0, 2], [0, 2, 1, 0], [4, 0, 1, 1], [1, 3, 0, 2]]
    context = context_from_counts(counts, ids=[901, 7, 9_007_199_254_740_993, 84])
    precision, expected_b, expected_scores = fraction_inverse_ease(counts, ridge)
    wrong_row_normalization = -precision / np.diag(precision)[:, None]
    np.fill_diagonal(wrong_row_normalization, 0)
    assert not np.allclose(expected_b, wrong_row_normalization)  # Fixture discriminates orientation.
    model = EASEModel(regularization=ridge).fit(context)
    assert_array_equal(model.active_item_ids_, [901, 7, 9_007_199_254_740_993, 84])
    assert np.isnan(model.score(0, context.data.item_ids)).all()  # Raw IDs are not matrix indices.
    assert_allclose(model.coefficients_, expected_b, rtol=3e-13, atol=2e-15)
    assert_array_equal(np.diag(model.coefficients_), 0)
    users, candidates = np.array([3, 0, 3, 1]), np.array([3, 1, 3, 0, 2])
    users.setflags(write=False)
    candidates.setflags(write=False)
    assert_allclose(model.score_batch(users, candidates), expected_scores[np.ix_(users, candidates)],
                    rtol=3e-13, atol=2e-15)
    assert_allclose(model.score(np.int64(3), candidates), expected_scores[3, candidates],
                    rtol=3e-13, atol=2e-15)


@pytest.mark.parametrize("transform,row_l2,idf,half_life", [
    ("raw", False, False, None), ("log", False, True, None),
    ("binary", True, True, None), ("raw", True, True, 2),
    ("log", True, False, 2), ("binary", False, True, 2),
])
def test_varied_feature_spaces_against_separate_target_regressions(transform, row_l2, idf, half_life):
    counts = [[0, 0, 0, 0, 0, 0], [3, 1, 0, 2, 0, 0], [1, 0, 0, 0, 0, 0],
              [0, 2, 1, 0, 0, 0], [4, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1],
              [0, 1, 0, 2, 0, 0]]
    context = context_from_counts(counts, ages=lambda u, j, r: (u + 2 * j + r) % 5)
    rows, columns, x, weights = independently_transformed_features(
        context, transform=transform, row_l2=row_l2, idf=idf, half_life=half_life, cap=4,
    )
    expected_b = separate_target_ridge(x, 2.5)
    model = EASEModel(regularization=2.5, transform=transform, row_l2=row_l2,
                      item_weighting="idf" if idf else "none", half_life_days=half_life,
                      max_active_items=4).fit(context)
    assert_array_equal(columns, [0, 1, 2, 3])
    assert_array_equal(rows, [1, 2, 3, 4, 6])
    assert_array_equal(model.active_user_indices_, rows)
    assert_array_equal(model.active_item_indices_, columns)
    assert_allclose(model.item_weights_, weights, rtol=2e-15)
    assert_allclose(model.training_matrix_.toarray(), x, rtol=3e-15, atol=1e-16)
    assert_allclose(model.coefficients_, expected_b, rtol=3e-13, atol=2e-15)
    assert_allclose(model.score_batch(rows, columns), evidence_scores(x, expected_b),
                    rtol=3e-13, atol=2e-15, equal_nan=True)
    assert np.isnan(model.score_batch([0, 5], [0, 1, 2, 3, 4, 5])).all()
    if idf:
        assert model.diagnostics["idf_training_users"] == 6  # Includes truncated-only user5.
        assert len(set(weights)) > 1  # Unlike an equal-df fixture, tests column weighting.


def test_binary_is_count_multiplicity_invariant_but_raw_and_log_are_not():
    first = [[3, 1, 0], [0, 2, 1], [4, 0, 1]]
    second = [[9, 2, 0], [0, 3, 5], [2, 0, 7]]
    binary = [EASEModel(regularization=3, transform="binary").fit(context_from_counts(x))
              for x in (first, second)]
    assert_array_equal(binary[0].coefficients_, binary[1].coefficients_)
    assert_array_equal(binary[0].score_batch([0, 1, 2], [0, 1, 2]),
                       binary[1].score_batch([0, 1, 2], [0, 1, 2]))
    for transform in ("raw", "log"):
        a, b = [EASEModel(regularization=3, transform=transform).fit(context_from_counts(x))
                for x in (first, second)]
        assert not np.allclose(a.coefficients_, b.coefficients_)


def test_negative_scores_and_rational_cancellation_are_evidence_not_fallback():
    counts = [[2, 1, 0], [0, 1, 1], [0, 5, 1], [0, 0, 1]]
    _, b, scores = fraction_inverse_ease(counts, 3)
    assert b[1, 0] == 1 / 12 and b[2, 0] == -1 / 12
    assert scores[1, 0] == 0 and scores[3, 0] == -1 / 12
    model = EASEModel(regularization=3).fit(context_from_counts(counts))
    actual = model.score_batch([1, 3], [0])[:, 0]
    assert np.isfinite(actual).all()
    assert abs(actual[0]) < 2e-15
    assert_allclose(actual[1], -1 / 12, rtol=3e-14)
    assert np.isnan(model.score(3, [2])[0])  # Sole-feature self has no diagonal evidence.
    assert np.isfinite(model.score(0, [0])[0])  # Other history may support a watched target.


def test_disconnected_components_isolates_cold_items_and_users_have_no_pseudo_evidence():
    counts = [[4, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0],
              [0, 0, 1, 3, 0, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0]]
    model = EASEModel(regularization=3).fit(context_from_counts(counts))
    scores = model.score_batch([0, 1, 2, 3, 4, -1, 5], np.arange(6))
    assert scores[0, 1] > 0 and np.isnan(scores[0, [0, 2, 3, 4, 5]]).all()
    assert np.isfinite(scores[1, [0, 1]]).all()  # No watched-item exclusion.
    assert np.isnan(scores[3:]).all()
    assert np.isnan(scores[:, 5]).all()
    assert model.diagnostics["internal_fallback"] is False


def test_raw_id_tie_truncation_mapping_and_row_column_permutation():
    counts = np.array([[0, 4, 0, 4, 2, 4, 0], [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0]])
    ids = np.array([1, 90, 800, 7, 500, 13, 9_007_199_254_740_993])
    first = context_from_counts(counts, ids=ids, candidates=[0, 2, 6],
                                ages=lambda u, j, r: 20 if ids[j] == 13 else 0)
    model = EASEModel(regularization=3, half_life_days=1, max_active_items=2).fit(first)
    assert_array_equal(model.active_item_indices_, [3, 5])
    assert_array_equal(model.active_item_ids_, [7, 13])  # Raw mass, not decay, IDF, or candidates.
    assert_array_equal(model.active_user_indices_, [0])
    assert_array_equal(model.user_index_to_row_, [0, -1, -1, -1, -1])
    assert_array_equal(model.item_index_to_column_, [-1, -1, -1, 0, -1, 1, -1])
    assert np.isnan(model.score_batch([1, 2, 3, 4], np.arange(7))).all()
    assert np.isnan(model.score(0, [0, 1, 2, 4, 6])).all()
    rows, columns = np.array([4, 0, 3, 1, 2]), np.array([6, 5, 4, 3, 2, 1, 0])
    permuted_ids = ids[columns]
    second = context_from_counts(counts[np.ix_(rows, columns)], ids=permuted_ids,
                                 ages=lambda u, j, r: 20 if permuted_ids[j] == 13 else 0)
    other = EASEModel(regularization=3, half_life_days=1, max_active_items=2).fit(second)
    assert sorted(other.active_item_ids_) == [7, 13]
    lookup = {int(raw): j for j, raw in enumerate(permuted_ids)}
    requested_raw = [13, 7, 13, 90, 1]
    original_lookup = {int(raw): j for j, raw in enumerate(ids)}
    assert_allclose(other.score(1, [lookup[v] for v in requested_raw]),
                    model.score(0, [original_lookup[v] for v in requested_raw]), equal_nan=True)
    assert model.diagnostics["training_count_coverage"] == 8 / 18
    assert model.diagnostics["coefficient_bytes"] == 2 * 2 * 8


def test_shared_backfill_resolves_raw_id_ties_once_without_internal_votes():
    context = context_from_counts([[4, 4, 4, 0, 0]], ids=[90, 7, 13, 1, 700])
    model = EASEModel(regularization=3, max_active_items=1).fit(context)
    candidates = np.array([4, 2, 3, 0, 1])
    scores = model.score(0, candidates)
    assert np.isnan(scores).all()
    result = rank_candidates(scores, candidates, context.data.item_ids,
                             context.data.interactions.sum(axis=0).A1)
    assert result["items"] == [7, 13, 90, 1, 700]
    assert result["fallback_positions"] == 5


def prepared_history():
    counts = [[3, 1, 0], [0, 2, 1], [4, 0, 1]]
    raw_ids = [901, 7, 84]
    schedules, events = [], []
    for j, item in enumerate(raw_ids):
        for repeat in range(max(row[j] for row in counts)):
            end = CUTOFF - repeat * 120
            row = {"airing_id": f"past-{item}-{repeat}", "channel_id": f"c{j}",
                   "start": end - 60, "end": end, "tv_show_id": item, "metadata": {"title": str(item)}}
            schedules.append(row)
            for u, values in enumerate(counts):
                if repeat < values[j]:
                    events.append({**{key: row[key] for key in ("airing_id", "start", "end", "tv_show_id")},
                                   "user_id": f"u{u}"})
    for position, item in enumerate([*raw_ids, 13, 15, 17, 19]):
        row = {"airing_id": f"future-{item}", "channel_id": "future-channel",
               "start": CUTOFF + position * 120, "end": CUTOFF + position * 120 + 60,
               "tv_show_id": item, "metadata": {"title": "permitted static forecast metadata"}}
        schedules.append(row)
        events.append({**{key: row[key] for key in ("airing_id", "start", "end", "tv_show_id")},
                       "user_id": "u0"})
    return PreparedData(["u0", "u1", "u2", "cold"], schedules, events, {})


def from_prepared(prepared):
    return make_context(prepared, CUTOFF, CUTOFF, CUTOFF + 4 * DAY,
                        history_start=CUTOFF - 100 * DAY)


def test_actual_future_label_mutation_and_cold_forecast_growth_leave_warm_fit_unchanged():
    prepared = prepared_history()
    first, original_labels, _ = from_prepared(prepared)
    changed = deepcopy(prepared)
    changed.events = [row for row in changed.events if row["end"] <= CUTOFF]
    for row in changed.schedules:
        if row["start"] >= CUTOFF:
            row["metadata"] = {"irrelevant_future_hint": 10**100}
    extra = {"airing_id": "extra-forecast-only", "channel_id": "never-viewed",
             "start": CUTOFF + 2000, "end": CUTOFF + 2060, "tv_show_id": 1, "metadata": {}}
    changed.schedules.append(extra)
    changed.events.append({**{key: extra[key] for key in ("airing_id", "start", "end", "tv_show_id")},
                           "user_id": "heldout-only"})
    second, changed_labels, _ = from_prepared(changed)
    assert original_labels != changed_labels
    assert first.data.user_ids == second.data.user_ids and "heldout-only" not in second.data.user_ids
    kwargs = {"regularization": 3, "transform": "log", "item_weighting": "idf",
              "row_l2": True, "half_life_days": 2}
    models = [EASEModel(**kwargs).fit(context) for context in (first, second)]
    assert_array_equal(models[0].active_item_ids_, models[1].active_item_ids_)
    assert_array_equal(models[0].coefficients_, models[1].coefficients_)
    raw_query = [901, 7, 84, 901]
    scores = []
    for context, model in zip((first, second), models):
        lookup = {int(item): j for j, item in enumerate(context.data.item_ids)}
        users = [context.data.user_ids.index(u) for u in ["u2", "u0", "cold"]]
        scores.append(model.score_batch(users, [lookup[item] for item in raw_query]))
    assert_array_equal(scores[0], scores[1])


def test_fit_and_queries_are_read_only_and_snapshot_survives_source_mutation():
    context = context_from_counts([[3, 1, 0], [0, 2, 1], [4, 0, 1]])
    before = input_state(context)
    model = EASEModel(regularization=3).fit(context)
    users, candidates = np.array([2, 0, 2, -1]), np.array([2, 0, 2, 1])
    users.setflags(write=False)
    candidates.setflags(write=False)
    baseline = model.score_batch(users, candidates)
    assert input_state(context) == before
    assert_array_equal(users, [2, 0, 2, -1])
    assert_array_equal(candidates, [2, 0, 2, 1])
    assert not users.flags.writeable and not candidates.flags.writeable
    ids = model.active_item_ids_.copy()
    context.data.interactions.data[:] = 99
    context.data.item_ids[:] += 12345
    context.data.metadata[:] = [{"future": "not a score feature"}] * 3
    assert_array_equal(model.score_batch(users, candidates), baseline)
    assert_array_equal(model.active_item_ids_, ids)
    previous_b = model.coefficients_
    with pytest.raises(ValueError, match="disagree"):
        model.fit(context)
    assert model.coefficients_ is previous_b
    assert_array_equal(model.score_batch(users, candidates), baseline)


def test_fit_rechecks_future_event_end_after_deliberate_constructor_bypass():
    # Fault-injection defense, not a claim that make_context admits future events.
    context = context_from_counts([[3, 1, 0], [0, 2, 1], [4, 0, 1]])
    model = EASEModel(regularization=3, half_life_days=2).fit(context)
    baseline = model.score_batch([0, 2], [2, 0, 1])
    previous_b = model.coefficients_
    ends = context.event_ends.copy()
    ends[0] = CUTOFF + 1
    ends.setflags(write=False)
    object.__setattr__(context, "event_ends", ends)
    with pytest.raises(ValueError, match="Future or invalid behavior"):
        model.fit(context)
    assert model.coefficients_ is previous_b
    assert_array_equal(model.score_batch([0, 2], [2, 0, 1]), baseline)


def test_forecast_horizon_schedule_channel_and_metadata_are_not_cf_statistics():
    context = context_from_counts([[3, 1, 0], [0, 2, 1], [4, 0, 1]])
    model = EASEModel(regularization=3, item_weighting="idf").fit(context)
    altered = replace(context, forecast_start=CUTOFF + DAY, forecast_end=CUTOFF + 300 * DAY,
                      candidate_indices=np.array([2, 0]), historical_schedules=(),
                      forecast_schedules=(Airing(0, "different", CUTOFF + DAY, CUTOFF + DAY + 60, True),))
    altered.data.metadata[:] = [{"heldout_counts": 10**100}] * 3
    altered.data.provenance["future_behavior_not_an_input"] = object()
    other = EASEModel(regularization=3, item_weighting="idf").fit(altered)
    assert_array_equal(model.coefficients_, other.coefficients_)
    assert_array_equal(model.item_weights_, other.item_weights_)
    assert_array_equal(model.score_batch([0, 1, 2], [2, 0, 1]), other.score_batch([0, 1, 2], [2, 0, 1]))


def test_no_file_or_network_entry_point_is_needed_for_guarded_fit_and_score(monkeypatch):
    context = context_from_counts([[3, 1, 0], [0, 2, 1], [4, 0, 1]])
    EASEModel(regularization=3).fit(context)  # Initialize native library/controller laziness first.

    def forbidden(*args, **kwargs):
        raise AssertionError("Model attempted filesystem/network I/O during guarded fit/score")

    for module, name in ((builtins, "open"), (io, "open"), (os, "open"),
                         (np, "load"), (np, "save"), (np, "savez"),
                         (np.lib.format, "open_memmap"), (socket, "create_connection")):
        monkeypatch.setattr(module, name, forbidden)
    model = EASEModel(regularization=3, transform="log", item_weighting="idf").fit(context)
    assert np.isfinite(model.score_batch([0, 2], [2, 0, 1])).all()


def test_only_compact_item_gram_is_densified_and_scoring_bound_precedes_allocation(monkeypatch):
    values = np.zeros((17, 11), dtype=int)
    values[2, [1, 7, 9]] = [4, 3, 1]
    values[8, [1, 7]] = [1, 2]
    context = context_from_counts(values)
    original_toarray, original_full = csr_matrix.toarray, np.full

    def compact_only(matrix, *args, **kwargs):
        assert matrix.shape == (2, 2), "Global user-item densification is not permitted"
        return original_toarray(matrix, *args, **kwargs)

    def bounded_full(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2:
            assert math.prod(shape) <= 6, "Output allocated before score budget check"
        return original_full(shape, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", compact_only)
    monkeypatch.setattr(np, "full", bounded_full)
    model = EASEModel(regularization=3, max_active_items=2, max_batch_scores=6).fit(context)
    assert model.score_batch([2, 8], [7, 1, 7]).shape == (2, 3)
    assert model.score_batch([], [1, 7]).shape == (0, 2)
    assert model.score_batch([2, 8], []).shape == (2, 0)
    for users, candidates in (([2, 8], [1, 7, 1, 7]), ([-1, 99], [99, 99, 99, 99])):
        with pytest.raises(ValueError, match="max_batch_scores"):
            model.score_batch(users, candidates)
    with pytest.raises(ValueError, match="max_batch_scores"):
        model.score(2, [1] * 7)
    assert not any(isinstance(value, np.ndarray) and value.shape == values.shape
                   for value in vars(model).values())


@pytest.mark.parametrize("value", [-1, 3, 2**63, 2**100, np.uint64(2**64 - 1)])
def test_unknown_integer_scalar_users_are_nan_without_negative_or_unsigned_wrap(value):
    model = EASEModel(regularization=3).fit(context_from_counts([[3, 1], [1, 2]]))
    assert np.isnan(model.score(value, [0, 1])).all()


def test_unknown_object_and_uint64_indices_preserve_duplicate_query_positions():
    model = EASEModel(regularization=3).fit(context_from_counts([[3, 1], [1, 2]]))
    candidates = np.array([1, 2**100, -1, 0, 1], dtype=object)
    users = np.array([1, -1, 2**100, 0, 1], dtype=object)
    candidates.setflags(write=False)
    users.setflags(write=False)
    result = model.score_batch(users, candidates)
    assert_allclose(result[[0, 4]][:, [0, 3, 4]],
                    np.vstack([model.score(1, [1, 0, 1])] * 2))
    assert np.isnan(result[1:3]).all() and np.isnan(result[:, 1:3]).all()
    assert np.isnan(model.score(0, np.array([2**64 - 1], dtype=np.uint64))).all()


@pytest.mark.parametrize("bad", [True, np.bool_(True), 0.0, "0", None, np.array(0)])
def test_invalid_scalar_user_types_are_not_coerced(bad):
    model = EASEModel(regularization=3).fit(context_from_counts([[3, 1], [1, 2]]))
    with pytest.raises(ValueError):
        model.score(bad, [0, 1])


@pytest.mark.parametrize("position", ["users", "candidates"])
def test_mixed_boolean_batch_coordinates_should_follow_scalar_rejection(position):
    model = EASEModel(regularization=3).fit(context_from_counts([[3, 1], [1, 2]]))
    with pytest.raises(ValueError):
        model.score_batch([True, 0] if position == "users" else [0],
                          [True, 0] if position == "candidates" else [0])


@pytest.mark.parametrize("position", ["users", "candidates"])
def test_unknown_integer_lists_should_not_depend_on_numpy_inferred_dtype(position):
    model = EASEModel(regularization=3).fit(context_from_counts([[3, 1], [1, 2]]))
    actual = model.score_batch([-1, 2**63] if position == "users" else [0],
                               [-1, 2**63] if position == "candidates" else [0])
    assert actual.shape == ((2, 1) if position == "users" else (1, 2))
    assert np.isnan(actual).all()


def test_tie_marginalization_matches_exhaustive_without_replacement_oracle():
    frequencies = {901: 9, 7: 5, 13: 5, 15: 5, 17: 5, 19: 5, 84: 5, 999: 1}
    predictions = [7, 901, 13, 999, 17]
    scores = []
    for chosen in combinations([7, 13, 15, 17, 19, 84], 4):
        labels = {901, *chosen}
        hits, score = 0, 0.0
        for rank, item in enumerate(predictions, 1):
            if item in labels:
                hits += 1
                score += hits / rank
        scores.append(score / 5)
    assert_allclose(tie_marginalized_ap(predictions, frequencies), math.fsum(scores) / len(scores),
                    rtol=2e-15)


def test_runner_frequency_diagnostics_should_match_shared_heldout_event_deduplication():
    prepared = prepared_history()
    original, original_labels, _ = from_prepared(prepared)
    before = label_frequencies(prepared, original, ["u0"])
    duplicate = next(row for row in prepared.events
                     if row["start"] >= CUTOFF and row["tv_show_id"] == 901)
    prepared.events.append(deepcopy(duplicate))
    context, labels, audit = from_prepared(prepared)
    assert labels == original_labels and audit["duplicate_events_ignored"] == 1
    assert before["u0"] == Counter({901: 1, 7: 1, 84: 1, 13: 1, 15: 1, 17: 1, 19: 1})
    assert label_frequencies(prepared, context, ["u0"]) == before
