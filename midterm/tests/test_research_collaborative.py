"""Independent small-matrix EASE oracles; network/file-free training and scoring."""

from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import csr_matrix

from research_v2 import collaborative
from research_v2.collaborative import EASEModel
from research_v2.contracts import Airing, TemporalContext
from sweettv.contracts import TrainingData


def make_context(values, *, ids=None, candidates=None, ages=None):
    """One event per integer broadcast count, on the SAME source-clock axis."""
    values = np.asarray(values, dtype=np.float64)
    n_users, n_items = values.shape
    ids = np.arange(101, 101 + n_items) if ids is None else np.asarray(ids)
    cutoff = 2_000_000
    data = TrainingData(
        [f"user-{i:03d}" for i in range(n_users)], ids, csr_matrix(values),
        [{} for _ in range(n_items)],
        {"train_end": cutoff},
    )
    users, items, starts, ends = [], [], [], []
    for user in range(n_users):
        for item in range(n_items):
            for repeat in range(int(values[user, item])):
                age = 0 if ages is None else ages(user, item, repeat)
                end = cutoff - int(age * 86400)
                users.append(user)
                items.append(item)
                starts.append(end - 60)
                ends.append(end)
    data.provenance["behavior_max_end"] = max(ends, default=None)
    return TemporalContext(
        data=data, candidate_indices=np.arange(n_items) if candidates is None else candidates,
        history_start=-10_000_000, train_end=cutoff,
        forecast_start=cutoff, forecast_end=cutoff + 7 * 86400,
        event_users=np.asarray(users, dtype=np.int64),
        event_items=np.asarray(items, dtype=np.int64),
        event_starts=np.asarray(starts, dtype=np.int64),
        event_ends=np.asarray(ends, dtype=np.int64),
        event_channels=np.array(["channel"] * len(users)),
        historical_schedules=(), forecast_schedules=(),
    )


@pytest.fixture
def context():
    # Compact X=[[2,1,0],[0,1,3],[1,0,1]]. Interleave cold global rows/columns;
    # raw IDs are neither compact nor global indices, nor in numeric ID order.
    values = np.zeros((5, 7))
    values[np.ix_([0, 2, 4], [1, 3, 6])] = [[2, 1, 0], [0, 1, 3], [1, 0, 1]]
    return make_context(values, ids=[999, 90, 700, 5, 888, 777, 32])


def inverse3(matrix):
    """Independent 3x3 cofactor/adjugate inverse: NO production or linalg call."""
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    cofactors = [
        [e * i - f * h, f * g - d * i, d * h - e * g],
        [c * h - b * i, a * i - c * g, b * g - a * h],
        [b * f - c * e, c * d - a * f, a * e - b * d],
    ]
    determinant = a * cofactors[0][0] + b * cofactors[0][1] + c * cofactors[0][2]
    return [[cofactors[col][row] / determinant for col in range(3)] for row in range(3)]


def independent_fit3(features, regularization):
    # Python scalar accumulation and the above adjugate, not X.T@X/solve.
    gram = [
        [math.fsum(row[i] * row[j] for row in features)
         + (regularization if i == j else 0) for j in range(3)]
        for i in range(3)
    ]
    precision = inverse3(gram)
    coefficients = [
        [0.0 if i == j else -precision[i][j] / precision[j][j] for j in range(3)]
        for i in range(3)
    ]
    scores = [
        [math.fsum(row[i] * coefficients[i][j] for i in range(3)) for j in range(3)]
        for row in features
    ]
    return np.asarray(coefficients), np.asarray(scores)


def test_default_configuration():
    model = EASEModel()
    assert model.regularization == 1000
    assert model.transform == "raw"
    assert model.half_life_days is None
    assert model.row_l2 is False
    assert model.item_weighting == "none"
    assert model.max_active_items == 4096
    assert model.num_threads == 1
    assert model.max_batch_scores == 1_000_000
    assert model.diagnostics == {"status": "not_fitted", "backend": None}


def test_closed_form_rational_fixture_coefficients_scores_mapping_and_diag(context):
    before = context.data.interactions.copy()
    model = EASEModel(regularization=2)
    assert model.fit(context) is model
    # G=[[7,2,1],[2,4,3],[1,3,12]], det=233,
    # P=1/233*[[39,-21,2],[-21,83,-19],[2,-19,24]].
    # COLUMN normalization is essential: B is generally asymmetric.
    expected_b = [[0, 21 / 83, -1 / 12], [7 / 13, 0, 19 / 24], [-2 / 39, 19 / 83, 0]]
    expected_s = [[7 / 13, 42 / 83, 5 / 8], [5 / 13, 57 / 83, 19 / 24],
                  [-2 / 39, 40 / 83, -1 / 12]]
    assert_allclose(model.coefficients_, expected_b, rtol=2e-14, atol=1e-15)
    assert_array_equal(np.diag(model.coefficients_), 0)
    assert not np.allclose(model.coefficients_, model.coefficients_.T)
    assert_allclose(model.score_batch([0, 2, 4], [1, 3, 6]), expected_s, rtol=2e-14)
    assert_array_equal(model.active_user_indices_, [0, 2, 4])
    assert_array_equal(model.active_item_indices_, [1, 3, 6])
    assert_array_equal(model.active_item_ids_, [90, 5, 32])
    assert_array_equal(model.user_index_to_row_, [0, -1, 1, -1, 2])
    assert_array_equal(model.item_index_to_column_, [-1, 0, -1, 1, -1, -1, 2])
    assert isinstance(model.training_matrix_, csr_matrix)
    assert model.training_matrix_.dtype == np.float64
    assert_array_equal(model.training_matrix_.toarray(), [[2, 1, 0], [0, 1, 3], [1, 0, 1]])
    for name in ("data", "indices", "indptr"):
        assert_array_equal(getattr(context.data.interactions, name), getattr(before, name))
        assert not np.shares_memory(getattr(model.training_matrix_, name), getattr(before, name))


@pytest.mark.parametrize("transform", ["raw", "log", "binary"])
@pytest.mark.parametrize("row_l2", [False, True])
@pytest.mark.parametrize("item_weighting", ["none", "idf"])
def test_each_feature_space_against_scalar_adjugate_oracle(
    context, transform, row_l2, item_weighting,
):
    x = [[2, 1, 0], [0, 1, 3], [1, 0, 1]]
    # Each item has exactly two of the three effective training users.
    weight = 1 + math.log(4 / 3) if item_weighting == "idf" else 1
    expected = []
    for row in x:
        transformed = [
            (0 if not count else math.log1p(count) if transform == "log"
             else 1 if transform == "binary" else count) * weight for count in row
        ]
        norm = math.sqrt(math.fsum(v * v for v in transformed)) if row_l2 else 1
        expected.append([v / norm for v in transformed])
    b, scores = independent_fit3(expected, 2.5)
    model = EASEModel(regularization=2.5, transform=transform, row_l2=row_l2,
                      item_weighting=item_weighting).fit(context)
    assert_allclose(model.item_weights_, [weight] * 3)
    assert_allclose(model.training_matrix_.toarray(), expected, rtol=2e-15)
    assert_allclose(model.coefficients_, b, rtol=4e-14, atol=1e-15)
    assert_allclose(model.score_batch([0, 2, 4], [1, 3, 6]), scores, rtol=4e-14, atol=1e-15)


def test_counts_are_regression_targets_not_binary_preferences(context):
    raw = EASEModel(regularization=2).fit(context)
    binary = EASEModel(regularization=2, transform="binary").fit(context)
    # Binary G has diag=4, all off-diagonals=1; off-diagonal B entries=1/5.
    assert_allclose(binary.coefficients_, [[0, .2, .2], [.2, 0, .2], [.2, .2, 0]])
    assert_allclose(binary.score(0, [1, 3, 6]), [.2, .2, .4])
    assert not np.allclose(raw.coefficients_, binary.coefficients_)
    assert not np.allclose(raw.score(0, [1, 3, 6]), binary.score(0, [1, 3, 6]))


@pytest.mark.parametrize("transform", ["raw", "log", "binary"])
def test_shared_half_life_counts_against_independent_event_weight_sums(transform):
    values = np.array([[2, 1, 0], [0, 1, 3], [1, 0, 1]])
    context = make_context(values, ages=lambda user, item, repeat: (user + item + repeat) % 4)
    x = []
    for user, row in enumerate(values):
        transformed = []
        for item, count in enumerate(row):
            mass = math.fsum(2 ** (-((user + item + repeat) % 4) / 2)
                             for repeat in range(count))
            transformed.append(math.log1p(mass) if transform == "log"
                               else float(mass > 0) if transform == "binary" else mass)
        x.append(transformed)
    b, scores = independent_fit3(x, 2)
    model = EASEModel(regularization=2, half_life_days=2, transform=transform).fit(context)
    assert_allclose(model.training_matrix_.toarray(), x, rtol=2e-15)
    assert_allclose(model.coefficients_, b, rtol=3e-14, atol=1e-15)
    assert_allclose(model.score_batch([0, 1, 2], [0, 1, 2]), scores, rtol=3e-14, atol=1e-15)
    assert model.diagnostics["half_life_days"] == 2


def test_order_duplicates_watched_items_and_signed_predictions(context):
    model = EASEModel(regularization=2).fit(context)
    candidates = np.array([6, 1, 6, 0, 3, -1, 7, 90, 2**63 - 1])
    assert_allclose(model.score(4, candidates), [-1 / 12, -2 / 39, -1 / 12, np.nan,
                                              40 / 83, np.nan, np.nan, np.nan, np.nan],
                    rtol=2e-14, equal_nan=True)
    # Watched items with OTHER history evidence remain eligible, including negatives.
    assert np.isfinite(model.score(4, [1, 6])).all()
    expected = np.vstack([model.score(u, candidates) for u in [4, 0, 4, -1, 1, 2]])
    assert_allclose(model.score_batch([4, 0, 4, -1, 1, 2], candidates), expected, equal_nan=True)
    assert model.score(0, candidates).dtype == np.float64


def test_cancellation_zero_with_nonzero_coefficients_is_finite():
    context = make_context([[1, 1, 0], [0, 1, 1], [0, 3, 1]])
    model = EASEModel(regularization=2).fit(context)
    # Regressing item0 on the other two columns gives [1/9,-1/9].
    assert_allclose(model.coefficients_[1:, 0], [1 / 9, -1 / 9], rtol=2e-14)
    actual = model.score(1, [0])[0]  # History [0,1,1]: exact rational prediction=0.
    assert np.isfinite(actual)
    assert abs(actual) < 1e-15


def test_no_path_and_zero_coefficients_do_not_turn_into_popularity():
    diagonal = EASEModel(regularization=2).fit(make_context([[3, 0, 0], [0, 2, 0], [0, 0, 1]]))
    assert_array_equal(diagonal.coefficients_, 0)
    assert np.isnan(diagonal.score_batch([0, 1, 2], [0, 1, 2])).all()
    assert diagonal.diagnostics["n_users_with_coefficient_evidence"] == 0
    # Separate co-occurrence components. User1 knows only item0: self is absent
    # evidence, item1 has a learned path, and both warm items in the other block don't.
    context = make_context([[1, 2, 0, 0], [3, 0, 0, 0], [0, 0, 2, 1]])
    model = EASEModel(regularization=2).fit(context)
    actual = model.score(1, [0, 1, 2, 3])
    assert np.isnan(actual[[0, 2, 3]]).all()
    assert actual[1] > 0
    assert "repeat" in model.diagnostics["sole_history_self_score"]
    assert model.diagnostics["watched_item_filter"] is False
    assert model.diagnostics["internal_fallback"] is False


@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (3, 0), (3, 4)])
def test_empty_all_zero_and_successful_empty_refit(shape, context, monkeypatch):
    model = EASEModel(regularization=2).fit(context)

    def forbidden(*args, **kwargs):
        raise AssertionError("Empty fits must not invoke dense native algebra")

    monkeypatch.setattr(collaborative, "solve", forbidden)
    assert model.fit(make_context(np.zeros(shape))) is model
    d = model.diagnostics
    assert d["status"] == "empty"
    assert d["training_shape"] == list(shape)
    assert d["compact_training_shape"] == [0, 0]
    assert d["coefficient_shape"] == [0, 0]
    assert d["training_seconds"] == 0
    assert d["training_blas_thread_limit_requested"] is None
    assert d["training_blas_control_status"] == "not_requested_empty_fit"
    assert d["training_blas_pools"] == []
    assert len(model.user_index_to_row_) == shape[0]
    assert len(model.item_index_to_column_) == shape[1]
    assert np.isnan(model.score(0, [0, -1, 99])).all()
    json.dumps(d, allow_nan=False)


def test_singleton_and_refit_different_shape_clear_old_state(context):
    model = EASEModel(regularization=2).fit(context)
    old_b = model.coefficients_
    model.fit(make_context([[0, 0], [0, 3]]))
    assert model.coefficients_ is not old_b
    assert_array_equal(model.coefficients_, [[0]])
    assert_array_equal(model.active_user_indices_, [1])
    assert_array_equal(model.active_item_indices_, [1])
    assert_array_equal(model.user_index_to_row_, [-1, 0])
    assert_array_equal(model.item_index_to_column_, [-1, 0])
    assert np.isnan(model.score_batch([0, 1, -1, 4], [0, 1, 6])).all()
    model.fit(context)
    assert_allclose(model.score(4, [1, 6]), [-2 / 39, -1 / 12], rtol=2e-14)


def test_train_only_raw_count_cap_ties_mapping_and_coverage():
    context = make_context([[9, 4, 4, 0, 0], [0, 0, 0, 3, 0]],
                           ids=[100, 90, 5, 1, 7], candidates=[1, 3, 4])
    model = EASEModel(regularization=2, max_active_items=2).fit(context)
    # Top raw counts: item100=9, then tied items5/90=4 resolved by RAW ID5.
    assert_array_equal(model.active_item_indices_, [0, 2])
    assert_array_equal(model.active_item_ids_, [100, 5])
    assert_array_equal(model.active_user_indices_, [0])
    assert np.isnan(model.score(1, [0, 1, 2, 3, 4])).all()
    assert np.isnan(model.score(0, [1, 3, 4])).all()
    assert np.isfinite(model.score(0, [0, 2])).all()
    d = model.diagnostics
    assert (d["n_warm_items"], d["n_active_items"], d["n_truncated_items"]) == (4, 2, 2)
    assert (d["n_warm_users"], d["n_active_users"]) == (2, 1)
    assert d["training_count_total"] == 20
    assert d["retained_training_count_total"] == 13
    assert d["training_count_coverage"] == 13 / 20
    assert d["training_nnz_coverage"] == 2 / 4
    assert d["training_item_coverage"] == 2 / 4
    assert d["training_user_coverage"] == 1 / 2
    # A matrix-column permutation must not change which raw IDs survive a tie.
    permutation = [3, 2, 4, 0, 1]
    shuffled = make_context(context.data.interactions.toarray()[:, permutation],
                            ids=context.data.item_ids[permutation])
    other = EASEModel(regularization=2, max_active_items=2).fit(shuffled)
    assert sorted(other.active_item_ids_) == [5, 100]
    assert_allclose(other.score(0, [3, 1]), model.score(0, [0, 2]))


def test_cap_ranking_uses_raw_counts_not_decay_or_candidates():
    context = make_context([[8, 2, 2, 0]], ids=[100, 90, 5, 1], candidates=[1],
                           ages=lambda user, item, repeat: 10 if item == 0 else 0)
    model = EASEModel(regularization=2, max_active_items=2, half_life_days=1).fit(context)
    assert_array_equal(model.active_item_ids_, [100, 5])


def test_idf_is_training_fitted_before_cap_including_truncated_only_users():
    context = make_context([[3, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]])
    model = EASEModel(regularization=2, max_active_items=2, item_weighting="idf").fit(context)
    assert_array_equal(model.active_item_indices_, [0, 1])
    assert model.diagnostics["idf_training_users"] == 3  # Not 2 retained or 4 global users.
    assert_allclose(model.item_weights_, [1 + math.log(4 / 3)] * 2)


def test_temporal_underflow_empty_fit_and_stable_tiny_row_normalization():
    context = make_context([[2, 1], [0, 1]], ages=lambda *args: 1)
    empty = EASEModel(half_life_days=1e-6).fit(context)
    assert empty.diagnostics["status"] == "empty"
    assert empty.diagnostics["n_items_lost_to_temporal_underflow"] == 2
    assert empty.diagnostics["training_count_coverage"] == 0
    assert np.isnan(empty.score(0, [0, 1])).all()
    # 2^-1000 is positive, but its square underflows in float64. Row L2 must
    # normalize it stably, not silently erase the history or divide by zero.
    tiny = EASEModel(regularization=2, half_life_days=.001, row_l2=True).fit(context)
    assert_allclose(tiny.training_matrix_.toarray(), [[2 / math.sqrt(5), 1 / math.sqrt(5)], [0, 1]])
    assert np.isfinite(tiny.score(0, [0, 1])).all()


@pytest.mark.parametrize("user", [-1, -2**63, 1, 3, 5, 10**100, np.uint64(2**64 - 1)])
def test_unknown_and_cold_users_have_no_evidence(context, user):
    model = EASEModel(regularization=2).fit(context)
    assert np.isnan(model.score(user, [1, 3, 6])).all()


def test_huge_unsigned_and_python_integer_candidates_never_wrap(context):
    model = EASEModel(regularization=2).fit(context)
    candidates = np.array([1, 6, 2**64 - 1], dtype=np.uint64)
    actual = model.score(4, candidates)
    assert_allclose(actual[:2], [-2 / 39, -1 / 12], rtol=2e-14)
    assert np.isnan(actual[2])
    assert np.isnan(model.score(0, [-2**100, 2**100])).all()
    actual = model.score_batch([4, 2**100, -2**100, 4], [6, 2**100, 1])
    assert_allclose(actual[[0, 3]][:, [0, 2]], [[-1 / 12, -2 / 39]] * 2, rtol=2e-14)
    assert np.isnan(actual[1:3]).all()
    assert np.isnan(actual[:, 1]).all()
    assert np.isnan(model.score(0, context.data.item_ids)).all()  # Raw IDs aren't indices.


@pytest.mark.parametrize("indices", [
    [True, 0], (False, np.int64(0)), [np.bool_(True), 0],
    np.array([0, True], dtype=object),
])
@pytest.mark.parametrize("axis", ["users", "candidates"])
def test_mixed_boolean_coordinates_rejected_before_numeric_coercion(context, indices, axis):
    model = EASEModel(regularization=2).fit(context)
    with pytest.raises(ValueError, match="integer"):
        model.score_batch(indices if axis == "users" else [0],
                          indices if axis == "candidates" else [1])
    if axis == "candidates":
        with pytest.raises(ValueError, match="integer"):
            model.score(0, indices)


@pytest.mark.parametrize("indices", [
    [-1, 2**63], [0, -1, 2**63, 2], [-1, 2**64 - 1],
    [-1, np.uint64(2**64 - 1)], [2**64, -2**64, 1],
    (np.int64(0), np.uint64(2**63), 0), [2**100, -2**100],
    (6, -1, 2**63, 1, 6),
])
@pytest.mark.parametrize("axis", ["users", "candidates"])
def test_arbitrary_integer_sequences_preserve_known_scores_order_and_unknown_nan(context, indices, axis):
    model = EASEModel(regularization=2).fit(context)
    users = indices if axis == "users" else [4, 0, 2, 4]
    candidates = indices if axis == "candidates" else [6, 1, 3, 6]
    # Independent rational oracle from the original fixture, not score/private
    # index helpers. Unknown and cold integer coordinates have absent evidence.
    expected_rows = {0: {1: 7 / 13, 3: 42 / 83, 6: 5 / 8},
                     2: {1: 5 / 13, 3: 57 / 83, 6: 19 / 24},
                     4: {1: -2 / 39, 3: 40 / 83, 6: -1 / 12}}
    expected = [[expected_rows.get(int(user), {}).get(int(item), np.nan)
                 for item in candidates] for user in users]
    assert_allclose(model.score_batch(users, candidates), expected, rtol=2e-14, equal_nan=True)
    # Explicit object arrays and Python sequences must agree, without casting
    # unknown integers through float64/int64 or mutating the supplied requests.
    object_users = np.array(users, dtype=object)
    object_items = np.array(candidates, dtype=object)
    object_users.setflags(write=False)
    object_items.setflags(write=False)
    assert_allclose(model.score_batch(object_users, object_items), expected,
                    rtol=2e-14, equal_nan=True)


@pytest.mark.parametrize("indices", [
    [0.0, 2**100], [0, "1"], (0, np.array(1)),
    np.array([0, 1.0], dtype=object), np.array([0, 1], dtype=float),
])
@pytest.mark.parametrize("axis", ["users", "candidates"])
def test_preserving_integer_sequences_does_not_accept_real_nonintegers(context, indices, axis):
    model = EASEModel(regularization=2).fit(context)
    with pytest.raises(ValueError, match="integer"):
        model.score_batch(indices if axis == "users" else [0],
                          indices if axis == "candidates" else [1])


@pytest.mark.parametrize("user", [2**63, 2**64 - 1, 2**64, -2**63 - 1, 2**100, 10**400])
def test_unknown_scalar_integer_boundaries_consistently_return_nan(context, user):
    model = EASEModel(regularization=2).fit(context)
    assert np.isnan(model.score(user, [1, 3, 6])).all()


def test_unknown_integer_input_does_not_depend_on_native_inference_overflow(context, monkeypatch):
    model = EASEModel(regularization=2).fit(context)
    native_asarray = collaborative.np.asarray

    def overflow_if_inferred(values, *args, **kwargs):
        # Named fault injection for NumPy/platform inference differences, not an
        # observed native overflow on this environment. Object conversion must
        # be selected BEFORE any lossy numeric inference, not after an exception.
        if isinstance(values, (list, tuple)) and any(
            isinstance(v, (int, np.integer)) and abs(int(v)) > 2**63 - 1 for v in values
        ) and kwargs.get("dtype") is not object:
            raise OverflowError("intentional huge-integer inference failure")
        return native_asarray(values, *args, **kwargs)

    monkeypatch.setattr(collaborative.np, "asarray", overflow_if_inferred)
    assert np.isnan(model.score(2**100, [1, 3, 6])).all()
    actual = model.score_batch([0, 2**100, 0], [6, -1, 2**63, 1, 6])
    assert np.isnan(actual[1]).all()
    assert np.isnan(actual[:, 1:3]).all()
    assert_allclose(actual[[0, 2]][:, [0, 3, 4]], [[5 / 8, 7 / 13, 5 / 8]] * 2, rtol=2e-14)


def test_typed_integer_requests_keep_fast_path_and_exact_canonical_predictions(context, monkeypatch):
    model = EASEModel(regularization=2).fit(context)
    users, candidates = np.array([4, 0, 4, 1]), np.array([6, 1, 6, 3])
    expected = [[-1 / 12, -2 / 39, -1 / 12, 40 / 83],
                [5 / 8, 7 / 13, 5 / 8, 42 / 83],
                [-1 / 12, -2 / 39, -1 / 12, 40 / 83], [np.nan] * 4]
    baseline = model.score_batch(users, candidates)
    native_asarray = collaborative.np.asarray

    def no_object_conversion_for_typed_arrays(values, *args, **kwargs):
        if isinstance(values, np.ndarray) and values.dtype.kind in "iu":
            assert kwargs.get("dtype") is not object
        return native_asarray(values, *args, **kwargs)

    monkeypatch.setattr(collaborative.np, "asarray", no_object_conversion_for_typed_arrays)
    assert_array_equal(model.score_batch(users, candidates), baseline)
    assert_allclose(baseline, expected, rtol=2e-14, equal_nan=True)
    assert_array_equal(model.score_batch(tuple(users), tuple(candidates)), baseline)
    # Boolean provenance already erased by the CALLER cannot be reconstructed;
    # explicitly typed integer arrays remain valid matrix-index arrays.
    assert_array_equal(model.score_batch(np.array([True, 0], dtype=int), candidates),
                       model.score_batch(np.array([1, 0]), candidates))


def test_empty_requests_and_bounds_precede_output_allocation(context, monkeypatch):
    model = EASEModel(regularization=2, max_batch_scores=6).fit(context)
    assert model.score_batch([], [1, 3]).shape == (0, 2)
    assert model.score_batch([0, 2], []).shape == (2, 0)
    assert model.score_batch([], []).shape == (0, 0)
    assert model.score(0, []).shape == (0,)
    assert model.score_batch([0, 0], [1, 1, 6]).shape == (2, 3)  # Count duplicates.
    original_full = collaborative.np.full

    def bounded_full(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2 and math.prod(shape) > 6:
            raise AssertionError("Output guard happened too late")
        return original_full(shape, *args, **kwargs)

    monkeypatch.setattr(collaborative.np, "full", bounded_full)
    for users, candidates in [([0, 2], [1] * 4), ([-1] * 4, [-1] * 2), ([0], [1] * 7)]:
        with pytest.raises(ValueError, match="max_batch_scores"):
            model.score_batch(users, candidates)
    with pytest.raises(ValueError, match="max_batch_scores"):
        model.score(0, [1] * 7)


def test_only_item_item_matrix_is_densified_and_no_global_cache(monkeypatch):
    values = np.zeros((100, 10))
    values[:10, [1]] = 2
    values[5:20, [7]] = 1
    context = make_context(values)
    original_toarray = csr_matrix.toarray

    def guarded_toarray(matrix, *args, **kwargs):
        assert matrix.shape == (2, 2), "Never densify global user-item training/score tables"
        return original_toarray(matrix, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", guarded_toarray)
    model = EASEModel(regularization=2, max_active_items=2).fit(context)
    model.score_batch([0, 5, 10, 99], np.arange(10))
    assert model.coefficients_.shape == (2, 2)
    assert not any(isinstance(v, np.ndarray) and v.shape == values.shape for v in vars(model).values())


def test_forecast_horizon_schedules_candidates_and_payload_do_not_fit_cf(context):
    original = EASEModel(regularization=2, row_l2=True, item_weighting="idf",
                         half_life_days=3).fit(context)
    changed = replace(
        context, forecast_start=context.train_end + 100,
        forecast_end=context.train_end + 400 * 86400,
        candidate_indices=np.array([6, 0, 1]),
        forecast_schedules=(Airing(0, "unknown", context.train_end + 100,
                                  context.train_end + 200, True),),
        historical_schedules=(Airing(1, "irrelevant", context.train_end - 200,
                                    context.train_end - 100),),
    )
    # Poison unrelated fields to establish that they're not read as features.
    changed.data.metadata[:] = [{"heldout_frequency": 10**100}] * 7
    changed.data.provenance["heldout_counts"] = object()
    other = EASEModel(regularization=2, row_l2=True, item_weighting="idf",
                      half_life_days=3).fit(changed)
    assert_array_equal(original.coefficients_, other.coefficients_)
    assert_array_equal(original.item_weights_, other.item_weights_)
    assert_array_equal(original.score_batch([0, 2, 4], [6, 1, 3]),
                       other.score_batch([0, 2, 4], [6, 1, 3]))


def test_cold_catalog_growth_and_raw_id_order_do_not_change_warm_predictions(context):
    first = EASEModel(regularization=2, max_active_items=2).fit(context)
    # Supply additional permitted forecast catalog entries, never future behavior.
    enlarged = make_context(np.column_stack([context.data.interactions.toarray(), np.zeros((5, 2))]),
                            ids=[*context.data.item_ids, 1, 2])
    second = EASEModel(regularization=2, max_active_items=2).fit(enlarged)
    assert_array_equal(first.active_item_ids_, second.active_item_ids_)
    assert_allclose(first.coefficients_, second.coefficients_)
    assert_allclose(first.score_batch([0, 2, 4], np.arange(7)),
                    second.score_batch([0, 2, 4], np.arange(7)), equal_nan=True)
    assert np.isnan(second.score(0, [7, 8])).all()


def test_fit_snapshot_survives_training_mutation_and_scoring_is_read_only(context):
    model = EASEModel(regularization=2).fit(context)
    users, candidates = np.array([4, 0, 4]), np.array([6, 1, 6, -1, 3])
    users_before, candidates_before = users.copy(), candidates.copy()
    baseline = model.score_batch(users, candidates)
    context.data.interactions.data[:] = 999
    context.data.item_ids[:] = np.arange(8000, 8007)
    context.data.user_ids[:] = [f"mutated-{i}" for i in range(5)]
    context.data.metadata[:] = [{"future": True}] * 7
    context.data.provenance["train_end"] = context.train_end + 999
    assert_allclose(model.score_batch(users, candidates), baseline, equal_nan=True)
    assert_array_equal(users, users_before)
    assert_array_equal(candidates, candidates_before)
    assert_array_equal(model.active_item_ids_, [90, 5, 32])


@pytest.mark.parametrize("defect", ["future", "provenance_cutoff", "provenance_max", "count_mismatch"])
def test_cutoff_and_event_count_rechecks_preserve_previous_fit(context, defect):
    model = EASEModel(regularization=2).fit(context)
    previous_b, previous_d = model.coefficients_, model.diagnostics
    baseline = model.score(0, [1, 3, 6])
    if defect == "future":
        # Deliberately bypass the frozen constructor, testing fit's own guard.
        ends = context.event_ends.copy()
        ends[0] = context.train_end + 1
        object.__setattr__(context, "event_ends", ends)
    elif defect == "provenance_cutoff":
        context.data.provenance["train_end"] -= 1
    elif defect == "provenance_max":
        context.data.provenance["behavior_max_end"] = context.train_end + 1
    else:
        context.data.interactions.data[0] += 1
    with pytest.raises(ValueError, match="cutoff|pre-cutoff"):
        model.fit(context)
    assert model.coefficients_ is previous_b
    assert model.diagnostics is previous_d
    assert_allclose(model.score(0, [1, 3, 6]), baseline)


@pytest.mark.parametrize("value", [-1, np.inf, np.nan])
def test_invalid_mutated_training_counts_fail_explicitly(context, value):
    context.data.interactions.data[0] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        EASEModel(regularization=2).fit(context)


def test_native_failure_preserves_successful_snapshot(context, monkeypatch):
    model = EASEModel(regularization=2).fit(context)
    previous_b, previous_d = model.coefficients_, model.diagnostics
    baseline = model.score(0, [1, 3, 6])

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("intentional numerical failure")

    monkeypatch.setattr(collaborative, "solve", fail)
    with pytest.raises(ValueError, match="ill-conditioned"):
        model.fit(context)
    assert model.coefficients_ is previous_b
    assert model.diagnostics is previous_d
    assert_array_equal(model.score(0, [1, 3, 6]), baseline)


def test_readonly_native_precision_can_be_normalized(context, monkeypatch):
    native_solve = collaborative.solve

    def readonly_solve(*args, **kwargs):
        result = native_solve(*args, **kwargs).copy()
        result.setflags(write=False)
        return result

    monkeypatch.setattr(collaborative, "solve", readonly_solve)
    model = EASEModel(regularization=2).fit(context)
    assert_allclose(model.score(4, [1, 3, 6]), [-2 / 39, 40 / 83, -1 / 12], rtol=2e-14)


@pytest.mark.parametrize("defect", ["shape", "nonfinite", "nonpositive_diagonal"])
def test_invalid_native_precision_is_not_accepted(context, monkeypatch, defect):
    def invalid_solve(*args, **kwargs):
        result = np.eye(2 if defect == "shape" else 3)
        if defect == "nonfinite":
            result[0, 1] = np.nan
        elif defect == "nonpositive_diagonal":
            result[0, 0] = 0
        return result

    monkeypatch.setattr(collaborative, "solve", invalid_solve)
    model = EASEModel(regularization=2)
    with pytest.raises(ValueError, match="precision matrix"):
        model.fit(context)
    with pytest.raises(RuntimeError, match="fit"):
        model.score(0, [1])


def test_successful_refits_and_instances_do_not_share_buffers(context):
    first = EASEModel(regularization=2).fit(context)
    second = EASEModel(regularization=2).fit(context)
    old_b, old_x = first.coefficients_, first.training_matrix_
    baseline = first.score(0, [1, 3, 6])
    first.fit(context)
    assert not np.shares_memory(first.coefficients_, old_b)
    assert not np.shares_memory(first.training_matrix_.data, old_x.data)
    assert not np.shares_memory(first.coefficients_, second.coefficients_)
    old_b[:] = 1234
    old_x.data[:] = 555
    assert_array_equal(first.score(0, [1, 3, 6]), baseline)
    assert_array_equal(second.score(0, [1, 3, 6]), baseline)


@pytest.mark.parametrize("reports,status", [
    ([], "unverified_no_reported_pools"),
    ([{"user_api": "openmp", "num_threads": 1}], "unverified_no_reported_pools"),
    ([{"user_api": "blas", "internal_api": "openblas", "num_threads": 2}],
     "reported_pools_at_requested_limit"),
    ([{"user_api": "blas", "internal_api": "mkl", "num_threads": 8}],
     "reported_pools_not_at_requested_limit"),
    ([{"user_api": "blas", "internal_api": "accelerate", "num_threads": 2}],
     "unverified_accelerate"),
])
def test_native_blas_request_is_not_confused_with_observation(context, monkeypatch, reports, status):
    # Fault-injected controller reports, NOT measurements of these libraries.
    monkeypatch.setattr(collaborative, "threadpool_info", lambda: reports)
    model = EASEModel(regularization=2, num_threads=2).fit(context)
    assert model.diagnostics["training_blas_thread_limit_requested"] == 2
    assert model.diagnostics["training_blas_control_status"] == status
    assert model.diagnostics["blas_reports_measure_runtime_thread_activity"] is False
    json.dumps(model.diagnostics, allow_nan=False)


def test_diagnostics_dimensions_counts_and_json(context):
    model = EASEModel(regularization=2).fit(context)
    d = model.diagnostics
    assert d["training_shape"] == [5, 7]
    assert d["compact_training_shape"] == [3, 3]
    assert d["coefficient_shape"] == [3, 3]
    assert d["coefficient_bytes"] == 3 * 3 * 8
    assert d["coefficient_support_bytes"] == 3 * 3
    assert d["training_count_total"] == 9
    assert d["nnz"] == d["feature_nnz"] == 6
    assert d["training_count_coverage"] == d["training_nnz_coverage"] == 1
    assert d["training_item_coverage"] == d["training_user_coverage"] == 1
    assert d["n_nonzero_coefficients"] == 6
    assert d["n_users_with_coefficient_evidence"] == 3
    assert d["n_items_with_incoming_coefficients"] == 3
    assert d["cutoff_checked"] is True
    assert d["train_end"] == d["behavior_max_end"] == context.train_end
    assert d["training_seconds"] > 0
    assert d["fit_seconds"] >= d["training_seconds"]
    json.dumps(d, allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"regularization": 0}, {"regularization": -1}, {"regularization": np.inf},
    {"regularization": np.nan}, {"regularization": True}, {"regularization": "2"},
    {"transform": "sqrt"}, {"half_life_days": 0}, {"half_life_days": -1},
    {"half_life_days": np.nan}, {"half_life_days": np.inf}, {"half_life_days": True},
    {"half_life_days": "2"}, {"row_l2": "yes"}, {"row_l2": 1},
    {"item_weighting": "future_idf"}, {"max_active_items": 0}, {"max_active_items": 4097},
    {"max_active_items": 1.5}, {"max_active_items": True}, {"num_threads": 0},
    {"num_threads": 1.5}, {"max_batch_scores": 0}, {"max_batch_scores": True},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        EASEModel(**kwargs)


@pytest.mark.parametrize("indices", [[1.0], [True], ["1"], [None], [[1, 3]], np.array(1)])
def test_malformed_index_arrays_rejected(context, indices):
    model = EASEModel(regularization=2).fit(context)
    with pytest.raises(ValueError):
        model.score(0, indices)
    with pytest.raises(ValueError):
        model.score_batch(indices, [1, 3])
    with pytest.raises(ValueError):
        model.score_batch([0], indices)


@pytest.mark.parametrize("user", [1.0, True, None, "0", [0]])
def test_malformed_scalar_user_rejected(context, user):
    model = EASEModel(regularization=2).fit(context)
    with pytest.raises(ValueError):
        model.score(user, [1, 3])


def test_scoring_requires_fit_and_fit_requires_temporal_context(context):
    model = EASEModel()
    with pytest.raises(RuntimeError, match="fit"):
        model.score(0, [0])
    with pytest.raises(RuntimeError, match="fit"):
        model.score_batch([0], [0])
    with pytest.raises(ValueError, match="TemporalContext"):
        model.fit(context.data)
