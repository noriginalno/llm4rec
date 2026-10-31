"""Independent dense linear-system oracles are deliberately test-only."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix, isspmatrix_csr

from sweettv.contracts import TrainingData
from sweettv.models import ppr
from sweettv.models.ppr import PPRConvergenceError, PersonalizedPageRankModel


def training(counts, *, item_ids=None, metadata=None):
    counts = csr_matrix(counts)
    users, items = counts.shape
    return TrainingData(
        user_ids=[f"user-{i}" for i in range(users)],
        item_ids=np.arange(1, items + 1) if item_ids is None else item_ids,
        interactions=counts,
        metadata=[{} for _ in range(items)] if metadata is None else metadata,
    )


def independent_transition(counts):
    """Build the mathematical graph without production graph/iteration helpers."""
    counts = np.asarray(counts, dtype=np.float64)
    users, items = counts.shape
    adjacency = np.zeros((users + items, users + items), dtype=np.float64)
    for user in range(users):
        for item in range(items):
            adjacency[user, users + item] = counts[user, item]
            adjacency[users + item, user] = counts[user, item]
    transition = np.zeros_like(adjacency)
    for node in range(users + items):
        degree = sum(adjacency[node])
        if degree:
            transition[node] = adjacency[node] / degree
    return transition


def independent_stationary(transition, source, alpha):
    """Solve (I - alpha*P_restart.T)p = (1-alpha)e_source directly."""
    transition = np.array(transition, dtype=np.float64, copy=True)
    restart = np.zeros(len(transition), dtype=np.float64)
    restart[source] = 1.0
    for row in range(len(transition)):
        if sum(transition[row]) == 0:
            transition[row] = restart
    return np.linalg.solve(
        np.eye(len(transition)) - alpha * transition.T,
        (1.0 - alpha) * restart,
    )


def test_non_square_weighted_graph_matches_independent_linear_system():
    counts = np.array([[3, 1, 0, 0], [0, 2, 4, 0]], dtype=np.float64)
    model = PersonalizedPageRankModel(tol=1e-12).fit(training(counts))
    transition = independent_transition(counts)
    assert_allclose(model.transition_.toarray(), transition, rtol=0, atol=1e-15)
    for user in [0, 1]:
        expected = independent_stationary(transition, user, model.alpha)
        actual = model.score(user, np.arange(4))
        assert_allclose(actual[:3], expected[2:5], rtol=0, atol=1e-12)
        assert np.isnan(actual[3])
        info = model.diagnostics["last_query"]["solves"][0]
        assert info["converged"]
        assert info["residual_l1"] <= model.tol
        assert info["error_bound_l1"] == info["residual_l1"] / (1 - model.alpha)
        assert_allclose(info["walk_mass"], 1.0, rtol=0, atol=2e-15)
        assert_allclose(info["item_mass"], expected[2:].sum(), rtol=0, atol=1e-12)


def test_all_node_probabilities_and_observed_residual_match_oracle():
    counts = np.array([[4, 1, 0], [0, 2, 7]])
    model = PersonalizedPageRankModel(tol=1e-12).fit(training(counts))
    dense_transition = independent_transition(counts)
    result = ppr._iterate_ppr(
        model._transition_transpose,
        model._dangling,
        np.array([1, 0]),
        alpha=model.alpha,
        tol=model.tol,
        max_iter=model.max_iter,
    )
    assert result.probabilities.dtype == np.float64
    for column, source in enumerate([1, 0]):
        expected = independent_stationary(dense_transition, source, model.alpha)
        actual = result.probabilities[:, column]
        assert_allclose(actual, expected, rtol=0, atol=1e-12)
        restart = np.eye(len(actual))[source]
        residual = np.abs(
            model.alpha * dense_transition.T @ actual
            + (1 - model.alpha) * restart
            - actual
        ).sum()
        assert_allclose(result.residual_l1[column], residual, rtol=0, atol=5e-16)
        assert np.abs(actual - expected).sum() <= result.residual_l1[column] / (
            1 - model.alpha
        ) + 1e-15
    assert_allclose(result.probabilities.sum(axis=0), 1.0, rtol=0, atol=2e-15)


def test_star_closed_form_repeats_and_item_mass_not_one():
    weights = np.array([1, 2, 4], dtype=np.float64)
    model = PersonalizedPageRankModel(tol=1e-12).fit(training(weights[None, :]))
    actual = model.score(0, np.arange(3))
    expected = model.alpha / (1 + model.alpha) * weights / weights.sum()
    assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert np.all(np.isfinite(actual))  # Every item is already watched.
    assert_allclose(actual.sum(), model.alpha / (1 + model.alpha), atol=1e-12)
    assert actual.sum() < 0.5
    info = model.diagnostics["last_query"]["solves"][0]
    assert_allclose(
        info["walk_mass"] - info["item_mass"], 1 / (1 + model.alpha), atol=1e-12
    )


def test_transition_is_row_normalized_not_column_normalized():
    model = PersonalizedPageRankModel().fit(training([[3, 1, 0], [0, 2, 4]]))
    transition = model.transition_.toarray()
    expected = np.array(
        [
            [0, 0, 3 / 4, 1 / 4, 0],
            [0, 0, 0, 1 / 3, 2 / 3],
            [1, 0, 0, 0, 0],
            [1 / 3, 2 / 3, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ]
    )
    assert_allclose(transition, expected, rtol=0, atol=1e-15)
    assert_allclose(transition.sum(axis=1), 1.0)
    assert transition[0, 3] != transition[3, 0]
    assert not np.allclose(transition.sum(axis=0), 1.0)


def test_qualifying_counts_are_weights_not_binary_edges():
    counts = np.array([[9, 1, 0], [0, 1, 5]])
    model = PersonalizedPageRankModel(tol=1e-12).fit(training(counts))
    actual = model.score(0, np.arange(3))
    binary_oracle = independent_stationary(independent_transition(counts > 0), 0, 0.85)[2:]
    weighted_oracle = independent_stationary(independent_transition(counts), 0, 0.85)[2:]
    assert_allclose(actual, weighted_oracle, atol=1e-12, rtol=0)
    assert not np.allclose(actual, binary_oracle)
    assert model.diagnostics["training_edges"] == 4
    assert model.diagnostics["training_weight"] == 16
    assert model.diagnostics["transition_nnz"] == 8


def test_dangling_mass_returns_to_each_queries_own_restart():
    # Directed fixture exercises a sink, which is unreachable from active users
    # in an ordinary undirected bipartite graph but still must be handled safely.
    transition = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
    result = ppr._iterate_ppr(
        csr_matrix(transition.T),
        np.array([False, True, True]),
        np.array([0, 1, 2]),
        alpha=0.85,
        tol=1e-12,
        max_iter=300,
    )
    assert np.all(result.converged)
    for column, source in enumerate([0, 1, 2]):
        expected = independent_stationary(transition, source, 0.85)
        assert_allclose(result.probabilities[:, column], expected, atol=1e-12, rtol=0)
    assert_allclose(result.probabilities.sum(axis=0), 1.0, atol=2e-15, rtol=0)
    assert_array_equal(result.iterations[1:], [0, 0])


def test_unknown_cold_and_disconnected_nodes_are_nan_without_negative_wraparound():
    model = PersonalizedPageRankModel().fit(training([[1, 0, 0], [0, 7, 0], [0, 0, 0]]))
    actual = model.score_batch(
        np.array([0, 1, 2, -1, 3, 99]), np.array([0, 1, 2, -1, 3])
    )
    assert np.isfinite(actual[0, 0])
    assert np.isfinite(actual[1, 1])
    assert np.isnan(actual[0, 1:]).all()
    assert np.isnan(actual[1, [0, 2, 3, 4]]).all()
    assert np.isnan(actual[2:]).all()
    assert model.diagnostics["active_components"] == 2
    assert model.diagnostics["dangling_nodes"] == 2


def test_candidate_and_user_order_duplicates_and_matrix_to_raw_id_mapping():
    model = PersonalizedPageRankModel(tol=1e-12).fit(
        training([[3, 1, 0, 0], [0, 2, 4, 0]], item_ids=[707, 11, 902, 5])
    )
    candidates = np.array([2, 0, 2, 1, 3, -1, 88])
    users = np.array([1, 0, 1, -1])
    actual = model.score_batch(users, candidates)
    for row, user in enumerate(users):
        assert_allclose(
            actual[row], model.score(int(user), candidates),
            rtol=0, atol=1e-14, equal_nan=True,
        )
    assert_array_equal(model.item_ids_, [707, 11, 902, 5])
    assert_allclose(actual[0, 0], actual[0, 2])
    assert_allclose(actual[0], actual[2], equal_nan=True)
    assert np.isnan(model.score(0, np.array([707, 11, 902, 5]))).all()


def test_float64_default_converges_below_float32_plateau():
    model = PersonalizedPageRankModel().fit(training([[1, 2, 3]]))
    scores = model.score(0, np.arange(3))
    assert scores.dtype == np.float64
    assert model.transition_.dtype == np.float64
    info = model.diagnostics["last_query"]["solves"][0]
    assert 100 < info["iterations"] < 300
    assert info["residual_l1"] <= 1e-9
    assert info["converged"]
    assert_allclose(info["walk_mass"], 1, atol=2e-15, rtol=0)


def test_forced_nonconvergence_raises_and_records_last_iterate_residual():
    model = PersonalizedPageRankModel(max_iter=1).fit(training([[1]]))
    with pytest.raises(
        PPRConvergenceError, match="last L1 fixed-point residual"
    ) as caught:
        model.score(0, np.array([0]))
    error = caught.value
    assert error.user_indices == (0,)
    assert error.iterations == (1,)
    # r_1=(1-alpha, alpha); ||F(r_1)-r_1||_1 = 2*alpha**2.
    assert_allclose(error.residual_l1, [2 * 0.85**2], atol=1e-15, rtol=0)
    query = model.diagnostics["last_query"]
    assert query["status"] == "nonconverged"
    assert query["converged"] is False
    info = query["solves"][0]
    assert info["iterations"] == 1
    assert info["residual_l1"] > model.tol
    assert info["converged"] is False
    assert query["operator_applications"] == 2
    assert_allclose(info["walk_mass"], 1.0)


def test_alpha_zero_has_only_restart_mass_and_no_item_walk_evidence():
    model = PersonalizedPageRankModel(alpha=0).fit(training([[2, 1]]))
    assert np.isnan(model.score(0, np.array([0, 1]))).all()
    info = model.diagnostics["last_query"]["solves"][0]
    assert info["iterations"] == 0
    assert info["residual_l1"] == 0
    assert info["walk_mass"] == 1
    assert info["item_mass"] == 0


def test_alpha_immediately_below_one_is_valid_but_not_assumed_converged():
    alpha = np.nextafter(1.0, 0.0)
    model = PersonalizedPageRankModel(alpha=alpha, max_iter=1).fit(training([[1]]))
    with pytest.raises(PPRConvergenceError):
        model.score(0, np.array([0]))
    assert model.diagnostics["last_query"]["converged"] is False


@pytest.mark.parametrize("alpha", [-0.01, 1, 1.01, np.nan, np.inf, -np.inf, True, "0.85"])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        PersonalizedPageRankModel(alpha=alpha)


@pytest.mark.parametrize("tol", [0, -1e-9, np.nan, np.inf, True, "1e-9"])
def test_invalid_tolerance(tol):
    with pytest.raises(ValueError, match="tol"):
        PersonalizedPageRankModel(tol=tol)


@pytest.mark.parametrize("name", ["max_iter", "batch_size"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.inf, "2"])
def test_invalid_iteration_or_batch_limits(name, value):
    with pytest.raises(ValueError, match=name):
        PersonalizedPageRankModel(**{name: value})


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (4, 0), (2, 5)])
def test_empty_interactions_and_non_square_shapes(shape):
    model = PersonalizedPageRankModel().fit(training(csr_matrix(shape)))
    assert model.transition_.shape == (sum(shape), sum(shape))
    assert isspmatrix_csr(model.transition_)
    assert model.transition_.nnz == 0
    assert np.isnan(model.score(0, np.array([0, -1, 99]))).all()
    assert model.score_batch(np.array([], dtype=int), np.array([0])).shape == (0, 1)
    assert model.score_batch(np.array([0, -1]), np.array([], dtype=int)).shape == (2, 0)
    assert model.diagnostics["last_query"]["converged"] is None


def test_fit_uses_only_training_counts_ignores_metadata_and_does_not_mutate_input():
    data = training(
        [[2, 1, 0], [0, 3, 0]],
        metadata=[{"tv_show_title": "same", "tv_show_genre_1": "same"} for _ in range(3)],
    )
    original_data = data.interactions.data.copy()
    original_indices = data.interactions.indices.copy()
    data.provenance["heldout_interactions"] = [[0, 0, 100], [0, 0, 100]]
    model = PersonalizedPageRankModel().fit(data)
    baseline = model.score(0, np.arange(3))
    assert np.isnan(baseline[2])  # Catalog metadata and held-out provenance are not edges.
    assert_array_equal(data.interactions.data, original_data)
    assert_array_equal(data.interactions.indices, original_indices)
    data.metadata[2] = {"tv_show_title": "future favorite", "actors": "future"}
    data.provenance["heldout_interactions"] = [[100, 0, 0], [100, 0, 0]]
    refit = PersonalizedPageRankModel().fit(data)
    assert_allclose(
        refit.score(0, np.arange(3)), baseline, equal_nan=True, rtol=0, atol=0
    )
    # A fitted graph is a snapshot, not a mutable alias of the input matrix.
    data.interactions.data[:] = 99
    assert_allclose(
        model.score(0, np.arange(3)), baseline, equal_nan=True, rtol=0, atol=0
    )


def test_batch_propagation_is_bounded_and_does_not_densify_graph(monkeypatch):
    counts = csr_matrix(
        (np.arange(1, 41), (np.arange(40), np.arange(40) % 11)), shape=(40, 17)
    )
    calls = []
    real_iterate = ppr._iterate_ppr

    def bounded_iterate(transition, dangling, sources, **kwargs):
        calls.append(len(sources))
        return real_iterate(transition, dangling, sources, **kwargs)

    def no_dense_graph(*args, **kwargs):
        raise AssertionError("Production PPR must not densify the sparse graph")

    monkeypatch.setattr(ppr, "_iterate_ppr", bounded_iterate)
    monkeypatch.setattr(csr_matrix, "toarray", no_dense_graph)
    monkeypatch.setattr(csr_matrix, "todense", no_dense_graph)
    model = PersonalizedPageRankModel(batch_size=3).fit(training(counts))
    scores = model.score_batch(
        np.array([39, 0, 8, 3, 2, 39, -1]), np.array([0, 8, 16])
    )
    assert scores.shape == (7, 3)
    assert calls == [3, 2]  # Five unique active queries; duplicate user is reused.
    assert model.transition_.nnz == 2 * counts.nnz
    query = model.diagnostics["last_query"]
    assert query["batches"] == 2
    assert query["max_batch_users"] == 3
    assert query["converged"] is True
    assert len(query["solves"]) == 5
    assert np.isnan(scores[-1]).all()
    assert np.isnan(scores[:, 2]).all()
    assert_allclose(scores[0], scores[5], equal_nan=True)


def test_fit_returns_self_refit_resets_graph_and_diagnostics():
    model = PersonalizedPageRankModel()
    assert model.fit(training([[1, 2]])) is model
    model.score(0, np.array([0]))
    model.fit(training([[0, 0], [0, 5]]))
    assert model.diagnostics["last_query"] is None
    assert np.isnan(model.score(0, np.array([0, 1]))).all()
    assert np.isfinite(model.score(1, np.array([1]))).all()


def test_score_requires_fit_and_integer_one_dimensional_indices():
    model = PersonalizedPageRankModel()
    with pytest.raises(RuntimeError, match="Fit"):
        model.score(0, np.array([0]))
    model.fit(training([[1]]))
    for candidates in [
        np.array([0.5]), np.array([[0]]), np.array(["0"]), np.array([True])
    ]:
        with pytest.raises(ValueError):
            model.score(0, candidates)
    with pytest.raises(ValueError):
        model.score_batch(np.array([0.5]), np.array([0]))


@pytest.mark.parametrize(
    "unknown_user",
    [
        10**30, -(10**30), 2**63, 2**64 - 1, 2**64, 2**64 + 1,
        -(2**64), -(2**64) + 1, np.uint64(2**64 - 1),
    ],
)
def test_scalar_big_integer_unknown_users_are_nan_without_wrapping(unknown_user):
    # Two real users catch a narrowing/modulo bug that would turn 2**64 or
    # 2**64+1 into valid user indices 0 or 1.
    model = PersonalizedPageRankModel(max_iter=1).fit(training([[2, 1], [0, 3]]))
    actual = model.score(unknown_user, np.array([1, 0, 1, -1, 2]))
    assert actual.shape == (5,)
    assert actual.dtype == np.float64
    assert np.isnan(actual).all()
    query = model.diagnostics["last_query"]
    assert query["status"] == "no_evidence"
    assert query["converged"] is None
    assert query["requested_users"] == 1
    assert query["requested_candidates"] == 5
    assert query["operator_applications"] == 0
    assert query["solves"] == []


@pytest.mark.parametrize(
    "bad_user", [0.0, np.float64(0), True, np.bool_(False), "0", None, [0]]
)
def test_scalar_user_bounds_fix_does_not_accept_floats_bools_or_malformed_types(bad_user):
    model = PersonalizedPageRankModel().fit(training([[1]]))
    with pytest.raises(ValueError, match="user_index must be an integer"):
        model.score(bad_user, np.array([0]))


def test_big_integer_unknown_user_preserves_fit_and_candidate_validation():
    model = PersonalizedPageRankModel()
    with pytest.raises(RuntimeError, match="Fit"):
        model.score(10**30, np.array([0]))
    model.fit(training([[1]]))
    for candidates in [
        np.array([0.0]), np.array([True]), np.array([[0]]), np.array(["0"])
    ]:
        with pytest.raises(ValueError, match="candidate_indices"):
            model.score(-(10**30), candidates)


def test_unsigned_index_normalization_checks_bounds_before_narrowing():
    users = np.array([0, 1, 2**63, 2**64 - 1], dtype=np.uint64)
    candidates = np.array([1, 0, 2**64 - 1, 2**63, 1], dtype=np.uint64)
    original_users, original_candidates = users.copy(), candidates.copy()
    assert_array_equal(ppr._indices(users, "user_indices"), [0, 1, -1, -1])
    assert_array_equal(
        ppr._indices(candidates, "candidate_indices"), [1, 0, -1, -1, 1]
    )
    model = PersonalizedPageRankModel().fit(training([[2, 1], [0, 3]]))
    actual = model.score_batch(users, candidates)
    assert np.isfinite(actual[:2, :2]).all()
    assert np.isnan(actual[2:]).all()
    assert np.isnan(actual[:, 2:4]).all()
    assert_allclose(actual[:2, 0], actual[:2, 4], rtol=0, atol=0)
    assert_array_equal(users, original_users)
    assert_array_equal(candidates, original_candidates)


def test_small_positive_alpha_and_numpy_scalar_parameters():
    model = PersonalizedPageRankModel(
        alpha=np.float64(0.01),
        tol=np.float64(1e-12),
        max_iter=np.int64(300),
        batch_size=np.int64(2),
    ).fit(training([[2, 1]]))
    assert_allclose(
        model.score(0, np.array([0, 1])),
        0.01 / 1.01 * np.array([2 / 3, 1 / 3]),
        atol=1e-12,
        rtol=0,
    )
