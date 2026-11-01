"""MF author's independent PPR challenge; production code is never modified.

The dense transition/linear-system/reference-update calculations below are
test-only oracles. No production graph builder or walk helper supplies expected
values. The private walk kernel is called only as a subject under test or a spy.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix, isspmatrix_csr

from sweettv.contracts import TrainingData
from sweettv.models import ppr
from sweettv.models.ppr import PPRConvergenceError, PersonalizedPageRankModel


def make_training(counts, *, raw_ids=None):
    matrix = csr_matrix(counts, dtype=np.float32)
    n_users, n_items = matrix.shape
    return TrainingData(
        user_ids=[f"challenge-{user}" for user in range(n_users)],
        item_ids=np.arange(101, 101 + n_items) if raw_ids is None else raw_ids,
        interactions=matrix,
        metadata=[{} for _ in range(n_items)],
    )


@pytest.fixture
def weighted_counts():
    # A weighted, cyclic, nonsquare component; a separate weighted star; and
    # interleaved user/item cold-start handling. No validation behavior is used.
    return np.array([
        [9, 1, 0, 0, 0, 0, 0],
        [0, 4, 2, 0, 0, 0, 0],
        [1, 0, 7, 3, 0, 0, 0],
        [0, 0, 0, 0, 5, 2, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float64)


def dense_transition(counts):
    """Compute the mathematical row transition with independent scalar loops."""
    counts = np.asarray(counts, dtype=np.float64)
    n_users, n_items = counts.shape
    transition = np.zeros((n_users + n_items, n_users + n_items))
    for user in range(n_users):
        user_weight = sum(float(counts[user, item]) for item in range(n_items))
        if user_weight:
            for item in range(n_items):
                transition[user, n_users + item] = counts[user, item] / user_weight
    for item in range(n_items):
        item_weight = sum(float(counts[user, item]) for user in range(n_users))
        if item_weight:
            for user in range(n_users):
                transition[n_users + item, user] = counts[user, item] / item_weight
    return transition


def query_transition(transition, source):
    """Complete each dangling row with THIS query's one-hot restart."""
    completed = np.array(transition, dtype=np.float64, copy=True)
    for node in range(len(completed)):
        if not any(completed[node]):
            completed[node, source] = 1.0
    return completed


def linear_solution(transition, source, alpha):
    completed = query_transition(transition, source)
    restart = np.zeros(len(completed))
    restart[source] = 1
    return np.linalg.solve(
        np.eye(len(completed)) - alpha * completed.T,
        (1 - alpha) * restart,
    )


def independent_residual(transition, source, alpha, probabilities):
    restart = np.zeros(len(transition))
    restart[source] = 1
    following = alpha * query_transition(transition, source).T @ probabilities
    following += (1 - alpha) * restart
    return float(np.abs(following - probabilities).sum())


def oracle_scores(counts, users, candidates, alpha):
    """Map independent exact item masses to requested global-index positions."""
    n_users, n_items = counts.shape
    transition = dense_transition(counts)
    expected = np.full((len(users), len(candidates)), np.nan)
    for row, user in enumerate(users):
        if not 0 <= user < n_users or not np.any(counts[user]):
            continue
        probabilities = linear_solution(transition, int(user), alpha)
        for column, item in enumerate(candidates):
            if 0 <= item < n_items and probabilities[n_users + item] > 0:
                expected[row, column] = probabilities[n_users + item]
    return expected


@pytest.mark.parametrize("alpha", [0.2, 0.63, 0.85])
def test_weighted_nonsquare_linear_oracle_full_mass_components_and_order(weighted_counts, alpha):
    counts = weighted_counts
    raw_ids = [907, 101, 880, 4001, 20, 650, 3]
    model = PersonalizedPageRankModel(alpha=alpha, tol=5e-13, max_iter=500).fit(
        make_training(counts, raw_ids=raw_ids)
    )
    transition = dense_transition(counts)
    assert_allclose(model.transition_.toarray(), transition, rtol=0, atol=2e-16)
    assert_allclose(transition.sum(axis=1)[[0, 1, 2, 3, 5, 6, 7, 8, 9, 10]], 1)
    assert not np.allclose(transition.sum(axis=0), transition.sum(axis=1))

    users = np.array([2, 4, -1, 0, 3, 2, 5, 1])
    candidates = np.array([5, 2, 5, -1, 6, 0, 7, 4, 1, 3, 907])
    expected = oracle_scores(counts, users, candidates, alpha)
    assert_allclose(model.score_batch(users, candidates), expected, rtol=0, atol=5e-12)
    assert_array_equal(model.item_ids_, raw_ids)
    assert model.diagnostics["active_components"] == 2
    assert model.diagnostics["dangling_nodes"] == 2
    assert model.diagnostics["training_weight"] == 34
    # Repeats are eligible, not just graph-discovered programs.
    for user in [0, 1, 2, 3]:
        assert np.isfinite(model.score(user, np.flatnonzero(counts[user]))).all()

    sources = np.array([2, 0, 3, 4])
    actual = ppr._iterate_ppr(
        model._transition_transpose, model._dangling, sources,
        alpha=alpha, tol=model.tol, max_iter=model.max_iter,
    )
    for column, source in enumerate(sources):
        exact = linear_solution(transition, int(source), alpha)
        returned = actual.probabilities[:, column]
        observed = independent_residual(transition, int(source), alpha, returned)
        assert_allclose(returned, exact, rtol=0, atol=5e-12)
        assert_allclose(actual.residual_l1[column], observed, rtol=0, atol=8e-16)
        assert observed <= model.tol + 8e-16
        assert np.abs(returned - exact).sum() <= observed / (1 - alpha) + 5e-15
        assert_allclose(returned.sum(), 1, rtol=0, atol=4e-15)
        # Total-node mass is one; item-only mass is NOT one.
        expected_item_mass = 0 if source == 4 else alpha / (1 + alpha)
        assert_allclose(returned[len(counts):].sum(), expected_item_mass, atol=5e-12, rtol=0)


def test_counts_not_binary_and_global_weight_rescaling_invariant(weighted_counts):
    counts = weighted_counts
    model = PersonalizedPageRankModel(tol=1e-12).fit(make_training(counts))
    scaled = PersonalizedPageRankModel(tol=1e-12).fit(make_training(13 * counts))
    candidates = np.arange(counts.shape[1])
    expected = oracle_scores(counts, [0, 1, 2, 3], candidates, model.alpha)
    actual = model.score_batch([0, 1, 2, 3], candidates)
    assert_allclose(actual, expected, rtol=0, atol=1e-11)
    assert_allclose(scaled.score_batch([0, 1, 2, 3], candidates), actual, rtol=0, atol=2e-15)
    binary = oracle_scores((counts > 0).astype(float), [0, 1, 2, 3], candidates, model.alpha)
    assert np.nanmax(np.abs(binary - actual)) > 0.03


def test_directed_dangling_mass_uses_each_queries_own_restart():
    # Directed sinks are deliberately a kernel fixture: active components of
    # the production undirected graph cannot reach their isolated cold nodes.
    transition = np.array([
        [0, 0.2, 0.8, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
    ], dtype=float)
    sources = np.array([3, 0, 2])
    result = ppr._iterate_ppr(
        csr_matrix(transition.T), np.array([False, False, True, False]), sources,
        alpha=0.73, tol=1e-12, max_iter=300,
    )
    assert result.converged.all()
    for column, source in enumerate(sources):
        expected = linear_solution(transition, int(source), 0.73)
        assert_allclose(result.probabilities[:, column], expected, rtol=0, atol=2e-12)
        assert_allclose(result.probabilities[:, column].sum(), 1, rtol=0, atol=2e-15)
        assert_allclose(
            result.residual_l1[column],
            independent_residual(transition, int(source), 0.73, result.probabilities[:, column]),
            rtol=0, atol=5e-16,
        )
    assert result.iterations[2] == 0


def first_acceptable_iterate(transition, source, alpha, tol):
    """Independent dense, single-query reference; never calls production code."""
    completed = query_transition(transition, source)
    restart = np.zeros(len(completed))
    restart[source] = 1
    probabilities = restart.copy()
    for updates in range(500):
        following = alpha * completed.T @ probabilities + (1 - alpha) * restart
        residual = float(np.abs(following - probabilities).sum())
        if residual <= tol:
            return probabilities, updates, residual
        probabilities = following
    raise AssertionError("The independent finite fixture did not converge")


def test_per_query_convergence_freezes_the_reported_iterate_not_the_next_update():
    transition = np.zeros((7, 7))
    transition[1, 2] = transition[2, 2] = 1
    transition[3, 4] = transition[4, 3] = 1
    transition[5, [0, 1]] = [0.25, 0.75]
    sources = np.array([3, 0, 5, 1, 6, 3])
    alpha, tol = 0.8, 0.05
    result = ppr._iterate_ppr(
        csr_matrix(transition.T), transition.sum(axis=1) == 0, sources,
        alpha=alpha, tol=tol, max_iter=100,
    )
    assert result.converged.all()
    for column, source in enumerate(sources):
        expected, updates, residual = first_acceptable_iterate(transition, int(source), alpha, tol)
        # Sparse dangling reinjection and the equivalent dense completed
        # transition can differ by a few rounding bits, not a different iterate.
        assert_allclose(result.probabilities[:, column], expected, rtol=0, atol=1e-15)
        assert result.iterations[column] == updates
        assert_allclose(result.residual_l1[column], residual, rtol=0, atol=3e-16)
        exact = linear_solution(transition, int(source), alpha)
        assert np.abs(expected - exact).sum() <= residual / (1 - alpha) + 3e-15
    assert_array_equal(result.iterations[[0, 1, 3, 4, 5]], [16, 0, 1, 0, 16])
    assert result.operator_applications == int(result.iterations.max()) + 1


def test_max_iter_boundary_checks_returned_residual_and_does_not_return_partial_scores():
    data = make_training([[1, 3]])
    # After exactly one update at alpha=.5 the full-node fixed-point residual
    # is 2 * .5**2 == .5. Equality should be accepted, not an extra update.
    accepted = PersonalizedPageRankModel(alpha=0.5, tol=0.5, max_iter=1).fit(data)
    assert_allclose(accepted.score(0, [0, 1]), [0.125, 0.375], rtol=0, atol=0)
    info = accepted.diagnostics["last_query"]["solves"][0]
    assert info["iterations"] == 1
    assert info["residual_l1"] == 0.5
    assert accepted.diagnostics["last_query"]["operator_applications"] == 2

    rejected = PersonalizedPageRankModel(
        alpha=0.5, tol=np.nextafter(0.5, 0), max_iter=1
    ).fit(data)
    with pytest.raises(PPRConvergenceError) as caught:
        rejected.score_batch([0, 0], [1, 0, 1])
    error = caught.value
    assert error.user_indices == (0,)
    assert error.iterations == (1,)
    assert error.residual_l1 == (0.5,)
    failed_query = rejected.diagnostics["last_query"]
    failed_snapshot = copy.deepcopy(failed_query)
    assert failed_query["status"] == "nonconverged"
    assert failed_query["converged"] is False
    # A subsequent valid solve must replace failure diagnostics, not reuse a
    # partial output or mutate a retained reference to the failed query.
    rejected.max_iter = 200
    rejected.tol = 1e-12
    assert_allclose(rejected.score(0, [0, 1]), [1 / 12, 1 / 4], rtol=0, atol=1e-12)
    assert rejected.diagnostics["last_query"]["status"] == "converged"
    assert failed_query == failed_snapshot


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_walk_columns_bounded_requested_output_uncapped_and_duplicates_reused(
    weighted_counts, monkeypatch, batch_size,
):
    users = np.array([2, 0, 3, 1, 2, 4, -1, 3])
    candidates = np.arange(weighted_counts.shape[1])
    expected = oracle_scores(weighted_counts, users, candidates, 0.85)
    calls = []
    actual_kernel = ppr._iterate_ppr

    def spy(transition, dangling, sources, **kwargs):
        assert isspmatrix_csr(transition)
        calls.append(tuple(map(int, sources)))
        return actual_kernel(transition, dangling, sources, **kwargs)

    def forbidden_dense(*args, **kwargs):
        raise AssertionError("A production sparse graph must not be densified")

    monkeypatch.setattr(ppr, "_iterate_ppr", spy)
    monkeypatch.setattr(csr_matrix, "toarray", forbidden_dense)
    monkeypatch.setattr(csr_matrix, "todense", forbidden_dense)
    model = PersonalizedPageRankModel(batch_size=batch_size, tol=1e-12).fit(
        make_training(weighted_counts)
    )
    actual = model.score_batch(users, candidates)
    assert_allclose(actual, expected, rtol=0, atol=1e-11)
    assert sorted(user for block in calls for user in block) == [0, 1, 2, 3]
    assert max(map(len, calls)) <= batch_size
    assert model.diagnostics["last_query"]["max_batch_users"] <= batch_size
    # Explicitly characterize the documented output-memory limitation: this is
    # R x C (8 x 7) even when propagation B is one. It is not an assertion that
    # batch_size imposes an output-size cap.
    assert actual.shape == (8, 7)
    assert actual.nbytes == 8 * 7 * 8


def test_input_snapshot_query_arrays_and_returned_scores_have_no_mutable_alias(weighted_counts):
    data = make_training(weighted_counts)
    original_users = tuple(data.user_ids)
    original_items = data.item_ids.copy()
    original_counts = data.interactions.toarray().copy()
    model = PersonalizedPageRankModel(tol=1e-12).fit(data)
    users = np.array([2, 0, 3, 2])
    candidates = np.array([3, 0, 5, 3, 6, -1])
    users_before, candidates_before = users.copy(), candidates.copy()
    expected = oracle_scores(weighted_counts, users, candidates, model.alpha)
    first = model.score_batch(users, candidates)
    assert_allclose(first, expected, rtol=0, atol=1e-11)
    assert_array_equal(users, users_before)
    assert_array_equal(candidates, candidates_before)
    assert_array_equal(data.interactions.toarray(), original_counts)
    assert not np.shares_memory(data.interactions.data, model.transition_.data)
    assert not np.shares_memory(data.item_ids, model.item_ids_)
    data.interactions.data[:] = 99
    data.user_ids[0] = "mutated-after-fit"
    data.item_ids[:] += 10_000
    assert model.user_ids_ == original_users
    assert_array_equal(model.item_ids_, original_items)
    first[:] = -999
    assert_allclose(model.score_batch(users, candidates), expected, rtol=0, atol=1e-11)


def test_refit_replaces_components_and_failed_refit_is_explicitly_fail_closed(weighted_counts):
    model = PersonalizedPageRankModel(tol=1e-12).fit(make_training(weighted_counts))
    model.score(0, [0, 1])
    replacement_counts = np.array([[0, 0, 0, 0], [0, 7, 0, 3]], dtype=float)
    replacement = make_training(replacement_counts)
    model.fit(replacement)
    assert model.diagnostics["last_query"] is None
    assert model.transition_.shape == (6, 6)
    assert model.diagnostics["active_components"] == 1
    assert_allclose(
        model.score_batch([1, 0, 2], [3, 1, 0, 4]),
        oracle_scores(replacement_counts, [1, 0, 2], [3, 1, 0, 4], model.alpha),
        rtol=0, atol=1e-11,
    )
    replacement.interactions.data[0] = np.nan
    with pytest.raises(ValueError, match="finite positive"):
        model.fit(replacement)
    with pytest.raises(RuntimeError, match="Fit"):
        model.score(1, [1])
    # This invalidates the previous fit rather than transactionally preserving
    # it. That is a safe, tested policy difference, not a contract violation.
    assert model.diagnostics == {}
    model.fit(make_training(np.zeros((0, 0))))
    assert model.transition_.shape == (0, 0)
    assert np.isnan(model.score(-1, [0, 1])).all()


def test_alpha_zero_and_one_endpoints_do_not_synthesize_item_evidence(weighted_counts):
    model = PersonalizedPageRankModel(alpha=0).fit(make_training(weighted_counts))
    assert np.isnan(model.score_batch([0, 3, 4], np.arange(7))).all()
    for solve in model.diagnostics["last_query"]["solves"]:
        assert solve["walk_mass"] == 1
        assert solve["item_mass"] == 0
        assert solve["residual_l1"] == 0
        assert solve["iterations"] == 0
    for invalid in [1, np.float64(1), -0.1, np.nan, np.inf, True]:
        with pytest.raises(ValueError, match="alpha"):
            PersonalizedPageRankModel(alpha=invalid)
    near_one = PersonalizedPageRankModel(alpha=np.nextafter(1.0, 0), max_iter=4).fit(
        make_training([[1]])
    )
    with pytest.raises(PPRConvergenceError):
        near_one.score(0, [0])


def test_unknown_uint64_indices_do_not_wrap_to_known_entities(weighted_counts):
    model = PersonalizedPageRankModel().fit(make_training(weighted_counts))
    users = np.array([0, 2**63, 2**64 - 1], dtype=np.uint64)
    candidates = np.array([0, 2**63, 2**64 - 1], dtype=np.uint64)
    users_before, candidates_before = users.copy(), candidates.copy()
    actual = model.score_batch(users, candidates)
    assert np.isfinite(actual[0, 0])
    assert np.isnan(actual[1:]).all()
    assert np.isnan(actual[:, 1:]).all()
    assert_array_equal(users, users_before)
    assert_array_equal(candidates, candidates_before)


@pytest.mark.parametrize("unknown_user", [10**30, -(10**30)])
def test_unknown_python_integer_outside_int64_returns_nan_not_dtype_error(unknown_user):
    """Contract edge regression: score accepts a Python int, not only int64.

    Deliberately not xfailed: the author should reproduce and fix the concrete
    failure, or explicitly resolve the integer-domain contract with the owner.
    """
    model = PersonalizedPageRankModel().fit(make_training([[1]]))
    actual = model.score(unknown_user, np.array([0, -1, 1]))
    assert actual.shape == (3,)
    assert np.isnan(actual).all()
