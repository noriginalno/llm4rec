"""PPR author's independent adversarial MF checks; no production MF edits.

The fixed backend is an API/mapping probe, not a training-performance claim.
Native checks additionally solve one complete ALS iteration with independent
dense normal equations and validate scores with scalar factor arithmetic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix

from sweettv.contracts import TrainingData
from sweettv.models import mf
from sweettv.models.mf import MatrixFactorizationModel


WARM_USERS = [1, 4]
WARM_ITEMS = [1, 5, 7]
COMPACT_COUNTS = np.array([[1, 5, 0], [0, 2, 8]], dtype=np.float32)
FIXED_USERS = np.array([[1, -2, 3], [-1, 2, 1]], dtype=np.float32)
FIXED_ITEMS = np.array([[2, 1, -1], [1, 2, 1], [-2, 0, 1]], dtype=np.float32)


def review_data() -> TrainingData:
    """Global 6x9 / compact 2x3, nonidentity IDs, internal and boundary cold IDs."""
    counts = np.zeros((6, 9), dtype=np.float32)
    counts[np.ix_(WARM_USERS, WARM_ITEMS)] = COMPACT_COUNTS
    return TrainingData(
        user_ids=["0001", "tenant/a", "42", "literal--", "unicode-ü", "0006"],
        item_ids=np.array([9007, 44, 701, 12, 5005, 101, 309, 29, 8001]),
        interactions=csr_matrix(counts),
        metadata=[{"tv_show_category": "shared-even-when-cold"} for _ in range(9)],
        provenance={"heldout_counts": [[999] * 9 for _ in range(6)]},
    )


def empty_data(shape: tuple[int, int]) -> TrainingData:
    users, items = shape
    return TrainingData(
        [f"empty-{user}" for user in range(users)],
        np.arange(101, 101 + items),
        csr_matrix(shape, dtype=np.float32),
        [{} for _ in range(items)],
    )


def scalar_factor_oracle(user, candidates, user_factors, item_factors):
    """Fixture-known global maps and scalar float64 arithmetic; no MF helpers."""
    expected = []
    for candidate in candidates:
        if user not in WARM_USERS or candidate not in WARM_ITEMS:
            expected.append(np.nan)
            continue
        left = user_factors[WARM_USERS.index(user)]
        right = item_factors[WARM_ITEMS.index(candidate)]
        expected.append(
            math.fsum(float(a) * float(b) for a, b in zip(left, right, strict=True))
        )
    return np.asarray(expected, dtype=np.float64)


@pytest.fixture
def backend_probe(monkeypatch):
    class ProbeALS:
        instances = []
        fail_next = False

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            type(self).instances.append(self)

        def fit(self, confidence, show_progress):
            self.received = confidence.copy()
            assert show_progress is False
            if type(self).fail_next:
                type(self).fail_next = False
                raise RuntimeError("adversarial native-fit failure")
            users, items = confidence.shape
            assert users <= len(FIXED_USERS) and items <= len(FIXED_ITEMS)
            assert self.kwargs["factors"] == 3
            self.user_factors = FIXED_USERS[:users].copy()
            self.item_factors = FIXED_ITEMS[:items].copy()

        def recommend(self, *args, **kwargs):
            raise AssertionError("MF must not delegate watched filtering/backfill")

    monkeypatch.setattr(mf, "_load_cpu_als", lambda: (ProbeALS, "adversarial-probe"))
    return ProbeALS


def test_all_global_users_and_items_match_fixed_scalar_oracle(backend_probe):
    data = review_data()
    model = MatrixFactorizationModel(factors=3).fit(data)
    users, candidates = np.arange(6), np.arange(9)
    expected = np.vstack(
        [scalar_factor_oracle(user, candidates, FIXED_USERS, FIXED_ITEMS) for user in users]
    )
    assert_allclose(model.score_batch(users, candidates), expected, equal_nan=True)
    for user in users:
        assert_allclose(model.score(user, candidates), expected[user], equal_nan=True)
    assert_array_equal(model.warm_user_indices, WARM_USERS)
    assert_array_equal(model.warm_item_indices, WARM_ITEMS)
    assert_array_equal(model.user_index_to_factor, [-1, 0, -1, -1, 1, -1])
    assert_array_equal(model.item_index_to_factor, [-1, 0, -1, -1, -1, 1, -1, 2, -1])
    assert np.isnan(model.score(1, data.item_ids)).all()  # Raw IDs are not indices.
    assert data.user_ids[0] == "0001"
    assert model.diagnostics["training_shape"] == [6, 9]
    assert model.diagnostics["compact_training_shape"] == [2, 3]


@pytest.mark.parametrize("alpha", [0.0, 2.3, 20.0])
def test_confidence_is_one_plus_log_count_once_with_users_by_items(alpha, backend_probe):
    data = review_data()
    before = data.interactions.copy()
    model = MatrixFactorizationModel(factors=3, alpha=alpha).fit(data)
    backend = backend_probe.instances[-1]
    expected = np.array(
        [
            [1 + alpha * math.log1p(float(count)) if count > 0 else 0 for count in row]
            for row in COMPACT_COUNTS
        ],
        dtype=np.float32,
    )
    assert isinstance(backend.received, csr_matrix)
    assert backend.received.shape == (2, 3), "Transposition changes the API semantics"
    assert backend.received.dtype == np.float32
    assert_array_equal(backend.received.toarray(), expected)
    assert backend.kwargs["alpha"] == 1.0, "Never multiply the transformed confidence twice"
    assert backend.kwargs["dtype"] is np.float32
    assert backend.kwargs["use_native"] is True
    assert backend.kwargs["use_cg"] is True
    assert_array_equal(data.interactions.data, before.data)
    assert_array_equal(data.interactions.indices, before.indices)
    assert_array_equal(data.interactions.indptr, before.indptr)
    assert model.diagnostics["confidence_min"] == float(expected[expected > 0].min())
    assert model.diagnostics["confidence_max"] == float(expected.max())


def test_orders_duplicates_repeat_eligibility_and_finite_negative_or_zero(backend_probe):
    model = MatrixFactorizationModel(factors=3).fit(review_data())
    users = np.array([4, 1, 4, -1, 5, 6])
    candidates = np.array([7, 1, 7, 5, 0, -1, 9, 1])
    expected = np.vstack(
        [scalar_factor_oracle(user, candidates, FIXED_USERS, FIXED_ITEMS) for user in users]
    )
    actual = model.score_batch(users, candidates)
    assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert_allclose(model.score(1, [1, 5, 7]), [-3, 0, 1], rtol=0, atol=0)
    assert np.isfinite(model.score(1, [1, 5])).all()  # Both watched, one negative and one zero.
    assert_array_equal(actual[0], actual[2])
    assert_array_equal(actual[:, 0], actual[:, 2])


def test_unknown_signed_unsigned_and_extreme_scalar_indices_are_nan(backend_probe):
    model = MatrixFactorizationModel(factors=3).fit(review_data())
    for user in [-1, -2**63, 6, 10**100, np.uint64(2**64 - 1)]:
        assert np.isnan(model.score(user, [1, 5, 7])).all()
    assert np.isnan(model.score(1, [-2**63, -1, 0, 9, 2**63 - 1])).all()
    unsigned = np.array([1, 7, 2**64 - 1], dtype=np.uint64)
    assert_allclose(model.score(1, unsigned)[:2], [-3, 1])
    assert np.isnan(model.score(1, unsigned)[2])
    actual = model.score_batch(np.array([4, 2**64 - 1], dtype=np.uint64), unsigned)
    assert_allclose(actual[0, :2], [-1, 3])
    assert np.isnan(actual[1]).all()


def test_scoring_does_not_mutate_index_arguments_or_input_data(backend_probe):
    data = review_data()
    model = MatrixFactorizationModel(factors=3).fit(data)
    users, candidates = np.array([4, 1, 4, -1]), np.array([7, 1, 5, -1, 1])
    original_users, original_candidates = users.copy(), candidates.copy()
    baseline = model.score_batch(users, candidates)
    model.score(1, candidates)
    assert_array_equal(users, original_users)
    assert_array_equal(candidates, original_candidates)
    data.interactions.data[:] = 999
    data.metadata[:] = [{"future_only": True} for _ in data.metadata]
    data.provenance["heldout_counts"] = None
    data.item_ids[:] = np.arange(100_001, 100_010)
    data.user_ids[:] = [f"mutated-{i}" for i in range(6)]
    assert_allclose(model.score_batch(users, candidates), baseline, equal_nan=True)


def test_model_instances_and_successful_refits_do_not_share_old_factor_buffers(backend_probe):
    data = review_data()
    first = MatrixFactorizationModel(factors=3).fit(data)
    second = MatrixFactorizationModel(factors=3).fit(data)
    assert first.als_model_ is not second.als_model_
    assert not np.shares_memory(first.user_factors, second.user_factors)
    assert not np.shares_memory(first.item_factors, second.item_factors)
    old_backend = first.als_model_
    first.fit(data)
    assert first.als_model_ is not old_backend
    assert not np.shares_memory(first.user_factors, old_backend.user_factors)
    assert not np.shares_memory(first.item_factors, old_backend.item_factors)
    baseline = first.score(1, [1, 5, 7])
    old_backend.user_factors[:] = 1234
    old_backend.item_factors[:] = -5678
    assert_allclose(first.score(1, [1, 5, 7]), baseline)
    assert_allclose(second.score(1, [1, 5, 7]), baseline)


def test_native_fit_exception_preserves_previous_successful_state(backend_probe):
    data = review_data()
    model = MatrixFactorizationModel(factors=3).fit(data)
    previous_backend, previous_diagnostics = model.als_model_, model.diagnostics
    previous_scores = model.score(1, [1, 5, 7])
    backend_probe.fail_next = True
    with pytest.raises(RuntimeError, match="adversarial native-fit failure"):
        model.fit(data)
    assert model.als_model_ is previous_backend
    assert model.diagnostics is previous_diagnostics
    assert_allclose(model.score(1, [1, 5, 7]), previous_scores)


@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (4, 0), (4, 5)])
def test_empty_fit_and_empty_refit_never_load_backend(shape, monkeypatch):
    def forbidden():
        raise AssertionError("No backend is needed for a graph without training evidence")

    monkeypatch.setattr(mf, "_load_cpu_als", forbidden)
    model = MatrixFactorizationModel(factors=3).fit(empty_data(shape))
    assert model.diagnostics["status"] == "empty"
    assert model.als_model_ is None
    assert np.isnan(model.score(0, [0, 1, -1])).all()
    assert model.score_batch([], [0]).shape == (0, 1)
    assert model.score_batch([0], []).shape == (1, 0)
    model.fit(empty_data((2, 7)))
    assert model.diagnostics["training_shape"] == [2, 7]
    assert len(model.user_index_to_factor) == 2
    assert len(model.item_index_to_factor) == 7


def test_warm_empty_then_differently_shaped_warm_refit_has_no_stale_mappings(backend_probe):
    model = MatrixFactorizationModel(factors=3).fit(review_data())
    model.fit(empty_data((2, 2)))
    assert np.isnan(model.score(1, [1])).all()
    changed = empty_data((3, 4))
    changed.interactions = csr_matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0]])
    model.fit(changed)
    assert_array_equal(model.warm_user_indices, [2])
    assert_array_equal(model.warm_item_indices, [2])
    assert_array_equal(model.user_index_to_factor, [-1, -1, 0])
    assert_array_equal(model.item_index_to_factor, [-1, -1, 0, -1])
    assert np.isnan(model.score(1, [2])).all()
    assert model.score(2, [2])[0] == -3


def test_batch_output_guard_precedes_dense_allocation_and_counts_duplicates(
    backend_probe, monkeypatch
):
    model = MatrixFactorizationModel(factors=3, max_batch_scores=12).fit(review_data())
    exact = model.score_batch([1, 4, 1], [7, 5, 1, 7])
    assert exact.shape == (3, 4)
    assert exact.nbytes == 12 * np.dtype(np.float32).itemsize
    real_full = np.full

    def guarded_full(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2 and math.prod(shape) > 12:
            raise AssertionError("Oversized requested output was allocated before the cap")
        return real_full(shape, *args, **kwargs)

    monkeypatch.setattr(mf.np, "full", guarded_full)
    with pytest.raises(ValueError, match="max_batch_scores"):
        model.score_batch([1, 4, 1], [7, 5, 1, 7, 1])
    with pytest.raises(ValueError, match="max_batch_scores"):
        model.score_batch([-1, -1, -1], [-1, -1, -1, -1, -1])


def test_missing_blas_controller_is_observed_as_empty_not_a_thread_measurement(
    backend_probe, monkeypatch
):
    # Reproduce an unsupported/no-controller environment without inventing a
    # pool. An empty list is NOT proof of one-thread BLAS execution.
    monkeypatch.setattr(mf, "threadpool_info", lambda: [])
    model = MatrixFactorizationModel(factors=3, num_threads=2).fit(review_data())
    assert model.diagnostics["training_blas_pools"] == []
    assert backend_probe.instances[-1].kwargs["num_threads"] == 2


@pytest.fixture
def native_available():
    dependency = pytest.importorskip("implicit")
    pytest.importorskip("implicit.cpu.als")
    return dependency


def independent_one_als_iteration(counts, alpha, regularization, seed, factors):
    """Direct linear solves, not implicit's CG solver or MF transformation helper.

    Version 0.7.3 initializes users and then items with default_rng and float32.
    The first exact alternating update only needs the initial item vectors.
    Observed preferences are BINARY, with missing confidence equal to one.
    """
    counts = np.asarray(counts, dtype=np.float64)
    users, items = counts.shape
    rng = np.random.default_rng(seed)
    rng.random((users, factors), dtype=np.float32)
    initial_items = (
        rng.random((items, factors), dtype=np.float32) * np.float32(0.01)
    ).astype(np.float64)
    confidence = np.ones((users, items), dtype=np.float64)
    for user in range(users):
        for item in range(items):
            if counts[user, item] > 0:
                confidence[user, item] = np.float32(
                    1 + alpha * math.log1p(counts[user, item])
                )
    preference = (counts > 0).astype(np.float64)
    user_factors = np.empty((users, factors), dtype=np.float64)
    for user in range(users):
        left = initial_items.T @ (confidence[user, :, None] * initial_items)
        right = initial_items.T @ (confidence[user] * preference[user])
        user_factors[user] = np.linalg.solve(
            left + regularization * np.eye(factors), right
        )
    item_factors = np.empty((items, factors), dtype=np.float64)
    for item in range(items):
        left = user_factors.T @ (confidence[:, item, None] * user_factors)
        right = user_factors.T @ (confidence[:, item] * preference[:, item])
        item_factors[item] = np.linalg.solve(
            left + regularization * np.eye(factors), right
        )
    return user_factors, item_factors


def test_native_one_iteration_matches_independent_binary_preference_normal_equations(
    native_available,
):
    assert native_available.__version__ == "0.7.3", "Oracle pins the audited initialization API"
    kwargs = dict(factors=2, iterations=1, alpha=2.3, regularization=0.2, seed=79)
    model = MatrixFactorizationModel(**kwargs, num_threads=1).fit(review_data())
    expected_users, expected_items = independent_one_als_iteration(
        COMPACT_COUNTS, kwargs["alpha"], kwargs["regularization"],
        kwargs["seed"], kwargs["factors"],
    )
    assert_allclose(model.user_factors, expected_users, rtol=3e-6, atol=3e-7)
    assert_allclose(model.item_factors, expected_items, rtol=3e-6, atol=3e-7)
    expected = np.vstack(
        [scalar_factor_oracle(user, np.arange(9), expected_users, expected_items)
         for user in np.arange(6)]
    )
    assert_allclose(
        model.score_batch(np.arange(6), np.arange(9)), expected,
        rtol=3e-6, atol=3e-7, equal_nan=True,
    )
    assert model.user_factors.dtype == np.float32
    assert model.item_factors.dtype == np.float32
    assert model.score(1, np.arange(9)).dtype == np.float32
    assert model.als_model_.alpha == 1.0
    assert model.als_model_.use_native is True
    assert model.als_model_.solver.func.__module__ == "implicit.cpu._als"
    assert model.diagnostics["backend"] == "implicit.cpu.als.AlternatingLeastSquares"


def test_native_seed_repeat_and_one_vs_four_threads_match_factor_oracle(native_available):
    data = review_data()
    models = [
        MatrixFactorizationModel(factors=3, iterations=5, seed=79, num_threads=threads).fit(data)
        for threads in [1, 1, 4, 4]
    ]
    candidates = np.array([7, 1, 5, 7, 0, -1, 9])
    for model in models:
        for user in WARM_USERS:
            expected = scalar_factor_oracle(
                user, candidates, model.user_factors, model.item_factors
            )
            assert_allclose(
                model.score(user, candidates), expected,
                rtol=3e-6, atol=3e-7, equal_nan=True,
            )
        assert_allclose(
            model.user_factors, models[0].user_factors, rtol=3e-6, atol=3e-7
        )
        assert_allclose(
            model.item_factors, models[0].item_factors, rtol=3e-6, atol=3e-7
        )
    # This is tolerance-based reproducibility on this tiny fixture, not a
    # guarantee of bitwise identity across BLAS backends, machines, or versions.
