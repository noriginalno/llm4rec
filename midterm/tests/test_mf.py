"""Independent scoring/API checks plus small, network-free native ALS checks."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix

from sweettv.contracts import TrainingData
from sweettv.models import mf
from sweettv.models.mf import MatrixFactorizationModel


def make_data(values: np.ndarray) -> TrainingData:
    n_users, n_items = values.shape
    return TrainingData(
        user_ids=[f"user-{i}" for i in range(n_users)],
        item_ids=np.arange(101, 101 + n_items),
        interactions=csr_matrix(values, dtype=np.float32),
        metadata=[{} for _ in range(n_items)],
    )


@pytest.fixture
def data() -> TrainingData:
    # Global 5 x 8, compact 3 x 5: both nonsquare, with interleaved cold rows
    # and columns so a transpose or identity-index shortcut cannot pass.
    return make_data(np.array([
        [0, 3, 0, 0, 2, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 2, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32))


@pytest.fixture
def fake_als(monkeypatch):
    class RecordingALS:
        """Test double for API/scoring checks, NOT evidence of ALS training."""

        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

        def fit(self, user_items, show_progress):
            self.received = user_items.copy()
            self.show_progress = show_progress
            users, items = user_items.shape
            factors = self.kwargs["factors"]
            self.user_factors = np.arange(1, users * factors + 1, dtype=np.float32).reshape(
                users, factors
            )
            self.item_factors = np.arange(1, items * factors + 1, dtype=np.float32).reshape(
                items, factors
            )
            self.item_factors[::2] *= -1

        def recommend(self, *args, **kwargs):
            raise AssertionError("MF must not call implicit.recommend or its backfill/filter")

    monkeypatch.setattr(mf, "_load_cpu_als", lambda: (RecordingALS, "test-double"))
    return RecordingALS


def independent_score(model, user_index, candidates):
    """Scalar arithmetic on fitted arrays, with fixture-derived index maps.

    Does not call score/score_batch, production index helpers, or recommend.
    """
    warm_users, warm_items = [0, 2, 3], [1, 2, 4, 5, 7]
    expected = []
    for candidate in candidates:
        if user_index not in warm_users or candidate not in warm_items:
            expected.append(np.nan)
        else:
            user = model.als_model_.user_factors[warm_users.index(user_index)]
            item = model.als_model_.item_factors[warm_items.index(candidate)]
            expected.append(sum(float(a) * float(b) for a, b in zip(user, item, strict=True)))
    return np.asarray(expected)


def test_defaults():
    model = MatrixFactorizationModel()
    assert (model.factors, model.iterations, model.regularization) == (32, 15, 0.1)
    assert (model.alpha, model.seed, model.num_threads) == (20.0, 42, 4)
    assert model.diagnostics == {"status": "not_fitted", "backend": None}


def test_confidence_orientation_and_compact_mapping(data, fake_als):
    original = data.interactions.copy()
    model = MatrixFactorizationModel(factors=2)
    assert model.fit(data) is model
    backend = fake_als.instances[-1]
    assert isinstance(backend.received, csr_matrix)
    assert backend.received.shape == (3, 5)  # NOT items x users
    # Independently construct observed values with Python's scalar math.log1p.
    counts = [[3, 0, 2, 0, 1], [0, 4, 0, 2, 0], [1, 1, 0, 1, 0]]
    expected = [[1 + 20 * math.log1p(c) if c else 0 for c in row] for row in counts]
    assert_allclose(backend.received.toarray(), expected, rtol=1e-7)
    assert backend.kwargs == {
        "factors": 2, "iterations": 15, "regularization": 0.1, "alpha": 1.0,
        "dtype": np.float32, "use_native": True, "use_cg": True,
        "calculate_training_loss": False, "num_threads": 4, "random_state": 42,
    }
    assert backend.show_progress is False
    assert_array_equal(model.warm_user_indices, [0, 2, 3])
    assert_array_equal(model.warm_item_indices, [1, 2, 4, 5, 7])
    assert_array_equal(model.user_index_to_factor, [0, -1, 1, 2, -1])
    assert_array_equal(model.item_index_to_factor, [-1, 0, 1, -1, 2, 3, -1, 4])
    assert model.user_factors.shape == (3, 2)
    assert model.item_factors.shape == (5, 2)
    assert_array_equal(data.interactions.toarray(), original.toarray())
    assert_array_equal(data.interactions.data, original.data)


def test_configurable_confidence_and_no_double_scaling(data, fake_als):
    model = MatrixFactorizationModel(factors=3, alpha=2.5, iterations=4, seed=17, num_threads=1)
    model.fit(data)
    backend = fake_als.instances[-1]
    assert backend.kwargs["alpha"] == 1
    assert backend.kwargs["random_state"] == 17
    assert backend.kwargs["iterations"] == 4
    assert backend.kwargs["num_threads"] == 1
    assert_allclose(backend.received.data[0], 1 + 2.5 * math.log1p(3), rtol=1e-7)
    MatrixFactorizationModel(factors=2, alpha=0).fit(data)
    assert_array_equal(fake_als.instances[-1].received.data, 1.0)


@pytest.mark.parametrize("user_index", [0, 2, 3])
def test_direct_dot_products_order_duplicates_and_watched_eligibility(data, fake_als, user_index):
    model = MatrixFactorizationModel(factors=2).fit(data)
    candidates = np.array([7, 1, 7, 0, 2, -1, 8, 4, 5, 101])
    actual = model.score(user_index, candidates)
    assert actual.shape == candidates.shape
    assert np.issubdtype(actual.dtype, np.floating)
    assert_allclose(actual, independent_score(model, user_index, candidates), equal_nan=True)
    assert actual[0] == actual[2]
    # Every training-watched program must remain eligible, not be filtered.
    watched = data.interactions[user_index].indices
    assert np.isfinite(model.score(user_index, watched)).all()


def test_negative_and_genuine_zero_dot_products_not_clipped(data, fake_als):
    model = MatrixFactorizationModel(factors=2).fit(data)
    model.user_factors[0] = [1, 2]
    model.item_factors[0] = [2, -1]  # nonzero factors, genuine zero dot product
    model.item_factors[1] = [-3, 1]  # genuine negative dot product
    assert_allclose(model.score(0, np.array([1, 2, 1])), [0, -1, 0])


@pytest.mark.parametrize("user", [-1, -20, 1, 4, 5, 999, 10**30, np.int64(2**62)])
def test_unknown_or_cold_users_have_no_evidence(data, fake_als, user):
    model = MatrixFactorizationModel(factors=2).fit(data)
    assert np.isnan(model.score(user, np.arange(8))).all()


def test_unknown_cold_and_unsigned_overflow_candidates(data, fake_als):
    model = MatrixFactorizationModel(factors=2).fit(data)
    assert np.isnan(model.score(0, np.array([-1, 0, 3, 6, 8, 999]))).all()
    actual = model.score(0, np.array([1, 7, 2**64 - 1], dtype=np.uint64))
    assert_allclose(actual[:2], independent_score(model, 0, [1, 7]))
    assert np.isnan(actual[2])
    assert model.score(0, np.array([], dtype=float)).shape == (0,)


def test_score_batch_independent_oracle_and_no_global_cache(data, fake_als):
    model = MatrixFactorizationModel(factors=2).fit(data)
    users = np.array([3, 1, -1, 0, 3, 5, 2])
    candidates = np.array([7, 4, 1, 7, -1, 3, 8])
    expected = np.vstack([independent_score(model, int(u), candidates) for u in users])
    assert_allclose(model.score_batch(users, candidates), expected, equal_nan=True)
    assert model.score_batch([], candidates).shape == (0, len(candidates))
    assert model.score_batch(users, []).shape == (len(users), 0)
    assert model.score_batch([], []).shape == (0, 0)
    assert not any(
        isinstance(value, np.ndarray) and value.shape == data.interactions.shape
        for value in vars(model).values()
    )


def test_score_batch_guards_allocation_and_unsigned_users(data, fake_als):
    model = MatrixFactorizationModel(factors=2, max_batch_scores=4).fit(data)
    with pytest.raises(ValueError, match="smaller batches"):
        model.score_batch([0, 2], [1, 2, 4])
    actual = model.score_batch(np.array([0, 2**64 - 1], dtype=np.uint64), [1, 2])
    assert_allclose(actual[0], independent_score(model, 0, [1, 2]))
    assert np.isnan(actual[1]).all()


@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (3, 0), (3, 4)])
def test_empty_and_all_zero_data_do_not_load_backend(shape, monkeypatch):
    def forbidden():
        raise AssertionError("Empty fits must not load implicit or create random factors")

    monkeypatch.setattr(mf, "_load_cpu_als", forbidden)
    model = MatrixFactorizationModel(factors=2).fit(make_data(np.zeros(shape)))
    assert model.diagnostics["status"] == "empty"
    assert model.diagnostics["backend"] is None
    assert model.diagnostics["implicit_version"] is None
    assert model.diagnostics["training_shape"] == list(shape)
    assert model.diagnostics["compact_training_shape"] == [0, 0]
    assert model.diagnostics["training_seconds"] == 0
    assert model.diagnostics["training_blas_thread_limit_requested"] is None
    assert model.diagnostics["training_blas_control_status"] == "not_requested_empty_fit"
    assert model.diagnostics["training_blas_pools"] == []
    assert model.als_model_ is None
    assert model.user_factors.shape == (0, 2)
    assert model.item_factors.shape == (0, 2)
    assert np.isnan(model.score(0, np.arange(shape[1]))).all()
    assert np.isnan(model.score_batch([0, -1], [0, 1])).all()


def test_empty_refit_clears_old_state(data, fake_als):
    model = MatrixFactorizationModel(factors=2).fit(data)
    model.fit(make_data(np.zeros((2, 3))))
    assert model.als_model_ is None
    assert model.diagnostics["nnz"] == 0
    assert len(model.user_index_to_factor) == 2
    assert len(model.item_index_to_factor) == 3
    assert np.isnan(model.score(0, [1, 2])).all()


def test_diagnostics_are_actual_and_json_serializable(data, fake_als):
    model = MatrixFactorizationModel(factors=2).fit(data)
    d = model.diagnostics
    assert d["backend"].endswith("RecordingALS")
    assert d["implicit_version"] == "test-double"
    assert d["device"] == "cpu"
    assert d["training_shape"] == [5, 8]
    assert d["compact_training_shape"] == [3, 5]
    assert (d["n_users"], d["n_items"], d["nnz"]) == (5, 8, 8)
    assert (d["n_warm_users"], d["n_warm_items"]) == (3, 5)
    assert (d["n_cold_users"], d["n_cold_items"]) == (2, 3)
    assert (d["user_factor_coverage"], d["item_factor_coverage"]) == (3 / 5, 5 / 8)
    assert (d["positive_preference"], d["missing_preference"], d["missing_confidence"]) == (1, 0, 1)
    assert d["training_seconds"] >= 0
    assert d["fit_seconds"] >= d["training_seconds"]
    assert "training_blas_thread_limit" not in d
    assert d["training_blas_thread_limit_requested"] == 1
    pools = d["training_blas_pools"]
    if not pools:
        assert d["training_blas_control_status"] == "unverified_no_reported_pools"
    elif all(pool["num_threads"] == 1 for pool in pools):
        assert d["training_blas_control_status"] == "reported_pools_at_requested_limit"
    else:
        assert d["training_blas_control_status"] == "reported_pools_not_at_requested_limit"
    json.dumps(d, allow_nan=False)


@pytest.mark.parametrize("reports, expected_status", [
    ([], "unverified_no_reported_pools"),
    ([{"user_api": "openmp", "num_threads": 4}], "unverified_no_reported_pools"),
    ([{"user_api": "blas", "internal_api": "openblas", "num_threads": 1}],
     "reported_pools_at_requested_limit"),
    ([{"user_api": "blas", "internal_api": "openblas", "num_threads": 1},
      {"user_api": "blas", "internal_api": "mkl", "num_threads": 4}],
     "reported_pools_not_at_requested_limit"),
])
def test_blas_limit_request_distinguished_from_reported_control(
    data, fake_als, monkeypatch, reports, expected_status,
):
    # Synthetic controller reports exercise classification, not actual hardware
    # control. In particular, no BLAS report must never vacuously verify a cap.
    monkeypatch.setattr(mf, "threadpool_info", lambda: reports)
    model = MatrixFactorizationModel(factors=2, num_threads=3).fit(data)
    d = model.diagnostics
    assert d["training_blas_thread_limit_requested"] == 1
    assert d["training_blas_control_status"] == expected_status
    assert len(d["training_blas_pools"]) == sum(r["user_api"] == "blas" for r in reports)
    assert fake_als.instances[-1].kwargs["num_threads"] == 3
    assert "training_blas_thread_limit" not in d
    json.dumps(d, allow_nan=False)


def test_zero_factor_vectors_are_not_evidence(data, fake_als, monkeypatch):
    original_fit = fake_als.fit

    def fit_with_collapsed_factors(self, *args, **kwargs):
        original_fit(self, *args, **kwargs)
        self.user_factors[1] = 0
        self.item_factors[0] = 0

    monkeypatch.setattr(fake_als, "fit", fit_with_collapsed_factors)
    model = MatrixFactorizationModel(factors=2).fit(data)
    assert np.isnan(model.score(2, [1, 2, 4])).all()
    assert np.isnan(model.score(0, [1])).all()
    assert np.isfinite(model.score(0, [2, 4])).all()
    assert np.isnan(model.score_batch([0, 2], [1, 2]))[1].all()
    assert model.diagnostics["n_users_with_factor_evidence"] == 2
    assert model.diagnostics["n_items_with_factor_evidence"] == 4


@pytest.mark.parametrize("indices, error", [
    ([1.0, 2.0], TypeError), ([True, False], TypeError), (["1"], TypeError),
    (np.array([[1, 2]]), ValueError), (np.array(1), ValueError),
])
def test_malformed_index_arrays_rejected(data, fake_als, indices, error):
    model = MatrixFactorizationModel(factors=2).fit(data)
    with pytest.raises(error):
        model.score(0, indices)
    with pytest.raises(error):
        model.score_batch(indices, [1, 2])
    with pytest.raises(error):
        model.score_batch([0], indices)


@pytest.mark.parametrize("user", [1.0, True, None, "0", [0]])
def test_malformed_scalar_users_rejected(data, fake_als, user):
    model = MatrixFactorizationModel(factors=2).fit(data)
    with pytest.raises(TypeError, match="user_index"):
        model.score(user, [1, 2])


def test_score_requires_fit():
    model = MatrixFactorizationModel()
    with pytest.raises(RuntimeError, match="fit"):
        model.score(0, [1])
    with pytest.raises(RuntimeError, match="fit"):
        model.score_batch([0], [1])


@pytest.mark.parametrize("kwargs, error", [
    ({"factors": 0}, ValueError), ({"factors": 2.5}, TypeError),
    ({"factors": True}, TypeError), ({"iterations": 0}, ValueError),
    ({"regularization": 0}, ValueError), ({"regularization": -0.1}, ValueError),
    ({"regularization": float("inf")}, ValueError), ({"alpha": -1}, ValueError),
    ({"alpha": float("nan")}, ValueError), ({"alpha": "20"}, TypeError),
    ({"seed": -1}, ValueError), ({"seed": None}, TypeError),
    ({"num_threads": 0}, ValueError), ({"max_batch_scores": 0}, ValueError),
])
def test_invalid_configuration(kwargs, error):
    with pytest.raises(error):
        MatrixFactorizationModel(**kwargs)


@pytest.mark.parametrize("bad_count", [-1, np.nan, np.inf])
def test_invalid_mutated_counts_fail_without_losing_previous_fit(data, fake_als, bad_count):
    model = MatrixFactorizationModel(factors=2).fit(data)
    previous = model.score(0, [1, 2])
    data.interactions.data[0] = bad_count
    with pytest.raises(ValueError, match="finite positive"):
        model.fit(data)
    assert_allclose(model.score(0, [1, 2]), previous)


def test_confidence_overflow_is_explicit(data, fake_als):
    with pytest.raises(ValueError, match="confidences"):
        MatrixFactorizationModel(alpha=1e308).fit(data)
    assert fake_als.instances == []


@pytest.mark.parametrize("defect", ["shape", "nonfinite"])
def test_native_bad_factors_are_not_silently_accepted(data, fake_als, monkeypatch, defect):
    original_fit = fake_als.fit

    def bad_fit(self, *args, **kwargs):
        original_fit(self, *args, **kwargs)
        if defect == "shape":
            self.user_factors = self.user_factors[:-1]
        else:
            self.item_factors[0, 0] = np.nan

    monkeypatch.setattr(fake_als, "fit", bad_fit)
    model = MatrixFactorizationModel(factors=2)
    with pytest.raises(RuntimeError, match="Native ALS"):
        model.fit(data)
    with pytest.raises(RuntimeError, match="fit"):
        model.score(0, [1])


@pytest.fixture
def native_als():
    # Core API tests above use a named test double. These checks actually fit
    # implicit's compiled CPU implementation if the dependency is available.
    implicit = pytest.importorskip("implicit", reason="native implicit wheel is not installed")
    pytest.importorskip("implicit.cpu.als", reason="implicit CPU extension is unavailable")
    return implicit


@pytest.mark.parametrize("num_threads", [1, 4])
def test_actual_cpu_als_smoke_orientation_dot_products_and_repeats(data, native_als, num_threads):
    model = MatrixFactorizationModel(factors=3, iterations=5, num_threads=num_threads).fit(data)
    assert model.diagnostics["backend"] == "implicit.cpu.als.AlternatingLeastSquares"
    assert model.diagnostics["implicit_version"] == native_als.__version__
    assert model.user_factors.shape == (3, 3)
    assert model.item_factors.shape == (5, 3)
    assert model.diagnostics["training_seconds"] > 0
    candidates = np.array([7, 1, 7, 2, 5, 4, 0, -1, 8])
    for user in [0, 2, 3]:
        assert_allclose(
            model.score(user, candidates), independent_score(model, user, candidates),
            rtol=2e-6, atol=2e-6, equal_nan=True,
        )
        assert np.isfinite(model.score(user, data.interactions[user].indices)).all()
    assert np.isnan(model.score(1, candidates)).all()
    assert np.isnan(model.score(-1, candidates)).all()
    assert_allclose(
        model.score_batch([3, 0, 3, 1], candidates),
        np.vstack([independent_score(model, u, candidates) for u in [3, 0, 3, 1]]),
        rtol=2e-6, atol=2e-6, equal_nan=True,
    )


def test_actual_cpu_als_seed_reproducibility_and_fresh_refit(data, native_als):
    kwargs = dict(factors=3, iterations=6, num_threads=1, seed=42)
    first = MatrixFactorizationModel(**kwargs).fit(data)
    second = MatrixFactorizationModel(**kwargs).fit(data)
    assert_allclose(first.user_factors, second.user_factors, rtol=2e-6, atol=2e-6)
    assert_allclose(first.item_factors, second.item_factors, rtol=2e-6, atol=2e-6)
    candidates = np.array([7, 1, 2, 4, 5])
    old_scores = first.score_batch([0, 2, 3], candidates)
    old_backend = first.als_model_
    first.fit(data)
    assert first.als_model_ is not old_backend
    assert_allclose(first.score_batch([0, 2, 3], candidates), old_scores, rtol=2e-6, atol=2e-6)


def test_actual_singleton_warm_component_and_cold_padding(native_als):
    data = make_data(np.array([[0, 0, 0], [0, 2, 0]], dtype=np.float32))
    model = MatrixFactorizationModel(factors=2, iterations=4, num_threads=1).fit(data)
    assert model.diagnostics["compact_training_shape"] == [1, 1]
    assert np.isfinite(model.score(1, [1])).all()
    assert np.isnan(model.score(0, [1])).all()
    assert np.isnan(model.score(1, [0, 2])).all()
