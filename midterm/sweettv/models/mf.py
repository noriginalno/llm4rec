"""CPU implicit-feedback ALS, not regression on broadcast counts.

The fitted objective is sum_ui c_ui * (p_ui - x_u @ y_i)**2 plus L2
regularization. Observed counts give p_ui=1 and c_ui=1+alpha*log1p(count);
missing entries give p_ui=0, c_ui=1. Frequency changes confidence in binary
affinity, not the target value. Scores are uncalibrated latent dot products,
including legitimate negative values, rather than predicted viewing counts.

API audit: implicit v0.7.3's implicit/cpu/als.py:fit accepts USERS x ITEMS CSR
and transposes it internally. Its alpha multiplies stored confidences, so we
pass alpha=1 after applying the transform ourselves. The native _als.pyx
least-squares solvers include Y.T@Y (unit confidence for missing entries),
add (confidence-1) corrections, and use binary positive preferences.
Sources: https://github.com/benfred/implicit/blob/v0.7.3/implicit/cpu/als.py
and https://github.com/benfred/implicit/blob/v0.7.3/implicit/cpu/_als.pyx.

Only warm rows/columns are factorized. Removing the cold rows/columns leaves
the warm optimum unchanged when their factors are zero, and avoids presenting
random initialization or zero/no-history factors as evidence. Public factor
arrays use COMPACT indices; warm_user_indices/warm_item_indices map their rows
back to TrainingData indices. Raw IDs are never used as matrix indices.

The default regularization, confidence scale, requested thread counts, and batch
limits are engineering choices, not organizer requirements. This explicitly selects the
native CPU backend: implicit's GPU backend requires CUDA, not Apple Metal.
There is no CUDA/MPS request, Python solver fallback, or PyTorch dependency.
"""

from __future__ import annotations

from numbers import Integral, Real
from time import perf_counter
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_info, threadpool_limits

from sweettv.contracts import TrainingData


def _load_cpu_als() -> tuple[Any, str]:
    """Load the native dependency only when there is something to train."""
    try:
        import implicit
        from implicit.cpu.als import AlternatingLeastSquares
    except ImportError as exc:
        raise ImportError(
            "Nonempty MF training requires implicit==0.7.3 with its native CPU "
            "extension (use a wheel matching Python, OS, and CPU architecture). "
            "No GPU or alternate algorithm is substituted."
        ) from exc
    return AlternatingLeastSquares, implicit.__version__


def _integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _finite_real(value: float, name: str, *, positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        sign = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {sign}")
    return value


def _indices(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    # An empty Python list normally becomes float64, but has no invalid values.
    if not result.size:
        return np.empty(0, dtype=np.intp)
    if result.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer matrix indices")
    return result


def _map_indices(indices: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    result = np.full(indices.size, -1, dtype=np.intp)
    valid = (indices >= 0) & (indices < mapping.size)
    # Test bounds BEFORE casting: uint64's maximum must not wrap to index -1.
    result[valid] = mapping[indices[valid].astype(np.intp, copy=False)]
    return result


class MatrixFactorizationModel:
    """Repeat-eligible, training-only ALS with explicit absent-evidence scores.

    ``alpha`` is the confidence-transform scale, NOT implicit's library alpha.
    ``num_threads`` is passed to native ALS, not a measured worker count.
    To reduce nested parallelism, threadpoolctl requests BLAS limits of one
    during fitting and ``num_threads`` during batch scoring, for supported
    controllers only. Some backends, including Apple Accelerate, may expose no
    controllable pool: an empty report means unverified, not an enforced cap.
    Nonempty reports describe controller configuration, not measured runtime
    thread activity or a whole-process cap. Diagnostics store fit-time pool
    reports only; single-user scoring uses a non-BLAS reduction.
    ``max_batch_scores`` bounds dense *requested batch* output (default 4 MB of
    float32 scores); split larger requests into user/candidate batches.

    Unknown/out-of-range integer indices and cold entities return NaN. Malformed
    index types/shapes raise rather than silently truncating or flattening them.
    Candidate order and duplicate positions are preserved. No watched-item
    filter, recommend/backfill call, or global user-item score cache is used.
    """

    def __init__(
        self,
        factors: int = 32,
        iterations: int = 15,
        regularization: float = 0.1,
        alpha: float = 20.0,
        seed: int = 42,
        num_threads: int = 4,
        max_batch_scores: int = 1_000_000,
    ) -> None:
        self.factors = _integer(factors, "factors")
        self.iterations = _integer(iterations, "iterations")
        self.regularization = _finite_real(regularization, "regularization", positive=True)
        self.alpha = _finite_real(alpha, "alpha", positive=False)
        self.seed = _integer(seed, "seed", minimum=0)
        self.num_threads = _integer(num_threads, "num_threads")
        self.max_batch_scores = _integer(max_batch_scores, "max_batch_scores")
        self.diagnostics: dict[str, Any] = {"status": "not_fitted", "backend": None}
        self.als_model_: Any = None
        self._is_fitted = False

    def fit(self, data: TrainingData) -> MatrixFactorizationModel:
        """Fit from qualifying training broadcast counts; never mutate input.

        Empty/all-zero training data are a successful empty fit without loading
        implicit. A new backend is created on every nonempty fit, so refitting
        does not reuse stale factors/random state. A failed fit preserves the
        previous successful state. Native failures propagate, not fall back.
        """
        started = perf_counter()
        counts = csr_matrix(data.interactions, dtype=np.float32, copy=True)
        n_users, n_items = len(data.user_ids), len(data.item_ids)
        if counts.shape != (n_users, n_items):
            raise ValueError("Interaction shape must be users x items")
        counts.sum_duplicates()
        counts.eliminate_zeros()
        counts.sort_indices()
        if not np.all(np.isfinite(counts.data)) or np.any(counts.data <= 0):
            raise ValueError("Stored interactions must be finite positive counts")

        warm_users = np.flatnonzero(np.diff(counts.indptr))
        warm_items = np.flatnonzero(counts.getnnz(axis=0))
        user_mapping = np.full(n_users, -1, dtype=np.intp)
        item_mapping = np.full(n_items, -1, dtype=np.intp)
        user_mapping[warm_users] = np.arange(warm_users.size)
        item_mapping[warm_items] = np.arange(warm_items.size)

        model = None
        version = None
        backend = None
        training_seconds = 0.0
        blas_pools: list[dict[str, Any]] = []
        blas_control_status = "not_requested_empty_fit"
        confidence_min = confidence_max = None
        user_factors = np.empty((0, self.factors), dtype=np.float32)
        item_factors = np.empty((0, self.factors), dtype=np.float32)

        if counts.nnz:
            confidence = counts[warm_users][:, warm_items].tocsr()
            # Compute in float64, validate, then provide the CPU API's float32.
            with np.errstate(over="ignore", invalid="ignore"):
                values = 1.0 + self.alpha * np.log1p(confidence.data.astype(np.float64))
            if not np.all(np.isfinite(values)) or np.any(values > np.finfo(np.float32).max):
                raise ValueError("Transformed confidences must be finite float32 values")
            confidence.data = values.astype(np.float32)
            confidence_min = float(confidence.data.min())
            confidence_max = float(confidence.data.max())
            als_class, version = _load_cpu_als()
            with threadpool_limits(limits=1, user_api="blas"):
                model = als_class(
                    factors=self.factors,
                    iterations=self.iterations,
                    regularization=self.regularization,
                    alpha=1.0,
                    dtype=np.float32,
                    use_native=True,
                    use_cg=True,
                    calculate_training_loss=False,
                    num_threads=self.num_threads,
                    random_state=self.seed,
                )
                blas_pools = [
                    {key: pool.get(key) for key in ("internal_api", "num_threads", "prefix")}
                    for pool in threadpool_info()
                    if pool.get("user_api") == "blas"
                ]
                if not blas_pools:
                    blas_control_status = "unverified_no_reported_pools"
                elif all(pool["num_threads"] == 1 for pool in blas_pools):
                    blas_control_status = "reported_pools_at_requested_limit"
                else:
                    blas_control_status = "reported_pools_not_at_requested_limit"
                native_started = perf_counter()
                model.fit(confidence, show_progress=False)  # users x items, NOT .T
                training_seconds = perf_counter() - native_started
            backend = f"{type(model).__module__}.{type(model).__qualname__}"
            user_factors = np.asarray(model.user_factors)
            item_factors = np.asarray(model.item_factors)
            if user_factors.shape != (warm_users.size, self.factors) or item_factors.shape != (
                warm_items.size, self.factors
            ):
                raise RuntimeError("Native ALS returned factors with incorrect users/items dimensions")
            if not np.all(np.isfinite(user_factors)) or not np.all(np.isfinite(item_factors)):
                raise RuntimeError("Native ALS returned nonfinite factors")

        # A zero factor vector is not evidence, even if a numerical collapse
        # occurred on a nominally warm row. A zero DOT PRODUCT of two nonzero
        # factor vectors, in contrast, remains a legitimate finite score.
        user_evidence = np.any(user_factors != 0, axis=1)
        item_evidence = np.any(item_factors != 0, axis=1)
        diagnostics = {
            "status": "fitted" if counts.nnz else "empty",
            "backend": backend,
            "device": "cpu",
            "implicit_version": version,
            "training_shape": [n_users, n_items],
            "compact_training_shape": [int(warm_users.size), int(warm_items.size)],
            "n_users": n_users,
            "n_items": n_items,
            "nnz": int(counts.nnz),
            "n_warm_users": int(warm_users.size),
            "n_warm_items": int(warm_items.size),
            "n_cold_users": n_users - int(warm_users.size),
            "n_cold_items": n_items - int(warm_items.size),
            "user_factor_shape": list(user_factors.shape),
            "item_factor_shape": list(item_factors.shape),
            "n_users_with_factor_evidence": int(user_evidence.sum()),
            "n_items_with_factor_evidence": int(item_evidence.sum()),
            "user_factor_coverage": float(user_evidence.sum() / n_users) if n_users else 0.0,
            "item_factor_coverage": float(item_evidence.sum() / n_items) if n_items else 0.0,
            "factors": self.factors,
            "iterations": self.iterations,
            "regularization": self.regularization,
            "confidence_scale": self.alpha,
            "confidence_transform": "1 + alpha * log1p(count)",
            "confidence_min": confidence_min,
            "confidence_max": confidence_max,
            "library_alpha": 1.0,
            "positive_preference": 1.0,
            "missing_preference": 0.0,
            "missing_confidence": 1.0,
            "seed": self.seed,
            # Native ALS parameter request, not an observed active thread count.
            "num_threads": self.num_threads,
            "training_blas_thread_limit_requested": 1 if counts.nnz else None,
            "training_blas_control_status": blas_control_status,
            "training_blas_pools": blas_pools,
            "max_batch_scores": self.max_batch_scores,
            "training_seconds": training_seconds,
            "fit_seconds": perf_counter() - started,
        }
        self.als_model_ = model
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.warm_user_indices = warm_users
        self.warm_item_indices = warm_items
        self.user_index_to_factor = user_mapping
        self.item_index_to_factor = item_mapping
        self._user_evidence = user_evidence
        self._item_evidence = item_evidence
        self.diagnostics = diagnostics
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("MatrixFactorizationModel.fit must be called before scoring")

    def _candidate_factors(self, candidates: np.ndarray) -> np.ndarray:
        mapped = _map_indices(candidates, self.item_index_to_factor)
        valid = np.flatnonzero(mapped >= 0)
        mapped[valid[~self._item_evidence[mapped[valid]]]] = -1
        return mapped

    def score(self, user_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        """Return one float32 dot product per requested candidate, else NaN."""
        self._check_fitted()
        candidates = _indices(candidate_indices, "candidate_indices")
        if isinstance(user_index, (bool, np.bool_)) or not isinstance(user_index, Integral):
            raise TypeError("user_index must be an integer matrix index")
        scores = np.full(candidates.size, np.nan, dtype=np.float32)
        if user_index < 0 or user_index >= self.user_index_to_factor.size:
            return scores
        user_row = self.user_index_to_factor[user_index]
        if user_row < 0 or not self._user_evidence[user_row]:
            return scores
        item_rows = self._candidate_factors(candidates)
        valid = item_rows >= 0
        # This contraction has no BLAS pool and never forms a user-item matrix.
        scores[valid] = np.einsum(
            "ij,j->i", self.item_factors[item_rows[valid]], self.user_factors[user_row],
            optimize=False,
        )
        return scores

    def score_batch(self, user_indices: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
        """Return a bounded users x candidates array, preserving both orders.

        The requested output must have at most max_batch_scores elements. It is
        not cached; callers must stream larger catalogs/cohorts in batches.
        The BLAS limit is a request for supported controllers, not a measured
        runtime cap. Fit-time pool reports do not verify batch-scoring control.
        """
        self._check_fitted()
        users = _indices(user_indices, "user_indices")
        candidates = _indices(candidate_indices, "candidate_indices")
        if users.size * candidates.size > self.max_batch_scores:
            raise ValueError("score_batch exceeds max_batch_scores; use smaller batches")
        scores = np.full((users.size, candidates.size), np.nan, dtype=np.float32)
        user_rows = _map_indices(users, self.user_index_to_factor)
        valid = np.flatnonzero(user_rows >= 0)
        user_rows[valid[~self._user_evidence[user_rows[valid]]]] = -1
        item_rows = self._candidate_factors(candidates)
        valid_users = np.flatnonzero(user_rows >= 0)
        valid_items = np.flatnonzero(item_rows >= 0)
        if valid_users.size and valid_items.size:
            with threadpool_limits(limits=self.num_threads, user_api="blas"):
                scores[np.ix_(valid_users, valid_items)] = (
                    self.user_factors[user_rows[valid_users]]
                    @ self.item_factors[item_rows[valid_items]].T
                )
        return scores
