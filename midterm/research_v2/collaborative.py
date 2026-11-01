"""Count-sensitive, training-only EASE linear collaborative filtering.

Method: Harald Steck (2019), *Embarrassingly Shallow Autoencoders for Sparse
Data*, WWW, pp. 3251-3257, https://doi.org/10.1145/3308558.3313710.
Equations 1-3 and 8 define min_B ||X-XB||_F^2 + lambda*||B||_F^2 subject to
diag(B)=0, with B_ij=-P_ij/P_jj off the diagonal, P=(X.T X+lambda I)^-1.
This is an original implementation of that method, not a neural/deep/GPU model.

Here X contains qualifying broadcast counts, optionally decayed using the
shared temporal_matrix (ages in unaltered source-clock seconds / 86400).
The feature order is raw/log1p/binary, optional training-fitted IDF, then
optional row L2 normalization. Both regression inputs AND targets are X in
that feature space. Scores X_u B are signed, uncalibrated reconstructions,
not guaranteed forecasts of future frequencies or probabilities. There is no
intercept, internal popularity fallback, watched-item exclusion, or file I/O.

Active items are bounded by 4096 and selected only from effective training
history, by descending RAW training count, then ascending numeric raw ID.
The forecast candidate list/schedules never select features or fit statistics.
Only the item-item algebra is dense; training profiles remain sparse and
scoring streams one compact user vector at a time, never a global U x I table.
"""

from __future__ import annotations

from numbers import Integral, Real
from time import perf_counter
from typing import Any
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, solve
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_info, threadpool_limits

from research_v2.contracts import TemporalContext, candidate_array, temporal_matrix, valid_user


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _indices(values) -> np.ndarray:
    # Preserve the original element types of untyped sequences BEFORE numeric
    # dtype inference can erase mixed Booleans or round signed/huge integers.
    # Typed arrays keep the existing fast path; a caller's earlier coercion
    # cannot be undone here. Object conversion also avoids inference overflow.
    array = (np.asarray(values) if isinstance(values, np.ndarray)
             else np.asarray(values, dtype=object))
    # Python integers outside uint64 are valid *unknown* indices. Validate the
    # preserved objects, then keep them exact until _mapped's bounds test.
    if array.ndim == 1 and array.dtype.kind == "O" and all(
        isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        for value in array
    ):
        return array
    return candidate_array(array)


def _mapped(indices: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    result = np.full(indices.size, -1, dtype=np.intp)
    valid = (indices >= 0) & (indices < mapping.size)
    result[valid] = mapping[indices[valid].astype(np.intp, copy=False)]
    return result


def _blas_observation(requested: int) -> tuple[list[dict[str, Any]], str]:
    pools = [
        {key: pool.get(key) for key in ("internal_api", "num_threads", "prefix")}
        for pool in threadpool_info() if pool.get("user_api") == "blas"
    ]
    if not pools:
        status = "unverified_no_reported_pools"
    elif any(
        any(word in str(pool).lower() for word in ("accelerate", "veclib"))
        for pool in pools
    ):
        status = "unverified_accelerate"
    elif all(pool["num_threads"] == requested for pool in pools):
        status = "reported_pools_at_requested_limit"
    else:
        status = "reported_pools_not_at_requested_limit"
    return pools, status


def _training_snapshot(context: TemporalContext) -> tuple[csr_matrix, np.ndarray]:
    """Recheck the cutoff and mutable TrainingData at fit time, without labels."""
    if not isinstance(context, TemporalContext):
        raise ValueError("EASEModel.fit requires a TemporalContext")
    boundaries = (context.history_start, context.train_end,
                  context.forecast_start, context.forecast_end)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in boundaries):
        raise ValueError("Context boundaries must be integer source-clock seconds")
    if not boundaries[0] < boundaries[1] <= boundaries[2] < boundaries[3]:
        raise ValueError("Invalid context source-clock cutoff boundaries")
    arrays = (context.event_users, context.event_items, context.event_starts, context.event_ends)
    if any(a.ndim != 1 or a.dtype.kind not in "iu" for a in arrays):
        raise ValueError("Training-event arrays must be one-dimensional integers")
    if any(len(a) != len(arrays[0]) for a in arrays):
        raise ValueError("Training-event arrays must align")
    if np.any(context.event_ends > context.train_end) or np.any(
        context.event_starts >= context.event_ends
    ):
        raise ValueError("Future or invalid behavior at the context cutoff")
    data = context.data
    n_users, n_items = len(data.user_ids), len(data.item_ids)
    if np.any(context.event_users < 0) or np.any(context.event_users >= n_users):
        raise ValueError("Training-event user index out of range")
    if np.any(context.event_items < 0) or np.any(context.event_items >= n_items):
        raise ValueError("Training-event item index out of range")
    # Only these two clock-audit fields are read; provenance/metadata may carry
    # arbitrary unrelated payloads and are never used as model features.
    for key in ("train_end", "behavior_max_end"):
        value = data.provenance.get(key)
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"Provenance {key} must be integer source-clock seconds")
            if (key == "train_end" and value != context.train_end) or value > context.train_end:
                raise ValueError(f"Provenance {key} disagrees with the context cutoff")
    ids = np.array(data.item_ids, copy=True)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids <= 0):
        raise ValueError("Raw program IDs must be positive integers")
    if len(np.unique(ids)) != n_items:
        raise ValueError("Raw program IDs must be unique")
    counts = csr_matrix(data.interactions, dtype=np.float64, copy=True)
    if counts.shape != (n_users, n_items):
        raise ValueError("Training interaction shape must be users x items")
    if not np.all(np.isfinite(counts.data)) or np.any(counts.data < 0):
        raise ValueError("Training counts must be finite and nonnegative")
    counts.sum_duplicates()
    counts.eliminate_zeros()
    counts.sort_indices()
    if not np.all(np.isfinite(counts.data)):
        raise ValueError("Aggregated training counts must be finite")
    # TemporalContext's constructor validates events but TrainingData is mutable.
    # Require the same raw count reconstruction for either feature-source mode.
    observed = temporal_matrix(context)
    if observed.shape != counts.shape or (observed != counts).nnz:
        raise ValueError("Training counts disagree with pre-cutoff context events")
    return counts, ids


class EASEModel:
    """Bounded CPU linear CF, with explicit coefficient-evidence semantics.

    ``regularization`` is the strictly positive ridge lambda on SUMMED squared
    feature reconstruction errors, not a per-user average. Defaults are research
    choices, not validated SWEET.TV hyperparameters. ``item_weighting='idf'``
    uses 1+log((1+N)/(1+df_i)), where N is the number of effective warm training
    users BEFORE item truncation and df_i is effective training user support.
    Normalization never fits from forecast schedules/candidates or query batches.

    Compact learned arrays end in ``_``; mappings translate global matrix
    indices, never raw program IDs. A score is NaN for cold/truncated entities
    or when none of the user's retained history features has a nonzero learned
    coefficient to the target. Negative values and genuine cancellation zeros
    remain finite. Evidence uses exact nonzero coefficients, with no epsilon
    pruning. Zero diagonal prevents recommending an item from its OWN feature;
    in particular, sole-history self scores are NaN. A separate repeat model
    must handle that signal; watched IDs are not otherwise excluded.

    ``num_threads`` requests fit-time BLAS control via threadpoolctl. Diagnostics
    distinguish that request from reported pool configuration, which is NOT a
    runtime thread-activity measurement or whole-process cap. Apple Accelerate
    and missing controller reports remain unverified. Sparse row scoring uses
    SciPy's sparse-dense kernels, not a dense batch BLAS multiplication.
    """

    MAX_ACTIVE_ITEMS = 4096

    def __init__(
        self,
        regularization: float = 1000.0,
        transform: str = "raw",
        half_life_days: float | None = None,
        row_l2: bool = False,
        item_weighting: str = "none",
        max_active_items: int = 4096,
        num_threads: int = 1,
        max_batch_scores: int = 1_000_000,
    ) -> None:
        self.regularization = _positive_real(regularization, "regularization")
        if transform not in ("raw", "log", "binary"):
            raise ValueError("transform must be 'raw', 'log', or 'binary'")
        self.transform = transform
        self.half_life_days = None if half_life_days is None else _positive_real(
            half_life_days, "half_life_days"
        )
        if not isinstance(row_l2, (bool, np.bool_)):
            raise ValueError("row_l2 must be boolean")
        self.row_l2 = bool(row_l2)
        if item_weighting not in ("none", "idf"):
            raise ValueError("item_weighting must be 'none' or 'idf'")
        self.item_weighting = item_weighting
        self.max_active_items = _positive_integer(max_active_items, "max_active_items")
        if self.max_active_items > self.MAX_ACTIVE_ITEMS:
            raise ValueError(f"max_active_items cannot exceed {self.MAX_ACTIVE_ITEMS}")
        self.num_threads = _positive_integer(num_threads, "num_threads")
        self.max_batch_scores = _positive_integer(max_batch_scores, "max_batch_scores")
        self.diagnostics: dict[str, Any] = {"status": "not_fitted", "backend": None}
        self._is_fitted = False

    def fit(self, context: TemporalContext) -> EASEModel:
        """Fit a fresh snapshot; an unsuccessful refit preserves previous state."""
        started = perf_counter()
        counts, ids = _training_snapshot(context)
        source = counts if self.half_life_days is None else temporal_matrix(
            context, half_life_days=self.half_life_days
        )
        n_users, n_items = counts.shape
        raw_mass = np.asarray(counts.sum(axis=0)).ravel()
        raw_warm_items = int(np.count_nonzero(counts.getnnz(axis=0)))
        raw_warm_users = int(np.count_nonzero(np.diff(counts.indptr)))
        effective_users = int(np.count_nonzero(np.diff(source.indptr)))
        support = np.asarray(source.getnnz(axis=0)).ravel()
        eligible = np.flatnonzero(support)
        order = np.lexsort((ids[eligible], -raw_mass[eligible]))
        active_items = np.sort(eligible[order[:self.max_active_items]])
        retained = source[:, active_items].tocsr()
        active_users = np.flatnonzero(np.diff(retained.indptr))
        features = retained[active_users].tocsr(copy=True)
        if self.transform == "log":
            features.data = np.log1p(features.data)
        elif self.transform == "binary":
            features.data.fill(1.0)
        weights = np.ones(active_items.size, dtype=np.float64)
        if self.item_weighting == "idf":
            weights += np.log((1.0 + effective_users) / (1.0 + support[active_items]))
            features.data *= weights[features.indices]
        if self.row_l2 and features.nnz:
            # Scale first: naive sum(x*x) can underflow for old decayed events.
            lengths = np.diff(features.indptr)
            maxima = np.maximum.reduceat(features.data, features.indptr[:-1])
            features.data /= np.repeat(maxima, lengths)
            norms = np.sqrt(np.asarray(features.multiply(features).sum(axis=1)).ravel())
            features.data /= np.repeat(norms, lengths)
        if not np.all(np.isfinite(features.data)) or np.any(features.data <= 0):
            raise ValueError("Transformed training features must be finite and positive")

        coefficients = np.empty((0, 0), dtype=np.float64)
        pools: list[dict[str, Any]] = []
        control_status = "not_requested_empty_fit"
        training_seconds = 0.0
        if active_items.size:
            native_started = perf_counter()
            with threadpool_limits(limits=self.num_threads, user_api="blas"):
                gram = (features.T @ features).toarray(order="F")
                gram.flat[::active_items.size + 1] += self.regularization
                if not np.all(np.isfinite(gram)):
                    raise ValueError("EASE Gram matrix must be finite")
                # Solve the SPD system against I, then reuse writable P memory
                # for B. Recent SciPy versions can return a read-only view even
                # with overwrite_b=True; never assume that flag grants writes.
                # Numerical loss of positive definiteness is an error, never a
                # silent pseudoinverse, altered lambda, or popularity fallback.
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", LinAlgWarning)
                        coefficients = solve(
                            gram, np.eye(active_items.size, order="F"), assume_a="pos",
                            overwrite_a=True, overwrite_b=True, check_finite=False,
                        )
                except (np.linalg.LinAlgError, LinAlgWarning) as exc:
                    raise ValueError(
                        "EASE solve is numerically ill-conditioned; increase regularization "
                        "or change the training feature transform"
                    ) from exc
                if coefficients.shape != (active_items.size, active_items.size):
                    raise ValueError("EASE precision matrix has incorrect item dimensions")
                if not coefficients.flags.writeable:
                    coefficients = coefficients.copy(order="F")
                diagonal = np.diag(coefficients).copy()
                if not np.all(np.isfinite(coefficients)) or np.any(diagonal <= 0):
                    raise ValueError("EASE precision matrix must be finite with positive diagonal")
                with np.errstate(over="ignore", invalid="ignore"):
                    coefficients /= -diagonal[np.newaxis, :]
                np.fill_diagonal(coefficients, 0.0)
                if not np.all(np.isfinite(coefficients)):
                    raise ValueError("EASE coefficients must be finite")
                pools, control_status = _blas_observation(self.num_threads)
            training_seconds = perf_counter() - native_started

        coefficient_support = coefficients != 0.0
        user_mapping = np.full(n_users, -1, dtype=np.intp)
        item_mapping = np.full(n_items, -1, dtype=np.intp)
        user_mapping[active_users] = np.arange(active_users.size)
        item_mapping[active_items] = np.arange(active_items.size)
        raw_total = float(raw_mass.sum())
        retained_raw = counts[:, active_items]
        retained_total = float(retained_raw.sum())
        users_with_coefficients = int(np.count_nonzero(
            features[:, np.any(coefficient_support, axis=1)].getnnz(axis=1)
        ))
        diagnostics = {
            "status": "fitted" if active_items.size else "empty",
            "backend": "scipy.linalg.solve(assume_a='pos')" if active_items.size else None,
            "device": "cpu",
            "training_shape": [n_users, n_items],
            "compact_training_shape": list(features.shape),
            "coefficient_shape": list(coefficients.shape),
            "coefficient_bytes": int(coefficients.nbytes),
            "coefficient_support_bytes": int(coefficient_support.nbytes),
            "nnz": int(counts.nnz),
            "feature_nnz": int(features.nnz),
            "n_warm_users": raw_warm_users,
            "n_warm_items": raw_warm_items,
            "n_effective_warm_users": effective_users,
            "n_effective_warm_items": int(eligible.size),
            "n_active_users": int(active_users.size),
            "n_active_items": int(active_items.size),
            "n_truncated_items": int(eligible.size - active_items.size),
            "n_items_lost_to_temporal_underflow": raw_warm_items - int(eligible.size),
            "active_item_selection": "effective training support; raw count descending, raw ID ascending",
            "compact_item_order": "global matrix index ascending",
            "training_count_total": raw_total,
            "retained_training_count_total": retained_total,
            "training_count_coverage": retained_total / raw_total if raw_total else 0.0,
            "training_nnz_coverage": retained_raw.nnz / counts.nnz if counts.nnz else 0.0,
            "training_item_coverage": active_items.size / raw_warm_items if raw_warm_items else 0.0,
            "training_user_coverage": active_users.size / raw_warm_users if raw_warm_users else 0.0,
            "effective_weight_total": float(source.sum()),
            "retained_effective_weight_total": float(retained.sum()),
            "n_nonzero_coefficients": int(coefficient_support.sum()),
            "n_items_with_incoming_coefficients": int(np.any(coefficient_support, axis=0).sum()),
            "n_users_with_coefficient_evidence": users_with_coefficients,
            "n_sole_retained_history_users": int(np.count_nonzero(np.diff(features.indptr) == 1)),
            "zero_diagonal": True,
            "sole_history_self_score": "NaN; repeat signal must be modeled separately",
            "coefficient_evidence_rule": "at least one retained history feature with B_ij != 0",
            "watched_item_filter": False,
            "internal_fallback": False,
            "regularization": self.regularization,
            "transform": self.transform,
            "half_life_days": self.half_life_days,
            "row_l2": self.row_l2,
            "item_weighting": self.item_weighting,
            "idf_training_users": effective_users if self.item_weighting == "idf" else None,
            "feature_min": float(features.data.min()) if features.nnz else None,
            "feature_max": float(features.data.max()) if features.nnz else None,
            "max_active_items": self.max_active_items,
            "max_batch_scores": self.max_batch_scores,
            "cutoff_checked": True,
            "train_end": int(context.train_end),
            "behavior_max_end": int(context.event_ends.max()) if context.event_ends.size else None,
            "clock": "unaltered source-clock seconds; no inferred UTC offset",
            "training_blas_thread_limit_requested": self.num_threads if active_items.size else None,
            "training_blas_control_status": control_status,
            "training_blas_pools": pools,
            "blas_reports_measure_runtime_thread_activity": False,
            "training_seconds": training_seconds,
            "fit_seconds": perf_counter() - started,
        }
        self.training_matrix_ = features
        self.coefficients_ = coefficients
        self._coefficient_support_ = coefficient_support
        self.active_user_indices_ = active_users
        self.active_item_indices_ = active_items
        self.active_item_ids_ = ids[active_items].copy()
        self.user_index_to_row_ = user_mapping
        self.item_index_to_column_ = item_mapping
        self.item_weights_ = weights
        self.diagnostics = diagnostics
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("EASEModel.fit must be called before scoring")

    def score(self, user_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        """Return a float64 vector in candidate order, including duplicates."""
        self._check_fitted()
        # Validate scalar types without casting huge unknown Python integers.
        valid_user(user_index, self.user_index_to_row_.size)
        return self.score_batch([user_index], candidate_indices)[0]

    def score_batch(self, user_indices: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
        """Return bounded requested users x candidates; no global score cache.

        Both orders and duplicate positions are preserved. The dense OUTPUT
        must fit max_batch_scores (default 8 MB of float64); it is allocated only
        after checking the bound, including for all-cold requests. Intermediate
        predictions/evidence are one compact active-item vector per user.
        """
        self._check_fitted()
        users, candidates = _indices(user_indices), _indices(candidate_indices)
        if users.size * candidates.size > self.max_batch_scores:
            raise ValueError("score_batch exceeds max_batch_scores; use smaller batches")
        scores = np.full((users.size, candidates.size), np.nan, dtype=np.float64)
        if not users.size or not candidates.size:
            return scores
        rows = _mapped(users, self.user_index_to_row_)
        columns = _mapped(candidates, self.item_index_to_column_)
        valid_columns = np.flatnonzero(columns >= 0)
        if not valid_columns.size:
            return scores
        for output_row in np.flatnonzero(rows >= 0):
            profile = self.training_matrix_.getrow(rows[output_row])
            values = np.asarray(profile @ self.coefficients_).ravel()
            evidence = np.asarray(profile.astype(bool) @ self._coefficient_support_).ravel()
            if not np.all(np.isfinite(values[evidence])):
                raise FloatingPointError("Nonfinite EASE scores; no fallback was substituted")
            values[~evidence] = np.nan
            scores[output_row, valid_columns] = values[columns[valid_columns]]
        return scores
