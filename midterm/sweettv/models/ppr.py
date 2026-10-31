"""Training-only, weighted bipartite Personalized PageRank.

``alpha`` is the probability of following an edge, not the restart probability.
The row-stochastic transition is stored as CSR; column probability vectors obey
``p = alpha * P.T @ p + (1 - alpha) * restart``. Dangling mass is also
returned to that query's restart user. Scores are item-node probability mass,
not a distribution renormalized over items. Already-watched items stay eligible.

Only bounded blocks of requested users are propagated. There is no dense square
graph, persistent user/item score cache, or metadata-derived edge. For block size
B, N nodes, E transitions, K updates, Q unique active queries, R requested users,
and C candidates, workspace is O(E + N*B), plus the requested R-by-C output.
Time is O((K+1)*(E+N)*Q + R*C + R*log(R)). A final residual check costs one
extra operator application; duplicate queries share their walk.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from operator import index
from typing import Any

import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.csgraph import connected_components

from sweettv.contracts import TrainingData


class PPRConvergenceError(RuntimeError):
    """A requested walk did not reach the configured fixed-point tolerance."""

    def __init__(
        self,
        user_indices: np.ndarray,
        iterations: np.ndarray,
        residual_l1: np.ndarray,
        tolerance: float,
    ) -> None:
        self.user_indices = tuple(map(int, user_indices))
        self.iterations = tuple(map(int, iterations))
        self.residual_l1 = tuple(map(float, residual_l1))
        self.tolerance = float(tolerance)
        super().__init__(
            f"PPR did not converge for {len(self.user_indices)} user(s) after "
            f"{max(self.iterations)} update(s): last L1 fixed-point residual "
            f"{max(self.residual_l1):.17g} exceeds {self.tolerance:.17g}"
        )


@dataclass
class _WalkResult:
    probabilities: np.ndarray
    iterations: np.ndarray
    residual_l1: np.ndarray
    converged: np.ndarray
    operator_applications: int


def _iterate_ppr(
    transition_transpose: csr_matrix,
    dangling: np.ndarray,
    source_indices: np.ndarray,
    *,
    alpha: float,
    tol: float,
    max_iter: int,
) -> _WalkResult:
    """Propagate one bounded block; report residuals of the returned iterates.

    Iteration zero is the one-hot restart. After at most ``max_iter`` updates,
    explicitly evaluate ``||F(p) - p||_1`` at the returned p. Do not normalize
    iterates or substitute an assumed convergence rate for an observed residual.
    The caller must reject a result with any false ``converged`` entry.
    """
    n_nodes = transition_transpose.shape[0]
    columns = np.arange(len(source_indices))
    probabilities = np.zeros((n_nodes, len(source_indices)), dtype=np.float64)
    probabilities[source_indices, columns] = 1.0
    iterations = np.zeros(len(source_indices), dtype=np.int64)
    residual_l1 = np.full(len(source_indices), np.inf, dtype=np.float64)
    converged = np.zeros(len(source_indices), dtype=bool)

    for iteration in range(max_iter + 1):
        following = transition_transpose @ probabilities
        following *= alpha
        restart_mass = np.full(len(source_indices), 1.0 - alpha, dtype=np.float64)
        if np.any(dangling):
            restart_mass += alpha * probabilities[dangling].sum(axis=0)
        following[source_indices, columns] += restart_mass

        difference = following - probabilities
        np.abs(difference, out=difference)
        current_residual = difference.sum(axis=0, dtype=np.float64)
        del difference
        active = ~converged
        residual_l1[active] = current_residual[active]
        iterations[active] = iteration
        converged |= active & (current_residual <= tol)

        if np.all(converged) or iteration == max_iter:
            return _WalkResult(
                probabilities,
                iterations,
                residual_l1,
                converged,
                iteration + 1,
            )

        # Keep early-converged columns at the iterate whose residual we recorded.
        if np.any(converged):
            following[:, converged] = probabilities[:, converged]
        probabilities = following

    raise AssertionError("Unreachable PPR iteration state")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _indices(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if result.size and not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must contain matrix indices, not raw IDs or floats")
    if np.issubdtype(result.dtype, np.unsignedinteger):
        outside_int64 = result > np.uint64(np.iinfo(np.int64).max)
        if np.any(outside_int64):
            # Bounds-check before narrowing; unknown unsigned values never wrap.
            bounded = np.full(result.shape, -1, dtype=np.int64)
            bounded[~outside_int64] = result[~outside_int64].astype(np.int64)
            return bounded
    return result.astype(np.int64, copy=False)


class PersonalizedPageRankModel:
    """Weighted user/program PPR with explicit convergence and missing evidence.

    ``0 <= alpha < 1`` and positive finite ``tol`` are required. At alpha zero
    all mass stays on the restart user, so all item scores have no walk evidence
    and are NaN. Unknown/cold users and disconnected/cold items also yield NaN.
    Nonconvergence raises :class:`PPRConvergenceError`; no partial scores escape.

    ``batch_size`` bounds dense propagation columns, not requested output size.
    Diagnostics expose the observed fixed-point residual and its mathematical
    L1 error bound ``residual / (1-alpha)``, not a claim of exact precision.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        tol: float = 1e-9,
        max_iter: int = 300,
        batch_size: int = 16,
    ) -> None:
        if (
            isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, Real)
            or not np.isfinite(alpha)
            or not 0 <= alpha < 1
        ):
            raise ValueError("alpha must be a finite follow probability in [0, 1)")
        if (
            isinstance(tol, (bool, np.bool_))
            or not isinstance(tol, Real)
            or not np.isfinite(tol)
            or tol <= 0
        ):
            raise ValueError("tol must be finite and positive")
        self.alpha = float(alpha)
        self.tol = float(tol)
        self.max_iter = _positive_integer(max_iter, "max_iter")
        self.batch_size = _positive_integer(batch_size, "batch_size")
        self.diagnostics: dict[str, Any] = {}
        self._fitted = False

    def fit(self, data: TrainingData) -> PersonalizedPageRankModel:
        self._fitted = False
        self.diagnostics = {}
        # Copy: canonicalization and normalization must not mutate TrainingData.
        counts = csr_matrix(data.interactions, dtype=np.float64, copy=True)
        counts.sum_duplicates()
        counts.eliminate_zeros()
        counts.sort_indices()
        self.n_users_, self.n_items_ = counts.shape
        if counts.shape != (len(data.user_ids), len(data.item_ids)):
            raise ValueError("Interaction shape must be users x items")
        if not np.all(np.isfinite(counts.data)) or np.any(counts.data <= 0):
            raise ValueError("Interactions must be finite positive training counts")
        self.user_ids_ = tuple(data.user_ids)
        self.item_ids_ = np.array(data.item_ids, dtype=np.int64, copy=True)

        # Each qualifying count supplies both directions of a bipartite edge.
        transition = bmat(
            [
                [csr_matrix((self.n_users_, self.n_users_)), counts],
                [counts.T, csr_matrix((self.n_items_, self.n_items_))],
            ],
            format="csr",
            dtype=np.float64,
        )
        self.node_degrees_ = np.asarray(transition.sum(axis=1)).ravel()
        self._dangling = self.node_degrees_ == 0
        inverse_degrees = np.zeros_like(self.node_degrees_)
        np.divide(
            1.0, self.node_degrees_, out=inverse_degrees, where=~self._dangling
        )
        transition.data *= np.repeat(inverse_degrees, np.diff(transition.indptr))
        transition.sort_indices()
        self.transition_ = transition
        self._transition_transpose = transition.T.tocsr()

        n_nodes = self.n_users_ + self.n_items_
        if n_nodes:
            n_components, self.component_labels_ = connected_components(
                transition, directed=False, return_labels=True
            )
        else:
            n_components = 0
            self.component_labels_ = np.empty(0, dtype=np.int32)
        n_dangling = int(np.count_nonzero(self._dangling))
        self.diagnostics = {
            "backend": "scipy.sparse.csr.float64",
            "n_users": self.n_users_,
            "n_items": self.n_items_,
            "n_nodes": n_nodes,
            "active_users": int(np.count_nonzero(self.node_degrees_[: self.n_users_])),
            "active_items": int(np.count_nonzero(self.node_degrees_[self.n_users_ :])),
            "training_edges": int(counts.nnz),
            "training_weight": float(counts.sum()),
            "transition_nnz": int(transition.nnz),
            "dangling_nodes": n_dangling,
            "active_components": int(n_components - n_dangling),
            "alpha": self.alpha,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
            "convergence_criterion": "L1 fixed-point residual <= tol",
            "score_semantics": "item-node probability mass, not item-normalized",
            "last_query": None,
        }
        self._fitted = True
        return self

    def score(self, user_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fit PersonalizedPageRankModel before scoring")
        if isinstance(user_index, (bool, np.bool_)):
            raise ValueError("user_index must be an integer matrix index")
        try:
            user = index(user_index)
        except TypeError as error:
            raise ValueError("user_index must be an integer matrix index") from error
        # Python integers need not fit any NumPy dtype. Normalize unknown users
        # using Python bounds before constructing the delegated integer array.
        if not 0 <= user < self.n_users_:
            user = -1
        return self.score_batch([user], candidate_indices)[0]

    def score_batch(
        self, user_indices: np.ndarray, candidate_indices: np.ndarray
    ) -> np.ndarray:
        """Return requested users x candidates, preserving both orders/duplicates."""
        if not self._fitted:
            raise RuntimeError("Fit PersonalizedPageRankModel before scoring")
        users = _indices(user_indices, "user_indices")
        candidates = _indices(candidate_indices, "candidate_indices")
        scores = np.full((len(users), len(candidates)), np.nan, dtype=np.float64)
        query: dict[str, Any] = {
            "status": "no_evidence",
            "converged": None,
            "requested_users": len(users),
            "requested_candidates": len(candidates),
            "batches": 0,
            "max_batch_users": 0,
            "operator_applications": 0,
            "solves": [],
        }
        self.diagnostics["last_query"] = query
        if not len(users) or not len(candidates):
            return scores

        valid_users = (users >= 0) & (users < self.n_users_)
        user_positions = np.flatnonzero(valid_users)
        valid_users[user_positions] &= self.node_degrees_[users[user_positions]] > 0
        valid_candidates = (candidates >= 0) & (candidates < self.n_items_)
        candidate_positions = np.flatnonzero(valid_candidates)
        valid_candidates[candidate_positions] &= (
            self.node_degrees_[self.n_users_ + candidates[candidate_positions]] > 0
        )
        candidate_positions = np.flatnonzero(valid_candidates)
        if not np.any(valid_users) or not len(candidate_positions):
            return scores

        sources, inverse = np.unique(users[valid_users], return_inverse=True)
        user_positions = np.flatnonzero(valid_users)
        rows_by_source: list[list[int]] = [[] for _ in sources]
        for row, source_position in zip(user_positions, inverse):
            rows_by_source[source_position].append(int(row))
        item_nodes = self.n_users_ + candidates[candidate_positions]
        item_components = self.component_labels_[item_nodes]
        for start in range(0, len(sources), self.batch_size):
            block_sources = sources[start : start + self.batch_size]
            result = _iterate_ppr(
                self._transition_transpose,
                self._dangling,
                block_sources,
                alpha=self.alpha,
                tol=self.tol,
                max_iter=self.max_iter,
            )
            query["batches"] += 1
            query["max_batch_users"] = max(query["max_batch_users"], len(block_sources))
            query["operator_applications"] += result.operator_applications
            for column, user in enumerate(block_sources):
                query["solves"].append(
                    {
                        "user_index": int(user),
                        "iterations": int(result.iterations[column]),
                        "residual_l1": float(result.residual_l1[column]),
                        "error_bound_l1": float(
                            result.residual_l1[column] / (1.0 - self.alpha)
                        ),
                        "converged": bool(result.converged[column]),
                        "walk_mass": float(result.probabilities[:, column].sum()),
                        "item_mass": float(
                            result.probabilities[self.n_users_ :, column].sum()
                        ),
                    }
                )

            if not np.all(result.converged):
                query["status"] = "nonconverged"
                query["converged"] = False
                failed = ~result.converged
                raise PPRConvergenceError(
                    block_sources[failed],
                    result.iterations[failed],
                    result.residual_l1[failed],
                    self.tol,
                )

            for column, user in enumerate(block_sources):
                values = result.probabilities[item_nodes, column].copy()
                has_evidence = (
                    item_components == self.component_labels_[user]
                ) & (values > 0)
                values[~has_evidence] = np.nan
                rows = rows_by_source[start + column]
                scores[np.ix_(rows, candidate_positions)] = values
            del result

        query["status"] = "converged"
        query["converged"] = True
        return scores
