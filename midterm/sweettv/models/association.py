"""Directed singleton association rules over explicit training-user baskets.

One transaction is one user's positive training history, restricted first to
training-frequency top items and then to that user's highest-frequency items.
Both ties use numeric raw program IDs. Users with an originally positive
history remain in N even if the item restriction empties their basket. Counts
are basket supports, not numbers of broadcasts or qualifying viewings.

The defaults below are engineering choices, not organizer requirements. For
antecedent a and consequent b, the retained weight is

    max(0, (J_ab + smoothing * A_b / N) / (A_a + smoothing) - A_b / N).

Training uses int64 sparse, chunked cross-products. Only chunk_rows x M counts
are dense (possibly M x M for a small catalog); there is no fixed 4096-item
padding or dense user-item score matrix.
The fitted, bounded history is also the inference history; discarded items
cannot reappear as antecedents. Scores sum log1p(original frequency) * weight.
Repeated/already-watched consequents remain eligible. Missing evidence is NaN,
never an internal popularity recommendation.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from sweettv.contracts import TrainingData


_SAFE_INT64_PRODUCT = math.isqrt(np.iinfo(np.int64).max)


def _sparse_bytes(matrix: csr_matrix) -> int:
    """Logical bytes in the three sparse arrays (not allocator overhead)."""
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _rank_training_items(
    interactions: csr_matrix, item_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rank exact represented frequencies, including ties beyond float64 precision.

    Nonnegative integral partial sums are exact in float64 when every item's
    total is at most 2**53. Canonical CSR has at most one entry per user/item,
    so U * ceil(max_count) is a conservative, integer-arithmetic upper bound.
    Check integrality in fixed-size chunks; ordinary broadcast counts retain
    vectorized bincount rather than paying for Python arithmetic per entry.

    Otherwise accumulate exact binary fixed-point integers. All finite float32
    counts supplied by TrainingData are multiples of 2**-149, including its
    smallest subnormal. One accumulator per catalog item suffices; no CSC copy,
    transaction pair list, or dense user-item matrix is needed. Exact ranking
    also distinguishes unequal totals that math.fsum would round to a tie.
    """
    counts = interactions.data
    max_count = float(counts.max()) if counts.size else 0.0
    safe_integer_sums = math.ceil(max_count) * interactions.shape[0] <= 2**53
    if safe_integer_sums:
        for start in range(0, len(counts), 65_536):
            chunk = counts[start : start + 65_536]
            if np.any(chunk != np.trunc(chunk)):
                safe_integer_sums = False
                break
    if safe_integer_sums:
        frequencies = np.bincount(
            interactions.indices, weights=counts, minlength=len(item_ids)
        )
        positive = np.flatnonzero(frequencies > 0)
        order = np.lexsort((item_ids[positive], -frequencies[positive]))
        return positive[order], {
            "frequency_aggregation": "float64_exact_integer_bincount",
            "frequency_ranking_exact": True,
        }

    float_info = np.finfo(counts.dtype)
    fractional_bits = int(float_info.nmant - float_info.minexp)
    totals = [0] * len(item_ids)
    for item, count in zip(interactions.indices, counts):
        numerator, denominator = float(count).as_integer_ratio()
        totals[int(item)] += numerator << (fractional_bits - denominator.bit_length() + 1)
    positive = [item for item, total in enumerate(totals) if total]
    positive.sort(key=lambda item: (-totals[item], int(item_ids[item])))
    return np.asarray(positive, dtype=np.intp), {
        "frequency_aggregation": "python_integer_fixed_point",
        "frequency_quantum_exponent": -fractional_bits,
        "frequency_ranking_exact": True,
    }


def _excess_joint(
    joint: np.ndarray, antecedent_count: int, consequent_counts: np.ndarray, n: int
) -> np.ndarray:
    """Return J*N - A_a*A_b without int64 overflow or independence roundoff."""
    if n <= _SAFE_INT64_PRODUCT:
        return joint * n - antecedent_count * consequent_counts
    # This population would already require enormous sparse history storage.
    # Keep the arithmetic correct anyway, using at most M Python integers at a
    # time, rather than allocating a Python list of transaction/item pairs.
    return np.fromiter(
        (
            float(int(j) * n - antecedent_count * int(b))
            for j, b in zip(joint, consequent_counts)
        ),
        dtype=np.float64,
        count=len(joint),
    )


class AssociationRulesModel:
    """Training-only, frequency-bounded directed association-rule recommender.

    ``max_items`` has a hard safety limit of 4096. Other positive integer
    settings are configurable; their effective sizes cannot exceed the fitted
    catalog. ``smoothing`` is a finite, nonnegative pseudo-count. The optional
    ``count_chunk_size`` bounds the number of dense count rows materialized at
    once. Rule pruning is by descending positive shrunk uplift, then ascending
    numeric raw consequent ID, separately for each antecedent.

    Item-cap frequency ranking is exact for the represented training counts.
    Ordinary integral totals use a provably exact vectorized path; fractional
    or potentially huge totals use bounded, exact fixed-point accumulation.

    Diagnostic indices, like score candidates, are *matrix indices*. Raw IDs
    appear separately in traces. ``trace_rule`` also recounts pruned pairs on
    demand, so its counts can be checked independently of the retained graph.
    """

    def __init__(
        self,
        *,
        max_items: int = 4096,
        max_basket_size: int = 16,
        max_neighbors: int = 64,
        min_joint_count: int = 5,
        smoothing: float = 10.0,
        count_chunk_size: int = 128,
    ) -> None:
        for name, value in (
            ("max_items", max_items),
            ("max_basket_size", max_basket_size),
            ("max_neighbors", max_neighbors),
            ("min_joint_count", min_joint_count),
            ("count_chunk_size", count_chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_items > 4096:
            raise ValueError("max_items must not exceed the 4096-item safety bound")
        if isinstance(smoothing, bool):
            raise ValueError("smoothing must be finite and nonnegative")
        try:
            smoothing = float(smoothing)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("smoothing must be finite and nonnegative") from exc
        if not math.isfinite(smoothing) or smoothing < 0:
            raise ValueError("smoothing must be finite and nonnegative")

        self.max_items = int(max_items)
        self.max_basket_size = int(max_basket_size)
        self.max_neighbors = int(max_neighbors)
        self.min_joint_count = int(min_joint_count)
        self.smoothing = smoothing
        self.count_chunk_size = int(count_chunk_size)
        self.diagnostics: dict[str, Any] = {"fitted": False}
        self._fitted = False

    def fit(self, data: TrainingData) -> AssociationRulesModel:
        """Fit only the supplied training counts; do not mutate ``data``."""
        self._fitted = False
        self.diagnostics = {"fitted": False}
        interactions = data.interactions
        self._n_users, self._n_items = interactions.shape
        self._item_ids = data.item_ids.copy()
        original_user_sizes = np.diff(interactions.indptr)
        self._n_transactions = int(np.count_nonzero(original_user_sizes))

        # Frequency, not support/schedule rows, sets the cap. Ranking must
        # preserve exact ties and small differences independently of user order.
        ordered_items, frequency_diagnostics = _rank_training_items(
            interactions, self._item_ids
        )
        self._selected_indices = ordered_items[: self.max_items].copy()
        m = len(self._selected_indices)
        self._catalog_to_local = np.full(self._n_items, -1, dtype=np.int32)
        self._catalog_to_local[self._selected_indices] = np.arange(m, dtype=np.int32)
        self._history, coverage = self._build_history(interactions)
        self._supports = np.bincount(self._history.indices, minlength=m).astype(
            np.int64, copy=False
        )
        self._rules, self._rule_joint_counts, counting = self._fit_rules()

        basket_sizes = np.diff(self._history.indptr)
        nonempty_baskets = int(np.count_nonzero(basket_sizes))
        has_outgoing = np.diff(self._rules.indptr) > 0
        active_rows = self._history.copy()
        # One bounded sparse pass identifies users with at least one antecedent
        # that actually owns a retained positive outgoing rule.
        active_rows.data = has_outgoing[active_rows.indices].astype(np.int8)
        active_rows.eliminate_zeros()
        users_with_rule_antecedents = int(np.count_nonzero(np.diff(active_rows.indptr)))
        original_frequency = float(interactions.data.sum(dtype=np.float64))
        retained_frequency = float(self._history.data.sum(dtype=np.float64))
        original_nnz = int(interactions.nnz)
        retained_nnz = int(self._history.nnz)
        persistent_bytes = (
            _sparse_bytes(self._history)
            + _sparse_bytes(self._rules)
            + self._rule_joint_counts.nbytes
            + self._supports.nbytes
            + self._item_ids.nbytes
            + self._selected_indices.nbytes
            + self._catalog_to_local.nbytes
        )
        self.diagnostics = {
            "fitted": True,
            "model": "directed_singleton_association_rules",
            "backend": "scipy_sparse_int64_chunked_crossproduct",
            **frequency_diagnostics,
            "frequency_coverage_dtype": "float64_approximate",
            "basket_policy": "train_frequency_top_items_then_user_frequency_cap",
            "tie_break": "numeric_raw_program_id_ascending",
            "max_items": self.max_items,
            "max_basket_size": self.max_basket_size,
            "max_neighbors": self.max_neighbors,
            "min_joint_count": self.min_joint_count,
            "smoothing": self.smoothing,
            "n_users": self._n_users,
            "n_catalog_items": self._n_items,
            "n_original_positive_items": len(ordered_items),
            "n_selected_items": m,
            "n_transactions": self._n_transactions,
            "n_nonempty_baskets": nonempty_baskets,
            "n_original_empty_users": self._n_users - self._n_transactions,
            "n_positive_history_users_empty_after_item_cap": (
                self._n_transactions - nonempty_baskets
            ),
            "n_zero_basket_antecedent_users": self._n_users - nonempty_baskets,
            "n_users_with_retained_rule_antecedents": users_with_rule_antecedents,
            "n_zero_retained_rule_antecedent_users": (
                self._n_users - users_with_rule_antecedents
            ),
            "n_positive_history_users_without_retained_rule_antecedents": (
                self._n_transactions - users_with_rule_antecedents
            ),
            "original_positive_interactions": original_nnz,
            **coverage,
            "retained_basket_interactions": retained_nnz,
            "dropped_positive_interactions": original_nnz - retained_nnz,
            "retained_positive_interaction_fraction": (
                retained_nnz / original_nnz if original_nnz else 0.0
            ),
            "original_qualifying_frequency": original_frequency,
            "retained_qualifying_frequency": retained_frequency,
            "qualifying_frequency_dropped_by_item_cap": max(
                0.0, original_frequency - coverage["qualifying_frequency_after_item_cap"]
            ),
            "qualifying_frequency_dropped_by_basket_cap": max(
                0.0, coverage["qualifying_frequency_after_item_cap"] - retained_frequency
            ),
            "dropped_qualifying_frequency": max(
                0.0, original_frequency - retained_frequency
            ),
            "retained_qualifying_frequency_fraction": (
                retained_frequency / original_frequency if original_frequency else 0.0
            ),
            "retained_rule_count": int(self._rules.nnz),
            "basket_matrix_shape": [self._n_users, m],
            "rule_matrix_shape": [m, m],
            "largest_basket_size": int(basket_sizes.max()) if self._n_users else 0,
            "persistent_array_bytes": int(persistent_bytes),
            **counting,
        }
        self._fitted = True
        return self

    def _build_history(self, interactions: csr_matrix) -> tuple[csr_matrix, dict[str, Any]]:
        """Build exact-size sparse buffers; no transaction pair lists."""
        m = len(self._selected_indices)
        cap = min(self.max_basket_size, m)
        # The first sparse pass counts retained entries. The second selects
        # them, avoiding an oversized U*cap allocation when top items are rare.
        sizes = np.empty(self._n_users, dtype=np.int64)
        after_item_cap_nnz = 0
        after_item_cap_frequency = 0.0
        for user in range(self._n_users):
            start, stop = interactions.indptr[user : user + 2]
            mask = self._catalog_to_local[interactions.indices[start:stop]] >= 0
            size = int(np.count_nonzero(mask))
            sizes[user] = min(size, cap)
            after_item_cap_nnz += size
            after_item_cap_frequency += float(
                interactions.data[start:stop][mask].sum(dtype=np.float64)
            )
        indptr = np.empty(self._n_users + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(sizes, out=indptr[1:])
        indices = np.empty(int(indptr[-1]), dtype=np.int32)
        counts = np.empty(int(indptr[-1]), dtype=np.float64)
        raw_ids = self._item_ids[self._selected_indices]
        for user in range(self._n_users):
            if not sizes[user]:
                continue
            start, stop = interactions.indptr[user : user + 2]
            local = self._catalog_to_local[interactions.indices[start:stop]]
            mask = local >= 0
            local = local[mask]
            values = interactions.data[start:stop][mask].astype(np.float64)
            if len(local) > cap:
                keep = np.lexsort((raw_ids[local], -values))[:cap]
                local, values = local[keep], values[keep]
            # Canonical sparse column order also fixes summation order.
            column_order = np.argsort(local)
            out_start, out_stop = indptr[user : user + 2]
            indices[out_start:out_stop] = local[column_order]
            counts[out_start:out_stop] = values[column_order]
        history = csr_matrix(
            (counts, indices, indptr), shape=(self._n_users, m), copy=False
        )
        return history, {
            "interactions_after_item_cap": after_item_cap_nnz,
            "interactions_dropped_by_item_cap": int(interactions.nnz) - after_item_cap_nnz,
            "interactions_dropped_by_basket_cap": after_item_cap_nnz - int(history.nnz),
            "qualifying_frequency_after_item_cap": after_item_cap_frequency,
        }

    def _fit_rules(self) -> tuple[csr_matrix, np.ndarray, dict[str, Any]]:
        m = len(self._selected_indices)
        diagnostics: dict[str, Any] = {
            "count_dtype": "int64",
            "count_chunk_size": self.count_chunk_size,
            "largest_dense_count_block_shape": [0, m],
            "largest_dense_count_block_bytes": 0,
            "largest_sparse_count_block_bytes": 0,
            "counting_sparse_operands_logical_bytes": 0,
            "full_dense_count_matrix_allocated": False,
        }
        if m < 2:
            reason = "fewer_than_two_selected_items"
        elif self._n_transactions < self.min_joint_count:
            reason = "population_below_min_joint_count"
        elif not np.any(np.diff(self._history.indptr) >= 2):
            reason = "no_multi_item_basket"
        else:
            reason = None
        diagnostics["counting_skipped_reason"] = reason
        if reason is not None:
            return csr_matrix((m, m), dtype=np.float64), np.empty(0, np.int64), diagnostics

        # int64 is deliberate: uint8/uint16 binary matrices overflow at 256 /
        # 65536 supporting users. Indices/indptr may share history buffers;
        # the logical-byte diagnostic does not claim process peak RSS.
        binary = csr_matrix(
            (
                np.ones(self._history.nnz, dtype=np.int64),
                self._history.indices,
                self._history.indptr,
            ),
            shape=self._history.shape,
            copy=False,
        )
        transposed = binary.T.tocsr()
        diagnostics["counting_sparse_operands_logical_bytes"] = (
            _sparse_bytes(binary) + _sparse_bytes(transposed)
        )
        raw_ids = self._item_ids[self._selected_indices]
        indptr = np.zeros(m + 1, dtype=np.int64)
        # At most M arrays per list, each with at most max_neighbors entries.
        # These are bounded retained rules, not unbounded transaction pairs.
        destinations: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        joint_counts: list[np.ndarray] = []
        n = self._n_transactions
        for start in range(0, m, self.count_chunk_size):
            stop = min(start + self.count_chunk_size, m)
            product = transposed[start:stop] @ binary
            diagnostics["largest_sparse_count_block_bytes"] = max(
                diagnostics["largest_sparse_count_block_bytes"], _sparse_bytes(product)
            )
            dense = product.toarray()
            if dense.shape == (m, m):
                diagnostics["full_dense_count_matrix_allocated"] = True
            if dense.nbytes > diagnostics["largest_dense_count_block_bytes"]:
                diagnostics["largest_dense_count_block_bytes"] = int(dense.nbytes)
                diagnostics["largest_dense_count_block_shape"] = list(dense.shape)
            del product
            for offset, antecedent in enumerate(range(start, stop)):
                row = dense[offset]
                candidates = np.flatnonzero(row >= self.min_joint_count)
                candidates = candidates[candidates != antecedent]
                a_count = int(self._supports[antecedent])
                joint = row[candidates]
                excess = _excess_joint(joint, a_count, self._supports[candidates], n)
                # Algebraically q - p_b, evaluated stably. In particular,
                # independence gives exactly zero, not a roundoff-only rule.
                uplift = excess.astype(np.float64) / n / (a_count + self.smoothing)
                positive = uplift > 0
                candidates, joint, uplift = (
                    candidates[positive], joint[positive], uplift[positive]
                )
                order = np.lexsort((raw_ids[candidates], -uplift))[: self.max_neighbors]
                destinations.append(candidates[order].astype(np.int32))
                weights.append(uplift[order])
                joint_counts.append(joint[order])
                indptr[antecedent + 1] = indptr[antecedent] + len(order)
            # Release before the next block's dense allocation.
            del dense
        rule_matrix = csr_matrix(
            (np.concatenate(weights), np.concatenate(destinations), indptr),
            shape=(m, m),
            copy=False,
        )
        return rule_matrix, np.concatenate(joint_counts), diagnostics

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("AssociationRulesModel must be fitted first")

    def _known_user(self, user_index: Any) -> bool:
        return (
            not isinstance(user_index, (bool, np.bool_))
            and isinstance(user_index, Integral)
            and 0 <= user_index < self._n_users
        )

    def score(self, user_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        """Return candidate-aligned scores, preserving order and duplicates.

        Unknown users and out-of-catalog/unsupported candidates receive NaN.
        Candidates must be a one-dimensional integer array (an empty sequence
        is also allowed). No watched-item exclusion or popularity backfill.
        """
        self._require_fit()
        candidates = np.asarray(candidate_indices)
        if candidates.ndim != 1:
            raise ValueError("candidate_indices must be one-dimensional")
        if candidates.size and candidates.dtype.kind not in "iu":
            raise TypeError("candidate_indices must contain integer matrix indices")
        result = np.full(candidates.shape, np.nan, dtype=np.float64)
        if not self._known_user(user_index) or not candidates.size or not self._rules.nnz:
            return result
        valid_positions = np.flatnonzero((candidates >= 0) & (candidates < self._n_items))
        local = self._catalog_to_local[candidates[valid_positions].astype(np.intp)]
        selected = local >= 0
        valid_positions, local = valid_positions[selected], local[selected]
        if not len(local):
            return result
        history = self._history.getrow(int(user_index))
        if not history.nnz:
            return result
        history.data = np.log1p(history.data)
        evidence = (history @ self._rules).tocsr()
        evidence.sort_indices()
        if not evidence.nnz:
            return result
        locations = np.searchsorted(evidence.indices, local)
        present = locations < evidence.nnz
        positions = np.flatnonzero(present)
        present[positions] = evidence.indices[locations[positions]] == local[positions]
        positions = np.flatnonzero(present)
        values = evidence.data[locations[positions]]
        positive = np.isfinite(values) & (values > 0)
        result[valid_positions[positions[positive]]] = values[positive]
        return result

    def user_basket(self, user_index: int) -> list[dict[str, Any]]:
        """Trace the exact inference basket, frequency/ID ordered; unknown => []."""
        self._require_fit()
        if not self._known_user(user_index):
            return []
        row = self._history.getrow(int(user_index))
        original = self._selected_indices[row.indices]
        raw_ids = self._item_ids[original]
        order = np.lexsort((raw_ids, -row.data))
        return [
            {
                "item_index": int(original[i]),
                "item_id": int(raw_ids[i]),
                "frequency": float(row.data[i]),
            }
            for i in order
        ]

    def trace_rule(self, antecedent_index: int, consequent_index: int) -> dict[str, Any]:
        """Expose counts and probabilities for any in-catalog directed pair.

        Retained joints come from the fitted cross-product; other pairs are
        recounted from the bounded histories in O(U + retained interactions).
        Undefined probabilities are NaN. Self-pair counts can be inspected,
        but self rules are never eligible/retained. ``shrunk_uplift`` uses the
        stable algebraic equivalent of q - p_b; ``retained_weight`` is zero
        when thresholding or outgoing-neighbor pruning removed the rule.
        """
        self._require_fit()
        for index in (antecedent_index, consequent_index):
            if (
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, Integral)
                or not 0 <= index < self._n_items
            ):
                raise IndexError("Rule diagnostics require in-catalog matrix indices")
        a = int(self._catalog_to_local[antecedent_index])
        b = int(self._catalog_to_local[consequent_index])
        a_count = int(self._supports[a]) if a >= 0 else 0
        b_count = int(self._supports[b]) if b >= 0 else 0
        joint, retained_weight = 0, 0.0
        if a >= 0 and b >= 0:
            start, stop = self._rules.indptr[a : a + 2]
            match = np.flatnonzero(self._rules.indices[start:stop] == b)
            if len(match):
                position = int(start + match[0])
                joint = int(self._rule_joint_counts[position])
                retained_weight = float(self._rules.data[position])
            elif a == b:
                joint = a_count
            else:
                pair = self._history[:, [a, b]]
                joint = int(np.count_nonzero(np.diff(pair.indptr) == 2))
        n = self._n_transactions
        support = joint / n if n else math.nan
        confidence = joint / a_count if a_count else math.nan
        base = b_count / n if n else math.nan
        denominator = a_count + self.smoothing
        shrunk = (joint + self.smoothing * base) / denominator if n and denominator else math.nan
        uplift = (joint * n - a_count * b_count) / n / denominator if n and denominator else math.nan
        weight = max(0.0, uplift) if math.isfinite(uplift) else 0.0
        eligible = (
            a >= 0 and b >= 0 and a != b and joint >= self.min_joint_count and weight > 0
        )
        return {
            "antecedent_index": int(antecedent_index),
            "consequent_index": int(consequent_index),
            "antecedent_item_id": int(self._item_ids[antecedent_index]),
            "consequent_item_id": int(self._item_ids[consequent_index]),
            "antecedent_selected": a >= 0,
            "consequent_selected": b >= 0,
            "n_transactions": n,
            "antecedent_count": a_count,
            "consequent_count": b_count,
            "joint_count": joint,
            "support": support,
            "confidence": confidence,
            "base_probability": base,
            "shrunk_probability": shrunk,
            "raw_uplift": confidence - base,
            "shrunk_uplift": uplift,
            "weight": weight,
            "meets_min_joint_count": joint >= self.min_joint_count,
            "eligible_before_neighbor_cap": bool(eligible),
            "retained": retained_weight > 0,
            "retained_weight": retained_weight,
        }
