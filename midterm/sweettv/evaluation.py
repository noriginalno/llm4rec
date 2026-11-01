"""Shared ranking and standard binary-relevance retrieval metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def average_precision_at_k(predicted: list[int], actual: list[int], k: int = 5) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(actual)
    if not relevant:
        return 0.0
    found: set[int] = set()
    precision_sum = 0.0
    for rank, item in enumerate(predicted[:k], 1):
        if item in relevant and item not in found:
            found.add(item)
            precision_sum += len(found) / rank
    return precision_sum / min(k, len(relevant))


def evaluate_predictions(
    predictions: dict[str, list[int]],
    labels: dict[str, list[int]],
    users: list[str],
    k: int = 5,
) -> dict[str, Any]:
    if k < 1 or len(set(users)) != len(users):
        raise ValueError("Positive k and unique evaluation users are required")
    per_user = []
    for user in users:
        if user not in predictions or user not in labels:
            raise ValueError(f"Missing predictions or labels for {user}")
        relevant = set(labels[user])
        hits = len(set(predictions[user][:k]) & relevant)
        per_user.append({
            "user_id": user,
            "ap": average_precision_at_k(predictions[user], labels[user], k),
            "precision": hits / k,
            "recall": hits / len(relevant) if relevant else 0.0,
            "hits": hits,
            "relevant_count": len(relevant),
        })
    def mean(key: str) -> float | None:
        return float(np.mean([row[key] for row in per_user])) if per_user else None
    return {
        f"map_at_{k}": mean("ap"),
        f"precision_at_{k}": mean("precision"),
        f"recall_at_{k}": mean("recall"),
        "user_count": len(users),
        "empty_label_users": sum(not labels[u] for u in users),
        "per_user": per_user,
    }


def rank_candidates(
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    item_ids: np.ndarray,
    popularity: np.ndarray,
    k: int = 5,
) -> dict[str, Any]:
    """Rank evidence, then backfill once; numeric raw ID resolves every tie."""
    scores = np.asarray(scores, dtype=float)
    candidates = np.asarray(candidate_indices)
    item_ids = np.asarray(item_ids)
    popularity = np.asarray(popularity, dtype=float)
    if k < 1 or candidates.ndim != 1 or scores.shape != candidates.shape:
        raise ValueError("k must be positive and score/candidate vectors must align")
    if candidates.dtype.kind not in "iu":
        raise ValueError("Candidate indices must be integers")
    if popularity.shape != item_ids.shape or item_ids.ndim != 1:
        raise ValueError("Popularity and item-ID vectors must align")
    if item_ids.dtype.kind not in "iu" or np.any(item_ids <= 0):
        raise ValueError("Catalog must contain positive integer raw program IDs")
    if len(set(map(int, item_ids))) != len(item_ids):
        raise ValueError("Raw program IDs must be unique across the catalog")
    if np.any(candidates < 0) or np.any(candidates >= len(item_ids)):
        raise ValueError("Candidate index out of range")
    if np.isinf(scores).any() or not np.isfinite(popularity).all() or np.any(popularity < 0):
        raise ValueError("Infinite model scores or invalid popularity")
    evidence: dict[int, float] = {}
    for index, score in zip(candidates, scores):
        index = int(index)
        if index in evidence and not (
            evidence[index] == float(score) or np.isnan(evidence[index]) and np.isnan(score)
        ):
            raise ValueError("Duplicate candidate has inconsistent scores")
        evidence[index] = float(score)
    if len(evidence) < k:
        raise ValueError(f"Need at least {k} distinct candidate programs")
    ordered = sorted(
        (i for i, score in evidence.items() if np.isfinite(score)),
        key=lambda i: (-evidence[i], int(item_ids[i])),
    )[:k]
    used = set(ordered)
    fallback = sorted(
        (i for i in evidence if i not in used),
        key=lambda i: (-popularity[i], int(item_ids[i])),
    )[:k - len(ordered)]
    result = ordered + fallback
    return {
        "items": [int(item_ids[i]) for i in result],
        "indices": result,
        "scores": [evidence[i] for i in ordered] + [None] * len(fallback),
        "fallback_positions": len(fallback),
    }


def reciprocal_rank_fusion(
    score_vectors: dict[str, np.ndarray],
    weights: dict[str, float],
    candidate_indices: np.ndarray,
    item_ids: np.ndarray,
    rank_constant: float = 60.0,
    depth: int = 200,
) -> np.ndarray:
    """Combine evidence-only ranks. Popularity backfill never receives votes."""
    if rank_constant < 0 or depth < 1:
        raise ValueError("Invalid fusion parameters")
    result = np.zeros(len(candidate_indices), dtype=float)
    supported = np.zeros(len(candidate_indices), dtype=bool)
    for name, weight in weights.items():
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Fusion weights must be finite and nonnegative")
        if weight == 0:
            continue
        scores = np.asarray(score_vectors[name], dtype=float)
        if scores.shape != result.shape or np.isinf(scores).any():
            raise ValueError("Invalid fusion score vector")
        positions = np.flatnonzero(np.isfinite(scores))
        positions = sorted(
            positions, key=lambda j: (-scores[j], int(item_ids[candidate_indices[j]]))
        )[:depth]
        for rank, position in enumerate(positions, 1):
            result[position] += weight / (rank_constant + rank)
            supported[position] = True
    result[~supported] = np.nan
    return result
