"""Stable model/data interface shared by independently implemented methods.

Behavioral interactions are training-only qualifying broadcast counts. Metadata
may include supplied forecast schedules, but never future viewing behavior.
Raw identifiers are kept separately from contiguous matrix indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class TrainingData:
    user_ids: list[str]
    item_ids: np.ndarray
    interactions: csr_matrix
    metadata: list[dict[str, Any]]
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.item_ids = np.asarray(self.item_ids, dtype=np.int64)
        self.interactions = csr_matrix(self.interactions, dtype=np.float32)
        self.interactions.sum_duplicates()
        self.interactions.eliminate_zeros()
        self.interactions.sort_indices()
        if self.interactions.shape != (len(self.user_ids), len(self.item_ids)):
            raise ValueError("Interaction shape must be users x items")
        if len(self.metadata) != len(self.item_ids):
            raise ValueError("Metadata must align with item indices")
        if any(not isinstance(u, str) or not u for u in self.user_ids):
            raise ValueError("User IDs must be nonempty strings")
        if len(set(self.user_ids)) != len(self.user_ids):
            raise ValueError("Duplicate user IDs")
        if len(set(map(int, self.item_ids))) != len(self.item_ids):
            raise ValueError("Duplicate program IDs")
        if np.any(self.item_ids <= 0):
            raise ValueError("Program IDs must be positive")
        if not np.all(np.isfinite(self.interactions.data)):
            raise ValueError("Interactions must be finite")
        if np.any(self.interactions.data <= 0):
            raise ValueError("Stored interactions must be positive counts")


class RecommendationModel(Protocol):
    """All score arrays align with candidate *matrix indices*, not raw IDs.

    Larger finite scores rank first. NaN denotes absent model evidence and is
    backfilled by the shared ranker. Already-watched items remain eligible.
    Models must not add popularity backfill internally. Intrinsic zero/no-path
    scores should be NaN when they contain no evidence; legitimate negative
    latent-factor scores are allowed. Unknown users return NaNs, never index -1.
    """

    diagnostics: dict[str, Any]

    def fit(self, data: TrainingData) -> RecommendationModel: ...

    def score(self, user_index: int, candidate_indices: np.ndarray) -> np.ndarray: ...
