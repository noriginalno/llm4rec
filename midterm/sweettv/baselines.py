"""Training-only reference methods, separate from the four requested models."""

from __future__ import annotations

import numpy as np

from .contracts import TrainingData


class PopularityModel:
    def fit(self, data: TrainingData) -> PopularityModel:
        self.popularity = np.asarray(data.interactions.sum(axis=0)).ravel()
        self.diagnostics = {"definition": "sum of training qualifying broadcast counts"}
        return self

    def score(self, user_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        scores = self.popularity[candidate_indices].astype(float)
        scores[scores <= 0] = np.nan
        return scores


class PersonalRepeatModel:
    def fit(self, data: TrainingData) -> PersonalRepeatModel:
        self.interactions = data.interactions
        self.diagnostics = {"definition": "user's training qualifying broadcast counts"}
        return self

    def score(self, user_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        if user_index < 0 or user_index >= self.interactions.shape[0]:
            return np.full(len(candidate_indices), np.nan)
        scores = self.interactions[user_index, candidate_indices].toarray().ravel().astype(float)
        scores[scores <= 0] = np.nan
        return scores
