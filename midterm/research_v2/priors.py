"""Public-heuristic-inspired airing priors, independently implemented.

Source idea: kinahem/gornyaki-final-solution, final scoring cells40/45/46.
Only its recency/airing-prior ideas are used. Its metric/preprocessing and any
outputs are not executed or copied. No numeric source score is assumed.
"""

from __future__ import annotations

import math

import numpy as np

from .contracts import TemporalContext, candidate_array, temporal_matrix, valid_user


PUBLIC_HALF_LIFE_DAYS = -60.0 * math.log(2.0) / math.log(0.8)


class AiringPriorRepeatModel:
    def __init__(self, half_life_days: float | None = PUBLIC_HALF_LIFE_DAYS,
                 lookback_days: float | None = None, prior_basis: str = "history",
                 prior_transform: str = "log1p", prior_power: float = 1.0):
        if prior_basis not in ("history", "forecast"):
            raise ValueError("prior_basis must be history or forecast")
        if prior_transform not in ("log1p", "log10"):
            raise ValueError("Unknown prior transformation")
        if isinstance(prior_power, bool) or not np.isfinite(prior_power) or prior_power < 0:
            raise ValueError("Prior power must be finite and nonnegative")
        self.half_life_days = half_life_days
        self.lookback_days = lookback_days
        self.prior_basis = prior_basis
        self.prior_transform = prior_transform
        self.prior_power = float(prior_power)
        self.diagnostics = {"status": "not_fitted"}

    def fit(self, context: TemporalContext):
        matrix = temporal_matrix(context, self.half_life_days, self.lookback_days)
        rows = context.historical_schedules if self.prior_basis == "history" else context.forecast_schedules
        counts = np.zeros(len(context.data.item_ids), dtype=np.float64)
        for airing in rows:
            counts[airing.item_index] += 1
        if self.prior_power == 0:
            factors = np.ones_like(counts)
        elif self.prior_transform == "log1p":
            factors = np.log1p(counts) ** self.prior_power
        else:
            factors = np.zeros_like(counts)
            positive = counts > 0
            factors[positive] = np.log10(counts[positive]) ** self.prior_power
        self._counts = matrix
        self._prior = factors
        self.diagnostics = {
            "status": "fitted", "half_life_days": self.half_life_days,
            "lookback_days": self.lookback_days, "prior_basis": self.prior_basis,
            "prior_transform": self.prior_transform, "prior_power": self.prior_power,
            "training_events": len(context.event_users), "active_prior_items": int((factors > 0).sum()),
            "interpretation": "Predictive airing-frequency prior, not personal exposure or a calibrated probability",
            "public_formula_difference": "Qualifying views count once rather than summing its unreconciled coverage ratings; log1p option avoids forcing one-airing programs to zero",
        }
        return self

    def score(self, user_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_counts"):
            raise RuntimeError("Fit the airing-prior model before scoring")
        candidates = candidate_array(candidate_indices)
        scores = np.full(len(candidates), np.nan)
        if not valid_user(user_index, self._counts.shape[0]):
            return scores
        valid = (candidates >= 0) & (candidates < self._counts.shape[1])
        selected = candidates[valid].astype(np.intp, copy=False)
        values = self._counts[user_index, selected].toarray().ravel() * self._prior[selected]
        values[values <= 0] = np.nan
        scores[valid] = values
        return scores
