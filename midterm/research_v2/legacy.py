"""Unmodified round-one methods behind the research context interface."""

import numpy as np

from sweettv.evaluation import reciprocal_rank_fusion
from sweettv.pipeline import create_model


class LegacyHybrid:
    def __init__(self, reverse_score_ties: bool = False):
        self.reverse_score_ties = reverse_score_ties
        self.weights = {"repeat": 1.0, "cbf": 0.5, "association": 0.5, "mf": 0.5, "ppr": 0.5}
        self.diagnostics = {}

    def fit(self, context):
        self.data = context.data
        self.models = {name: create_model(name).fit(context.data) for name in self.weights}
        self.diagnostics = {"round_one_parameters_unchanged": not self.reverse_score_ties,
                            "reverse_score_ties_sensitivity_only": self.reverse_score_ties, "weights": self.weights,
                            "components": {name: model.diagnostics for name, model in self.models.items()}}
        return self

    def score_batch(self, users, candidates):
        unique, inverse = np.unique(candidates, return_inverse=True)
        if not len(users):
            return np.empty((0, len(candidates)), dtype=np.float64)
        vectors = {}
        for name, model in self.models.items():
            if hasattr(model, "score_batch"):
                values = np.array(model.score_batch(users, unique), dtype=float, copy=True)
            else:
                values = np.vstack([np.array(model.score(int(u), unique), dtype=float, copy=True) for u in users])
            vectors[name] = values
        tie_ids = -self.data.item_ids if self.reverse_score_ties else self.data.item_ids
        return np.vstack([reciprocal_rank_fusion({name: scores[r] for name, scores in vectors.items()},
                                                 self.weights, unique, tie_ids)
                          for r in range(len(users))])[:, inverse]

    def score(self, user, candidates):
        return self.score_batch(np.array([user]), candidates)[0]
