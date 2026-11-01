"""Small evidence-only fusions; no constituent popularity backfill is imported."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import inspect
from pathlib import Path
from numbers import Integral

import numpy as np

from sweettv.evaluation import reciprocal_rank_fusion
from .contracts import candidate_array, valid_user


class EvidenceBlendModel:
    def __init__(self, components: list[dict], mode: str = "rank",
                 rank_constant: float = 60.0, depth: int = 200,
                 reverse_score_ties: bool = False):
        if mode not in ("rank", "count"):
            raise ValueError("mode must be rank or count")
        if not components or len({c["name"] for c in components}) != len(components):
            raise ValueError("Use nonempty uniquely named components")
        if not np.isfinite(rank_constant) or rank_constant < 0 or depth < 1:
            raise ValueError("Invalid rank parameters")
        weights = [c.get("weight", 1.0) for c in components]
        if any(isinstance(w, bool) or not np.isfinite(w) or w < 0 for w in weights) or not any(weights):
            raise ValueError("Use finite nonnegative weights and at least one positive weight")
        self.components = deepcopy(components)
        self.mode, self.rank_constant, self.depth = mode, rank_constant, depth
        self.reverse_score_ties = reverse_score_ties
        self.diagnostics = {}
        self._models = None

    def fit(self, context):
        fitted, source_hashes = {}, {}
        for component in self.components:
            if component.get("weight", 1.0) == 0:
                continue
            module, cls_name = component["class"].split(":", 1)
            if not module.startswith("research_v2.") or cls_name == "EvidenceBlendModel":
                raise ValueError("Use direct project-owned research components")
            cls = getattr(importlib.import_module(module), cls_name)
            path = Path(inspect.getfile(cls))
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            model = cls(**component.get("parameters", {})).fit(context)
            if hashlib.sha256(path.read_bytes()).hexdigest() != before:
                raise RuntimeError("Constituent source changed during fit")
            source_hashes[str(path)] = before
            fitted[component["name"]] = model
        self._models = fitted
        self._weights = {c["name"]: float(c.get("weight", 1.0)) for c in self.components
                         if c.get("weight", 1.0) > 0}
        self._raw_ids = context.data.item_ids.copy()
        self._n_users = len(context.data.user_ids)
        self.diagnostics = {
            "mode": self.mode, "components": self.components,
            "rank_constant": self.rank_constant, "depth": self.depth,
            "reverse_score_ties": self.reverse_score_ties,
            "sources": source_hashes,
            "fits": {name: model.diagnostics for name, model in fitted.items()},
            "internal_backfill": False,
            "count_mode_units": "unscaled historical-count proxies; weights are development choices, not probability calibration",
        }
        return self

    def score(self, user_index, candidate_indices):
        if self._models is None:
            raise RuntimeError("Fit EvidenceBlendModel before scoring")
        candidates = candidate_array(candidate_indices)
        result = np.full(len(candidates), np.nan)
        if not valid_user(user_index, self._n_users):
            return result
        valid = (candidates >= 0) & (candidates < len(self._raw_ids))
        selected = candidates[valid].astype(np.intp, copy=False)
        if not len(selected):
            return result
        vectors = {name: np.array(model.score(user_index, selected), dtype=np.float64, copy=True)
                   for name, model in self._models.items()}
        result[valid] = self._blend(vectors, selected)
        return result

    def _blend(self, vectors, selected):
        if any(v.shape != selected.shape or np.isinf(v).any() for v in vectors.values()):
            raise ValueError("Invalid constituent score shape/values")
        if self.mode == "rank":
            tie_ids = -self._raw_ids if self.reverse_score_ties else self._raw_ids
            # Deduplicate before assigning ranks; duplicate query positions must
            # not receive distinct rank votes or consume depth.
            unique, first, inverse = np.unique(selected, return_index=True, return_inverse=True)
            fused = reciprocal_rank_fusion({n: v[first] for n, v in vectors.items()},
                                            self._weights, unique, tie_ids,
                                            rank_constant=self.rank_constant, depth=self.depth)
            values = fused[inverse]
        else:
            values = np.zeros(len(selected), dtype=np.float64)
            supported = np.zeros(len(selected), dtype=bool)
            for name, vector in vectors.items():
                observed = np.isfinite(vector)
                with np.errstate(over="raise", invalid="raise"):
                    values[observed] += self._weights[name] * vector[observed]
                supported |= observed
            if not np.isfinite(values[supported]).all():
                raise FloatingPointError("Nonfinite supported count fusion; no evidence was discarded")
            values[~supported] = np.nan
        if np.isinf(values).any():
            raise FloatingPointError("Fusion overflow; no backfill was substituted")
        return values

    def score_batch(self, user_indices, candidate_indices):
        """Batch users, never candidate-axis chunks (RRF is listwise)."""
        if self._models is None:
            raise RuntimeError("Fit EvidenceBlendModel before scoring")
        users = np.asarray(user_indices, dtype=object)
        if users.ndim != 1 or any(isinstance(u, (bool, np.bool_)) or not isinstance(u, Integral) for u in users):
            raise ValueError("Users must be a one-dimensional integer-index vector")
        candidates = candidate_array(candidate_indices)
        if len(users) * len(candidates) > 1_000_000:
            raise ValueError("Fusion score_batch exceeds one million requested cells")
        result = np.full((len(users), len(candidates)), np.nan, dtype=np.float64)
        known_rows = [r for r, user in enumerate(users) if 0 <= user < self._n_users]
        valid = (candidates >= 0) & (candidates < len(self._raw_ids))
        selected = candidates[valid].astype(np.intp, copy=False)
        if not known_rows or not len(selected):
            return result
        query_users = np.array([users[r] for r in known_rows], dtype=np.int64)
        query_users.setflags(write=False)
        selected.setflags(write=False)
        vectors = {}
        for name, model in self._models.items():
            if hasattr(model, "score_batch"):
                values = np.array(model.score_batch(query_users, selected), dtype=np.float64, copy=True)
            else:
                values = np.vstack([np.array(model.score(int(u), selected), dtype=np.float64, copy=True) for u in query_users])
            if values.shape != (len(query_users), len(selected)):
                raise ValueError("Invalid constituent batch dimensions")
            vectors[name] = values
        positions = np.flatnonzero(valid)
        for row, output_row in enumerate(known_rows):
            result[output_row, positions] = self._blend({n: v[row] for n, v in vectors.items()}, selected)
        return result
