"""Training-only channel/clock habits integrated over supplied forecast airings.

Scheduled airings are opportunity proxies, not measured personal exposure. The
exact hierarchy and capped repeat blend are documented in
``reviews/research-v2-schedule.md``. No raw logs, labels, files, or network are
consulted by this module.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from research_v2.contracts import TemporalContext, candidate_array, valid_user


def _positive(value, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _slots(starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Source-clock arithmetic only; epoch-anchor Thursday has weekday index 3."""
    hours = (starts // 3600) % 24
    weekhours = (((starts // 86400) + 3) % 7) * 24 + hours
    return hours, weekhours


def _counts(users, groups, weights, shape) -> csr_matrix:
    matrix = coo_matrix((weights, (users, groups)), shape=shape, dtype=np.float64).tocsr()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _opportunities(groups, weights, size: int) -> np.ndarray:
    return np.bincount(groups, weights=weights, minlength=size).astype(np.float64)


def _rate_diagnostics(counts: csr_matrix, opportunities: np.ndarray) -> dict:
    cells = counts.tocoo(copy=False)
    if not cells.nnz:
        return {"completion_cells": 0, "min_observed_rate": None, "max_observed_rate": None}
    denominator = opportunities[cells.col]
    if np.any(denominator <= 0) or np.any(cells.data > denominator + 1e-9):
        raise ValueError("Completions exceed matched historical opportunities")
    rates = cells.data / denominator
    return {
        "completion_cells": int(cells.nnz),
        "min_observed_rate": float(rates.min()),
        "max_observed_rate": float(rates.max()),
    }


class ChannelScheduleModel:
    """Sparse channel -> hour -> weekhour rates with a personal-repeat blend.

    ``half_life_days=None`` disables decay. ``lookback_days`` is an optional
    inclusive end-age window. The zero-centered channel prior and zero-preserving
    activity gates prevent unsupported channel/time transfer. Only a query's
    channel/time-group vector and requested candidate vector are densified.
    Parameter edits take effect on the next fit. A failed fit invalidates scoring
    rather than silently continuing with an earlier context.
    """

    def __init__(
        self,
        *,
        half_life_days: float | None = 28.0,
        lookback_days: float | None = None,
        slot_mode: str = "weekhour",
        prior_strength: float = 4.0,
        programme_blend: float = 0.5,
        max_repeat_multiplier: float = 3.0,
        quarantined_future_policy: str = "exclude",
    ):
        self.half_life_days = half_life_days
        self.lookback_days = lookback_days
        self.slot_mode = slot_mode
        self.prior_strength = prior_strength
        self.programme_blend = programme_blend
        self.max_repeat_multiplier = max_repeat_multiplier
        self.quarantined_future_policy = quarantined_future_policy
        self.diagnostics: dict = {}
        self._fitted = False
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        _positive(self.half_life_days, "half_life_days", optional=True)
        _positive(self.lookback_days, "lookback_days", optional=True)
        _positive(self.prior_strength, "prior_strength")
        _positive(self.max_repeat_multiplier, "max_repeat_multiplier")
        if self.slot_mode not in {"channel", "hour", "weekhour"}:
            raise ValueError("slot_mode must be channel, hour, or weekhour")
        if self.quarantined_future_policy not in {"include", "exclude"}:
            raise ValueError("quarantined_future_policy must be include or exclude")
        if (isinstance(self.programme_blend, (bool, np.bool_))
                or not isinstance(self.programme_blend, Real)
                or not np.isfinite(self.programme_blend)
                or not 0 <= self.programme_blend <= 1):
            raise ValueError("programme_blend must be finite and in [0, 1]")

    def _weights(self, ends: np.ndarray, context: TemporalContext) -> np.ndarray:
        ages = (context.train_end - ends).astype(np.float64) / 86400.0
        weights = np.ones(len(ends), dtype=np.float64)
        if self.half_life_days is not None:
            weights = np.exp2(-ages / self.half_life_days)
        weights[(ends <= context.history_start) | (ends > context.train_end)] = 0.0
        if self.lookback_days is not None:
            weights[ages > self.lookback_days] = 0.0
        return weights

    @staticmethod
    def _airings(airings, n_items: int) -> tuple[dict, int]:
        """Deduplicate full broadcast identity, conservatively OR-ing flags."""
        unique = {}
        duplicates = 0
        for airing in airings:
            if (isinstance(airing.item_index, (bool, np.bool_))
                    or not isinstance(airing.item_index, Integral)
                    or not 0 <= airing.item_index < n_items):
                raise ValueError("Airing item index out of range")
            if not isinstance(airing.channel_id, str) or not airing.channel_id:
                raise ValueError("Airing channel must be a nonempty string")
            if any(isinstance(t, (bool, np.bool_)) or not isinstance(t, Integral)
                   for t in (airing.start, airing.end)) or airing.start >= airing.end:
                raise ValueError("Airing times must be integer seconds and start < end")
            key = (int(airing.item_index), airing.channel_id, int(airing.start), int(airing.end))
            if key in unique:
                duplicates += 1
            unique[key] = unique.get(key, False) or bool(airing.quarantined)
        return unique, duplicates

    def fit(self, context: TemporalContext) -> ChannelScheduleModel:
        self._fitted = False
        self.diagnostics = {}
        self._validate_parameters()
        n_users, n_items = context.data.interactions.shape
        historical, h_duplicates = self._airings(context.historical_schedules, n_items)
        forecast_all, f_duplicates = self._airings(context.forecast_schedules, n_items)
        h_keys = list(historical)
        h_ends = np.array([key[3] for key in h_keys], dtype=np.int64)
        h_weights = self._weights(h_ends, context)
        keep = h_weights > 0
        h_keys = [key for key, retained in zip(h_keys, keep) if retained]
        h_weights = h_weights[keep]
        historical_index = {key: i for i, key in enumerate(h_keys)}
        flagged = [key for key, flag in forecast_all.items() if flag]
        f_keys = [key for key, flag in forecast_all.items()
                  if not flag or self.quarantined_future_policy == "include"]

        channel_ids = sorted({key[1] for key in h_keys} | {key[1] for key in f_keys})
        channel_index = {channel: i for i, channel in enumerate(channel_ids)}
        n_channels = len(channel_ids)
        h_items = np.array([key[0] for key in h_keys], dtype=np.int64)
        h_channels = np.array([channel_index[key[1]] for key in h_keys], dtype=np.int64)
        h_hours, h_weekhours = _slots(np.array([key[2] for key in h_keys], dtype=np.int64))
        f_items = np.array([key[0] for key in f_keys], dtype=np.int64)
        f_channels = np.array([channel_index[key[1]] for key in f_keys], dtype=np.int64)
        f_hours, f_weekhours = _slots(np.array([key[2] for key in f_keys], dtype=np.int64))

        # Match completions to identical opportunities: no synthetic denominators
        # and no use of count CSR/metadata as additional behavioral observations.
        event_weights = self._weights(context.event_ends, context)
        users, airing_indices, seen = [], [], set()
        unmatched = event_duplicates = 0
        for position in np.flatnonzero(event_weights > 0):
            key = (int(context.event_items[position]), str(context.event_channels[position]),
                   int(context.event_starts[position]), int(context.event_ends[position]))
            airing_index = historical_index.get(key)
            if airing_index is None:
                unmatched += 1
                continue
            user = int(context.event_users[position])
            if (user, airing_index) in seen:
                event_duplicates += 1
                continue
            seen.add((user, airing_index))
            users.append(user)
            airing_indices.append(airing_index)
        users = np.array(users, dtype=np.int64)
        airing_indices = np.array(airing_indices, dtype=np.int64)
        e_weights = h_weights[airing_indices]
        e_channels = h_channels[airing_indices]
        e_items = h_items[airing_indices]
        e_hours = h_hours[airing_indices]
        e_weekhours = h_weekhours[airing_indices]

        channel_counts = _counts(users, e_channels, e_weights, (n_users, n_channels))
        hour_counts = _counts(users, e_hours, e_weights, (n_users, 24))
        weekhour_counts = _counts(users, e_weekhours, e_weights, (n_users, 168))
        ch_hour_counts = _counts(users, e_channels * 24 + e_hours, e_weights,
                                 (n_users, n_channels * 24))
        ch_weekhour_counts = _counts(users, e_channels * 168 + e_weekhours, e_weights,
                                     (n_users, n_channels * 168))
        programme_counts = _counts(users, e_items, e_weights, (n_users, n_items))
        channel_opportunities = _opportunities(h_channels, h_weights, n_channels)
        ch_hour_opportunities = _opportunities(h_channels * 24 + h_hours, h_weights, n_channels * 24)
        ch_weekhour_opportunities = _opportunities(
            h_channels * 168 + h_weekhours, h_weights, n_channels * 168)
        historical_item_opportunities = _opportunities(h_items, h_weights, n_items)
        forecast_item_opportunities = _opportunities(f_items, np.ones(len(f_items)), n_items)
        n_slots = {"channel": 1, "hour": 24, "weekhour": 168}[self.slot_mode]
        f_groups = (f_channels if n_slots == 1 else
                    f_channels * n_slots + (f_hours if n_slots == 24 else f_weekhours))
        forecast_group_items = coo_matrix(
            (np.ones(len(f_items)), (f_groups, f_items)),
            shape=(n_channels * n_slots, n_items), dtype=np.float64,
        ).tocsc()
        forecast_group_items.sort_indices()
        repeat_ratio = forecast_item_opportunities / (historical_item_opportunities + self.prior_strength)
        repeat_multiplier = np.minimum(repeat_ratio, self.max_repeat_multiplier)
        active_users = np.asarray(programme_counts.sum(axis=1)).ravel() > 0
        # Use matched counts, not just source-array presence, for the conservative
        # sufficient condition that flagged airings cannot change any user's score.
        trained_channels = {channel_ids[i] for i in np.flatnonzero(
            np.asarray(channel_counts.sum(axis=0)).ravel() > 0)}
        trained_items = set(np.flatnonzero(np.asarray(programme_counts.sum(axis=0)).ravel() > 0))
        supported_flags = sum(key[1] in trained_channels or key[0] in trained_items for key in flagged)
        candidates = np.unique(context.candidate_indices)
        candidate_forecast = candidates[forecast_item_opportunities[candidates] > 0]
        levels = {
            "channel": _rate_diagnostics(channel_counts, channel_opportunities),
            "hour": _rate_diagnostics(ch_hour_counts, ch_hour_opportunities),
            "weekhour": _rate_diagnostics(ch_weekhour_counts, ch_weekhour_opportunities),
        }

        self.n_users_, self.n_items_, self.n_channels_ = n_users, n_items, n_channels
        self.item_ids_ = np.array(context.data.item_ids, copy=True)
        self.item_ids_.setflags(write=False)
        self.user_ids_ = tuple(context.data.user_ids)
        self.channel_ids_ = tuple(channel_ids)
        self.channel_counts_, self.hour_counts_ = channel_counts, hour_counts
        self.weekhour_counts_ = weekhour_counts
        self.channel_hour_counts_ = ch_hour_counts
        self.channel_weekhour_counts_ = ch_weekhour_counts
        self.programme_counts_ = programme_counts
        self.channel_opportunities_ = channel_opportunities
        self.channel_hour_opportunities_ = ch_hour_opportunities
        self.channel_weekhour_opportunities_ = ch_weekhour_opportunities
        self.historical_item_opportunities_ = historical_item_opportunities
        self.forecast_item_opportunities_ = forecast_item_opportunities
        self.forecast_group_items_ = forecast_group_items
        self.repeat_multiplier_ = repeat_multiplier
        self._active_users = active_users
        # Parameter edits take effect on refit, never halfway through scoring a
        # fitted forecast matrix with a different group dimensionality.
        self._fitted_slot_mode = self.slot_mode
        self._fitted_prior_strength = self.prior_strength
        self._fitted_programme_blend = self.programme_blend
        self._fitted_max_repeat_multiplier = self.max_repeat_multiplier
        self.diagnostics = {
            "model": "ChannelScheduleModel",
            "backend": "scipy.sparse CPU; float64",
            "parameters": {
                "half_life_days": self.half_life_days, "lookback_days": self.lookback_days,
                "slot_mode": self.slot_mode, "prior_strength": self.prior_strength,
                "programme_blend": self.programme_blend,
                "max_repeat_multiplier": self.max_repeat_multiplier,
                "quarantined_future_policy": self.quarantined_future_policy,
            },
            "users": n_users, "items": n_items, "channels": n_channels,
            "active_users": int(active_users.sum()), "hierarchy": levels,
            "hierarchy_order": "channel -> channel/hour -> channel/weekhour; zero-preserving activity gates",
            "history_window": {
                "history_start": int(context.history_start), "train_end": int(context.train_end),
                "forecast_start": int(context.forecast_start), "forecast_end": int(context.forecast_end),
                "end_rule": "history_start < end <= train_end; end_age_days <= lookback_days",
            },
            "source_clock": "naive source-clock epoch anchor; Monday=0; no time-zone inference",
            "behavior_source": "context training-only >=80% completed broadcasts; partial logs unavailable",
            "opportunity_interpretation": "scheduled airing proxy, not personal exposure or calibrated probability",
            "metadata_rows": len(context.data.metadata), "descriptive_metadata_used": False,
            "historical_airings_input": len(context.historical_schedules),
            "historical_airings_retained": len(h_keys),
            "historical_opportunity_weight": float(h_weights.sum()),
            "historical_duplicates_ignored": h_duplicates,
            "completion_events_input": len(context.event_users),
            "completion_events_retained": len(users),
            "completion_weight": float(e_weights.sum()),
            "completion_events_outside_weight_window": int(np.count_nonzero(event_weights == 0)),
            "unmatched_completion_events_excluded": unmatched,
            "duplicate_completion_events_ignored": event_duplicates,
            "forecast_airings_input": len(context.forecast_schedules),
            "forecast_airings_retained": len(f_keys), "forecast_duplicates_ignored": f_duplicates,
            "forecast_items_with_opportunities": int(np.count_nonzero(forecast_item_opportunities)),
            "forecast_group_item_shape": list(forecast_group_items.shape),
            "forecast_group_item_nnz": int(forecast_group_items.nnz),
            "candidate_coverage": {
                "unique_input_candidates": len(candidates),
                "with_retained_forecast_airings": len(candidate_forecast),
                "cold_training_programmes_with_forecast": sum(
                    int(item) not in trained_items for item in candidate_forecast),
                "repeat_only_user_candidate_pairs": int(programme_counts[:, candidates].nnz),
                "full_personal_channel_slot_coverage": "measured per query, not inferred from repeat coverage",
            },
            "quarantined_forecast_rows": int(sum(a.quarantined for a in context.forecast_schedules)),
            "quarantined_forecast_airings": len(flagged),
            "quarantined_forecast_items": len({key[0] for key in flagged}),
            "quarantined_airings_excluded": len(flagged) if self.quarantined_future_policy == "exclude" else 0,
            "quarantine_changes_opportunity_counts": bool(flagged),
            "quarantined_airings_with_any_training_support": int(supported_flags),
            "quarantine_score_irrelevance_proven": supported_flags == 0,
            "candidate_ids_removed": 0,
            "programme_repeat_cells": int(programme_counts.nnz),
            "repeat_uplift_bound_per_weighted_completion": float(
                self.programme_blend * self.max_repeat_multiplier),
            "forecast_items_repeat_multiplier_capped": int(np.count_nonzero(
                (repeat_ratio > self.max_repeat_multiplier) & (historical_item_opportunities > 0))),
            "max_repeat_multiplier_used": float(repeat_multiplier.max()) if n_items else 0.0,
        }
        self._fitted = True
        return self

    def _group_rates(self, user_index: int) -> np.ndarray:
        a = self._fitted_prior_strength
        channel = self.channel_counts_.getrow(user_index).toarray().ravel() / (self.channel_opportunities_ + a)
        if self._fitted_slot_mode == "channel":
            return channel
        activity = self.hour_counts_.getrow(user_index).toarray().ravel()
        activity = activity / (activity + a)
        hour = (self.channel_hour_counts_.getrow(user_index).toarray().ravel()
                + a * np.repeat(channel, 24) * np.tile(activity, self.n_channels_)) / (
                    self.channel_hour_opportunities_ + a)
        if self._fitted_slot_mode == "hour":
            return hour
        activity = self.weekhour_counts_.getrow(user_index).toarray().ravel()
        activity = activity / (activity + a)
        parent = hour.reshape(self.n_channels_, 24)[:, np.arange(168) % 24].ravel()
        return (self.channel_weekhour_counts_.getrow(user_index).toarray().ravel()
                + a * parent * np.tile(activity, self.n_channels_)) / (
                    self.channel_weekhour_opportunities_ + a)

    def score(self, user_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fit ChannelScheduleModel before scoring")
        candidates = candidate_array(candidate_indices)
        scores = np.full(len(candidates), np.nan, dtype=np.float64)
        known_user = valid_user(user_index, self.n_users_)
        valid = (candidates >= 0) & (candidates < self.n_items_)
        query = {"candidate_count": len(candidates), "finite_scores": 0,
                 "repeat_caps_applied": 0, "positive_group_rates": 0, "max_group_rate": 0.0}
        self.diagnostics["last_query"] = query
        if not known_user or not self._active_users[user_index] or not np.any(valid):
            query["status"] = "unknown_user" if not known_user else (
                "cold_user" if not self._active_users[user_index] else "no_valid_candidates")
            return scores
        indices, inverse = np.unique(candidates[valid].astype(np.int64), return_inverse=True)
        rates = self._group_rates(int(user_index))
        base = np.asarray(self.forecast_group_items_[:, indices].T @ rates).ravel()
        personal = self.programme_counts_.getrow(user_index)[:, indices].toarray().ravel()
        seen = personal > 0
        history = self.historical_item_opportunities_[indices]
        a, blend = self._fitted_prior_strength, self._fitted_programme_blend
        values = base.copy()
        values[seen] = ((1 - blend * history[seen] / (history[seen] + a))
                        * base[seen] + blend * personal[seen]
                        * self.repeat_multiplier_[indices[seen]])
        values[(values <= 0) | (self.forecast_item_opportunities_[indices] <= 0)] = np.nan
        scores[valid] = values[inverse]
        query.update({
            "status": "scored", "unique_valid_candidates": len(indices),
            "finite_scores": int(np.isfinite(scores).sum()),
            "positive_group_rates": int(np.count_nonzero(rates > 0)),
            "max_group_rate": float(rates.max()) if rates.size else 0.0,
            "channel_slot_evidence_items": int(np.count_nonzero(base > 0)),
            "personal_programme_evidence_items": int(np.count_nonzero(seen)),
            "repeat_caps_applied": int(np.count_nonzero(seen & (
                self.forecast_item_opportunities_[indices] / (history + a)
                > self._fitted_max_repeat_multiplier))) if blend > 0 else 0,
        })
        return scores
