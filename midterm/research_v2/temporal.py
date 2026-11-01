"""Training-only repeat counts and a bounded schedule-opportunity reranker.

Let T be the training cutoff, h the half-life in days and L the optional
lookback. For event/airing end t, w(t) = 2**(-(T-t)/(86400*h)), or 1 without
decay, and w=0 when (T-t)/86400 > L. The unchanged shared helpers implement
these weights, including the inclusive lookback boundary. Both event and
historical-airing features additionally require history_start < end <= T.

R_ui is the sum of qualifying-event weights. H_i is the sum of the SAME
weights over unique historical schedule airings; F_i counts unique, fully
contained forecast airings supplied by TemporalContext. Forecast quarantine
flags are retained as metadata, not silently filtered (historical quarantine
is already excluded by the context contract).

For historical span D=min((T-history_start)/86400, L), omitting L if absent,
the effective historical duration E is the integral of w(age) over [0,D]:
E=D without decay, otherwise h/log(2) * (1-2**(-D/h)), in weighted days.
For forecast duration Q days, B_i=H_i*Q/E is in forecast-horizon airings.
The ratio is clip((F_i+s)/(B_i+s), 1/cap, cap), where s is a prior of
scheduled airings PER FORECAST HORIZON, not days or historical-window airings.
Thus unchanged broadcast intensity (H_i=lambda_i*E, F_i=lambda_i*Q) gives
ratio=1 for both rare and frequent programs. Scores are R_ui*ratio**power.

Positive power requires F_i>0; a boundary-only candidate remains in place as
NaN, never removed from the caller's catalog. Power zero deliberately bypasses
ALL schedule adjustment/availability masking and exactly reproduces repeat.
R_ui=0 is always NaN; neither expert supplies its own fallback. This is an
available-airing proxy, not measured personal exposure or a causal probability.
"""

from __future__ import annotations

from dataclasses import replace
import math
from numbers import Integral, Real
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from .contracts import (
    Airing,
    TemporalContext,
    candidate_array,
    temporal_matrix,
    temporal_weights,
    valid_user,
)


def _finite_number(name: str, value: Any, *, minimum: float = 0.0,
                   inclusive: bool = False, maximum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if (not math.isfinite(number) or number < minimum
            or (not inclusive and number == minimum)
            or (maximum is not None and number > maximum)):
        comparison = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be finite, {comparison} {minimum}"
                         + (f" and <= {maximum}" if maximum is not None else ""))
    return number


def _decay_parameters(half_life_days, lookback_days):
    half_life = None if half_life_days is None else _finite_number("half_life_days", half_life_days)
    lookback = None if lookback_days is None else _finite_number("lookback_days", lookback_days)
    return half_life, lookback


def _history_duration(context: TemporalContext, half_life: float | None,
                      lookback: float | None) -> tuple[float, float]:
    span = (context.train_end - context.history_start) / 86400.0
    duration = span if lookback is None else min(span, lookback)
    if half_life is None:
        return duration, duration
    # expm1 avoids cancellation for a long half-life and a short window.
    exponent = math.log(2.0) * (duration / half_life)
    if exponent == 0:
        effective = duration  # The valid positive ratio underflowed to zero.
    elif math.isinf(exponent):
        effective = half_life / math.log(2.0)
    else:
        # duration * (...) avoids overflowing h/log(2) for a valid huge h.
        effective = duration * (-math.expm1(-exponent) / exponent)
    return duration, effective


def _candidate_coverage(matrix: csr_matrix, candidates: np.ndarray,
                        available: np.ndarray | None = None) -> dict[str, Any]:
    unique = np.unique(candidates)
    eligible = unique if available is None else unique[available[unique]]
    subset = matrix[:, eligible]
    possible = matrix.shape[0] * len(unique)
    return {
        "unique_candidates": int(len(unique)),
        "eligible_candidates": int(len(eligible)),
        "supported_candidates": int(len(np.unique(subset.indices))),
        "supported_users": int(np.count_nonzero(np.diff(subset.indptr))),
        "supported_user_candidate_pairs": int(subset.nnz),
        "user_candidate_pair_coverage": float(subset.nnz / possible) if possible else 0.0,
    }


def _repeat_state(context: TemporalContext, half_life: float | None,
                  lookback: float | None) -> dict[str, Any]:
    if not isinstance(context, TemporalContext):
        raise TypeError("fit requires a training-only TemporalContext")
    in_history = context.event_ends > context.history_start
    feature_context = context
    if not np.all(in_history):
        # The shared factory can retain older behavioral events when its nominal
        # history_start is shortened. Restrict only our feature view, leaving the
        # original context, count CSR, helper, candidates, and labels untouched.
        fields = ("event_users", "event_items", "event_starts", "event_ends", "event_channels")
        feature_context = replace(context, **{name: getattr(context, name)[in_history] for name in fields})
    # This is a new CSR, never a replacement for data.interactions.
    matrix = temporal_matrix(feature_context, half_life, lookback)
    weights = temporal_weights(feature_context, half_life, lookback)
    ages = (context.train_end - context.event_ends).astype(np.float64) / 86400.0
    feature_ages = ages[in_history]
    duration, effective = _history_duration(context, half_life, lookback)
    diagnostics = {
        "model": "TemporalRepeatModel",
        "backend": "scipy.sparse.csr",
        "weight_dtype": str(matrix.dtype),
        "parameters": {"half_life_days": half_life, "lookback_days": lookback},
        "users": int(matrix.shape[0]),
        "items": int(matrix.shape[1]),
        "training_events": int(len(context.event_ends)),
        "events_in_history_window": int(np.count_nonzero(in_history)),
        "events_before_history_start_excluded": int(np.count_nonzero(~in_history)),
        "events_in_lookback": int(len(weights) if lookback is None else np.count_nonzero(feature_ages <= lookback)),
        "positive_weight_events": int(np.count_nonzero(weights > 0)),
        "temporal_user_item_pairs": int(matrix.nnz),
        "active_users": int(np.count_nonzero(np.diff(matrix.indptr))),
        "active_items": int(len(np.unique(matrix.indices))),
        "total_temporal_count": float(matrix.data.sum(dtype=np.float64)),
        "history_window": {
            "history_start": int(context.history_start),
            "train_end": int(context.train_end),
            "forecast_start": int(context.forecast_start),
            "forecast_end": int(context.forecast_end),
            "end_age_rule": "history_start < event/airing end <= train_end; age_days <= lookback_days",
            "history_span_days": (context.train_end - context.history_start) / 86400.0,
            "window_span_days": duration,
            "effective_weighted_history_days": effective,
            "observed_min_age_days": float(ages.min()) if len(ages) else None,
            "observed_max_age_days": float(ages.max()) if len(ages) else None,
        },
        "candidate_coverage": _candidate_coverage(matrix, context.candidate_indices),
        "availability_policy": "repeat evidence only; no forecast-availability mask",
    }
    return {
        "repeat_counts_": matrix,
        "user_ids_": tuple(context.data.user_ids),
        "item_ids_": context.data.item_ids.copy(),
        "half_life_days": half_life,
        "lookback_days": lookback,
        "diagnostics": diagnostics,
    }


def _unique_airings(airings: tuple[Airing, ...], item_count: int) -> tuple[Airing, ...]:
    seen = set()
    unique = []
    for airing in airings:
        if (isinstance(airing.item_index, (bool, np.bool_))
                or not isinstance(airing.item_index, Integral)
                or not 0 <= airing.item_index < item_count):
            raise ValueError("Schedule item index out of range")
        if not isinstance(airing.channel_id, str) or not airing.channel_id:
            raise ValueError("Schedule channels must be nonempty strings")
        if any(isinstance(t, (bool, np.bool_)) or not isinstance(t, Integral)
               for t in (airing.start, airing.end)):
            raise ValueError("Schedule timestamps must be integer source-clock seconds")
        if airing.start >= airing.end:
            raise ValueError("Schedule airings must have positive duration")
        # An airing is an interval for a program on a channel, not a metadata row.
        key = (airing.item_index, airing.channel_id, airing.start, airing.end)
        if key not in seen:
            seen.add(key)
            unique.append(airing)
    return tuple(unique)


def _summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(values.size),
        "min": float(values.min()) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "max": float(values.max()) if values.size else None,
    }


class TemporalRepeatModel:
    """Sum end-age-weighted qualifying events, retaining previously seen items."""

    def __init__(self, half_life_days: float | None = 28.0,
                 lookback_days: float | None = None):
        self.half_life_days, self.lookback_days = _decay_parameters(half_life_days, lookback_days)
        self.repeat_counts_: csr_matrix | None = None
        self.diagnostics: dict[str, Any] = {}

    def fit(self, context: TemporalContext) -> TemporalRepeatModel:
        half_life, lookback = _decay_parameters(self.half_life_days, self.lookback_days)
        state = _repeat_state(context, half_life, lookback)
        # Commit only after successful construction; refits never mix old/new state.
        self.__dict__.update(state)
        return self

    def score(self, user_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        if self.repeat_counts_ is None:
            raise RuntimeError("TemporalRepeatModel must be fitted before scoring")
        candidates = candidate_array(candidate_indices)
        scores = np.full(len(candidates), np.nan, dtype=np.float64)
        if not valid_user(user_index, self.repeat_counts_.shape[0]):
            return scores
        valid = (candidates >= 0) & (candidates < self.repeat_counts_.shape[1])
        if np.any(valid):
            indices = candidates[valid].astype(np.intp, copy=False)
            values = self.repeat_counts_[int(user_index), indices].toarray().ravel()
            values[values <= 0] = np.nan
            scores[valid] = values
        return scores


class ScheduleAdjustedRepeatModel(TemporalRepeatModel):
    """Temporal repeat times a shrunk, horizon-normalized availability ratio.

    ``smoothing_airings`` is measured in scheduled airings per forecast horizon.
    ``ratio_cap >= 1`` clips both tails, and ``0 <= schedule_power <= 1`` bounds
    adjustment between cap**(-power) and cap**power. With positive power, no
    fully contained forecast occurrence is unsupported, not prior-only evidence.
    Zero power is an exact ablation of every schedule-dependent score change.
    """

    def __init__(self, half_life_days: float | None = 28.0,
                 lookback_days: float | None = None, schedule_power: float = 0.5,
                 smoothing_airings: float = 10.0, ratio_cap: float = 10.0):
        super().__init__(half_life_days, lookback_days)
        self.schedule_power = _finite_number("schedule_power", schedule_power, inclusive=True, maximum=1.0)
        self.smoothing_airings = _finite_number("smoothing_airings", smoothing_airings)
        self.ratio_cap = _finite_number("ratio_cap", ratio_cap, minimum=1.0, inclusive=True)

    def fit(self, context: TemporalContext) -> ScheduleAdjustedRepeatModel:
        half_life, lookback = _decay_parameters(self.half_life_days, self.lookback_days)
        power = _finite_number("schedule_power", self.schedule_power, inclusive=True, maximum=1.0)
        smoothing = _finite_number("smoothing_airings", self.smoothing_airings)
        cap = _finite_number("ratio_cap", self.ratio_cap, minimum=1.0, inclusive=True)
        state = _repeat_state(context, half_life, lookback)
        item_count = len(context.data.item_ids)
        historical_all = _unique_airings(context.historical_schedules, item_count)
        historical = tuple(a for a in historical_all if a.end > context.history_start)
        forecast = _unique_airings(context.forecast_schedules, item_count)
        history_ends = np.fromiter((a.end for a in historical), dtype=np.int64, count=len(historical))
        # The unchanged shared helper depends only on these two numeric fields.
        # A small view avoids copying/synthesizing full behavioral contexts.
        history_weights = temporal_weights(
            SimpleNamespace(train_end=context.train_end, event_ends=history_ends),
            half_life, lookback,
        )
        history_items = np.fromiter((a.item_index for a in historical), dtype=np.intp, count=len(historical))
        forecast_items = np.fromiter((a.item_index for a in forecast), dtype=np.intp, count=len(forecast))
        history_counts = np.bincount(history_items, weights=history_weights, minlength=item_count).astype(np.float64)
        forecast_counts = np.bincount(forecast_items, minlength=item_count).astype(np.float64)
        duration, effective = _history_duration(context, half_life, lookback)
        forecast_days = (context.forecast_end - context.forecast_start) / 86400.0
        log_scale = math.log(forecast_days) - math.log(effective)
        # Log-domain addition protects valid, extremely short half-lives/priors
        # from overflow in H*Q/E. The final ratios and factors remain finite.
        with np.errstate(divide="ignore"):
            log_ratio = np.logaddexp(np.log(forecast_counts), math.log(smoothing)) - np.logaddexp(
                np.log(history_counts) + log_scale, math.log(smoothing)
            )
        log_cap = math.log(cap)
        ratios = np.clip(np.exp(np.clip(log_ratio, -log_cap, log_cap)), 1.0 / cap, cap)
        factors = np.ones(item_count, dtype=np.float64) if power == 0 else ratios ** power
        matrix = state["repeat_counts_"]
        # A finite but astronomically large user-supplied cap can still make
        # R*factor unrepresentable. Reject that configuration before refit commit
        # rather than emit infinity or silently alter the scoring formula.
        limits = np.finfo(np.float64).max / np.maximum(factors[matrix.indices], 1.0)
        if np.any(matrix.data > limits):
            raise ValueError("Schedule parameters would overflow float64 repeat scores")
        available = forecast_counts > 0
        diagnostics = state["diagnostics"]
        diagnostics.update({
            "model": "ScheduleAdjustedRepeatModel",
            "parameters": {"half_life_days": half_life, "lookback_days": lookback,
                           "schedule_power": power, "smoothing_airings": smoothing, "ratio_cap": cap},
            "availability_policy": (
                "power=0: exact repeat, including boundary-only candidates" if power == 0 else
                "power>0: F=0 is NaN; fully contained forecast metadata, including quarantine flags"
            ),
            "schedule_counts": {
                "historical_rows": int(len(context.historical_schedules)),
                "unique_historical_airings": int(len(historical_all)),
                "historical_airings_in_history_window": int(len(historical)),
                "historical_airings_before_history_start_excluded": int(len(historical_all) - len(historical)),
                "positive_weight_historical_airings": int(np.count_nonzero(history_weights > 0)),
                "weighted_historical_airings": float(history_counts.sum()),
                "historical_programs": int(np.count_nonzero(history_counts)),
                "forecast_rows": int(len(context.forecast_schedules)),
                "unique_forecast_airings": int(len(forecast)),
                "forecast_quarantined_rows": int(sum(a.quarantined for a in forecast)),
                "forecast_programs": int(np.count_nonzero(available)),
                "forecast_programs_without_weighted_history": int(np.count_nonzero(available & (history_counts == 0))),
            },
            "availability_ratios": {
                "normalization": "B=H*forecast_days/effective_weighted_history_days; (F+s)/(B+s)",
                "smoothing_units": "scheduled airings per forecast horizon",
                "history_window_days": duration,
                "effective_weighted_history_days": effective,
                "forecast_days": forecast_days,
                "history_to_forecast_log_scale": log_scale,
                "raw_log_ratio_available": _summary(log_ratio[available]),
                "clipped_ratio_available": _summary(ratios[available]),
                "factor_available": _summary(factors[available]),
                "lower_clipped_programs": int(np.count_nonzero(available & (log_ratio < -log_cap))),
                "upper_clipped_programs": int(np.count_nonzero(available & (log_ratio > log_cap))),
            },
            "candidate_coverage": _candidate_coverage(
                state["repeat_counts_"], context.candidate_indices, None if power == 0 else available,
            ),
        })
        state.update({
            "schedule_power": power,
            "smoothing_airings": smoothing,
            "ratio_cap": cap,
            "_fitted_schedule_power": power,
            "historical_counts_": history_counts,
            "forecast_counts_": forecast_counts,
            "availability_ratios_": ratios,
            "schedule_factors_": factors,
        })
        self.__dict__.update(state)
        return self

    def score(self, user_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        scores = super().score(user_index, candidate_indices)
        if self._fitted_schedule_power == 0:
            return scores  # Bit-for-bit exact, including NaN locations.
        candidates = candidate_array(candidate_indices)
        supported = np.isfinite(scores)
        if np.any(supported):
            indices = candidates[supported].astype(np.intp, copy=False)
            scores[supported] *= self.schedule_factors_[indices]
            positions = np.flatnonzero(supported)
            scores[positions[self.forecast_counts_[indices] == 0]] = np.nan
        return scores
