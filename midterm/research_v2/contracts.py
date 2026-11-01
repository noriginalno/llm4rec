"""Training-only temporal context shared by independently researched models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from sweettv.contracts import TrainingData
from sweettv.data import PreparedData, _prepare_schedules, build_fold


@dataclass(frozen=True, slots=True)
class Airing:
    item_index: int
    channel_id: str
    start: int
    end: int
    quarantined: bool = False


@dataclass(frozen=True)
class TemporalContext:
    data: TrainingData
    candidate_indices: np.ndarray
    history_start: int
    train_end: int
    forecast_start: int
    forecast_end: int
    event_users: np.ndarray
    event_items: np.ndarray
    event_starts: np.ndarray
    event_ends: np.ndarray
    event_channels: np.ndarray
    historical_schedules: tuple[Airing, ...]
    forecast_schedules: tuple[Airing, ...]

    def __post_init__(self) -> None:
        if not self.history_start < self.train_end <= self.forecast_start < self.forecast_end:
            raise ValueError("Invalid context time boundaries")
        for name in ("candidate_indices", "event_users", "event_items", "event_starts", "event_ends"):
            supplied = np.asarray(getattr(self, name))
            if supplied.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if supplied.size and supplied.dtype.kind not in "iu":
                raise ValueError(f"{name} must contain integer coordinates/indices")
            if supplied.size and supplied.dtype.kind == "u" and np.any(supplied > np.iinfo(np.int64).max):
                raise ValueError(f"{name} exceeds the signed index/time coordinate range")
            array = np.array(supplied, dtype=np.int64, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        channels = np.array(self.event_channels, dtype=str, copy=True)
        if channels.ndim != 1:
            raise ValueError("event_channels must be one-dimensional")
        channels.setflags(write=False)
        object.__setattr__(self, "event_channels", channels)
        n = len(self.event_users)
        if any(len(getattr(self, key)) != n for key in ("event_items", "event_starts", "event_ends", "event_channels")):
            raise ValueError("Training-event arrays must align")
        if np.any(self.event_ends > self.train_end) or np.any(self.event_starts >= self.event_ends):
            raise ValueError("Future or invalid behavior in model context")
        if np.any(self.event_users < 0) or np.any(self.event_users >= len(self.data.user_ids)):
            raise ValueError("Event user index out of range")
        if np.any(self.event_items < 0) or np.any(self.event_items >= len(self.data.item_ids)):
            raise ValueError("Event item index out of range")
        if np.any(self.candidate_indices < 0) or np.any(self.candidate_indices >= len(self.data.item_ids)):
            raise ValueError("Candidate index out of range")
        if any(a.end > self.train_end or a.quarantined for a in self.historical_schedules):
            raise ValueError("Historical opportunities must be resolved and end before cutoff")
        if any(not (self.forecast_start <= a.start < a.end <= self.forecast_end)
               for a in self.forecast_schedules):
            raise ValueError("Forecast opportunities must be fully contained")


def temporal_weights(context: TemporalContext, half_life_days: float | None = None,
                     lookback_days: float | None = None) -> np.ndarray:
    ages = (context.train_end - context.event_ends).astype(np.float64) / 86400.0
    weights = np.ones(len(ages), dtype=np.float64)
    if half_life_days is not None:
        if isinstance(half_life_days, (bool, np.bool_)) or not np.isfinite(half_life_days) or half_life_days <= 0:
            raise ValueError("Half-life must be finite and positive")
        weights = np.exp2(-ages / half_life_days)
    if lookback_days is not None:
        if isinstance(lookback_days, (bool, np.bool_)) or not np.isfinite(lookback_days) or lookback_days <= 0:
            raise ValueError("Lookback must be finite and positive")
        weights[ages > lookback_days] = 0.0
    return weights


def temporal_matrix(context: TemporalContext, half_life_days: float | None = None,
                    lookback_days: float | None = None) -> csr_matrix:
    weights = temporal_weights(context, half_life_days, lookback_days)
    matrix = coo_matrix((weights, (context.event_users, context.event_items)),
                        shape=context.data.interactions.shape).tocsr()
    matrix.eliminate_zeros()
    return matrix


def candidate_array(values) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError("Candidate indices must be a one-dimensional integer array")
    return array


def valid_user(value, n: int) -> bool:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("User must be an integer matrix index")
    return 0 <= value < n


def make_context(prepared: PreparedData, train_end: int, forecast_start: int,
                 forecast_end: int, history_start: int = 1583712000):
    """Keep labels outside the model context; immutable forecast metadata is allowed."""
    data, candidates, labels, audit = build_fold(prepared, train_end, forecast_start, forecast_end)
    users = {u: i for i, u in enumerate(data.user_ids)}
    items = {int(v): i for i, v in enumerate(data.item_ids)}
    canonical_schedules, _ = _prepare_schedules(prepared.schedules)
    schedules = {row["airing_id"]: row for row in canonical_schedules}
    selected, seen_events = [], {}
    for event in prepared.events:
        if event["end"] > train_end:
            continue
        identity = (event["user_id"], event["airing_id"])
        signature = (event["tv_show_id"], event["start"], event["end"])
        if identity in seen_events:
            if seen_events[identity] != signature:
                raise ValueError("Conflicting duplicate training event")
            continue
        seen_events[identity] = signature
        selected.append(event)
    context = TemporalContext(
        data=data, candidate_indices=candidates, history_start=history_start,
        train_end=train_end, forecast_start=forecast_start, forecast_end=forecast_end,
        event_users=np.array([users[e["user_id"]] for e in selected], dtype=np.int64),
        event_items=np.array([items[e["tv_show_id"]] for e in selected], dtype=np.int64),
        event_starts=np.array([e["start"] for e in selected], dtype=np.int64),
        event_ends=np.array([e["end"] for e in selected], dtype=np.int64),
        event_channels=np.array([schedules[e["airing_id"]]["channel_id"] for e in selected], dtype=str),
        historical_schedules=tuple(Airing(items[r["tv_show_id"]], r["channel_id"], r["start"], r["end"])
            for r in canonical_schedules if r["tv_show_id"] in items
            and history_start < r["end"] <= train_end and not r.get("quarantined", False)),
        forecast_schedules=tuple(Airing(items[r["tv_show_id"]], r["channel_id"], r["start"], r["end"], bool(r.get("quarantined", False)))
            for r in canonical_schedules if r["tv_show_id"] in items
            and forecast_start <= r["start"] < r["end"] <= forecast_end),
    )
    observed = temporal_matrix(context)
    if observed.shape != data.interactions.shape or (observed != data.interactions).nnz:
        raise ValueError("Context event counts disagree with the unchanged production fold")
    return context, labels, audit
