"""Raw, union-covered channel/clock occupancy integrated over future schedules.

This is a coverage/frequency *proxy*, not a probability of >=80% completion or
personal exposure. Source timestamps remain naive wall-clock seconds on the
project's common epoch anchor. No timezone conversion is performed.

Build the source-bound Parquet once with ``python -m research_v2.raw_habits``.
Fitting reads that cache, selects users and clips behavior BEFORE estimation.
Only supplied forecast airing metadata, never future viewing, enters scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from numbers import Integral, Real
from pathlib import Path
import re
import time
import uuid

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix, csr_matrix

from research_v2.contracts import TemporalContext, candidate_array, valid_user


DAY = 86400
DEFAULT_CACHE = ".cache/research-v2/raw-watch-union.parquet"
DEFAULT_SOURCE = "data/raw/dataset11-30.csv"
DEFAULT_MANIFEST = "data/raw/source-manifest.json"
CLOCK = "naive source-wall-clock epoch anchor; Monday=0; no physical UTC inference"
CACHE_FORMAT = "research-v2-raw-watch-union-v1"
CACHE_SCHEMA = pa.schema([
    ("user_id", pa.string()), ("channel_id", pa.string()),
    ("start", pa.int64()), ("end", pa.int64()),
])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _guard(path: Path) -> dict:
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _sql_string(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _connection(temp_dir: Path):
    temp_dir.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(config={
        "threads": 4, "memory_limit": "4GB", "temp_directory": str(temp_dir.resolve()),
        "enable_external_access": True,
    })


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _manifest_entry(manifest_path: Path, source_path: Path) -> tuple[dict, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get("files", {}).get(source_path.name, {})
    if (not isinstance(entry.get("bytes"), int)
            or not isinstance(entry.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
        raise ValueError("Source must have a bytes/SHA-256 entry in the existing manifest")
    return entry, _sha256(manifest_path)


def _validated_cache(cache_path: Path) -> dict:
    """Verify typed cache bytes and original-source stat/manifest guards.

    The source SHA is verified in full at build time. Subsequent fits trust the
    recorded size/mtime guard, not a fresh 396MB source scan; this is not defense
    against deliberately replacing a file while preserving both stat fields.
    """
    meta = json.loads(_meta_path(cache_path).read_text(encoding="utf-8"))
    if meta.get("format") != CACHE_FORMAT or meta.get("clock") != CLOCK:
        raise ValueError("Unsupported raw-watch cache format/clock")
    source = meta["source"]
    source_path, manifest_path = Path(source["path"]), Path(source["manifest_path"])
    if _guard(source_path) != source["guard"]:
        raise ValueError("Raw source size/mtime guard changed; rebuild to a NEW cache path")
    entry, manifest_sha = _manifest_entry(manifest_path, source_path)
    if (entry["sha256"] != source["sha256"] or entry["bytes"] != source["guard"]["bytes"]
            or manifest_sha != source["manifest_sha256"]):
        raise ValueError("Source manifest binding changed")
    before = _guard(cache_path)
    if before != meta["cache"]["guard"] or _sha256(cache_path) != meta["cache"]["sha256"]:
        raise ValueError("Raw-watch cache size/mtime/SHA-256 changed")
    if _guard(cache_path) != before:
        raise ValueError("Raw-watch cache changed during validation")
    if not pq.read_schema(cache_path).equals(CACHE_SCHEMA, check_metadata=False):
        raise ValueError("Raw-watch cache must use string IDs and int64 start/end")
    return meta


def build_cache(
    source_path=DEFAULT_SOURCE,
    cache_path=DEFAULT_CACHE,
    manifest_path=DEFAULT_MANIFEST,
    *,
    temp_dir=None,
) -> dict:
    """Offline, all-user interval union; never modify raw files or an old cache.

    DuckDB reads CSV with ``all_varchar=True`` (opaque user/channel strings),
    four threads and a 4GB memory limit. ISO-naive second-resolution endpoints
    are authoritative; ``duraton`` is audited only. Adjacent, duplicate, nested
    and overlapping device/fragment intervals are unioned per user/channel.
    Existing cache+sidecar pairs are verified and reused, not overwritten.
    """
    started = time.perf_counter()
    source_path, cache_path = Path(source_path).resolve(), Path(cache_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    sidecar = _meta_path(cache_path)
    if cache_path in {source_path, manifest_path} or sidecar in {source_path, manifest_path}:
        raise ValueError("Cache output must not overwrite a source or manifest")
    if cache_path.exists() or sidecar.exists():
        meta = _validated_cache(cache_path)
        if (meta["source"]["path"] != str(source_path)
                or meta["source"]["manifest_path"] != str(manifest_path)):
            raise ValueError("Existing cache belongs to a different source/manifest")
        return {**meta, "reused": True}
    entry, manifest_sha = _manifest_entry(manifest_path, source_path)
    source_guard = _guard(source_path)
    if source_guard["bytes"] != entry["bytes"]:
        raise ValueError("Raw source size differs from existing manifest")
    hash_start = time.perf_counter()
    if _sha256(source_path) != entry["sha256"] or _guard(source_path) != source_guard:
        raise ValueError("Raw source SHA-256 differs from manifest or source changed")
    timings = {"source_sha256": time.perf_counter() - hash_start}
    temp_dir = Path(temp_dir) if temp_dir is not None else cache_path.parent / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = temp_dir / f"raw-union-{token}.parquet"
    temporary_meta = temp_dir / f"raw-union-{token}.meta.json"
    published_cache = published_meta = False
    try:
        with _connection(temp_dir) as con:
            parse_start = time.perf_counter()
            con.execute("""
                CREATE TEMP TABLE raw_intervals AS
                SELECT user_id, vsetv_id AS channel_id,
                       CAST(epoch(try_strptime(start_time,
                           ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'])) AS BIGINT) AS s,
                       CAST(epoch(try_strptime(stop_time,
                           ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'])) AS BIGINT) AS e,
                       try_cast(duraton AS DOUBLE) AS reported_duration,
                       (regexp_full_match(start_time, '[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}')
                        AND regexp_full_match(stop_time, '[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}')) AS naive_format
                FROM read_csv(?, header=true, all_varchar=true, delim=',', strict_mode=true)
            """, [str(source_path)])
            audit = con.execute("""
                SELECT count(*) AS raw_rows,
                       count(*) FILTER (WHERE user_id IS NULL OR trim(user_id) = ''
                           OR channel_id IS NULL OR trim(channel_id) = ''
                           OR s IS NULL OR e IS NULL OR s >= e OR NOT coalesce(naive_format, false)) AS invalid_rows,
                       count(*) FILTER (WHERE reported_duration IS NULL OR NOT isfinite(reported_duration)) AS duration_invalid_rows,
                       count(*) FILTER (WHERE isfinite(reported_duration) AND reported_duration != e-s) AS duration_mismatch_rows,
                       coalesce(sum(e-s), 0)::BIGINT AS ununioned_seconds
                FROM raw_intervals
            """).to_arrow_table().to_pylist()[0]
            if audit["invalid_rows"]:
                raise ValueError(f"Invalid raw IDs/naive timestamps/intervals: {audit['invalid_rows']}")
            timings["read_parse_audit"] = time.perf_counter() - parse_start
            union_start = time.perf_counter()
            # Running MAX, not LAG(end): a long enclosing interval must keep
            # later nested fragments in the same connected component.
            con.execute(f"""
                COPY (
                    WITH previous AS (
                        SELECT *, max(e) OVER (PARTITION BY user_id, channel_id
                            ORDER BY s, e ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_end
                        FROM raw_intervals
                    ), islands AS (
                        SELECT *, sum(CASE WHEN prior_end IS NULL OR s > prior_end THEN 1 ELSE 0 END)
                            OVER (PARTITION BY user_id, channel_id ORDER BY s, e ROWS UNBOUNDED PRECEDING) AS island
                        FROM previous
                    )
                    SELECT user_id::VARCHAR AS user_id, channel_id::VARCHAR AS channel_id,
                           min(s)::BIGINT AS "start", max(e)::BIGINT AS "end"
                    FROM islands GROUP BY user_id, channel_id, island
                    ORDER BY user_id, channel_id, "start"
                ) TO {_sql_string(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
            audit.update(con.execute("""
                SELECT count(*) AS union_rows, count(DISTINCT user_id) AS users,
                       count(DISTINCT channel_id) AS channels,
                       min("start") AS minimum_start, max("end") AS maximum_end,
                       coalesce(sum("end"-"start"), 0)::BIGINT AS union_seconds
                FROM read_parquet(?)
            """, [str(temporary)]).to_arrow_table().to_pylist()[0])
            timings["union_write_audit"] = time.perf_counter() - union_start
        if _guard(source_path) != source_guard or _sha256(manifest_path) != manifest_sha:
            raise ValueError("Source or manifest changed during cache build")
        hash_start = time.perf_counter()
        cache_sha = _sha256(temporary)
        timings["cache_sha256"] = time.perf_counter() - hash_start
        timings["total"] = time.perf_counter() - started
        meta = {
            "format": CACHE_FORMAT, "clock": CLOCK,
            "source": {"path": str(source_path), "guard": source_guard,
                       "sha256": entry["sha256"], "sha256_verified_at_build": True,
                       "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha},
            "cache": {"path": str(cache_path), "guard": _guard(temporary), "sha256": cache_sha,
                      "schema": {"user_id": "string", "channel_id": "string", "start": "int64", "end": "int64"}},
            "audit": audit, "timings_seconds": timings,
            "construction": {"users": "all raw users; training fits select context users only",
                             "union": "half-open per-user/channel union; adjacency merged; endpoints authoritative",
                             "duration": "duraton ignored except audit", "duckdb": duckdb.__version__,
                             "threads": 4, "memory_limit": "4GB", "temp_dir": str(temp_dir.resolve()),
                             "implementation_sha256": _sha256(Path(__file__))},
        }
        temporary_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Exclusive links never silently replace a concurrently created output.
        os.link(temporary, cache_path)
        published_cache = True
        os.link(temporary_meta, sidecar)
        published_meta = True
        return {**meta, "reused": False}
    except BaseException:
        if published_cache:
            cache_path.unlink()
        if published_meta:
            sidecar.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)
        temporary_meta.unlink(missing_ok=True)


def _positive(value, name: str, *, optional=False) -> None:
    if optional and value is None:
        return
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")


def _pieces(start: int, end: int, width: int):
    """Exact half-open overlap lengths with absolute source-clock bins."""
    for absolute_bin in range(start // width, (end - 1) // width + 1):
        yield absolute_bin, min(end, (absolute_bin + 1) * width) - max(start, absolute_bin * width)


class RawScheduleModel:
    """Sparse raw-watch occupancy -> duration-normalized forecast airing sums.

    ``half_life_days=None`` disables decay; lookback is floored to whole seconds.
    ``slot_minutes`` must divide a source-clock day; ``slot_mode`` is ``week``
    (Monday-first) or ``day``. A full calendar-bin midpoint supplies the SAME
    recency weight to observed and available seconds, including partial bins.
    Availability means only seconds in the observation window, not known user
    activity or channel exposure. No smoothing, popularity or watched exclusion.

    With ``coverage_power=1``, integrate linear occupancy exactly over each
    airing, divide by its FULL duration, then sum by program. Other positive
    powers transform each airing's mean first, never the aggregate sum. Scoring
    densifies at most one user's ``score_block_size`` output cells at a time.
    Parameters changed after fit take effect only on the next fit.
    """

    def __init__(
        self,
        raw_cache_path=DEFAULT_CACHE,
        half_life_days: float | None = 28.0,
        lookback_days: float | None = None,
        slot_minutes: int = 30,
        coverage_power: float = 1.0,
        *,
        slot_mode: str = "week",
        score_block_size: int = 4096,
        quarantined_future_policy: str = "exclude",
    ):
        self.raw_cache_path = raw_cache_path
        self.half_life_days = half_life_days
        self.lookback_days = lookback_days
        self.slot_minutes = slot_minutes
        self.coverage_power = coverage_power
        self.slot_mode = slot_mode
        self.score_block_size = score_block_size
        self.quarantined_future_policy = quarantined_future_policy
        self.diagnostics: dict = {}
        self._fitted = False
        self._validate_parameters()

    def _validate_parameters(self):
        _positive(self.half_life_days, "half_life_days", optional=True)
        _positive(self.lookback_days, "lookback_days", optional=True)
        _positive(self.coverage_power, "coverage_power")
        if (isinstance(self.slot_minutes, (bool, np.bool_)) or not isinstance(self.slot_minutes, Integral)
                or self.slot_minutes <= 0 or 1440 % self.slot_minutes):
            raise ValueError("slot_minutes must be a positive integer divisor of 1440")
        if (isinstance(self.score_block_size, (bool, np.bool_))
                or not isinstance(self.score_block_size, Integral) or self.score_block_size <= 0):
            raise ValueError("score_block_size must be a positive integer")
        if self.slot_mode not in {"week", "day"}:
            raise ValueError("slot_mode must be week or day")
        if self.quarantined_future_policy not in {"exclude", "include"}:
            raise ValueError("quarantined_future_policy must be exclude or include")

    @staticmethod
    def _forecast(context: TemporalContext, n_items: int):
        unique, duplicates = {}, 0
        for airing in context.forecast_schedules:
            if (isinstance(airing.item_index, (bool, np.bool_))
                    or not isinstance(airing.item_index, Integral) or not 0 <= airing.item_index < n_items):
                raise ValueError("Airing item index out of range")
            if not isinstance(airing.channel_id, str) or not airing.channel_id:
                raise ValueError("Airing channel must be a nonempty string")
            if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
                    for v in (airing.start, airing.end))
                    or not context.forecast_start <= airing.start < airing.end <= context.forecast_end):
                raise ValueError("Airing must be fully contained in the forecast horizon")
            key = (int(airing.item_index), airing.channel_id, int(airing.start), int(airing.end))
            duplicates += int(key in unique)
            unique[key] = unique.get(key, False) or bool(airing.quarantined)
        return unique, duplicates

    def fit(self, context: TemporalContext) -> RawScheduleModel:
        self._fitted = False
        self.diagnostics = {}
        self._validate_parameters()
        started = time.perf_counter()
        cache_path = Path(self.raw_cache_path).resolve()
        meta = _validated_cache(cache_path)
        width = int(self.slot_minutes) * 60
        slots_per_day = DAY // width
        n_slots = slots_per_day * (7 if self.slot_mode == "week" else 1)
        offset = 3 * slots_per_day  # epoch's Thursday -> Monday-first arithmetic
        lower, cutoff = int(context.history_start), int(context.train_end)
        if self.lookback_days is not None:
            # Avoid converting opaque IDs or timestamps through floating point.
            span = float(self.lookback_days) * DAY
            if span < cutoff - lower:
                lower = cutoff - math.floor(span)
        n_users, n_items = context.data.interactions.shape
        forecast_all, duplicates = self._forecast(context, n_items)
        forecast = [key for key, flag in forecast_all.items()
                    if not flag or self.quarantined_future_policy == "include"]

        absolute_bins = np.arange(lower // width, (cutoff - 1) // width + 1, dtype=np.int64)
        observed = np.maximum(0, np.minimum(cutoff, (absolute_bins + 1) * width)
                              - np.maximum(lower, absolute_bins * width)).astype(np.float64)
        slots = (absolute_bins + offset) % n_slots
        weights = np.ones(len(absolute_bins), dtype=np.float64)
        if self.half_life_days is not None:
            ages = np.maximum(0.0, (cutoff - (absolute_bins * width + width / 2)) / DAY)
            with np.errstate(over="ignore", under="ignore", divide="ignore"):
                weights = np.exp2(-ages / self.half_life_days)
        denominator = np.bincount(slots, weights=observed * weights, minlength=n_slots)
        user_ids = tuple(context.data.user_ids)
        half_life = None if self.half_life_days is None else float(self.half_life_days)
        lookback = None if self.lookback_days is None else float(self.lookback_days)
        user_table = pa.table({"user_id": pa.array(user_ids, type=pa.string()),
                               "user_index": pa.array(np.arange(n_users), type=pa.int64())})
        bin_table = pa.table({"absolute_bin": absolute_bins, "slot": slots, "weight": weights})
        with _connection(cache_path.parent / "tmp") as con:
            con.register("training_users", user_table)
            con.register("clock_bins", bin_table)
            # Temporal filtering and clipping precede channel selection, slot
            # expansion, recency weighting, aggregation and all feature fitting.
            con.execute("""
                CREATE TEMP TABLE clipped AS
                SELECT u.user_index, r.channel_id,
                       greatest(r."start", ?) AS s, least(r."end", ?) AS e
                FROM read_parquet(?) r JOIN training_users u USING (user_id)
                WHERE r."start" < ? AND r."end" > ?
            """, [lower, cutoff, str(cache_path), cutoff, lower])
            behavior = con.execute("""
                SELECT count(*) AS selected_union_intervals,
                       coalesce(sum(e-s), 0)::BIGINT AS clipped_union_seconds,
                       min(s) AS minimum_behavior_start, max(e) AS maximum_behavior_end
                FROM clipped WHERE s < e
            """).to_arrow_table().to_pylist()[0]
            channel_ids = sorted({row[0] for row in con.execute("SELECT DISTINCT channel_id FROM clipped WHERE s < e").fetchall()}
                                 | {key[1] for key in forecast})
            channel_index = {channel: i for i, channel in enumerate(channel_ids)}
            con.register("training_channels", pa.table({
                "channel_id": pa.array(channel_ids, type=pa.string()),
                "channel_index": pa.array(np.arange(len(channel_ids)), type=pa.int64()),
            }))
            grouped = con.execute(f"""
                SELECT user_index, c.channel_index * {n_slots} + b.slot AS feature,
                       sum((least(e, (b.absolute_bin+1)*{width})
                            - greatest(s, b.absolute_bin*{width})) * b.weight) AS seconds
                FROM clipped
                JOIN training_channels c USING (channel_id)
                CROSS JOIN LATERAL range(CAST(floor(s / {width}) AS BIGINT),
                    CAST(floor((e-1) / {width}) AS BIGINT)+1) piece(absolute_bin)
                JOIN clock_bins b USING (absolute_bin)
                WHERE s < e AND b.weight > 0
                GROUP BY user_index, feature
            """).to_arrow_table()
        if _guard(cache_path) != meta["cache"]["guard"] or _guard(Path(meta["source"]["path"])) != meta["source"]["guard"]:
            raise ValueError("Raw source/cache changed during training extraction")
        users = grouped["user_index"].to_numpy()
        features = grouped["feature"].to_numpy()
        seconds = grouped["seconds"].to_numpy()
        retained = seconds > 0
        users, features, seconds = users[retained], features[retained], seconds[retained]
        values = seconds / denominator[features % n_slots]
        if (not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1 + 1e-10)):
            raise ValueError("Union-covered occupancy must be in [0, 1]")
        occupancy = coo_matrix((np.minimum(values, 1), (users, features)),
                               shape=(n_users, len(channel_ids) * n_slots), dtype=np.float64).tocsr()
        occupancy.eliminate_zeros()
        occupancy.sort_indices()
        extraction_seconds = time.perf_counter() - started

        rows, columns, fractions = [], [], []
        linear = self.coverage_power == 1
        airing_items = np.array([key[0] for key in forecast], dtype=np.int64)
        for airing_index, (item, channel, start, end) in enumerate(forecast):
            base = channel_index[channel] * n_slots
            for absolute_bin, overlap in _pieces(start, end, width):
                rows.append(base + (absolute_bin + offset) % n_slots)
                columns.append(item if linear else airing_index)
                fractions.append(overlap / (end - start))
        forecast_features = coo_matrix(
            (fractions, (rows, columns)),
            shape=(len(channel_ids) * n_slots, n_items if linear else len(forecast)), dtype=np.float64,
        ).tocsc()
        forecast_features.sum_duplicates()
        forecast_features.sort_indices()
        airing_counts = np.bincount(airing_items, minlength=n_items)
        self.n_users_, self.n_items_, self.n_slots_ = n_users, n_items, n_slots
        self.user_ids_, self.channel_ids_ = user_ids, tuple(channel_ids)
        self.item_ids_ = np.array(context.data.item_ids, copy=True)
        self.item_ids_.setflags(write=False)
        self.occupancy_, self.observable_seconds_ = occupancy, denominator
        self.forecast_features_ = forecast_features
        self.forecast_airing_items_ = airing_items
        self.forecast_item_airings_ = airing_counts
        self._airing_order = np.argsort(airing_items, kind="stable")
        self._airing_indptr = np.r_[0, np.cumsum(airing_counts)]
        self._active_users = np.diff(occupancy.indptr) > 0
        self._fitted_power = float(self.coverage_power)
        self._fitted_block_size = int(self.score_block_size)
        fingerprint = hashlib.sha256(json.dumps({
            "users": user_ids, "channels": channel_ids, "start": lower, "cutoff": cutoff,
            "slot_minutes": int(self.slot_minutes), "slot_mode": self.slot_mode,
            "half_life_days": half_life,
        }, sort_keys=True).encode())
        for array in (occupancy.indptr.astype("<i8"), occupancy.indices.astype("<i8"), occupancy.data.astype("<f8")):
            fingerprint.update(array.tobytes())
        self.diagnostics = {
            "model": "RawScheduleModel", "backend": "DuckDB 4 threads / 4GB + SciPy sparse CPU",
            "parameters": {"raw_cache_path": str(cache_path), "half_life_days": half_life,
                           "lookback_days": lookback, "slot_minutes": int(self.slot_minutes),
                           "slot_mode": self.slot_mode, "coverage_power": float(self.coverage_power),
                           "score_block_size": int(self.score_block_size),
                           "quarantined_future_policy": self.quarantined_future_policy},
            "clock": CLOCK, "users": n_users, "items": n_items, "channels": len(channel_ids),
            "active_users": int(self._active_users.sum()),
            "occupancy_shape": list(occupancy.shape), "occupancy_nnz": int(occupancy.nnz),
            "minimum_nonzero_occupancy": float(occupancy.data.min()) if occupancy.nnz else None,
            "maximum_occupancy": float(occupancy.data.max()) if occupancy.nnz else None,
            "training_feature_sha256": fingerprint.hexdigest(),
            "source": {**meta["source"], "guard_checked_at_fit": True,
                       "cache_sha256": meta["cache"]["sha256"], "cache_path": str(cache_path)},
            "history_window": {"history_start": int(context.history_start), "effective_start": lower,
                               "train_end": cutoff, "source_cutoff_exclusive": cutoff,
                               "forecast_start": int(context.forecast_start), "forecast_end": int(context.forecast_end),
                               "policy": "clip raw union to [effective_start, train_end) before estimation"},
            **behavior,
            "behavior_source": "all raw partial/full watch intervals; no completion labels/counts used",
            "recency": "same full calendar-bin midpoint weights for numerator and denominator; age floored at zero",
            "observable_denominator": "weighted seconds in history window, NOT known active-user availability/exposure",
            "score_interpretation": "sum of duration-normalized airing mean-coverage powers; NOT calibrated completion probability",
            "forecast_airings_input": len(context.forecast_schedules), "forecast_airings_retained": len(forecast),
            "forecast_duplicates_ignored": duplicates,
            "quarantined_forecast_airings": sum(forecast_all.values()),
            "forecast_feature_shape": list(forecast_features.shape), "forecast_feature_nnz": int(forecast_features.nnz),
            "fit_seconds": {"validate_and_extract": extraction_seconds, "total": time.perf_counter() - started},
        }
        self._fitted = True
        return self

    def score(self, user_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Fit RawScheduleModel before scoring")
        candidates = candidate_array(candidate_indices)
        scores = np.full(len(candidates), np.nan, dtype=np.float64)
        known_user = valid_user(user_index, self.n_users_)
        valid = (candidates >= 0) & (candidates < self.n_items_)
        if not known_user or not self._active_users[user_index] or not np.any(valid):
            return scores
        indices, inverse = np.unique(candidates[valid].astype(np.int64), return_inverse=True)
        row = self.occupancy_.getrow(int(user_index))
        values = np.zeros(len(indices), dtype=np.float64)
        block_size = self._fitted_block_size
        if self._fitted_power == 1:
            for start in range(0, len(indices), block_size):
                block = indices[start:start + block_size]
                values[start:start + len(block)] = (row @ self.forecast_features_[:, block]).toarray().ravel()
        else:
            counts = self.forecast_item_airings_[indices]
            # Per-query arrays only, never users x all forecast airings.
            selected = [self._airing_order[self._airing_indptr[i]:self._airing_indptr[i + 1]] for i in indices]
            airings = np.concatenate(selected)
            owners = np.repeat(np.arange(len(indices)), counts)
            for start in range(0, len(airings), block_size):
                block = airings[start:start + block_size]
                means = (row @ self.forecast_features_[:, block]).toarray().ravel()
                means = np.clip(means, 0, 1)
                values += np.bincount(owners[start:start + len(block)],
                                      weights=means ** self._fitted_power, minlength=len(indices))
        values[(values <= 0) | ~np.isfinite(values)] = np.nan
        scores[valid] = values[inverse]
        return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline source-bound raw-watch interval union (no network)")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--temp-dir", default=None)
    args = parser.parse_args()
    result = build_cache(args.source, args.cache, args.manifest, temp_dir=args.temp_dir)
    print(json.dumps({key: result[key] for key in ("cache", "audit", "timings_seconds", "reused")}, indent=2))


if __name__ == "__main__":
    main()
