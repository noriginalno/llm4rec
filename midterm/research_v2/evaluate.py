"""Shared experimental evaluation with fixed labels/candidates and disk scores."""

from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from sweettv.evaluation import average_precision_at_k, evaluate_predictions, rank_candidates
from sweettv.pipeline import save_json, training_fingerprint
from sweettv.submission import validate_submission, write_submission


def tie_marginalized_ap(predicted: list[int], frequencies: dict[int, int], k: int = 5) -> float:
    """Expected AP when the boundary tie uniformly supplies the remaining labels.

    A sensitivity diagnostic, not the official competition metric. Different tied
    items have dependent inclusion indicators (sampling without replacement).
    """
    if k < 1:
        raise ValueError("k must be positive")
    if len(set(predicted[:k])) != len(predicted[:k]):
        raise ValueError("Predictions must be unique")
    if len(frequencies) <= k:
        return average_precision_at_k(predicted, list(frequencies), k)
    threshold = sorted(frequencies.values(), reverse=True)[k - 1]
    certain = {i for i, n in frequencies.items() if n > threshold}
    boundary = {i for i, n in frequencies.items() if n == threshold}
    remaining = k - len(certain)
    p = remaining / len(boundary)
    p2 = remaining * (remaining - 1) / (len(boundary) * (len(boundary) - 1)) if len(boundary) > 1 else 0.0
    categories = [2 if i in certain else 1 if i in boundary else 0 for i in predicted[:k]]
    total = 0.0
    for rank, category in enumerate(categories, 1):
        if not category:
            continue
        marginal = 1.0 if category == 2 else p
        joint_sum = 0.0
        for earlier in categories[:rank - 1]:
            if earlier:
                joint_sum += p2 if earlier == category == 1 else (p if 1 in (earlier, category) else 1.0)
        total += (marginal + joint_sum) / rank
    return total / k


def label_frequencies(prepared, context, users: list[str]) -> dict[str, Counter]:
    result = {u: Counter() for u in users}
    seen = {}
    for event in prepared.events:
        if event["user_id"] not in result or not (context.forecast_start <= event["start"]
                                                  and event["end"] <= context.forecast_end):
            continue
        identity = (event["user_id"], event["airing_id"])
        signature = (event["tv_show_id"], event["start"], event["end"])
        if identity in seen:
            if seen[identity] != signature:
                raise ValueError("Conflicting duplicate held-out event")
            continue
        seen[identity] = signature
        result[event["user_id"]][event["tv_show_id"]] += 1
    return result


def context_fingerprint(context) -> str:
    digest = hashlib.sha256(training_fingerprint(context.data).encode())
    digest.update(json.dumps([context.history_start, context.train_end,
                              context.forecast_start, context.forecast_end]).encode())
    for key in ("event_users", "event_items", "event_starts", "event_ends", "event_channels", "candidate_indices"):
        array = getattr(context, key)
        header = json.dumps([key, array.dtype.str, list(array.shape), array.nbytes]).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes())
    for key in ("historical_schedules", "forecast_schedules"):
        schedules = getattr(context, key)
        header = json.dumps([key, len(schedules)]).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        for a in schedules:
            payload = json.dumps([a.item_index, a.channel_id, a.start, a.end, a.quarantined]).encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    """Bind the local execution dependency closure, not just the model adapter."""
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root / "sweettv").rglob("*.py")),
             *sorted((root / "research_v2").rglob("*.py")),
             root / "tools/run_research.py", root / "tools/prepare_official.py",
             root / "requirements.lock"]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def evaluate_model(model, context, labels, users: list[str], frequencies, output: Path,
                   *, parameters: dict | None = None, batch_size: int = 16,
                   score_cache_root: Path = Path(".cache/research-v2/scores")) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    candidate_indices = context.candidate_indices
    raw_ids = context.data.item_ids
    user_map = {u: i for i, u in enumerate(context.data.user_ids)}
    popularity = np.asarray(context.data.interactions.sum(axis=0)).ravel().astype(float)
    before = context_fingerprint(context)
    class_path = Path(inspect.getfile(type(model))).resolve()
    code_before = hashlib.sha256(class_path.read_bytes()).hexdigest()
    source_bindings = source_manifest()
    started = perf_counter()
    model.fit(context)
    fit_seconds = perf_counter() - started
    if context_fingerprint(context) != before:
        raise RuntimeError("Model mutated training/context inputs")
    score_name = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:24] + ".npy"
    score_cache_root.mkdir(parents=True, exist_ok=True)
    score_path = score_cache_root / score_name
    if score_path.exists():
        raise FileExistsError(score_path)
    # Persist requested-cohort scores batchwise, without allocating a global
    # user-item score matrix in RAM. Preserve float64 for exact later fusions.
    stored = np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float64,
                                       shape=(len(users), len(candidate_indices)))
    predictions, fallback_users, fallback_positions = {}, 0, 0
    scoring_seconds, ranking_seconds = 0.0, 0.0
    size = min(batch_size, max(1, 500_000 // len(candidate_indices)))
    for start in range(0, len(users), size):
        batch_users = users[start:start + size]
        indices = np.array([user_map[u] for u in batch_users], dtype=np.int64)
        queries, candidates = indices.copy(), candidate_indices.copy()
        queries.setflags(write=False)
        candidates.setflags(write=False)
        began = perf_counter()
        if hasattr(model, "score_batch"):
            scores = np.array(model.score_batch(queries, candidates), dtype=float, copy=True)
        else:
            scores = np.vstack([np.array(model.score(int(i), candidates), dtype=float, copy=True) for i in queries])
        scoring_seconds += perf_counter() - began
        if scores.shape != (len(batch_users), len(candidates)) or np.isinf(scores).any():
            raise ValueError("Invalid model scores")
        if not np.array_equal(queries, indices) or not np.array_equal(candidates, candidate_indices):
            raise RuntimeError("Model mutated query alignment")
        stored[start:start + len(batch_users)] = scores
        began = perf_counter()
        for row, user in enumerate(batch_users):
            ranking = rank_candidates(scores[row], candidate_indices, raw_ids, popularity)
            predictions[user] = ranking["items"]
            fallback_positions += ranking["fallback_positions"]
            fallback_users += int(ranking["fallback_positions"] > 0)
        ranking_seconds += perf_counter() - began
    stored.flush()
    del stored
    if context_fingerprint(context) != before or hashlib.sha256(class_path.read_bytes()).hexdigest() != code_before:
        raise RuntimeError("Model inputs/source changed during experiment")
    if source_manifest() != source_bindings:
        raise RuntimeError("Shared research/evaluation source changed during experiment")
    primary = [u for u in users if len(labels[u]) == 5]
    reverse = {u: sorted(frequencies[u], key=lambda i: (-frequencies[u][i], -i))[:5] for u in users}
    standard = evaluate_predictions(predictions, labels, primary)
    diagnostic = evaluate_predictions(predictions, labels, users)
    reversed_metric = evaluate_predictions(predictions, reverse, primary)
    expected_tie = float(np.mean([tie_marginalized_ap(predictions[u], frequencies[u]) for u in primary])) if primary else None
    result = {
        "class": type(model).__name__, "parameters": parameters or {},
        "source_sha256": code_before, "shared_source_sha256": source_bindings, "context_sha256": before,
        "primary_users": len(primary), "all_users": len(users),
        "map_at_5": standard["map_at_5"], "all_user_map_at_5": diagnostic["map_at_5"],
        "reverse_tie_map_at_5": reversed_metric["map_at_5"],
        "reverse_label_tie_map_at_5": reversed_metric["map_at_5"],
        "tie_marginalized_map_at_5": expected_tie,
        "fit_seconds": fit_seconds, "score_seconds": scoring_seconds, "rank_seconds": ranking_seconds,
        "fallback_users": fallback_users, "fallback_positions": fallback_positions,
        "score_file": str(score_path), "score_shape": [len(users), len(candidate_indices)],
        "diagnostics": model.diagnostics, "kaggle_score": None, "submitted": False,
    }
    save_json(output / "result.json", result)
    save_json(output / "predictions.json", predictions)
    save_json(output / "primary_metrics.json", standard)
    save_json(output / "diagnostic_metrics.json", diagnostic)
    write_submission(output / "submission.csv", users, predictions)
    validate_submission(output / "submission.csv", users, raw_ids[candidate_indices])
    return result
