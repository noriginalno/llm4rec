"""Common bounded-batch prediction, evaluation, and evidence export."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import numpy as np

from .baselines import PersonalRepeatModel, PopularityModel
from .contracts import TrainingData
from .data import PreparedData, build_fold
from .evaluation import evaluate_predictions, rank_candidates, reciprocal_rank_fusion
from .submission import validate_submission, write_submission


MODEL_NAMES = ("popularity", "repeat", "cbf", "association", "mf", "ppr")
DEFAULT_FUSIONS = {
    "hybrid_repeat": {"repeat": 1.0, "cbf": 0.25, "association": 0.25, "mf": 0.25, "ppr": 0.25},
    "hybrid_balanced": {"repeat": 1.0, "cbf": 0.5, "association": 0.5, "mf": 0.5, "ppr": 0.5},
}


def save_json(path: Path, value: Any) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Cannot serialize {type(obj).__name__}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=convert, allow_nan=False)
        handle.write("\n")


def training_fingerprint(data: TrainingData) -> str:
    digest = hashlib.sha256()
    structure = {
        "matrix_shape": list(data.interactions.shape),
        "users": data.user_ids, "metadata": data.metadata,
        "provenance": data.provenance,
    }
    digest.update(json.dumps(structure, ensure_ascii=False, sort_keys=True).encode())
    for name, array in (("item_ids", data.item_ids), ("indptr", data.interactions.indptr),
                        ("indices", data.interactions.indices), ("counts", data.interactions.data)):
        header = json.dumps([name, array.dtype.str, list(array.shape), array.nbytes]).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes())
    return digest.hexdigest()


def implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}


def create_model(name: str, parameters: dict[str, Any] | None = None):
    if name == "popularity":
        cls = PopularityModel
    elif name == "repeat":
        cls = PersonalRepeatModel
    elif name == "cbf":
        from .models.cbf import ContentBasedModel
        cls = ContentBasedModel
    elif name == "association":
        from .models.association import AssociationRulesModel
        cls = AssociationRulesModel
    elif name == "mf":
        from .models.mf import MatrixFactorizationModel
        cls = MatrixFactorizationModel
    elif name == "ppr":
        from .models.ppr import PersonalizedPageRankModel
        cls = PersonalizedPageRankModel
    else:
        raise ValueError(f"Unknown model: {name}")
    return cls(**(parameters or {}))


def select_example_users(data: TrainingData, template_users: list[str]) -> tuple[list[str], dict]:
    mapping = {user: i for i, user in enumerate(data.user_ids)}
    descriptions = {}
    for user in template_users:
        row = data.interactions.getrow(mapping[user])
        total = float(row.data.sum())
        descriptions[user] = {
            "qualifying_broadcasts": total,
            "distinct_programs": int(row.nnz),
            "top_program_share": float(row.data.max() / total) if total else 0.0,
        }
    active = sorted(
        (u for u in template_users if descriptions[u]["qualifying_broadcasts"] > 0),
        key=lambda u: (descriptions[u]["qualifying_broadcasts"], u),
    )
    if len(active) >= 4:
        pool = active[len(active) // 4: max(len(active) // 4 + 2, 3 * len(active) // 4)]
    else:
        pool = active
    pool = sorted(pool, key=lambda u: (descriptions[u]["top_program_share"], u))
    if len(pool) >= 2:
        chosen = [pool[0], pool[-1]]
    else:
        chosen = list(pool)
        chosen += [u for u in template_users if u not in chosen][:2 - len(chosen)]
    return chosen, {
        "selection": "training-only middle-activity pool; minimum/maximum top-program share; string-ID ties",
        "users": {u: descriptions[u] for u in chosen},
    }


def run_fold(
    prepared: PreparedData,
    train_end: int,
    forecast_start: int,
    forecast_end: int,
    output: Path,
    *,
    model_parameters: dict[str, dict] | None = None,
    fusions: dict[str, dict[str, float]] | None = None,
    evaluate: bool = True,
    synthetic: bool = False,
    batch_size: int = 16,
) -> dict:
    """Fail on model errors; never silently replace a failed method with baseline."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = implementation_hashes()
    data, candidates, labels, split_audit = build_fold(prepared, train_end, forecast_start, forecast_end)
    if len(candidates) < 5:
        raise ValueError("Forecast schedule has fewer than five candidate programs")
    model_parameters = model_parameters or {}
    fusions = fusions or {}
    unknown = set(model_parameters) - set(MODEL_NAMES)
    if unknown or set(fusions) & set(MODEL_NAMES):
        raise ValueError("Unknown parameters or duplicate method names")
    if any(set(weights) - set(MODEL_NAMES) for weights in fusions.values()):
        raise ValueError("Fusion refers to an unknown constituent")
    popularity = np.asarray(data.interactions.sum(axis=0)).ravel().astype(float)
    before = training_fingerprint(data)
    models, timings = {}, {}
    for name in MODEL_NAMES:
        started = perf_counter()
        model = create_model(name, model_parameters.get(name))
        initialization_seconds = perf_counter() - started
        started = perf_counter()
        models[name] = model.fit(data)
        timings[name] = {"initialization_seconds": initialization_seconds,
                         "fit_seconds": perf_counter() - started, "score_seconds": 0.0}
        if training_fingerprint(data) != before:
            raise RuntimeError(f"{name} mutated shared training data")
    user_index = {u: i for i, u in enumerate(data.user_ids)}
    users = list(prepared.template_users)
    example_users, example_selection = select_example_users(data, users)
    predictions = {name: {} for name in (*MODEL_NAMES, *fusions)}
    support = {name: {"fallback_positions": 0, "fallback_users": 0, "no_evidence_users": 0}
               for name in predictions}
    query_diagnostics = {name: [] for name in MODEL_NAMES}
    for name in fusions:
        timings[name] = {"initialization_seconds": 0.0, "fit_seconds": 0.0, "score_seconds": 0.0}
    evidence_path = output / "recommendations.csv"
    fieldnames = ["method", "user_id", "rank", "tv_show_id", "title", "category", "score", "source", "heldout_relevant"]
    # MF caps a requested dense batch at one million entries. PPR separately caps
    # walk columns. Stream batches; do not retain all user-candidate score arrays.
    actual_batch_size = min(batch_size, max(1, 500_000 // len(candidates)))
    with evidence_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for start in range(0, len(users), actual_batch_size):
            batch_users = users[start:start + actual_batch_size]
            indices = np.asarray([user_index[u] for u in batch_users], dtype=np.int64)
            vectors = {}
            for name, model in models.items():
                query_users = indices.copy()
                query_candidates = candidates.copy()
                query_users.setflags(write=False)
                query_candidates.setflags(write=False)
                began = perf_counter()
                if hasattr(model, "score_batch"):
                    scores = np.array(model.score_batch(query_users, query_candidates), dtype=float, copy=True)
                else:
                    scores = np.vstack([
                        np.array(model.score(int(i), query_candidates), dtype=float, copy=True)
                        for i in query_users
                    ])
                timings[name]["score_seconds"] += perf_counter() - began
                if not np.array_equal(query_users, indices) or not np.array_equal(query_candidates, candidates):
                    raise RuntimeError(f"{name} mutated query user/candidate alignment")
                if scores.shape != (len(batch_users), len(candidates)) or np.isinf(scores).any():
                    raise RuntimeError(f"Invalid score shape/values from {name}")
                vectors[name] = scores
                query = model.diagnostics.get("last_query")
                if query is not None:
                    # Freeze diagnostics before the next call replaces last_query.
                    query_diagnostics[name].append(json.loads(json.dumps(query)))
            for row, user in enumerate(batch_users):
                one_user = {name: scores[row] for name, scores in vectors.items()}
                for name, weights in fusions.items():
                    began = perf_counter()
                    one_user[name] = reciprocal_rank_fusion(one_user, weights, candidates, data.item_ids)
                    timings[name]["score_seconds"] += perf_counter() - began
                for name, scores in one_user.items():
                    result = rank_candidates(scores, candidates, data.item_ids, popularity)
                    predictions[name][user] = result["items"]
                    fallback = result["fallback_positions"]
                    support[name]["fallback_positions"] += fallback
                    support[name]["fallback_users"] += int(fallback > 0)
                    support[name]["no_evidence_users"] += int(not np.isfinite(scores).any())
                    for rank, (item, index, score) in enumerate(zip(result["items"], result["indices"], result["scores"]), 1):
                        metadata = data.metadata[index]
                        writer.writerow({
                            "method": name, "user_id": user, "rank": rank, "tv_show_id": item,
                            "title": metadata.get("tv_show_title") or "", "category": metadata.get("tv_show_category") or "",
                            "score": "" if score is None else repr(score),
                            "source": "training_popularity_backfill" if score is None else "model_evidence",
                            "heldout_relevant": int(item in labels[user]) if evaluate else "",
                        })
    if training_fingerprint(data) != before:
        raise RuntimeError("Scoring mutated shared training data")
    if implementation_hashes() != source_hashes:
        raise RuntimeError("Implementation source changed during this run; rerun a stable version")
    primary = [u for u in users if len(labels[u]) == 5]
    sparse = [u for u in users if data.interactions.getrow(user_index[u]).nnz < 3]
    metrics, submissions = {}, {}
    for name in predictions:
        path = output / f"submission_{name}.csv"
        write_submission(path, users, predictions[name])
        submissions[name] = validate_submission(path, users, data.item_ids[candidates])
        save_json(output / f"predictions_{name}.json", predictions[name])
        if evaluate:
            metrics[name] = {
                "primary_five_label_cohort": evaluate_predictions(predictions[name], labels, primary),
                "all_template_users_diagnostic": evaluate_predictions(predictions[name], labels, users),
                "sparse_history_users_diagnostic": evaluate_predictions(predictions[name], labels, sparse),
            }
    history_path = output / "two_user_histories.csv"
    with history_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "tv_show_id", "title", "category", "qualifying_training_broadcasts"])
        for user in example_users:
            row = data.interactions.getrow(user_index[user])
            for i, count in sorted(zip(row.indices, row.data), key=lambda pair: (-pair[1], int(data.item_ids[pair[0]]))):
                meta = data.metadata[i]
                writer.writerow([user, int(data.item_ids[i]), meta.get("tv_show_title") or "", meta.get("tv_show_category") or "", int(count)])
    save_json(output / "labels.json", {u: labels[u] for u in users})
    save_json(output / "evaluation_users.json", primary)
    save_json(output / "template_users.json", users)
    save_json(output / "candidate_ids.json", data.item_ids[candidates])
    save_json(output / "two_user_selection.json", example_selection)
    save_json(output / "timings.json", timings)
    save_json(output / "query_diagnostics.json", query_diagnostics)
    save_json(output / "metrics.json", metrics)
    save_json(output / "model_diagnostics.json", {name: model.diagnostics for name, model in models.items()})
    summary = {
        "synthetic": synthetic,
        "official_kaggle_data": False if synthetic else None,
        "data_source_status": "synthetic" if synthetic else "unverified_source_identity",
        "kaggle_score": None, "submitted": False,
        "training_fingerprint": before, "implementation_sha256": source_hashes,
        "model_parameters": model_parameters,
        "fusions": fusions, "split": split_audit, "submissions": submissions,
        "support": support, "example_users": example_users,
        "primary_user_count": len(primary) if evaluate else None,
        "primary_map_at_5": {name: value["primary_five_label_cohort"]["map_at_5"] for name, value in metrics.items()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()},
        "selection_warning": "Temporal validation/model selection, not independent test or a returned Kaggle score",
    }
    save_json(output / "summary.json", summary)
    return summary


def best_method(summary: dict, names: list[str] | None = None) -> str:
    values = summary["primary_map_at_5"]
    names = names or list(values)
    usable = [name for name in names if values[name] is not None]
    if not usable:
        raise ValueError("No nonempty primary evaluation cohort; cannot select a model")
    priority = ["repeat", "popularity", "cbf", "association", "mf", "ppr", *sorted(set(names) - set(MODEL_NAMES))]
    return min(usable, key=lambda name: (-values[name], priority.index(name)))
