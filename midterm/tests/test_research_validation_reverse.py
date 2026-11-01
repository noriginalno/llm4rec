"""EASE author's reverse challenge of context/evaluation, synthetic inputs only.

Strict xfails are newly reproduced shortcomings, NOT passed correctness checks
and NOT the already-fixed C1-C4. No real models, cached data, archived metrics,
credentials, or submissions are loaded. All evaluation artifacts use tmp_path.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from research_v2.contracts import make_context, temporal_matrix
from research_v2.evaluate import (
    evaluate_model,
    label_frequencies,
    tie_marginalized_ap,
)
from sweettv.data import PreparedData


DAY = 86400
HISTORY = 1583712000
CUTOFF = HISTORY + 3 * DAY
START = CUTOFF + DAY  # Deliberate nonzero train-to-forecast gap.
END = START + 2 * DAY
BIG_USER = "18446744073709551615"
USERS = ["cold", BIG_USER, "active"]  # Template order differs from matrix order.
ROOT = Path(__file__).resolve().parents[1]


def _schedule(name, item, start, end, channel="c"):
    return dict(airing_id=name, tv_show_id=item, start=start, end=end,
                channel_id=channel, metadata={"tv_show_title": name}, quarantined=False)


def _event(row, user="active"):
    return {key: row[key] for key in ("airing_id", "tv_show_id", "start", "end")} | {
        "user_id": user, "coverage": .8,
    }


def _prepared(frequencies=(1, 1, 1, 1, 1, 1)):
    past = [
        _schedule("past11a", 11, CUTOFF - 100, CUTOFF - 90),
        _schedule("past11b", 11, CUTOFF - 80, CUTOFF - 70),
        _schedule("past21", 21, CUTOFF - 60, CUTOFF - 50),
        _schedule("cutoff31", 31, CUTOFF - 10, CUTOFF),
    ]
    future = []
    for item, count in zip([11, 21, 31, 41, 51, 61], frequencies, strict=True):
        for repeat in range(count):
            start = START + len(future) * 20
            if item == 61 and repeat == count - 1:
                start = END - 10  # Whole airing ending exactly at forecast end.
            future.append(_schedule(f"future{item}-{repeat}", item, start, start + 10))
    gap = _schedule("gap81", 81, CUTOFF + 100, CUTOFF + 110)
    crossing = _schedule("cross71", 71, END - 20, END + 20, "cross-channel")
    late = _schedule("late91", 91, END, END + 10)
    events = [*(_event(row) for row in past[:3]), _event(past[3], BIG_USER),
              *(_event(row) for row in future),
              _event(next(row for row in future if row["tv_show_id"] == 21), BIG_USER),
              _event(gap), _event(crossing), _event(late, "heldout-only")]
    return PreparedData(USERS.copy(), past + future + [gap, crossing, late], events,
                        {"fixture": "reverse validation challenge; synthetic only"})


def _context(prepared=None):
    return make_context(_prepared() if prepared is None else prepared, CUTOFF, START, END,
                        history_start=HISTORY)


def _independent_counts(prepared):
    """Direct broadcast-identity scan; never call build_fold/label helpers."""
    seen, training, frequencies = set(), Counter(), {u: Counter() for u in USERS}
    for event in prepared.events:
        identity = (event["user_id"], event["airing_id"])
        if identity in seen:
            continue
        seen.add(identity)
        if event["end"] <= CUTOFF:
            training[event["user_id"], event["tv_show_id"]] += 1
        if (event["user_id"] in frequencies
                and START <= event["start"] < event["end"] <= END):
            frequencies[event["user_id"]][event["tv_show_id"]] += 1
    return training, frequencies


def _scalar_ap(prediction, relevant, k=5):
    """Independent exact rational AP oracle, not shared AP/evaluator code."""
    relevant = set(relevant)
    if not relevant:
        return Fraction(0)
    found, total = set(), Fraction(0)
    for rank, item in enumerate(prediction[:k], 1):
        if item in relevant and item not in found:
            found.add(item)
            total += Fraction(len(found), rank)
    return total / min(k, len(relevant))


class ScoreProbe:
    """Alignment/fallback probe, never evidence of a trained recommender."""

    def __init__(self, mode="partial", mutation=None, phase="fit"):
        self.mode, self.mutation, self.phase = mode, mutation, phase
        self.calls = []
        self.diagnostics = {"synthetic_probe": True}

    def _mutate(self):
        if self.mutation == "count":
            self.context.data.interactions.data[0] += 1
        elif self.mutation == "metadata":
            self.context.data.metadata[0]["changed"] = True
        elif self.mutation == "event_bytes":
            ends = self.context.event_ends.copy()
            ends[0] -= 1
            object.__setattr__(self.context, "event_ends", ends)
        elif self.mutation == "event_shape":
            object.__setattr__(self.context, "event_channels",
                               self.context.event_channels.reshape(1, -1))
        elif self.mutation == "event_dtype":
            object.__setattr__(self.context, "event_channels",
                               self.context.event_channels.view(np.int32))
        elif self.mutation == "schedule_partition":
            # Move the final historical airing to the forecast tuple, retaining
            # the same concatenated airing sequence. This is intentional fault
            # injection, not a claim that a real model performs this mutation.
            history = self.context.historical_schedules
            object.__setattr__(self.context, "historical_schedules", history[:-1])
            object.__setattr__(self.context, "forecast_schedules",
                               (history[-1], *self.context.forecast_schedules))

    def fit(self, context):
        self.context = context
        if self.phase == "fit":
            self._mutate()
        return self

    def score_batch(self, users, candidates):
        assert not users.flags.writeable and not candidates.flags.writeable
        self.calls.append((users.copy(), candidates.copy()))
        scores = np.full((len(users), len(candidates)), np.nan)
        if self.mode == "ties":
            scores[:] = 1
        elif self.mode == "partial":
            for row, user in enumerate(users):
                if self.context.data.user_ids[user] == "active":
                    raw = self.context.data.item_ids[candidates]
                    scores[row, raw == 41] = -2
                    scores[row, raw == 61] = -3
        if self.phase == "score":
            self._mutate()
        return scores


def _evaluate(tmp_path, model=None, prepared=None, suffix="trial", context=None):
    prepared = _prepared() if prepared is None else prepared
    built, labels, _ = _context(prepared)
    context = built if context is None else context
    frequencies = label_frequencies(prepared, context, USERS)
    result = evaluate_model(
        ScoreProbe() if model is None else model, context, labels, USERS.copy(), frequencies,
        tmp_path / suffix, batch_size=2, score_cache_root=tmp_path / "synthetic-scores",
    )
    return result, context


def test_gap_whole_airing_cutoffs_and_shared_catalog_against_scalar_scan():
    prepared = _prepared()
    context, labels, _ = _context(prepared)
    training, frequencies = _independent_counts(prepared)
    for user_index, user in enumerate(context.data.user_ids):
        for item_index, item in enumerate(context.data.item_ids):
            assert context.data.interactions[user_index, item_index] == training[user, int(item)]
    assert_array_equal(temporal_matrix(context).toarray(), context.data.interactions.toarray())
    expected_catalog = sorted({row["tv_show_id"] for row in prepared.schedules if row["start"] < END})
    expected_candidates = sorted({row["tv_show_id"] for row in prepared.schedules
                                  if START <= row["start"] < END})
    assert_array_equal(context.data.item_ids, expected_catalog)
    assert_array_equal(context.data.item_ids[context.candidate_indices], expected_candidates)
    assert 71 in expected_candidates and 81 not in expected_candidates and 91 not in expected_catalog
    assert context.data.interactions.sum() == 4  # Gap/crossing/forecast behavior is never training.
    assert "heldout-only" not in context.data.user_ids
    assert frequencies == label_frequencies(prepared, context, USERS)
    for user in USERS:
        assert labels[user] == sorted(frequencies[user], key=lambda i: (-frequencies[user][i], i))[:5]
    assert frequencies["active"][61] == 1  # Whole airing ending at END is included.
    assert all(71 not in row and 81 not in row for row in frequencies.values())


def test_common_cohorts_candidate_order_one_time_fallback_and_raw_nan_cache(tmp_path):
    context, _, _ = _context()
    permutation = np.array([6, 2, 0, 5, 1, 4, 3])
    context = replace(context, candidate_indices=context.candidate_indices[permutation])
    probe = ScoreProbe()
    partial, _ = _evaluate(tmp_path, probe, context=context, suffix="partial")
    absent, _ = _evaluate(tmp_path, ScoreProbe(mode="absent"), suffix="absent")
    for result in (partial, absent):
        assert result["primary_users"] == 1
        assert result["all_users"] == 3
        assert result["score_shape"] == [3, 7]
        output = tmp_path / ("partial" if result is partial else "absent")
        strict = json.loads((output / "primary_metrics.json").read_text())
        all_users = json.loads((output / "diagnostic_metrics.json").read_text())
        assert [row["user_id"] for row in strict["per_user"]] == ["active"]
        assert [row["user_id"] for row in all_users["per_user"]] == USERS
        assert all_users["empty_label_users"] == 1
    mapping = {user: i for i, user in enumerate(context.data.user_ids)}
    assert_array_equal(np.concatenate([users for users, _ in probe.calls]), [mapping[u] for u in USERS])
    for _, candidates in probe.calls:
        assert_array_equal(candidates, context.candidate_indices)
    predictions = json.loads((tmp_path / "partial" / "predictions.json").read_text())
    # Negative finite evidence comes before fallback, even for less-popular IDs.
    assert predictions["active"] == [41, 61, 11, 21, 31]
    assert predictions["cold"] == predictions[BIG_USER] == [11, 21, 31, 41, 51]
    assert partial["fallback_positions"] == 13  # 3 + 5 + 5, exactly once.
    assert absent["fallback_positions"] == 15
    saved = np.load(partial["score_file"], mmap_mode="r")
    assert saved.shape == (3, 7)
    assert np.count_nonzero(np.isfinite(saved)) == 2
    assert np.isnan(saved[:2]).all()  # Cached fusion evidence has no backfill votes.
    assert sorted(saved[2, np.isfinite(saved[2])]) == [-3, -2]


@pytest.mark.parametrize("mutation", ["count", "metadata", "event_bytes"])
@pytest.mark.parametrize("phase", ["fit", "score"])
def test_existing_byte_level_mutation_guards_stop_results(tmp_path, mutation, phase):
    with pytest.raises(RuntimeError, match="mutated|changed"):
        _evaluate(tmp_path, ScoreProbe(mutation=mutation, phase=phase))
    assert not (tmp_path / "trial" / "result.json").exists()


@pytest.mark.parametrize("counts,prediction,k", [
    ({11: 2, 21: 1, 31: 1}, [21, 11], 2),
    ({11: 8, 21: 8, 31: 3, 41: 3, 51: 3, 61: 3, 71: 1}, [31, 71, 41, 11, 61], 5),
    ({11: 1, 21: 1, 31: 1, 41: 1, 51: 1, 61: 1, 71: 1}, [11, 21, 999, 31, 41], 5),
    ({11: 3, 21: 1}, [21, 99, 11], 5),
    ({}, [11, 21, 31], 5),
])
def test_label_boundary_tie_expectation_matches_exact_subset_enumeration(counts, prediction, k):
    if len(counts) <= k:
        truths = [list(counts)]
    else:
        threshold = sorted(counts.values(), reverse=True)[k - 1]
        certain = [item for item, n in counts.items() if n > threshold]
        boundary = [item for item, n in counts.items() if n == threshold]
        truths = [certain + list(subset) for subset in itertools.combinations(boundary, k - len(certain))]
    expected = sum((_scalar_ap(prediction, truth, k) for truth in truths), Fraction(0)) / len(truths)
    assert tie_marginalized_ap(prediction, counts, k) == pytest.approx(float(expected), abs=1e-15)


@pytest.mark.parametrize("mutation", ["event_shape", "event_dtype"])
def test_context_array_interpretation_mutation_must_fail_closed(tmp_path, mutation):
    with pytest.raises(RuntimeError, match="mutated|changed"):
        _evaluate(tmp_path, ScoreProbe(mutation=mutation))


def test_schedule_partition_mutation_must_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match="mutated|changed"):
        _evaluate(tmp_path, ScoreProbe(mutation="schedule_partition"))


def test_fingerprint_collision_reproductions_are_distinct_semantics():
    # Explicit characterization accompanying R1/R2; not a passed integrity guard.
    for mutation in ("event_shape", "event_dtype", "schedule_partition"):
        context, _, _ = _context()
        before_array_bytes = context.event_channels.tobytes()
        before_airings = (*context.historical_schedules, *context.forecast_schedules)
        probe = ScoreProbe(mutation=mutation)
        probe.fit(context)
        assert context.event_channels.tobytes() == before_array_bytes
        assert (*context.historical_schedules, *context.forecast_schedules) == before_airings
        if mutation == "event_shape":
            assert context.event_channels.ndim == 2
        elif mutation == "event_dtype":
            assert context.event_channels.dtype.kind == "i"
        else:
            assert context.forecast_schedules[0].end <= CUTOFF


def test_label_frequency_diagnostics_must_share_broadcast_deduplication():
    prepared = _prepared()
    duplicated = next(row for row in prepared.events
                      if row["user_id"] == "active" and row["tv_show_id"] == 61 and row["start"] >= START)
    prepared.events.extend([deepcopy(duplicated), deepcopy(duplicated)])
    context, labels, audit = _context(prepared)
    _, expected = _independent_counts(prepared)
    assert audit["duplicate_events_ignored"] == 2
    assert labels["active"] == [11, 21, 31, 41, 51]  # Shared fold still counts each broadcast once.
    assert label_frequencies(prepared, context, USERS) == expected


def test_trial_source_bindings_cover_executed_shared_entry_and_export_code(tmp_path):
    result, _ = _evaluate(tmp_path)
    actual = {Path(path).resolve() for path in result["shared_source_sha256"]}
    expected = {ROOT / path for path in (
        "research_v2/evaluate.py", "research_v2/contracts.py", "sweettv/data.py",
        "sweettv/evaluation.py", "sweettv/pipeline.py", "sweettv/submission.py",
        "sweettv/contracts.py", "tools/run_research.py", "requirements.lock",
    )}
    assert expected <= actual


def test_recorded_hashes_match_files_that_are_bound(tmp_path):
    result, _ = _evaluate(tmp_path)
    assert result["source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for path, digest in result["shared_source_sha256"].items():
        assert digest == hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_label_tie_metrics_do_not_measure_prediction_tie_robustness(tmp_path):
    # Novel characterization, not a claim that an existing diagnostic computes
    # its stated label-only expectation incorrectly. No label frequency ties!
    prepared = _prepared((6, 5, 4, 3, 2, 1))
    result, context = _evaluate(tmp_path, ScoreProbe(mode="ties"), prepared=prepared)
    truth = [11, 21, 31, 41, 51]
    ascending = sorted(map(int, context.data.item_ids[context.candidate_indices]))[:5]
    descending = sorted(map(int, context.data.item_ids[context.candidate_indices]), reverse=True)[:5]
    assert result["map_at_5"] == result["reverse_tie_map_at_5"] == result["tie_marginalized_map_at_5"] == 1
    assert _scalar_ap(ascending, truth) == 1
    assert _scalar_ap(descending, truth) < Fraction(1, 2)
    assert not any("score_tie" in key or "prediction_tie" in key for key in result)


def test_parent_cli_preserves_one_synthetic_fold_for_two_nonmodel_probes(tmp_path, monkeypatch):
    from research_v2 import contracts
    from tools import run_research

    prepared = _prepared()
    digest = "f" * 64
    loads, evaluations = [], []

    def fake_load(cache, *, data_source_sha256):
        loads.append((cache, data_source_sha256))
        return prepared

    def temporary_evaluate(*args, **kwargs):
        kwargs["score_cache_root"] = tmp_path / "runner-synthetic-scores"
        evaluations.append((id(args[1]), tuple(args[3]), args[1].candidate_indices.copy()))
        return evaluate_model(*args, **kwargs)

    monkeypatch.setattr(run_research, "load_prepared", fake_load)
    monkeypatch.setattr(run_research, "preparation_signature", lambda raw: digest)
    monkeypatch.setattr(run_research, "evaluate_model", temporary_evaluate)
    monkeypatch.setattr(run_research, "FOLDS", {"synthetic": (CUTOFF, START, END)})
    monkeypatch.setattr(contracts, "ReverseReviewProbe", ScoreProbe, raising=False)
    grid = [
        {"name": "partial", "class": "research_v2.contracts:ReverseReviewProbe", "parameters": {}},
        {"name": "absent", "class": "research_v2.contracts:ReverseReviewProbe", "parameters": {"mode": "absent"}},
    ]
    grid_path = tmp_path / "synthetic-grid.json"
    grid_path.write_text(json.dumps(grid))
    output = tmp_path / "runner-output"
    monkeypatch.setattr("sys.argv", ["run_research.py", "--grid", str(grid_path), "--folds",
                                     "synthetic", "--out", str(output), "--cache", "synthetic-only"])
    run_research.main()
    assert loads == [(Path("synthetic-only"), digest)]
    assert len(evaluations) == 2 and evaluations[0][0] == evaluations[1][0]
    assert evaluations[0][1] == evaluations[1][1] == tuple(USERS)
    assert_array_equal(evaluations[0][2], evaluations[1][2])
    fold = output / "synthetic"
    assert json.loads((fold / "template_users.json").read_text()) == USERS
    assert json.loads((fold / "candidate_ids.json").read_text()) == [11, 21, 31, 41, 51, 61, 71]
    results = json.loads((output / "results.json").read_text())["synthetic"]
    assert set(results) == {"partial", "absent"}
    assert results["partial"]["context_sha256"] == results["absent"]["context_sha256"]
    assert all(row["primary_users"] == 1 and row["all_users"] == 3 for row in results.values())
