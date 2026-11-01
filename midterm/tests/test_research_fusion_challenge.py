"""Independent fusion challenge; synthetic inputs, no research result files.

Expected values use scalar Python/Fraction arithmetic, never the shared RRF or
ranking helpers. In-memory table components isolate the fusion from model fit
quality; separate integration checks use actual local research components.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from fractions import Fraction
import hashlib
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix

from research_v2.contracts import TemporalContext, make_context
from research_v2.collaborative import EASEModel
from research_v2.fusion import EvidenceBlendModel
import research_v2.priors as prior_module
from research_v2.priors import AiringPriorRepeatModel
from research_v2.temporal import TemporalRepeatModel
from sweettv.contracts import TrainingData
from sweettv.data import PreparedData


DAY = 86_400
HISTORY_START = 1_583_712_000
TRAIN_END = HISTORY_START + 10 * DAY
FORECAST_END = TRAIN_END + 4 * DAY
NAN = float("nan")


class FusionChallengeTableModel:
    """Local synthetic evidence provider, not a production recommendation model."""

    def __init__(self, values, fail_at_train_end=None, bad_shape=False,
                 forbid_score=False):
        self.values = deepcopy(values)
        self.fail_at_train_end = fail_at_train_end
        self.bad_shape = bad_shape
        self.forbid_score = forbid_score
        self.diagnostics = {}

    def fit(self, context):
        if context.train_end == self.fail_at_train_end:
            raise ValueError("synthetic constituent fit rejected this context")
        supplied = np.array(self.values, dtype=np.float64, copy=True)
        if supplied.ndim == 1:
            self.table = np.full(context.data.interactions.shape, np.nan)
            self.table[0] = supplied
        else:
            self.table = supplied
        assert self.table.shape == context.data.interactions.shape
        self.table.setflags(write=False)
        self.diagnostics = {"synthetic_evidence_only": True, "internal_backfill": False}
        return self

    def score(self, user, candidates):
        if self.forbid_score:
            raise AssertionError("an empty/invalid query must not reach a component")
        result = self.table[user, candidates].copy()
        result.setflags(write=False)
        return result[:, None] if self.bad_shape else result


@pytest.fixture
def table_components(monkeypatch):
    # No module/source files are generated. inspect.getfile still identifies this
    # real test file, so the production constituent source hash remains exercised.
    monkeypatch.setattr(prior_module, "FusionChallengeTableModel",
                        FusionChallengeTableModel, raising=False)

    def configure(tables, weights=None):
        if weights is None:
            weights = [1.0] * len(tables)
        return [
            {"name": f"part-{i}",
             "class": "research_v2.priors:FusionChallengeTableModel",
             "parameters": {"values": deepcopy(table)}, "weight": weight}
            for i, (table, weight) in enumerate(zip(tables, weights, strict=True))
        ]

    return configure


def empty_context(raw_ids):
    ids = np.asarray(raw_ids, dtype=np.int64)
    return TemporalContext(
        data=TrainingData(["0007", "18446744073709551615"], ids,
                          csr_matrix((2, len(ids))), [{} for _ in ids]),
        candidate_indices=np.arange(len(ids)), history_start=HISTORY_START,
        train_end=TRAIN_END, forecast_start=TRAIN_END, forecast_end=FORECAST_END,
        event_users=np.array([], dtype=int), event_items=np.array([], dtype=int),
        event_starts=np.array([], dtype=int), event_ends=np.array([], dtype=int),
        event_channels=np.array([], dtype=str), historical_schedules=(),
        forecast_schedules=(),
    )


def scalar_oracle(raw_ids, tables, weights, query, *, mode, depth=200,
                  rank_constant=60, reverse=False):
    """Independent set-of-items oracle, not a call into any production ranker."""
    ids = [int(value) for value in raw_ids]
    unique = {int(i) for i in query if 0 <= int(i) < len(ids)}
    terms = {i: [] for i in unique}
    for table, weight in zip(tables, weights, strict=True):
        if weight == 0:
            continue
        observed = [i for i in unique if math.isfinite(float(table[i]))]
        if mode == "rank":
            ordered = sorted(observed, key=lambda i: (
                -float(table[i]), -ids[i] if reverse else ids[i],
            ))
            for rank, i in enumerate(ordered[:depth], 1):
                vote = Fraction(str(weight)) / (Fraction(str(rank_constant)) + rank)
                terms[i].append(float(vote))
        else:
            for i in observed:
                terms[i].append(float(weight) * float(table[i]))
    return np.array([
        math.fsum(terms[int(i)]) if int(i) in terms and terms[int(i)] else NAN
        for i in query
    ])


def evidence_order(scores, raw_ids):
    """Independent final ordering; deliberately does not perform backfill."""
    return sorted((i for i, score in enumerate(scores) if math.isfinite(score)),
                  key=lambda i: (-float(scores[i]), int(raw_ids[i])))


def snapshot(context):
    """Snapshot all exposed inputs without the production fingerprint helper."""
    arrays = tuple(
        (field.name, value.dtype.str, value.shape, value.tobytes(), value.flags.writeable)
        for field in fields(context)
        if isinstance((value := getattr(context, field.name)), np.ndarray)
    )
    matrix = context.data.interactions
    return (
        tuple(context.data.user_ids), context.data.item_ids.tobytes(), matrix.shape,
        matrix.data.tobytes(), matrix.indices.tobytes(), matrix.indptr.tobytes(),
        deepcopy(context.data.metadata), deepcopy(context.data.provenance), arrays,
        context.history_start, context.train_end, context.forecast_start,
        context.forecast_end, context.historical_schedules, context.forecast_schedules,
    )


def test_hand_derived_rrf_signed_zero_and_missing_evidence(table_components):
    ids = [50, 10, 30, 20, 90, 60]
    tables = [[4, 4, -2, NAN, 0, NAN], [NAN, 1, -1, -5, NAN, NAN]]
    model = EvidenceBlendModel(table_components(tables, [2, 0.5])).fit(empty_context(ids))
    # First component: raw IDs 10,50,90,30. Second: raw IDs 10,30,20.
    expected = [2 / 62, 2.5 / 61, 2 / 64 + 0.5 / 62, 0.5 / 63, 2 / 63, NAN]
    assert_allclose(model.score(0, np.arange(6)), expected, rtol=1e-14, atol=0)
    assert np.isnan(model.score(1, np.arange(6))).all()


def test_hand_derived_count_blend_is_not_probability_or_rank_blend(table_components):
    tables = [[4, 4, -2, NAN, 0, NAN], [NAN, 1, -1, -5, NAN, NAN]]
    model = EvidenceBlendModel(table_components(tables, [2, 0.5]), mode="count")
    model.fit(empty_context([50, 10, 30, 20, 90, 60]))
    assert_array_equal(model.score(0, np.arange(6)), [8, 8.5, -4.5, -2.5, 0, NAN])
    assert "unscaled" in model.diagnostics["count_mode_units"]
    assert "not probability" in model.diagnostics["count_mode_units"]


def test_count_cancellation_zero_still_has_evidence(table_components):
    tables = [[1, -2, 0, NAN, NAN], [-1, 1, 0, -3, NAN]]
    model = EvidenceBlendModel(table_components(tables), mode="count")
    model.fit(empty_context([100, 200, 300, 400, 500]))
    assert_array_equal(model.score(0, np.arange(5)), [0, -1, 0, -3, NAN])


@pytest.mark.parametrize("mode", ["rank", "count"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_seeded_scalar_oracle_with_permutations_duplicates_and_cap(
    table_components, mode, reverse, seed,
):
    rng = np.random.default_rng(9201 + seed)
    ids = [53, 7, 2**63 - 1, 18, 49, 2, 301, 7009, 88]
    tables = rng.integers(-4, 5, size=(3, len(ids))).astype(float)
    tables[rng.random(tables.shape) < 0.35] = np.nan
    weights = [0.25, 1.0, 2.0]
    # Every item is queried; duplicates must neither consume depth nor alter rank.
    query = np.r_[rng.permutation(len(ids)), [3, 3, 0, 8, -1, 10**9]]
    query.setflags(write=False)
    before = query.copy()
    ctx = empty_context(ids)
    original = snapshot(ctx)
    model = EvidenceBlendModel(table_components(tables.tolist(), weights), mode=mode,
                               depth=3, reverse_score_ties=reverse).fit(ctx)
    expected = scalar_oracle(ids, tables, weights, query, mode=mode, depth=3,
                             reverse=reverse)
    assert_allclose(model.score(0, query), expected, rtol=1e-13, atol=1e-15)
    plain = model.score(0, np.arange(len(ids)))
    assert_allclose(model.score(0, query[:-2]), plain[query[:-2]], equal_nan=True)
    assert_array_equal(query, before)
    assert not query.flags.writeable
    assert snapshot(ctx) == original


@pytest.mark.parametrize("reverse", [False, True])
def test_default_depth_200_is_per_component_and_tie_sensitive(table_components, reverse):
    ids = np.arange(1000, 1205, dtype=np.int64)[::-1]
    table = np.ones(len(ids))
    model = EvidenceBlendModel(table_components([table.tolist()]),
                               reverse_score_ties=reverse).fit(empty_context(ids))
    query = np.arange(len(ids))
    scores = model.score(0, query)
    assert np.isfinite(scores).sum() == 200
    kept = set(ids[np.isfinite(scores)].tolist())
    assert kept == (set(range(1005, 1205)) if reverse else set(range(1000, 1200)))
    assert_allclose(scores, scalar_oracle(ids, [table], [1], query, mode="rank",
                                          reverse=reverse), rtol=1e-14)
    # Duplicating cutoff-adjacent items must not move rank 201 into/out of the cap.
    repeated = np.r_[query[::-1], np.repeat(query[199], 20), query[200], query[200]]
    assert_array_equal(model.score(0, repeated), scores[repeated])
    assert model.diagnostics["rank_constant"] == 60
    assert model.diagnostics["depth"] == 200
    assert model.diagnostics["reverse_score_ties"] is reverse


def test_rank_cap_is_not_a_final_global_output_cap(table_components):
    first = [float(10 - i) if i < 4 else NAN for i in range(8)]
    second = [NAN if i < 4 else float(10 - i) for i in range(8)]
    model = EvidenceBlendModel(table_components([first, second]), depth=3)
    model.fit(empty_context(range(1, 9)))
    scores = model.score(0, np.arange(8))
    assert np.isfinite(scores).sum() == 6  # Union of two distinct top-three lists.
    assert np.isnan(scores[[3, 7]]).all()


def test_count_mode_ignores_rank_cap_and_reversal(table_components):
    table = [0, -3, 4, 2, 1]
    ctx = empty_context([5, 1, 9, 2, 8])
    for reverse in (False, True):
        model = EvidenceBlendModel(table_components([table]), mode="count", depth=1,
                                   reverse_score_ties=reverse).fit(ctx)
        assert_array_equal(model.score(0, np.arange(5)), table)


@pytest.mark.parametrize("mode", ["rank", "count"])
def test_rank_is_query_relative_so_candidate_chunking_is_not_equivalent(table_components, mode):
    model = EvidenceBlendModel(table_components([[9, 5, 1]]), mode=mode)
    model.fit(empty_context([10, 20, 30]))
    complete = model.score(0, np.arange(3))
    singleton = model.score(0, np.array([2]))
    if mode == "rank":
        assert complete[2] == pytest.approx(1 / 63)
        assert singleton[0] == pytest.approx(1 / 61)
    else:
        assert complete[2] == singleton[0] == 1


def test_reversing_component_ties_does_not_reverse_untied_or_final_equal_scores(table_components):
    ctx = empty_context([50, 10, 70])
    strict = table_components([[1, 3, -2], [2, -5, 4]])
    forward = EvidenceBlendModel(strict).fit(ctx).score(0, np.arange(3))
    reversed_scores = EvidenceBlendModel(strict, reverse_score_ties=True).fit(ctx).score(0, np.arange(3))
    assert_array_equal(forward, reversed_scores)
    # Opposed strict component orders cancel to a final tie in either direction.
    opposed = table_components([[2, 1, NAN], [1, 2, NAN]])
    for reverse in (False, True):
        model = EvidenceBlendModel(opposed, reverse_score_ties=reverse).fit(ctx)
        scores = model.score(0, np.arange(3))
        assert scores[0] == scores[1]
        assert evidence_order(scores, ctx.data.item_ids) == [1, 0]


@pytest.mark.parametrize("mode", ["rank", "count"])
def test_zero_weight_component_is_neither_imported_fitted_nor_voted(table_components, mode):
    cfg = table_components([[NAN, -2, 0]])
    cfg.append({"name": "disabled", "class": "research_v2.does_not_exist:NoFit",
                "parameters": {"must_not_be_read": True}, "weight": 0})
    model = EvidenceBlendModel(cfg, mode=mode).fit(empty_context([3, 2, 1]))
    scores = model.score(0, np.arange(3))
    assert np.isnan(scores[0])
    assert_allclose(scores[1:], [1 / 62, 1 / 61] if mode == "rank" else [-2, 0])
    assert set(model._models) == set(model._weights) == {"part-0"}
    assert set(model.diagnostics["fits"]) == {"part-0"}


@pytest.mark.parametrize("mode", ["rank", "count"])
def test_invalid_and_empty_queries_do_not_invoke_components(table_components, mode):
    cfg = table_components([[3, NAN, -1]])
    cfg[0]["parameters"]["forbid_score"] = True
    model = EvidenceBlendModel(cfg, mode=mode).fit(empty_context([8, 4, 19]))
    assert model.score(0, np.array([], dtype=float)).shape == (0,)
    assert np.isnan(model.score(0, np.array([-1, 10**9]))).all()
    assert np.isnan(model.score(0, np.array([2**64 - 1], dtype=np.uint64))).all()
    for user in (-1, 2, 2**100):
        assert np.isnan(model.score(user, np.arange(3))).all()
    for user in (True, np.bool_(False), 0.0, "0", None):
        with pytest.raises(ValueError):
            model.score(user, np.arange(3))
    for query in (np.array(0), [[0]], [1.0], [True]):
        with pytest.raises(ValueError):
            model.score(0, query)


@pytest.mark.parametrize("mode", ["rank", "count"])
def test_malformed_component_scores_are_rejected(table_components, mode):
    cfg = table_components([[1, 2]])
    cfg[0]["parameters"]["bad_shape"] = True
    with pytest.raises(ValueError, match="shape/values"):
        EvidenceBlendModel(cfg, mode=mode).fit(empty_context([1, 2])).score(0, np.arange(2))
    cfg = table_components([[1, np.inf]])
    with pytest.raises(ValueError, match="shape/values"):
        EvidenceBlendModel(cfg, mode=mode).fit(empty_context([1, 2])).score(0, np.arange(2))


@pytest.mark.parametrize("mode", ["rank", "count"])
def test_original_configuration_and_read_only_component_outputs_are_not_mutated(table_components, mode):
    cfg = table_components([[5, NAN, -3]], [0.5])
    model = EvidenceBlendModel(cfg, mode=mode)
    cfg[0]["weight"] = 123
    cfg[0]["parameters"]["values"][0] = 1000
    cfg[0]["class"] = "research_v2.no_such_module:NoSuchClass"
    ctx = empty_context([100, 2, 7])
    before = snapshot(ctx)
    model.fit(ctx)
    underlying = model._models["part-0"].table.copy()
    expected = [0.5 / 61, NAN, 0.5 / 62] if mode == "rank" else [2.5, NAN, -1.5]
    assert_allclose(model.score(0, np.arange(3)), expected)
    assert_array_equal(model._models["part-0"].table, underlying)
    assert not model._models["part-0"].table.flags.writeable
    assert snapshot(ctx) == before
    assert model.diagnostics["components"][0]["weight"] == 0.5


@pytest.mark.parametrize("mode", ["rank", "count"])
def test_positive_component_rescaling_count_tradeoff_but_rank_invariance(table_components, mode):
    ctx = empty_context([20, 10])
    unscaled = EvidenceBlendModel(table_components([[100, NAN], [NAN, 1]]), mode=mode).fit(ctx)
    scaled = EvidenceBlendModel(table_components([[100, NAN], [NAN, 101]]), mode=mode).fit(ctx)
    first, second = unscaled.score(0, np.arange(2)), scaled.score(0, np.arange(2))
    if mode == "rank":
        assert_array_equal(first, [1 / 61, 1 / 61])
        assert_array_equal(first, second)
    else:
        assert_array_equal(first, [100, 1])
        assert_array_equal(second, [100, 101])
        assert evidence_order(first, ctx.data.item_ids)[0] == 0
        assert evidence_order(second, ctx.data.item_ids)[0] == 1


def test_repeat_discovery_dominance_threshold_including_exact_id_tie(table_components):
    # Repeat fifth rank: w_R/65. Discovery first rank: w_D/61.
    ctx = empty_context([10, 20, 30, 40, 50, 1])
    tables = [[5, 4, 3, 2, 1, NAN], [NAN, NAN, NAN, NAN, NAN, 1]]
    for repeat_weight, fifth_above in ((64, False), (66, True)):
        model = EvidenceBlendModel(table_components(tables, [repeat_weight, 61])).fit(ctx)
        scores = model.score(0, np.arange(6))
        assert bool(scores[4] > scores[5]) is fifth_above
        assert (5 not in evidence_order(scores, ctx.data.item_ids)[:5]) is fifth_above
    boundary = EvidenceBlendModel(table_components(tables, [65, 61])).fit(ctx)
    scores = boundary.score(0, np.arange(6))
    assert scores[4] == scores[5] == 1
    assert 5 in evidence_order(scores, ctx.data.item_ids)[:5]  # Smaller raw ID wins.


def test_one_repeat_plus_four_quarter_weight_votes_does_not_forbid_discovery(table_components):
    ctx = empty_context([10, 20, 30, 40, 50, 1])
    repeat = [5, 4, 3, 2, 1, NAN]
    novel = [NAN, NAN, NAN, NAN, NAN, 1]
    model = EvidenceBlendModel(table_components([repeat, novel, novel, novel, novel],
                                               [1, 0.25, 0.25, 0.25, 0.25])).fit(ctx)
    scores = model.score(0, np.arange(6))
    assert scores[5] == pytest.approx(1 / 61)
    assert scores[5] > scores[4] == pytest.approx(1 / 65)
    assert 5 in evidence_order(scores, ctx.data.item_ids)[:5]


def prepared_fixture():
    def airing(name, item, start, channel="007"):
        return {"airing_id": name, "tv_show_id": item, "channel_id": channel,
                "start": start, "end": start + 100, "quarantined": False,
                "metadata": {"tv_show_title": name}}

    past = [airing("a-old", 101, TRAIN_END - 6 * DAY),
            airing("a-recent", 101, TRAIN_END - 2 * DAY),
            airing("b", 307, TRAIN_END - DAY)]
    past += [airing(f"popular-{i}", 9007, TRAIN_END - DAY + 500 * i + 200)
             for i in range(4)]
    future = [airing(f"future-{i}", item, TRAIN_END + 500 * i)
              for i, item in enumerate((101, 307, 9007, 60001, 18))]

    def viewing(row, user):
        return {key: row[key] for key in ("airing_id", "tv_show_id", "start", "end")} | {
            "user_id": user, "coverage": 0.8,
        }

    events = [viewing(row, "alice") for row in past[:3]]
    events += [viewing(row, "other") for row in past[3:]]
    events += [viewing(future[3], "alice"), viewing(future[2], "cold")]
    return PreparedData(["alice", "cold", "other"], past + future, events, {})


def production_components():
    return [
        {"name": "repeat", "class": "research_v2.temporal:TemporalRepeatModel",
         "parameters": {"half_life_days": None}, "weight": 1.0},
        {"name": "prior-ablation", "class": "research_v2.priors:AiringPriorRepeatModel",
         "parameters": {"half_life_days": None, "prior_power": 0}, "weight": 0.5},
    ]


def fit_context(prepared):
    return make_context(prepared, TRAIN_END, TRAIN_END, FORECAST_END,
                        history_start=HISTORY_START)


@pytest.mark.parametrize("mode", ["rank", "count"])
def test_real_local_components_have_no_popularity_votes_and_no_input_mutation(mode):
    ctx, _, _ = fit_context(prepared_fixture())
    original = snapshot(ctx)
    model = EvidenceBlendModel(production_components(), mode=mode).fit(ctx)
    alice, cold = (ctx.data.user_ids.index(user) for user in ("alice", "cold"))
    query = ctx.candidate_indices
    raw = ctx.data.item_ids[query]
    table = [2 if i == 101 else 1 if i == 307 else NAN for i in raw]
    expected = scalar_oracle(raw, [table, table], [1, 0.5], np.arange(len(raw)), mode=mode)
    assert_allclose(model.score(alice, query), expected, rtol=1e-14)
    assert np.isnan(model.score(cold, query)).all()
    assert ctx.data.interactions.sum(axis=0).max() == 4  # Popular item 9007 is unsupported for Alice.
    assert np.isnan(model.score(alice, query)[raw.tolist().index(9007)])
    assert snapshot(ctx) == original
    assert model.diagnostics["internal_backfill"] is False
    expected_sources = {}
    for cls in (TemporalRepeatModel, AiringPriorRepeatModel):
        path = Path(inspect.getfile(cls))
        expected_sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert model.diagnostics["sources"] == expected_sources


@pytest.mark.parametrize("mode", ["rank", "count"])
def test_actual_fit_and_scores_invariant_to_future_behavior_with_fixed_epg(mode):
    original = prepared_fixture()
    changed = deepcopy(original)
    changed.events = [event for event in changed.events if event["end"] <= TRAIN_END]
    future = next(row for row in changed.schedules if row["airing_id"] == "future-0")
    changed.events.append({key: future[key] for key in ("airing_id", "tv_show_id", "start", "end")} | {
        "user_id": "heldout-only-not-a-training-user", "coverage": 1.0,
    })
    first, first_labels, _ = fit_context(original)
    second, second_labels, _ = fit_context(changed)
    assert first_labels != second_labels  # The mutation really changes synthetic evaluation labels.
    assert snapshot(first) == snapshot(second)
    one = EvidenceBlendModel(production_components(), mode=mode).fit(first)
    two = EvidenceBlendModel(production_components(), mode=mode).fit(second)
    for user in range(len(first.data.user_ids)):
        assert_array_equal(one.score(user, first.candidate_indices),
                           two.score(user, second.candidate_indices))


def test_generic_real_ease_component_count_blend_has_independent_small_system_oracle():
    ctx, _, _ = fit_context(prepared_fixture())
    cfg = production_components()[:1]
    cfg.append({"name": "ease", "class": "research_v2.collaborative:EASEModel",
                "parameters": {"regularization": 2.0, "num_threads": 1}, "weight": 0.5})
    before = snapshot(ctx)
    model = EvidenceBlendModel(cfg, mode="count").fit(ctx)
    # Alice's features are (2,1,0); other's are (0,0,4). For the first two
    # columns G+2I=[[6,2],[2,3]], so B_01=1/3, B_10=2/3. Her reconstruction
    # is (2/3,2/3,0), with no coefficient evidence for the isolated third item.
    expected = [2 + 1 / 3 if raw == 101 else 1 + 1 / 3 if raw == 307 else NAN
                for raw in ctx.data.item_ids[ctx.candidate_indices]]
    assert_allclose(model.score(ctx.data.user_ids.index("alice"), ctx.candidate_indices),
                    expected, rtol=1e-13)
    assert snapshot(ctx) == before
    path = Path(inspect.getfile(EASEModel))
    assert model.diagnostics["sources"][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_successful_refit_replaces_state_and_failed_refit_preserves_old_fit(table_components):
    cfg = table_components([[3, 1, NAN], [2, NAN, -1]], [0.5, 1.0])
    cfg[1]["parameters"]["fail_at_train_end"] = TRAIN_END + DAY
    ctx = empty_context([9, 5, 100])
    model = EvidenceBlendModel(cfg, mode="count").fit(ctx)
    expected = [3.5, 0.5, -1]
    assert_array_equal(model.score(0, np.arange(3)), expected)
    previous_diagnostics = deepcopy(model.diagnostics)
    changed = replace(ctx, train_end=TRAIN_END + DAY, forecast_start=TRAIN_END + DAY)
    with pytest.raises(ValueError, match="synthetic constituent fit rejected"):
        model.fit(changed)
    assert_array_equal(model.score(0, np.arange(3)), expected)
    assert model.diagnostics == previous_diagnostics
    previous_objects = tuple(model._models.values())
    model.fit(ctx)
    assert all(new is not old for new, old in zip(model._models.values(), previous_objects, strict=True))
    assert_array_equal(model.score(0, np.arange(3)), expected)


def test_source_changed_during_fit_is_rejected_without_writing_any_source(table_components, monkeypatch):
    path = Path(inspect.getfile(FusionChallengeTableModel)).resolve()
    real_read_bytes = Path.read_bytes
    reads = 0

    def changing_read_bytes(self):
        nonlocal reads
        data = real_read_bytes(self)
        if self.resolve() == path:
            reads += 1
            if reads > 1:
                return data + b"\n# simulated concurrent source edit; disk remains untouched\n"
        return data

    model = EvidenceBlendModel(table_components([[1, 2]]))
    monkeypatch.setattr(Path, "read_bytes", changing_read_bytes)
    with pytest.raises(RuntimeError, match="source changed during fit"):
        model.fit(empty_context([10, 20]))
    assert model._models is None
    assert reads == 2


def test_opposite_count_overflows_must_not_masquerade_as_missing_evidence(table_components):
    # All inputs/weights are finite. Exact arithmetic cancels to supported zero;
    # explicitly rejecting unrepresentable intermediate products is also safe.
    # Current source instead silently returns NaN, which means 'no evidence'.
    tables, weights = [[2], [-2]], [1e308, 1e308]
    exact = sum(Fraction(str(w)) * Fraction(table[0])
                for table, w in zip(tables, weights, strict=True))
    assert exact == 0
    model = EvidenceBlendModel(table_components(tables, weights), mode="count")
    model.fit(empty_context([10]))
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            result = model.score(0, np.array([0]))
        except FloatingPointError:
            return  # Explicit overflow rejection is safe too.
    assert np.isfinite(result).all(), f"supported cancellation returned {result.tolist()}"
    assert_array_equal(result, [0])


# Follow-up challenge: FUS-1 author fix and user-axis-only batch implementation.
# Earlier cases above are retained; these additions do not change parent tests.


class FusionChallengeBatchTableModel(FusionChallengeTableModel):
    def __init__(self, values, bad_batch_shape=False, **kwargs):
        super().__init__(values, **kwargs)
        self.bad_batch_shape = bad_batch_shape

    def fit(self, context):
        super().fit(context)
        self.batch_calls = []
        self.scalar_calls = []
        return self

    def score(self, user, candidates):
        self.scalar_calls.append((user, candidates.copy()))
        return super().score(user, candidates)

    def score_batch(self, users, candidates):
        if self.forbid_score:
            raise AssertionError("an empty/invalid query must not reach a component")
        self.batch_calls.append((users.copy(), candidates.copy(),
                                 users.flags.writeable, candidates.flags.writeable))
        result = self.table[np.ix_(users, candidates)].copy()
        if self.bad_batch_shape:
            result = result[:, :0]
        result.setflags(write=False)
        return result


@pytest.fixture
def batch_components(monkeypatch, table_components):
    monkeypatch.setattr(prior_module, "FusionChallengeBatchTableModel",
                        FusionChallengeBatchTableModel, raising=False)

    def configure(tables, weights=None, native=(0,)):
        cfg = table_components(tables, weights)
        for i in native:
            cfg[i]["class"] = "research_v2.priors:FusionChallengeBatchTableModel"
        return cfg

    return configure


def independent_batch_oracle(ids, component_tables, weights, users, candidates,
                             *, mode, reverse=False, depth=200):
    rows = []
    n_users = len(component_tables[0])
    for user in users:
        if 0 <= int(user) < n_users:
            rows.append(scalar_oracle(ids, [table[int(user)] for table in component_tables],
                                      weights, candidates, mode=mode, reverse=reverse,
                                      depth=depth))
        else:
            rows.append(np.full(len(candidates), np.nan))
    return np.array(rows, dtype=float).reshape(len(users), len(candidates))


@pytest.mark.parametrize("mode", ["rank", "count"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_batch_has_independent_oracle_for_native_and_scalar_only_components(
    batch_components, mode, reverse, seed,
):
    rng = np.random.default_rng(7700 + seed)
    ids = [53, 7, 2**63 - 1, 18, 49, 2, 301, 7009, 88]
    tables = rng.integers(-5, 6, size=(3, 2, len(ids))).astype(float)
    tables[rng.random(tables.shape) < 0.3] = np.nan
    tables[:, :, -1] = np.nan  # No fallback may create evidence for this column.
    weights = [1, 0.25, 0.5]
    cfg = batch_components(tables.tolist(), weights, native=(0, 2))
    cfg.append({"name": "disabled", "weight": 0,
                "class": "research_v2.no_such_batch_component:NoImport"})
    ctx = empty_context(ids)
    before = snapshot(ctx)
    users = np.array([1, 0, 1, -1, np.int64(0), 2**100, np.uint64(2**64 - 1)],
                     dtype=object)
    query = np.r_[rng.permutation(len(ids)), [3, 0, 3, -1, 10**9]]
    users.setflags(write=False)
    query.setflags(write=False)
    saved_users, saved_query = users.copy(), query.copy()
    model = EvidenceBlendModel(cfg, mode=mode, reverse_score_ties=reverse).fit(ctx)
    actual = model.score_batch(users, query)
    expected = independent_batch_oracle(ids, tables, weights, users, query,
                                       mode=mode, reverse=reverse)
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    selected = query[(query >= 0) & (query < len(ids))]
    for name in ("part-0", "part-2"):
        component = model._models[name]
        assert len(component.batch_calls) == 1
        assert component.scalar_calls == []
        sent_users, sent_candidates, users_writeable, candidates_writeable = component.batch_calls[0]
        assert_array_equal(sent_users, [1, 0, 1, 0])
        assert_array_equal(sent_candidates, selected)  # Full vector, including duplicate positions.
        assert not users_writeable and not candidates_writeable
    scalar = np.vstack([model.score(user, query) for user in users])
    assert_allclose(actual, scalar, rtol=1e-13, atol=1e-15)
    # User partitioning is safe; no candidate partitioning occurs in any call.
    partitioned = np.vstack([model.score_batch(users[:2], query),
                             model.score_batch(users[2:], query)])
    assert_allclose(actual, partitioned, rtol=1e-13, atol=1e-15)
    for name in ("part-0", "part-2"):
        for _, sent_candidates, _, _ in model._models[name].batch_calls:
            assert_array_equal(sent_candidates, selected)
    assert_array_equal(users, saved_users)
    assert_array_equal(query, saved_query)
    assert not users.flags.writeable and not query.flags.writeable
    assert snapshot(ctx) == before
    # Results must not alias a model cache or one another.
    actual[:] = 999
    assert_allclose(model.score_batch(users, query), expected, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("reverse", [False, True])
def test_batch_full_candidate_rrf_keeps_global_depth_and_duplicate_ties(batch_components, reverse):
    ids = np.arange(1000, 1205, dtype=np.int64)[::-1]
    tables = np.ones((1, 2, len(ids)))
    model = EvidenceBlendModel(batch_components(tables.tolist()),
                               reverse_score_ties=reverse).fit(empty_context(ids))
    query = np.r_[np.arange(len(ids)), [199, 200, 199, -1]]
    users = [1, 0, 1]
    actual = model.score_batch(users, query)
    expected = independent_batch_oracle(ids, tables, [1], users, query,
                                       mode="rank", reverse=reverse)
    assert_allclose(actual, expected, rtol=1e-14)
    for row in actual:
        first = row[:len(ids)]
        assert np.isfinite(first).sum() == 200
        kept = set(ids[np.isfinite(first)].tolist())
        assert kept == (set(range(1005, 1205)) if reverse else set(range(1000, 1200)))
    native = model._models["part-0"]
    assert len(native.batch_calls) == 1
    assert_array_equal(native.batch_calls[0][1], query[:-1])
    # The candidate at full-list rank 201 is unsupported in the full request,
    # but querying it alone gives rank-one evidence. User batching must not do that.
    omitted = next(i for i, value in enumerate(actual[0, :len(ids)]) if np.isnan(value))
    assert model.score_batch([1], [omitted])[0, 0] == pytest.approx(1 / 61)


@pytest.mark.parametrize("mode", ["rank", "count"])
def test_batch_empty_unknown_and_uint64_queries_do_not_invoke_components(batch_components, mode):
    cfg = batch_components([[3, NAN, -1]])
    cfg[0]["parameters"]["forbid_score"] = True
    model = EvidenceBlendModel(cfg, mode=mode).fit(empty_context([8, 4, 19]))
    assert model.score_batch([], [0, 1]).shape == (0, 2)
    assert model.score_batch([0, 1], []).shape == (2, 0)
    assert model.score_batch([], []).shape == (0, 0)
    assert np.isnan(model.score_batch([-1, 2, 2**100], [0, 1, 2])).all()
    assert np.isnan(model.score_batch([0, 1], [-1, 10**9])).all()
    huge = np.array([2**64 - 1], dtype=np.uint64)
    assert np.isnan(model.score_batch(huge, [0])).all()
    assert np.isnan(model.score_batch([0], huge)).all()
    assert model._models["part-0"].batch_calls == []


@pytest.mark.parametrize("bad_users", [0, [[0]], [True, 0], [np.bool_(False)],
                                        [0.0], ["0"], [None], np.array(0)])
def test_batch_rejects_invalid_user_representations(batch_components, bad_users):
    model = EvidenceBlendModel(batch_components([[1, NAN]])).fit(empty_context([10, 20]))
    with pytest.raises(ValueError, match="one-dimensional integer-index"):
        model.score_batch(bad_users, [0, 1])


@pytest.mark.parametrize("bad_candidates", [np.array(0), [[0]], [True], [1.0], ["1"]])
def test_batch_rejects_invalid_candidate_representations(batch_components, bad_candidates):
    model = EvidenceBlendModel(batch_components([[1, NAN]])).fit(empty_context([10, 20]))
    with pytest.raises(ValueError, match="one-dimensional integer"):
        model.score_batch([0], bad_candidates)


def test_batch_requires_fit_and_enforces_one_million_requested_cells(batch_components):
    cfg = batch_components([[1, 2]])
    cfg[0]["parameters"]["forbid_score"] = True
    model = EvidenceBlendModel(cfg)
    with pytest.raises(RuntimeError, match="before scoring"):
        model.score_batch([], [])
    model.fit(empty_context([10, 20]))
    # Exactly one million requested cells is allowed even when all users are
    # unknown. More is rejected independently of the number of supported cells.
    exact = model.score_batch([-1] * 1000, np.full(1000, -1, dtype=int))
    assert exact.shape == (1000, 1000) and np.isnan(exact).all()
    assert exact.dtype == np.float64
    for users, candidates in (([-1] * 1001, np.full(1000, -1, dtype=int)),
                              ([-1] * 1000, np.full(1001, -1, dtype=int))):
        with pytest.raises(ValueError, match="one million requested cells"):
            model.score_batch(users, candidates)
    assert model._models["part-0"].batch_calls == []


@pytest.mark.parametrize("mode", ["rank", "count"])
def test_batch_rejects_wrong_native_dimensions_and_infinite_evidence(batch_components, mode):
    cfg = batch_components([[1, 2]])
    cfg[0]["parameters"]["bad_batch_shape"] = True
    with pytest.raises(ValueError, match="batch dimensions"):
        EvidenceBlendModel(cfg, mode=mode).fit(empty_context([10, 20])).score_batch([0, 1], [0, 1])
    model = EvidenceBlendModel(batch_components([[[1, 2], [NAN, np.inf]]]), mode=mode)
    model.fit(empty_context([10, 20]))
    with pytest.raises(ValueError, match="shape/values"):
        model.score_batch([0, 1], [0, 1])


def test_fus1_closure_scalar_and_batch_raise_but_missing_rows_remain_nan(batch_components):
    model = EvidenceBlendModel(batch_components([[2], [-2]], [1e308, 1e308]), mode="count")
    model.fit(empty_context([10]))
    with pytest.raises(FloatingPointError):
        model.score(0, [0])
    with pytest.raises(FloatingPointError):
        model.score_batch([0, 1, 0], [0, 0, -1])
    assert np.isnan(model.score_batch([1, -1], [0, 0, -1])).all()


@pytest.mark.parametrize("mode", ["rank", "count"])
def test_batch_actual_context_invariance_to_future_behavior(mode):
    original = prepared_fixture()
    changed = deepcopy(original)
    changed.events = [event for event in changed.events if event["end"] <= TRAIN_END]
    first, first_labels, _ = fit_context(original)
    second, second_labels, _ = fit_context(changed)
    assert first_labels != second_labels
    assert snapshot(first) == snapshot(second)
    before = snapshot(first)
    one = EvidenceBlendModel(production_components(), mode=mode).fit(first)
    two = EvidenceBlendModel(production_components(), mode=mode).fit(second)
    users = np.array([2, 0, 1, 2, -1, 0])
    users.setflags(write=False)
    assert_array_equal(one.score_batch(users, first.candidate_indices),
                       two.score_batch(users, second.candidate_indices))
    assert snapshot(first) == before


@pytest.mark.parametrize("mode", ["rank", "count"])
def test_real_legacy_ppr_batch_path_scalar_parity_and_nested_independent_rrf_oracle(mode):
    ctx, _, _ = fit_context(prepared_fixture())
    cfg = production_components()[:1]
    cfg.append({"name": "legacy", "class": "research_v2.legacy:LegacyHybrid", "weight": 0.5})
    before = snapshot(ctx)
    model = EvidenceBlendModel(cfg, mode=mode).fit(ctx)
    users = [ctx.data.user_ids.index("other"), ctx.data.user_ids.index("alice"),
             ctx.data.user_ids.index("cold"), ctx.data.user_ids.index("other"), -1, 2**100]
    query = ctx.candidate_indices
    actual = model.score_batch(users, query)
    legacy = model._models["legacy"]
    ppr_query = legacy.models["ppr"].diagnostics["last_query"]
    assert ppr_query["requested_users"] == 4  # Unknown rows were removed by the fusion.
    assert ppr_query["max_batch_users"] == 2  # Both distinct active users propagated together.
    assert ppr_query["batches"] == 1
    assert ppr_query["requested_candidates"] == len(query)
    expected_rows = []
    all_items = np.arange(len(ctx.data.item_ids))
    for user in users:
        if not 0 <= user < len(ctx.data.user_ids):
            expected_rows.append(np.full(len(query), np.nan))
            continue
        # Inner legacy fusion is independently reconstructed from constituent
        # evidence, not from LegacyHybrid.score or the shared RRF helper.
        names = list(legacy.weights)
        tables = [legacy.models[name].score(user, all_items) for name in names]
        inner = scalar_oracle(ctx.data.item_ids, tables,
                              [legacy.weights[name] for name in names], all_items, mode="rank")
        repeat = []
        for raw in ctx.data.item_ids:
            count = sum(event["user_id"] == ctx.data.user_ids[user]
                        and event["tv_show_id"] == raw and event["end"] <= TRAIN_END
                        for event in prepared_fixture().events)
            repeat.append(count if count else NAN)
        expected_rows.append(scalar_oracle(ctx.data.item_ids, [repeat, inner], [1, 0.5],
                                           query, mode=mode))
    assert_allclose(actual, np.vstack(expected_rows), rtol=1e-13, atol=1e-15)
    scalar = np.vstack([model.score(user, query) for user in users])
    assert_array_equal(actual, scalar)
    split = np.vstack([model.score_batch(users[:2], query), model.score_batch(users[2:], query)])
    assert_array_equal(actual, split)
    assert snapshot(ctx) == before


def test_real_nested_legacy_duplicate_queries_do_not_consume_inner_rank_depth():
    ctx, _, _ = fit_context(prepared_fixture())
    cfg = production_components()[:1]
    cfg.append({"name": "legacy", "class": "research_v2.legacy:LegacyHybrid", "weight": 0.5})
    model = EvidenceBlendModel(cfg).fit(ctx)
    user = ctx.data.user_ids.index("alice")
    unique = np.arange(len(ctx.data.item_ids))
    warm = ctx.data.item_ids.tolist().index(101)
    repeated = np.r_[unique, np.repeat(warm, 200)]
    legacy = model._models["legacy"]
    names = list(legacy.weights)
    tables = [legacy.models[name].score(user, unique) for name in names]
    inner = scalar_oracle(ctx.data.item_ids, tables,
                          [legacy.weights[name] for name in names], unique, mode="rank")
    repeat = [2 if raw == 101 else 1 if raw == 307 else NAN for raw in ctx.data.item_ids]
    expected_unique = scalar_oracle(ctx.data.item_ids, [repeat, inner], [1, 0.5],
                                    unique, mode="rank")
    assert_allclose(model.score(user, unique), expected_unique, rtol=1e-13)
    expected_repeated = expected_unique[repeated]
    scalar = model.score(user, repeated)
    batched = model.score_batch([user, user], repeated)
    assert_array_equal(batched, np.vstack([scalar, scalar]))  # Both entry points share this defect.
    assert_allclose(scalar, expected_repeated, rtol=1e-13,
                    err_msg="duplicate positions consumed the nested LegacyHybrid depth budget")
