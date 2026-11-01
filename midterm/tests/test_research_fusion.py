import numpy as np
import pytest

from research_v2.fusion import EvidenceBlendModel
from test_research_priors import context


def components():
    return [
        {"name": "repeat", "class": "research_v2.priors:AiringPriorRepeatModel",
         "parameters": {"half_life_days": None, "prior_power": 0}, "weight": 1},
        {"name": "forecast", "class": "research_v2.priors:AiringPriorRepeatModel",
         "parameters": {"half_life_days": None, "prior_basis": "forecast"}, "weight": 0.5},
    ]


def test_evidence_rank_votes_use_no_popularity_fallback():
    model = EvidenceBlendModel(components()).fit(context())
    scores = model.score(0, np.arange(3))
    assert scores[0] == pytest.approx(1.5 / 61)
    assert scores[1] == pytest.approx(1.5 / 62)
    assert np.isnan(scores[2])
    assert np.isnan(model.score(1, np.arange(3))).all()


def test_count_sum_is_literal_count_proxy_not_rank_sum():
    model = EvidenceBlendModel(components(), mode="count").fit(context())
    np.testing.assert_allclose(model.score(0, np.array([1, 0])),
                               [1 + 0.5 * np.log(2), 1 + 0.5 * np.log(3)])


@pytest.mark.parametrize("mode", ["count", "rank"])
def test_duplicate_queries_preserve_scores_and_not_rank_depth(mode):
    model = EvidenceBlendModel(components(), mode=mode).fit(context())
    plain = model.score(0, np.arange(3))
    query = np.array([1, 0, 1, 0, 2])
    query.setflags(write=False)
    np.testing.assert_equal(model.score(0, query), plain[query])
    assert np.isnan(model.score(-1, query)).all()
    assert np.isnan(model.score(0, np.array([-1, 10**9]))).all()


def test_fit_does_not_read_labels_and_preserves_original_counts():
    ctx = context()
    original = ctx.data.interactions.copy()
    model = EvidenceBlendModel(components()).fit(ctx)
    assert (ctx.data.interactions != original).nnz == 0
    assert model.diagnostics["internal_backfill"] is False


def test_reverse_score_ties_is_explicit_not_best_direction_selection():
    cfg = components()[:1]
    forward = EvidenceBlendModel(cfg).fit(context()).score(0, np.arange(3))
    backward = EvidenceBlendModel(cfg, reverse_score_ties=True).fit(context()).score(0, np.arange(3))
    np.testing.assert_equal(forward[:2], backward[:2][::-1])


def test_zero_weight_components_receive_no_votes_or_fit():
    cfg = components()
    cfg[1]["weight"] = 0
    cfg[1]["class"] = "research_v2.nonexistent:NoFit"
    model = EvidenceBlendModel(cfg).fit(context())
    assert set(model._models) == {"repeat"}


def test_not_fitted_or_invalid_weight():
    with pytest.raises(RuntimeError):
        EvidenceBlendModel(components()).score(0, np.arange(3))
    for value in (-1, np.inf, np.nan, True):
        cfg = components()
        cfg[0]["weight"] = value
        with pytest.raises(ValueError):
            EvidenceBlendModel(cfg)


@pytest.mark.parametrize("mode", ["rank", "count"])
def test_user_batch_matches_scalar_full_catalog_calls(mode):
    model = EvidenceBlendModel(components(), mode=mode).fit(context())
    candidates = np.array([2, 0, 1, 0, -1, 999])
    users = [0, 1, 0, -1, 2**100]
    expected = np.vstack([model.score(u, candidates) for u in users])
    np.testing.assert_equal(model.score_batch(users, candidates), expected)
    assert model.score_batch([], candidates).shape == (0, len(candidates))
    assert model.score_batch([0], []).shape == (1, 0)
    with pytest.raises(ValueError):
        model.score_batch([True, 0], candidates)
    with pytest.raises(ValueError):
        model.score_batch([0] * 200_000, candidates)
