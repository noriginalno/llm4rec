import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from research_v2.contracts import Airing, TemporalContext
from research_v2.priors import AiringPriorRepeatModel, PUBLIC_HALF_LIFE_DAYS
from sweettv.contracts import TrainingData


def context():
    return TemporalContext(
        data=TrainingData(["u", "cold"], np.array([10, 20, 30]), csr_matrix([[1, 1, 0], [0, 0, 0]]), [{}, {}, {}]),
        candidate_indices=np.arange(3), history_start=0, train_end=100,
        forecast_start=100, forecast_end=200,
        event_users=np.array([0, 0]), event_items=np.array([0, 1]),
        event_starts=np.array([10, 20]), event_ends=np.array([20, 30]), event_channels=np.array(["a", "b"]),
        historical_schedules=(Airing(0, "a", 10, 20), Airing(1, "b", 20, 30), Airing(1, "b", 40, 50)),
        forecast_schedules=(Airing(0, "a", 110, 120), Airing(0, "a", 130, 140), Airing(1, "b", 150, 160)),
    )


def test_public_decay_equivalence():
    for age in (0, 30, 60, 120):
        assert 2 ** (-age / PUBLIC_HALF_LIFE_DAYS) == pytest.approx(0.8 ** (age / 60))


def test_history_prior_uses_known_airings_not_future_counts():
    model = AiringPriorRepeatModel(half_life_days=None).fit(context())
    np.testing.assert_allclose(model.score(0, np.array([1, 0])), [math.log(3), math.log(2)])


def test_forecast_prior_is_an_explicit_metadata_only_extension():
    model = AiringPriorRepeatModel(half_life_days=None, prior_basis="forecast").fit(context())
    np.testing.assert_allclose(model.score(0, np.array([0, 1])), [math.log(3), math.log(2)])


def test_zero_prior_power_matches_repeat_and_preserves_query_order():
    model = AiringPriorRepeatModel(half_life_days=None, prior_power=0).fit(context())
    np.testing.assert_allclose(model.score(0, np.array([1, 0, 1])), [1, 1, 1])
    assert np.isnan(model.score(0, np.array([2]))).all()


def test_log10_source_variant_really_zeros_single_airing_prior():
    model = AiringPriorRepeatModel(half_life_days=None, prior_transform="log10").fit(context())
    scores = model.score(0, np.arange(3))
    assert np.isnan(scores[0]) and scores[1] == pytest.approx(math.log10(2))


def test_cold_user_invalid_indices_and_input_immutability():
    ctx = context()
    original = ctx.data.interactions.copy()
    model = AiringPriorRepeatModel(half_life_days=None).fit(ctx)
    assert np.isnan(model.score(1, np.arange(3))).all()
    assert np.isnan(model.score(10**30, np.arange(3))).all()
    assert np.isnan(model.score(0, np.array([-1, 999]))).all()
    assert (ctx.data.interactions != original).nnz == 0
