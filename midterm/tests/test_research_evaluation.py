import itertools

import pytest

from research_v2.evaluate import tie_marginalized_ap
from sweettv.evaluation import average_precision_at_k


def test_tie_expected_ap_matches_literal_two_outcomes():
    # Truth always includes1; exactly one of2/3 is selected. For[2,1], AP2 is
    # 1 if2 is selected and1/4 otherwise: expected5/8, not independent hits.
    assert tie_marginalized_ap([2, 1], {1: 2, 2: 1, 3: 1}, 2) == pytest.approx(5 / 8)


def test_tie_expected_matches_exhaustive_without_replacement_oracle():
    counts = {1: 10, 2: 10, 3: 3, 4: 3, 5: 3, 6: 3, 7: 1}
    prediction = [3, 7, 4, 1, 6]
    outcomes = [[1, 2, *choice] for choice in itertools.combinations([3, 4, 5, 6], 3)]
    expected = sum(average_precision_at_k(prediction, truth) for truth in outcomes) / len(outcomes)
    assert tie_marginalized_ap(prediction, counts) == pytest.approx(expected)


def test_no_boundary_tie_empty_or_short_labels():
    assert tie_marginalized_ap([1, 2, 3, 4, 5], {}) == 0
    assert tie_marginalized_ap([1, 9, 2], {1: 4, 2: 2}) == pytest.approx(5 / 6)
    assert tie_marginalized_ap([1, 2, 3, 4, 5], {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}) == 1


def test_duplicate_predictions_not_supported_by_sensitivity_shortcut():
    with pytest.raises(ValueError):
        tie_marginalized_ap([1, 1], {1: 3}, 2)
