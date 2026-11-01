"""Independent tiny basket/count oracles for the directed rule recommender."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sweettv.contracts import TrainingData
from sweettv.models.association import AssociationRulesModel


def make_data(item_ids, histories):
    """Histories map *matrix column* to positive qualifying frequency."""
    rows, columns, counts = [], [], []
    for user, history in enumerate(histories):
        for item, count in history.items():
            rows.append(user)
            columns.append(item)
            counts.append(count)
    matrix = csr_matrix(
        (counts, (rows, columns)), shape=(len(histories), len(item_ids)), dtype=np.float32
    )
    return TrainingData(
        user_ids=[f"user-{i}" for i in range(len(histories))],
        item_ids=np.array(item_ids, dtype=np.int64),
        interactions=matrix,
        metadata=[{"tv_show_title": f"Show {raw}"} for raw in item_ids],
    )


def tiny_data():
    # A=70, B=10, C=90; ID=20 is metadata-only. N=5, not six users.
    # A occurs in three baskets, B in two, A&B in two; counts do not
    # increment J repeatedly. A/B have unequal directed confidence/weight.
    return make_data(
        [70, 10, 90, 20],
        [{0: 4, 1: 1}, {0: 2, 1: 3}, {0: 1}, {2: 2}, {2: 1}, {}],
    )


def exhaustive_oracle(item_ids, histories, *, max_items, basket_cap, neighbors, min_joint, tau):
    """Pure Python direct scans; never call production counting/scoring helpers."""
    totals = {
        item: sum(history.get(item, 0) for history in histories)
        for item in range(len(item_ids))
    }
    selected = sorted(
        (item for item, frequency in totals.items() if frequency > 0),
        key=lambda item: (-totals[item], item_ids[item]),
    )[:max_items]
    baskets = []
    for history in histories:
        keep = sorted(
            (item for item in history if item in selected),
            key=lambda item: (-history[item], item_ids[item]),
        )[:basket_cap]
        baskets.append({item: history[item] for item in keep})
    n = sum(bool(history) for history in histories)
    supports = {a: sum(a in basket for basket in baskets) for a in range(len(item_ids))}
    pairs, rules = {}, {}
    for a in range(len(item_ids)):
        outgoing = []
        for b in range(len(item_ids)):
            joint = sum(a in basket and b in basket for basket in baskets)
            aa, ab = supports[a], supports[b]
            pb = ab / n if n else math.nan
            confidence = joint / aa if aa else math.nan
            q = (joint + tau * pb) / (aa + tau) if n and aa + tau else math.nan
            weight = max(0.0, q - pb) if math.isfinite(q) else 0.0
            pairs[a, b] = (aa, ab, joint, pb, confidence, q, weight)
            # Integer inequality prevents roundoff-only independence rules in
            # this separate, formula-based oracle as well.
            if a != b and joint >= min_joint and joint * n > aa * ab:
                outgoing.append((b, weight))
        rules[a] = dict(sorted(outgoing, key=lambda rule: (-rule[1], item_ids[rule[0]]))[:neighbors])
    scores = np.full((len(histories), len(item_ids)), np.nan)
    for user, basket in enumerate(baskets):
        for b in range(len(item_ids)):
            contributions = [
                math.log1p(frequency) * rules[a][b]
                for a, frequency in basket.items()
                if b in rules[a]
            ]
            if contributions:
                scores[user, b] = sum(contributions)
    return n, selected, baskets, pairs, rules, scores


def test_hand_derived_direction_support_and_shrinkage():
    model = AssociationRulesModel(min_joint_count=1).fit(tiny_data())
    forward = model.trace_rule(0, 1)
    reverse = model.trace_rule(1, 0)
    assert forward["antecedent_item_id"] == 70
    assert forward["consequent_item_id"] == 10
    assert forward["n_transactions"] == 5
    assert (forward["antecedent_count"], forward["consequent_count"], forward["joint_count"]) == (3, 2, 2)
    assert forward["support"] == reverse["support"] == pytest.approx(2 / 5)
    assert forward["confidence"] == pytest.approx(2 / 3)
    assert reverse["confidence"] == 1
    assert forward["base_probability"] == pytest.approx(2 / 5)
    assert forward["shrunk_probability"] == pytest.approx(6 / 13)
    assert forward["retained_weight"] == pytest.approx(4 / 65)
    assert reverse["shrunk_probability"] == pytest.approx(2 / 3)
    assert reverse["retained_weight"] == pytest.approx(1 / 15)
    assert forward["retained_weight"] != reverse["retained_weight"]
    # With three active items the default 128-row chunk is the complete tiny
    # matrix, only 72 bytes, not padded to the 4096-item safety limit.
    assert model.diagnostics["full_dense_count_matrix_allocated"]
    assert model.diagnostics["largest_dense_count_block_bytes"] == 3 * 3 * 8


def test_watched_consequents_stay_eligible_and_use_log_original_frequency():
    model = AssociationRulesModel(min_joint_count=1).fit(tiny_data())
    scores = model.score(0, np.array([1, 0, 1, 2, 3, 70, -1]))
    np.testing.assert_allclose(
        scores[:3], [math.log(5) * 4 / 65, math.log(2) / 15, math.log(5) * 4 / 65]
    )
    assert np.isnan(scores[3:]).all()
    # User 2 has only A: B is supported, but A cannot recommend itself.
    assert np.isnan(model.score(2, np.array([0]))[0])
    assert model.score(2, np.array([1]))[0] == pytest.approx(math.log(2) * 4 / 65)
    assert not model.trace_rule(0, 0)["retained"]
    assert not model.trace_rule(0, 0)["eligible_before_neighbor_cap"]


def test_n_keeps_users_whose_original_positive_history_is_empty_after_item_cap():
    data = make_data([30, 20, 10], [{0: 100, 1: 100}, {0: 1, 1: 1}, {2: 1}, {}])
    model = AssociationRulesModel(max_items=2, min_joint_count=1).fit(data)
    trace = model.trace_rule(0, 1)
    assert (trace["n_transactions"], trace["antecedent_count"], trace["joint_count"]) == (3, 2, 2)
    assert trace["support"] == pytest.approx(2 / 3)
    assert trace["retained_weight"] == pytest.approx(1 / 18)
    # Dropping the C-only user's transaction would make A/B independent and
    # remove this rule. The input-empty user, in contrast, is not in N.
    diagnostics = model.diagnostics
    assert diagnostics["n_positive_history_users_empty_after_item_cap"] == 1
    assert diagnostics["n_original_empty_users"] == 1
    assert diagnostics["n_zero_basket_antecedent_users"] == 2
    assert diagnostics["n_zero_retained_rule_antecedent_users"] == 2
    assert diagnostics["original_positive_interactions"] == 5
    assert diagnostics["interactions_dropped_by_item_cap"] == 1
    assert diagnostics["interactions_dropped_by_basket_cap"] == 0
    assert diagnostics["retained_qualifying_frequency"] == 202
    assert diagnostics["dropped_qualifying_frequency"] == 1
    assert diagnostics["qualifying_frequency_dropped_by_item_cap"] == 1
    assert diagnostics["qualifying_frequency_dropped_by_basket_cap"] == 0
    assert diagnostics["retained_positive_interaction_fraction"] == pytest.approx(4 / 5)
    assert model.user_basket(2) == []
    assert np.isnan(model.score(2, np.arange(3))).all()


def test_item_cap_uses_qualifying_frequency_not_basket_support_or_schedule_metadata():
    # ID=900 has only one supporting user but more qualifying viewings.
    data = make_data([900, 5, 42, 1], [{0: 100}, {1: 1}, {1: 1}, {2: 1}])
    data.metadata[3]["schedule_airings"] = 1_000_000
    model = AssociationRulesModel(max_items=1).fit(data)
    assert model.user_basket(0) == [{"item_index": 0, "item_id": 900, "frequency": 100.0}]
    assert model.user_basket(1) == []
    assert not model.trace_rule(3, 0)["antecedent_selected"]
    assert model.diagnostics["n_transactions"] == 4
    assert model.diagnostics["n_selected_items"] == 1
    assert np.isnan(model.score(0, np.arange(4))).all()


def test_equal_training_frequency_item_cap_uses_numeric_raw_id_not_column():
    data = make_data([100, 7, 50], [{0: 2}, {1: 2}, {2: 2}])
    model = AssociationRulesModel(max_items=2).fit(data)
    assert not model.trace_rule(0, 1)["antecedent_selected"]
    assert model.trace_rule(1, 2)["antecedent_selected"]
    assert model.trace_rule(1, 2)["consequent_selected"]


@pytest.mark.parametrize("huge", [2**54, float(np.finfo(np.float32).max)])
@pytest.mark.parametrize(
    "tails",
    [
        [{0: 1, 1: 4}, {0: 1}, {0: 1}, {0: 1}],  # Exact integral tie.
        [{1: 1}],  # Unequal totals that even fsum rounds to a float64 tie.
        [{1: 0.5}],
        [{1: float(np.nextafter(np.float32(0), np.float32(1)))}],
        [{0: 0.5, 1: 0.25}, {1: 0.25}],  # Exact fractional tie.
    ],
    ids=["integral-tie", "integer-gap", "fractional-gap", "subnormal-gap", "fractional-tie"],
)
def test_item_cap_uses_exact_represented_totals_under_user_and_column_permutations(huge, tails):
    rows = [{0: huge, 1: huge}, *tails]
    ids = [1, 99]
    for user_order in [list(range(len(rows))), list(reversed(range(len(rows))))]:
        for column_order in [[0, 1], [1, 0]]:
            inverse = {old: new for new, old in enumerate(column_order)}
            data = make_data(
                [ids[i] for i in column_order],
                [{inverse[i]: count for i, count in rows[user].items()} for user in user_order],
            )
            # Independent exact rational totals of the actual float32 values;
            # no production ranking helper or rounded aggregate is an oracle.
            totals = [Fraction(0), Fraction(0)]
            for item, count in zip(data.interactions.indices, data.interactions.data):
                totals[int(item)] += Fraction.from_float(float(count))
            expected_index = min(range(2), key=lambda i: (-totals[i], int(data.item_ids[i])))
            expected_raw_id = int(data.item_ids[expected_index])
            model = AssociationRulesModel(max_items=1, min_joint_count=1).fit(data)
            actual = {entry["item_id"] for user in range(len(rows)) for entry in model.user_basket(user)}
            assert actual == {expected_raw_id}
            assert model.diagnostics["frequency_aggregation"] == "python_integer_fixed_point"
            assert model.diagnostics["frequency_quantum_exponent"] == -149
            assert model.diagnostics["frequency_ranking_exact"]


def test_ordinary_integral_frequencies_keep_vectorized_exact_path():
    model = AssociationRulesModel(min_joint_count=1).fit(tiny_data())
    assert model.diagnostics["frequency_aggregation"] == "float64_exact_integer_bincount"
    assert model.diagnostics["frequency_ranking_exact"]
    assert model.trace_rule(0, 1)["retained_weight"] == pytest.approx(4 / 65)


def test_positive_fractional_counts_use_exact_ranking_without_changing_basket_supports():
    data = tiny_data()
    data.interactions.data *= np.float32(0.5)
    model = AssociationRulesModel(min_joint_count=1).fit(data)
    assert model.diagnostics["frequency_aggregation"] == "python_integer_fixed_point"
    trace = model.trace_rule(0, 1)
    assert (trace["n_transactions"], trace["antecedent_count"], trace["joint_count"]) == (5, 3, 2)
    assert trace["retained_weight"] == pytest.approx(4 / 65)
    assert model.score(0, np.array([1]))[0] == pytest.approx(math.log1p(2) * 4 / 65)


def test_bounded_basket_ties_and_column_permutation_are_deterministic():
    ids = [90, 10, 70, 20]
    histories = [{0: 1, 1: 1, 2: 1, 3: 1}, {0: 3}, {2: 3}, {}]
    model = AssociationRulesModel(max_basket_size=2, min_joint_count=1).fit(make_data(ids, histories))
    assert [entry["item_id"] for entry in model.user_basket(0)] == [10, 20]
    assert model.trace_rule(1, 3)["joint_count"] == 1
    assert model.trace_rule(0, 1)["joint_count"] == 0
    assert model.diagnostics["interactions_dropped_by_basket_cap"] == 2
    assert model.diagnostics["largest_basket_size"] == 2
    permutation = [2, 3, 0, 1]
    inverse = {old: new for new, old in enumerate(permutation)}
    permuted = [{inverse[item]: count for item, count in history.items()} for history in histories]
    other = AssociationRulesModel(max_basket_size=2, min_joint_count=1).fit(
        make_data([ids[i] for i in permutation], permuted)
    )
    assert [entry["item_id"] for entry in other.user_basket(0)] == [10, 20]
    for user in range(len(histories)):
        np.testing.assert_allclose(
            model.score(user, np.arange(4)),
            other.score(user, np.array([inverse[i] for i in range(4)])),
            equal_nan=True,
        )


def test_inference_uses_same_capped_basket_not_all_original_antecedents():
    data = make_data(
        [100, 200, 300, 400],
        [{0: 10, 1: 9, 2: 1}, {2: 2, 3: 2}, {2: 1, 3: 1}, {0: 1}, {1: 1}],
    )
    model = AssociationRulesModel(max_basket_size=2, min_joint_count=1).fit(data)
    assert model.trace_rule(2, 3)["retained"]
    assert [entry["item_id"] for entry in model.user_basket(0)] == [100, 200]
    assert np.isnan(model.score(0, np.array([3]))[0])
    assert np.isfinite(model.score(1, np.array([3]))[0])


def test_outgoing_rule_cap_breaks_equal_weight_ties_by_raw_consequent_id():
    data = make_data([500, 99, 4, 777], [{0: 1, 1: 1, 2: 1}, {0: 1}, {3: 1}, {3: 1}])
    model = AssociationRulesModel(max_neighbors=1, min_joint_count=1).fit(data)
    chosen, pruned = model.trace_rule(0, 2), model.trace_rule(0, 1)
    assert chosen["retained"]
    assert not pruned["retained"]
    assert pruned["eligible_before_neighbor_cap"]
    assert chosen["weight"] == pytest.approx(pruned["weight"])
    scores = model.score(1, np.array([1, 2, 2]))
    assert np.isnan(scores[0])
    assert scores[1] == scores[2] > 0


def test_joint_support_counts_each_pair_once_per_transaction_and_threshold_is_inclusive():
    data = make_data([8, 3, 42], [{0: 1000, 1: 999}, {0: 2, 1: 2}, {2: 1}])
    model = AssociationRulesModel(min_joint_count=2).fit(data)
    assert model.trace_rule(0, 1)["joint_count"] == 2
    assert model.trace_rule(0, 1)["retained"]
    assert not AssociationRulesModel(min_joint_count=3).fit(data).trace_rule(0, 1)["retained"]
    default = AssociationRulesModel().fit(data)
    assert default.min_joint_count == 5
    assert default.diagnostics["retained_rule_count"] == 0
    assert np.isnan(default.score(0, np.arange(3))).all()


def test_zero_smoothing_is_unshrunk_confidence_minus_baseline():
    model = AssociationRulesModel(min_joint_count=1, smoothing=0).fit(tiny_data())
    assert model.trace_rule(0, 1)["retained_weight"] == pytest.approx(2 / 3 - 2 / 5)
    assert model.trace_rule(1, 0)["retained_weight"] == pytest.approx(1 - 3 / 5)
    assert model.trace_rule(3, 1)["confidence"] != model.trace_rule(3, 1)["confidence"]
    assert not model.trace_rule(3, 1)["retained"]


def test_independence_and_no_cooccurrence_are_nan_not_popularity():
    data = make_data([10, 20, 30], [{0: 1, 1: 1}, {0: 2}, {1: 3}, {2: 100}])
    model = AssociationRulesModel(min_joint_count=1).fit(data)
    assert model.trace_rule(0, 1)["joint_count"] == 1
    assert model.trace_rule(0, 1)["shrunk_uplift"] == 0
    assert model.trace_rule(0, 2)["weight"] == 0
    assert model.diagnostics["retained_rule_count"] == 0
    assert model.diagnostics["n_zero_retained_rule_antecedent_users"] == 4
    for user in range(4):
        assert np.isnan(model.score(user, np.arange(3))).all()


@pytest.mark.parametrize("unknown", [-1, 6, 1000, None, "user-0", True])
def test_unknown_users_do_not_index_last_user_or_receive_fallback(unknown):
    model = AssociationRulesModel(min_joint_count=1).fit(tiny_data())
    assert np.isnan(model.score(unknown, np.array([1, 0, 1]))).all()
    assert model.user_basket(unknown) == []


def test_original_empty_user_and_unsupported_unsigned_candidates():
    model = AssociationRulesModel(min_joint_count=1).fit(tiny_data())
    assert np.isnan(model.score(5, np.arange(4))).all()
    candidates = np.array([1, np.iinfo(np.uint64).max, 3, 1], dtype=np.uint64)
    scores = model.score(np.int64(0), candidates)
    assert scores[0] == scores[3] > 0
    assert np.isnan(scores[1:3]).all()


@pytest.mark.parametrize("seed", [0, 12, 71])
def test_independent_exhaustive_directed_oracle(seed):
    rng = np.random.default_rng(seed)
    ids = [600, 2, 91, 7, 1003, 45, 9999]
    dense = rng.integers(0, 5, size=(13, 7))
    dense[rng.random(dense.shape) < 0.55] = 0
    dense[0] = 0
    dense[:, -1] = 0
    histories = [{i: int(value) for i, value in enumerate(row) if value} for row in dense]
    parameters = dict(max_items=4, basket_cap=2, neighbors=2, min_joint=1, tau=3.0)
    n, selected, baskets, pairs, rules, expected = exhaustive_oracle(ids, histories, **parameters)
    model = AssociationRulesModel(
        max_items=4,
        max_basket_size=2,
        max_neighbors=2,
        min_joint_count=1,
        smoothing=3,
        count_chunk_size=2,
    ).fit(make_data(ids, histories))
    assert model.diagnostics["n_transactions"] == n
    assert model.diagnostics["n_selected_items"] == len(selected)
    for user, basket in enumerate(baskets):
        traced = {entry["item_index"]: entry["frequency"] for entry in model.user_basket(user)}
        assert traced == basket
        candidates = np.array([5, 2, 0, 2, 6, -1, 100])
        wanted = np.r_[expected[user, [5, 2, 0, 2, 6]], np.nan, np.nan]
        actual = model.score(user, candidates)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(wanted))
        np.testing.assert_allclose(actual, wanted, rtol=1e-12, atol=1e-14, equal_nan=True)
    for a in range(len(ids)):
        for b in range(len(ids)):
            aa, ab, joint, pb, confidence, q, weight = pairs[a, b]
            trace = model.trace_rule(a, b)
            assert (trace["antecedent_count"], trace["consequent_count"], trace["joint_count"]) == (aa, ab, joint)
            assert trace["support"] == pytest.approx(joint / n)
            np.testing.assert_allclose(
                [trace["base_probability"], trace["confidence"], trace["shrunk_probability"], trace["weight"]],
                [pb, confidence, q, weight], atol=1e-14, equal_nan=True,
            )
            assert trace["retained"] == (b in rules[a])
            assert trace["retained_weight"] == pytest.approx(rules[a].get(b, 0), abs=1e-14)


def test_int64_counts_above_uint8_range_and_actual_chunk_memory_without_padding():
    histories = [{0: 1, 1: 1} for _ in range(300)] + [{2: 1} for _ in range(100)]
    model = AssociationRulesModel(min_joint_count=5, count_chunk_size=2).fit(
        make_data([99, 10, 50, 900], histories)
    )
    trace = model.trace_rule(0, 1)
    assert trace["joint_count"] == 300
    assert trace["support"] == pytest.approx(0.75)
    assert trace["retained"]
    diagnostics = model.diagnostics
    assert diagnostics["count_dtype"] == "int64"
    assert diagnostics["basket_matrix_shape"] == [400, 3]
    assert diagnostics["rule_matrix_shape"] == [3, 3]
    assert diagnostics["largest_dense_count_block_shape"] == [2, 3]
    assert diagnostics["largest_dense_count_block_bytes"] == 2 * 3 * np.dtype(np.int64).itemsize
    assert not diagnostics["full_dense_count_matrix_allocated"]
    assert diagnostics["persistent_array_bytes"] < 100_000


@pytest.mark.parametrize("n_users,n_items", [(0, 0), (3, 0), (0, 4), (3, 4)])
def test_empty_catalog_users_or_interactions(n_users, n_items):
    data = make_data(list(range(1, n_items + 1)), [{} for _ in range(n_users)])
    model = AssociationRulesModel().fit(data)
    assert model.diagnostics["n_transactions"] == 0
    assert model.diagnostics["n_selected_items"] == 0
    assert model.diagnostics["retained_rule_count"] == 0
    assert model.diagnostics["largest_dense_count_block_bytes"] == 0
    assert model.diagnostics["n_zero_basket_antecedent_users"] == n_users
    assert np.isnan(model.score(0, np.arange(n_items))).all()
    assert model.score(-1, np.array([], dtype=np.int64)).shape == (0,)


def test_basket_size_one_cannot_create_a_rule_from_discarded_history():
    model = AssociationRulesModel(max_basket_size=1, min_joint_count=1).fit(tiny_data())
    assert model.diagnostics["counting_skipped_reason"] == "no_multi_item_basket"
    assert model.trace_rule(0, 1)["joint_count"] == 0
    assert np.isnan(model.score(0, np.arange(4))).all()


def test_fitted_snapshot_is_independent_of_input_mutation_and_refit_clears_rules():
    data = tiny_data()
    model = AssociationRulesModel(min_joint_count=1).fit(data)
    before = model.score(0, np.arange(4))
    data.interactions.data[:] = 100
    data.item_ids[:] = [800, 900, 1000, 1100]
    np.testing.assert_allclose(model.score(0, np.arange(4)), before, equal_nan=True)
    assert model.trace_rule(0, 1)["antecedent_item_id"] == 70
    assert model.user_basket(0)[0]["frequency"] == 4
    assert model.fit(make_data([123], [{}])) is model
    assert model.diagnostics["n_catalog_items"] == 1
    assert model.diagnostics["retained_rule_count"] == 0
    assert np.isnan(model.score(0, np.array([0, 1, 0]))).all()


@pytest.mark.parametrize(
    "parameters",
    [
        {"max_items": 4097}, {"max_items": 0}, {"max_items": 2.5},
        {"max_basket_size": 0}, {"max_neighbors": -1}, {"min_joint_count": 0},
        {"count_chunk_size": 0}, {"max_neighbors": True}, {"smoothing": -1},
        {"smoothing": np.nan}, {"smoothing": np.inf}, {"smoothing": True},
    ],
)
def test_invalid_configuration_is_explicit(parameters):
    with pytest.raises(ValueError):
        AssociationRulesModel(**parameters)


def test_unfitted_or_malformed_candidate_requests_are_explicit():
    model = AssociationRulesModel(min_joint_count=1)
    with pytest.raises(RuntimeError):
        model.score(0, np.array([0]))
    model.fit(tiny_data())
    with pytest.raises(ValueError):
        model.score(0, np.array([[0, 1]]))
    with pytest.raises(TypeError):
        model.score(0, np.array([0.5, 1.0]))
    with pytest.raises(IndexError):
        model.trace_rule(70, 10)
    assert model.score(0, []).dtype == np.float64
    assert model.score(0, []).shape == (0,)
