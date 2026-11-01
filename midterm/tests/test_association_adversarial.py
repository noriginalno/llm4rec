"""CBF author's independent challenge of the association-rule implementation.

The tiny oracle uses Fraction arithmetic, Python sets/dicts, and direct scans;
it never imports the production counting, basket, pruning, or scoring helpers.
Dense reference scores exist only for tiny fixtures, not the large sparse test.
"""

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix, issparse

from sweettv.contracts import TrainingData
from sweettv.models.association import AssociationRulesModel


def data_from_rows(item_ids, rows):
    positions, columns, values = [], [], []
    for user, history in enumerate(rows):
        for item, count in history.items():
            positions.append(user)
            columns.append(item)
            values.append(count)
    matrix = csr_matrix(
        (values, (positions, columns)),
        shape=(len(rows), len(item_ids)),
        dtype=np.float32,
    )
    return data_from_matrix(item_ids, matrix)


def data_from_matrix(item_ids, matrix):
    return TrainingData(
        user_ids=[f"adversarial-{user}" for user in range(matrix.shape[0])],
        item_ids=np.array(item_ids, dtype=np.int64),
        interactions=matrix,
        metadata=[{} for _ in item_ids],
    )


def direct_oracle(
    data, *, max_items=4096, max_basket_size=16, max_neighbors=64,
    min_joint_count=1, smoothing=10.0,
):
    """Exhaustively enumerate a *tiny* fixture using exact represented input counts."""
    raw_ids = list(map(int, data.item_ids))
    rows = []
    for user in range(len(data.user_ids)):
        start, stop = data.interactions.indptr[user : user + 2]
        rows.append({
            int(column): Fraction.from_float(float(value))
            for column, value in zip(
                data.interactions.indices[start:stop], data.interactions.data[start:stop]
            )
        })
    totals = {
        item: sum((row.get(item, Fraction(0)) for row in rows), Fraction(0))
        for item in range(len(raw_ids))
    }
    selected = sorted(
        (item for item, total in totals.items() if total > 0),
        key=lambda item: (-totals[item], raw_ids[item]),
    )[:max_items]
    selected_set = set(selected)
    baskets = []
    for row in rows:
        kept = sorted(
            (item for item in row if item in selected_set),
            key=lambda item: (-row[item], raw_ids[item]),
        )[:max_basket_size]
        baskets.append({item: row[item] for item in kept})
    n = sum(bool(row) for row in rows)
    supports = [sum(item in basket for basket in baskets) for item in range(len(raw_ids))]
    tau = Fraction.from_float(float(smoothing))
    pairs, rules = {}, {}
    for a in range(len(raw_ids)):
        outgoing = []
        for b in range(len(raw_ids)):
            joint = sum(a in basket and b in basket for basket in baskets)
            excess = joint * n - supports[a] * supports[b]
            denominator = n * (supports[a] + tau)
            uplift = Fraction(excess, 1) / denominator if denominator else None
            pairs[a, b] = (supports[a], supports[b], joint, uplift)
            if a != b and joint >= min_joint_count and uplift is not None and uplift > 0:
                outgoing.append((b, uplift))
        rules[a] = dict(sorted(outgoing, key=lambda edge: (-edge[1], raw_ids[edge[0]]))[:max_neighbors])
    scores = np.full((len(rows), len(raw_ids)), np.nan)
    for user, basket in enumerate(baskets):
        for b in range(len(raw_ids)):
            terms = [
                math.log1p(float(count)) * float(rules[a][b])
                for a, count in basket.items() if b in rules[a]
            ]
            if terms:
                scores[user, b] = math.fsum(terms)
    return {
        "n": n, "totals": totals, "selected": selected, "baskets": baskets,
        "pairs": pairs, "rules": rules, "scores": scores,
    }


def check_tiny_model_against_oracle(data, **parameters):
    oracle_parameters = {key: value for key, value in parameters.items() if key != "count_chunk_size"}
    expected = direct_oracle(data, **oracle_parameters)
    model = AssociationRulesModel(**{"min_joint_count": 1, **parameters})
    assert model.fit(data) is model
    assert model.diagnostics["n_transactions"] == expected["n"]
    for user, basket in enumerate(expected["baskets"]):
        actual_basket = {
            entry["item_index"]: Fraction.from_float(entry["frequency"])
            for entry in model.user_basket(user)
        }
        assert actual_basket == basket
        candidates = np.array(list(reversed(range(len(data.item_ids)))) + [0, 0, -1, len(data.item_ids)])
        wanted = np.r_[expected["scores"][user, candidates[:-2]], np.nan, np.nan]
        assert_allclose(model.score(user, candidates), wanted, rtol=1e-12, atol=1e-14, equal_nan=True)
    for (a, b), (a_count, b_count, joint, uplift) in expected["pairs"].items():
        trace = model.trace_rule(a, b)
        assert (trace["antecedent_count"], trace["consequent_count"], trace["joint_count"]) == (
            a_count, b_count, joint
        )
        assert trace["retained"] == (b in expected["rules"][a])
        assert trace["retained_weight"] == pytest.approx(float(expected["rules"][a].get(b, 0)))
        assert trace["eligible_before_neighbor_cap"] == (
            a in expected["selected"] and b in expected["selected"] and a != b
            and joint >= parameters.get("min_joint_count", 1)
            and uplift is not None and uplift > 0
        )
    return model, expected


def test_population_includes_post_cap_empty_baskets_but_not_originally_empty_users():
    data = data_from_rows([90, 10, 70], [{0: 7, 1: 7}, {0: 1, 1: 1}, {2: 1}, {}, {}])
    model, expected = check_tiny_model_against_oracle(
        data, max_items=2, smoothing=3, count_chunk_size=1
    )
    assert expected["n"] == 3
    assert expected["rules"][0][1] == Fraction(2, 15)
    assert model.diagnostics["n_original_empty_users"] == 2
    assert model.diagnostics["n_positive_history_users_empty_after_item_cap"] == 1
    assert np.isnan(model.score(2, np.array([0, 1]))).all()


@pytest.mark.parametrize("smoothing", [0.0, 3.0, 10.0, 1_000_000.0])
def test_directed_conditionals_and_shrinkage_not_symmetric_joint_scores(smoothing):
    data = data_from_rows(
        [91, 3, 77],
        [{0: 8, 1: 2}, {0: 4, 1: 1}, {0: 1}, {0: 1}, {2: 1}, {2: 1}, {}],
    )
    model, expected = check_tiny_model_against_oracle(data, smoothing=smoothing)
    forward, reverse = model.trace_rule(0, 1), model.trace_rule(1, 0)
    assert forward["confidence"] == 0.5
    assert reverse["confidence"] == 1
    assert forward["base_probability"] == pytest.approx(1 / 3)
    assert reverse["base_probability"] == pytest.approx(2 / 3)
    assert forward["retained_weight"] < reverse["retained_weight"]
    assert_allclose(
        model.score(0, np.array([1, 0, 1])),
        [math.log(9) * float(expected["rules"][0][1]),
         math.log(3) * float(expected["rules"][1][0]),
         math.log(9) * float(expected["rules"][0][1])],
        rtol=1e-12,
    )


def test_duplicate_csr_entries_count_once_for_support_and_joint_but_sum_frequency():
    raw = csr_matrix(
        (np.array([1, 3, 2, 5, 1, 1, 1], dtype=np.float32),
         np.array([0, 0, 1, 1, 0, 1, 2]), np.array([0, 4, 6, 7, 7])),
        shape=(4, 3),
    )
    assert not raw.has_canonical_format
    data = data_from_matrix([90, 1, 50], raw)
    assert data.interactions.has_canonical_format
    assert data.interactions.nnz == 5
    model, _ = check_tiny_model_against_oracle(data, min_joint_count=2, smoothing=2)
    trace = model.trace_rule(0, 1)
    assert (trace["antecedent_count"], trace["consequent_count"], trace["joint_count"]) == (2, 2, 2)
    assert trace["retained_weight"] == pytest.approx(1 / 6)
    assert_allclose(model.score(0, np.array([1, 0])), [math.log(5) / 6, math.log(8) / 6])


def test_multiple_antecedents_sum_log_frequency_evidence_and_watched_repeats_are_eligible():
    data = data_from_rows(
        [8, 5, 99, 1],
        [{0: 5, 1: 3}, {0: 1, 1: 1, 2: 1}, {0: 1, 2: 1},
         {1: 1, 2: 1}, {3: 1}, {3: 1}, {3: 1}],
    )
    model, expected = check_tiny_model_against_oracle(data, smoothing=2)
    assert expected["rules"][0][2] == expected["rules"][1][2] == Fraction(1, 7)
    assert model.score(0, np.array([2]))[0] == pytest.approx((math.log(6) + math.log(4)) / 7)
    assert np.isfinite(model.score(1, np.array([0, 1, 2]))).all()
    assert model._rules.diagonal().sum() == 0
    for item in range(4):
        assert not model.trace_rule(item, item)["retained"]


def test_training_and_query_baskets_share_both_caps_no_discarded_antecedent_can_vote():
    data = data_from_rows(
        [101, 102, 103, 104, 105],
        [{0: 10, 1: 9, 2: 1}, {2: 2, 3: 2}, {2: 1, 3: 1},
         {0: 1}, {1: 1}, {4: 0.5}, {}],
    )
    model, _ = check_tiny_model_against_oracle(data, max_items=4, max_basket_size=2)
    assert [entry["item_id"] for entry in model.user_basket(0)] == [101, 102]
    assert model.trace_rule(2, 3)["retained"]
    assert np.isnan(model.score(0, np.array([3]))).all()
    assert np.isfinite(model.score(1, np.array([3]))).all()
    assert np.isnan(model.score(5, np.arange(5))).all()
    assert np.isnan(model.score(6, np.arange(5))).all()


def test_high_int64_raw_id_item_basket_and_neighbor_ties_are_not_float_or_column_ties():
    ids = [2**62 + 23, 2**62 + 1, 2**62 + 4, 7]
    rows = [{0: 1, 1: 1, 2: 1, 3: 1}, {0: 2}, {1: 2}, {2: 2}, {3: 2}]
    model, _ = check_tiny_model_against_oracle(
        data_from_rows(ids, rows), max_items=3, max_basket_size=2, max_neighbors=1
    )
    assert not model.trace_rule(0, 1)["antecedent_selected"]
    assert [entry["item_id"] for entry in model.user_basket(0)] == [7, 2**62 + 1]
    neighbor_data = data_from_rows(
        ids, [{0: 1, 1: 1, 2: 1}, {0: 1}, {3: 1}, {3: 1}]
    )
    neighbor, _ = check_tiny_model_against_oracle(neighbor_data, max_neighbors=1)
    assert neighbor.trace_rule(0, 1)["retained"]
    assert not neighbor.trace_rule(0, 2)["retained"]


def test_user_and_column_permutations_preserve_ordinary_count_raw_id_semantics():
    ids = [2**62 + 10, 2**62 + 1, 500, 7, 40]
    rows = [{0: 1, 1: 1, 2: 1}, {0: 1, 1: 1}, {2: 2, 3: 2},
            {2: 1, 3: 1}, {4: 1}, {}, {0: 1}]
    parameters = dict(max_items=4, max_basket_size=2, max_neighbors=1, min_joint_count=1)
    baseline = AssociationRulesModel(**parameters).fit(data_from_rows(ids, rows))
    column_order = [3, 1, 4, 0, 2]
    inverse = {old: new for new, old in enumerate(column_order)}
    user_order = [5, 4, 3, 2, 1, 0, 6]
    other_rows = [{inverse[item]: count for item, count in rows[user].items()} for user in user_order]
    other = AssociationRulesModel(**parameters).fit(
        data_from_rows([ids[item] for item in column_order], other_rows)
    )
    for new_user, old_user in enumerate(user_order):
        assert [entry["item_id"] for entry in baseline.user_basket(old_user)] == [
            entry["item_id"] for entry in other.user_basket(new_user)
        ]
        assert_allclose(
            baseline.score(old_user, np.arange(5)),
            other.score(new_user, np.array([inverse[item] for item in range(5)])),
            rtol=1e-12, atol=1e-14, equal_nan=True,
        )


@pytest.mark.parametrize("seed", [7, 19, 91, 400])
def test_fraction_oracle_for_random_tiny_cap_and_pruning_counterexamples(seed):
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 5, size=(17, 6))
    counts[rng.random(counts.shape) < 0.5] = 0
    counts[0] = 0
    rows = [{item: int(count) for item, count in enumerate(row) if count} for row in counts]
    check_tiny_model_against_oracle(
        data_from_rows([101, 4, 700, 2, 9, 30], rows),
        max_items=5, max_basket_size=3, max_neighbors=2,
        min_joint_count=2, smoothing=3.5, count_chunk_size=2,
    )


def test_independence_negative_association_and_singletons_do_not_become_self_or_popularity_rules():
    independent = data_from_rows([1, 2, 3], [{0: 1, 1: 1}, {0: 1}, {1: 1}, {2: 100}])
    model, _ = check_tiny_model_against_oracle(independent, smoothing=0.1)
    assert model.trace_rule(0, 1)["shrunk_uplift"] == 0
    assert model.trace_rule(0, 2)["shrunk_uplift"] < 0
    assert model.diagnostics["retained_rule_count"] == 0
    for user in range(4):
        assert np.isnan(model.score(user, np.arange(3))).all()
    only_singletons = data_from_rows([11, 90], [{0: 100}, {0: 1}, {1: 1}])
    model, _ = check_tiny_model_against_oracle(only_singletons)
    assert model.trace_rule(0, 0)["joint_count"] == 2
    assert not model.trace_rule(0, 0)["retained"]
    assert np.isnan(model.score(0, np.array([0, 1]))).all()


def test_unknown_users_and_unsigned_invalid_candidates_preserve_nan_positions():
    data = data_from_rows([999, 3, 50, 1], [{0: 1, 1: 1}, {0: 1}, {2: 1}, {}])
    model = AssociationRulesModel(min_joint_count=1).fit(data)
    candidates = np.array([1, np.iinfo(np.uint64).max, 3, 1], dtype=np.uint64)
    actual = model.score(1, candidates)
    assert actual[0] == actual[3] > 0
    assert np.isnan(actual[1:3]).all()
    for unknown in [-1, 4, 2**65, None, 0.0, np.bool_(True)]:
        assert np.isnan(model.score(unknown, candidates)).all()
        assert model.user_basket(unknown) == []


def test_state_is_not_aliased_to_training_arrays_or_returned_traces_and_refits_reset_it():
    data = data_from_rows([70, 3, 10], [{0: 4, 1: 2}, {0: 1, 1: 1}, {2: 1}, {}])
    for array in [data.item_ids, data.interactions.data, data.interactions.indices, data.interactions.indptr]:
        array.flags.writeable = False
    parameters = dict(min_joint_count=1, count_chunk_size=1)
    model = AssociationRulesModel(**parameters).fit(data)
    before = model.score(0, np.array([1, 0, 1, 2]))
    basket = model.user_basket(0)
    basket[0]["frequency"] = -1000
    basket.clear()
    trace = model.trace_rule(0, 1)
    trace["retained_weight"] = -1000
    data.interactions = csr_matrix(np.array([[0, 0, 1]] * 4, dtype=np.float32))
    data.item_ids = np.array([80, 90, 100], dtype=np.int64)
    data.metadata[:] = [{"heldout_label": 1}] * 3
    assert_allclose(model.score(0, np.array([1, 0, 1, 2])), before, equal_nan=True)
    assert model.user_basket(0)[0]["frequency"] == 4
    assert model.trace_rule(0, 1)["antecedent_item_id"] == 70
    assert model.fit(data_from_rows([], [{}, {}])) is model
    assert model.diagnostics["n_selected_items"] == 0
    assert model.user_basket(0) == []
    assert np.isnan(model.score(0, np.array([0, 1]))).all()
    fresh_data = data_from_rows([1, 2, 3], [{0: 1, 1: 1}, {0: 1}, {2: 1}])
    fresh = AssociationRulesModel(**parameters).fit(fresh_data)
    model.fit(fresh_data)
    assert_allclose(model.score(0, np.arange(3)), fresh.score(0, np.arange(3)), equal_nan=True)
    assert np.isnan(model.score(3, np.arange(3))).all()


def test_no_labels_or_metadata_values_enter_rule_learning():
    rows = [{0: 3, 1: 2}, {0: 1}, {2: 1}, {}]
    first = data_from_rows([1, 2, 3], rows)
    second = data_from_rows([1, 2, 3], rows)
    first.provenance["heldout_labels"] = [1]
    second.provenance["heldout_labels"] = [2, 3]
    second.metadata[:] = [{"tv_show_title": "future labels must not matter"}] * 3
    a = AssociationRulesModel(min_joint_count=1).fit(first)
    b = AssociationRulesModel(min_joint_count=1).fit(second)
    assert a.diagnostics == b.diagnostics
    for user in range(4):
        assert_allclose(a.score(user, np.arange(3)), b.score(user, np.arange(3)), equal_nan=True)


def test_int32_product_overflow_and_max_float32_frequencies_do_not_corrupt_binary_joint_counts():
    paired, other = 50_000, 20_000
    n = paired + other
    assert paired * n > np.iinfo(np.int32).max
    assert paired * paired > np.iinfo(np.int32).max
    row_indices = np.r_[np.arange(paired), np.arange(paired), np.arange(paired, n)]
    columns = np.r_[np.zeros(paired, dtype=int), np.ones(paired, dtype=int), np.full(other, 2)]
    values = np.ones(len(columns), dtype=np.float32)
    values[0] = np.finfo(np.float32).max
    data = data_from_matrix([91, 3, 70, 1], csr_matrix((values, (row_indices, columns)), shape=(n, 4)))
    model = AssociationRulesModel(min_joint_count=1, smoothing=10, count_chunk_size=1).fit(data)
    trace = model.trace_rule(0, 1)
    expected_weight = Fraction(paired * n - paired * paired, n * (paired + 10))
    assert (trace["n_transactions"], trace["antecedent_count"], trace["joint_count"]) == (n, paired, paired)
    assert trace["retained_weight"] == pytest.approx(float(expected_weight))
    assert_allclose(
        model.score(0, np.array([1, 0, 1, 3])),
        [math.log1p(float(values[0])) * float(expected_weight), math.log(2) * float(expected_weight),
         math.log1p(float(values[0])) * float(expected_weight), np.nan],
        rtol=1e-12, equal_nan=True,
    )
    assert model.diagnostics["count_dtype"] == "int64"
    assert model.diagnostics["largest_dense_count_block_shape"] == [1, 3]
    assert model.diagnostics["largest_dense_count_block_bytes"] == 24


def test_sparse_fit_densifies_only_actual_item_chunks_not_users_or_4096_padding(monkeypatch):
    users, items, chunk = 59, 17, 3
    rows = [{(3 * user + offset) % items: offset + 1 for offset in range(4)} for user in range(users)]
    data = data_from_rows(list(range(100, 100 + items)), rows)
    calls = []
    original_toarray = csr_matrix.toarray

    def tracked_toarray(matrix, *args, **kwargs):
        calls.append((matrix.shape, matrix.dtype))
        assert matrix.shape[0] <= chunk and matrix.shape[1] == items
        assert matrix.dtype == np.dtype(np.int64)
        return original_toarray(matrix, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", tracked_toarray)
    model = AssociationRulesModel(
        max_items=4096, max_basket_size=1_000_000, max_neighbors=2,
        min_joint_count=1, count_chunk_size=chunk,
    ).fit(data)
    assert len(calls) == math.ceil(items / chunk)
    assert issparse(model._history) and issparse(model._rules)
    assert model._history.shape == (users, items)
    assert model._history.nnz == users * 4
    assert model._rules.nnz <= items * 2
    diagnostics = model.diagnostics
    assert diagnostics["basket_matrix_shape"] == [users, items]
    assert diagnostics["rule_matrix_shape"] == [items, items]
    assert diagnostics["largest_dense_count_block_shape"] == [chunk, items]
    assert diagnostics["largest_dense_count_block_bytes"] == chunk * items * 8
    assert not diagnostics["full_dense_count_matrix_allocated"]
    observed_bytes = sum(
        array.nbytes for matrix in [model._history, model._rules]
        for array in [matrix.data, matrix.indices, matrix.indptr]
    ) + sum(array.nbytes for array in [
        model._rule_joint_counts, model._supports, model._item_ids,
        model._selected_indices, model._catalog_to_local,
    ])
    assert diagnostics["persistent_array_bytes"] == observed_bytes


@pytest.mark.parametrize("huge_first", [True, False], ids=["huge-count-first", "huge-count-last"])
def test_exact_frequency_ties_use_raw_id_and_not_user_row_accumulation_order(huge_first):
    """Real low-severity regression: huge-first currently retains raw ID 99, not 1."""
    huge = 2**54  # Exactly representable by the TrainingData float32 interface.
    rows = [{0: huge, 1: huge}, {0: 1, 1: 4}, {0: 1}, {0: 1}, {0: 1}]
    if not huge_first:
        rows = rows[1:] + rows[:1]
    data = data_from_rows([1, 99], rows)
    expected = direct_oracle(data, max_items=1)
    assert expected["totals"][0] == expected["totals"][1] == huge + 4
    assert expected["selected"] == [0]
    model = AssociationRulesModel(max_items=1, min_joint_count=1).fit(data)
    retained_raw_ids = {
        entry["item_id"] for user in range(5) for entry in model.user_basket(user)
    }
    assert retained_raw_ids == {1}, (
        f"Exact totals tie at {huge + 4}; raw ID 1 must win independently of user order, "
        f"but retained raw IDs are {retained_raw_ids}"
    )
