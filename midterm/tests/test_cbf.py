"""Independent cosine arithmetic and contract checks for the content model."""

import inspect
from fractions import Fraction
import unicodedata

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix, issparse

from sweettv.contracts import TrainingData
from sweettv.models.cbf import ContentBasedModel


def make_data(metadata, counts, *, item_ids=None, provenance=None):
    counts = csr_matrix(counts, dtype=np.float32)
    if item_ids is None:
        item_ids = np.arange(1, len(metadata) + 1, dtype=np.int64)
    return TrainingData(
        user_ids=[f"user-{i}" for i in range(counts.shape[0])],
        item_ids=np.asarray(item_ids, dtype=np.int64),
        interactions=counts,
        metadata=metadata,
        provenance={} if provenance is None else provenance,
    )


def basic_data():
    return make_data(
        [
            {"tv_show_title": "red"},
            {"tv_show_title": "blue"},
            {"tv_show_title": "red blue"},
            {"tv_show_title": "green"},
            {},
        ],
        [[3, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 2]],
        item_ids=[9_000_000_000_000_001, 11, 44, 8, 27],
    )


def test_independent_cosines_counts_candidate_order_duplicates_and_repeats():
    data = basic_data()
    model = ContentBasedModel(ngram_range=(1, 1))
    assert model.fit(data) is model
    # Equal red/blue IDFs cancel. X_red=e1, X_blue=e2,
    # X_mix=(e1+e2)/sqrt(2), and the user profile is (3,1)/sqrt(10).
    candidates = np.array([2, 1, 0, 2, 4, 3])
    expected = [4 / np.sqrt(20), 1 / np.sqrt(10), 3 / np.sqrt(10),
                4 / np.sqrt(20), np.nan, np.nan]
    assert_allclose(model.score(0, candidates), expected, equal_nan=True, atol=1e-12)
    # Both watched items are scored; metadata-only item 2 also has evidence.
    assert np.isfinite(model.score(0, np.array([0, 1, 2]))).all()
    assert np.isnan(model.score(1, candidates)).all()  # Cold user.
    assert np.isnan(model.score(2, candidates)).all()  # History has no text evidence.
    # Raw int64 program IDs are not contiguous matrix indices.
    assert np.isnan(model.score(0, np.array([data.item_ids[0]]))).all()


def test_tf_smoothed_idf_and_weighted_profile_against_small_dense_oracle():
    data = make_data(
        [{"tv_show_title": title} for title in ["red red blue", "red", "blue green"]],
        [[0, 2, 1]],
    )
    model = ContentBasedModel(ngram_range=(1, 1)).fit(data)
    # N=3, df(red)=df(blue)=2, df(green)=1. Derive without sklearn/model helpers.
    a, b = 1 + np.log(4 / 3), 1 + np.log(4 / 2)
    oracle_items = np.array([[2 * a, a, 0], [a, 0, 0], [0, a, b]])
    oracle_items /= np.linalg.norm(oracle_items, axis=1)[:, None]
    oracle_profile = 2 * oracle_items[1] + oracle_items[2]
    oracle_profile /= np.linalg.norm(oracle_profile)
    expected = oracle_items @ oracle_profile
    assert_allclose(model.score(0, np.arange(3)), expected, atol=1e-12)


def test_field_weights_and_namespaces_have_independent_expected_cosines():
    data = make_data(
        [
            {"tv_show_title": "drama", "tv_show_genre_1": "drama"},
            {"tv_show_title": "drama"},
            {"tv_show_genre_3": "drama"},
            {"tv_show_category": "drama"},
        ],
        [[1, 0, 0, 0]],
    )
    model = ContentBasedModel(
        field_weights={"title": 2, "genres": 1, "category": 1}, ngram_range=(1, 1)
    ).fit(data)
    # Title and genre IDFs are equal, but their field weights are 2:1.
    assert_allclose(
        model.score(0, np.array([2, 1, 0, 3])),
        [1 / np.sqrt(5), 2 / np.sqrt(5), 1, np.nan],
        equal_nan=True,
        atol=1e-12,
    )


def test_genre_slot_order_is_not_a_feature_and_optional_credits_can_be_enabled():
    data = make_data(
        [
            {"tv_show_genre_1": "Наука", "tv_show_genre_3": "Історія"},
            {"tv_show_genre_2": ["Історія", "Наука"]},
            {"actors": ["Ada Lovelace"], "director": "Grace Hopper"},
            {"actors": "Ada Lovelace", "director": "Grace Hopper"},
        ],
        [[1, 0, 0, 0], [0, 0, 1, 0]],
    )
    default = ContentBasedModel().fit(data)
    assert_allclose(default.score(0, np.array([1])), [1], atol=1e-12)
    assert np.isnan(default.score(1, np.array([2, 3]))).all()
    credits = ContentBasedModel(field_weights={"actors": 1, "director": 1}).fit(data)
    assert_allclose(credits.score(1, np.array([3])), [1], atol=1e-12)


@pytest.mark.parametrize(
    ("watched", "candidate"),
    [
        ("ПРИВІТ світ", "привіт світ"),
        ("Café", "CAFE\u0301"),
        ("я", "Я"),
        ("फ़िल्म", "फ़िल्म"),
        ("映画", "映画"),
        ("the", "the"),  # Not deleted by an English stopword policy.
    ],
)
def test_unicode_one_character_words_combining_marks_and_no_english_stopwords(
    watched, candidate
):
    model = ContentBasedModel().fit(
        make_data([{"tv_show_title": watched}, {"tv_show_title": candidate}], [[1, 0]])
    )
    assert_allclose(model.score(0, np.array([1])), [1], atol=1e-12)


def test_canonical_mark_reordering_precedes_casefold_for_all_spelling_variants():
    first, second = "\u03b1\u0345\u0300", "\u03b1\u0300\u0345"
    composed = unicodedata.normalize("NFC", first)
    assert composed == unicodedata.normalize("NFC", second)
    data = make_data(
        [{"tv_show_title": spelling} for spelling in [first, second, composed]],
        [[1, 0, 0]],
    )
    model = ContentBasedModel(ngram_range=(1, 1)).fit(data)
    assert model.diagnostics["n_features"] == 1
    assert_allclose(model.score(0, np.array([2, 1, 0, 1])), [1, 1, 1, 1], atol=1e-12)


@pytest.mark.parametrize(
    "scale",
    [1.0, 1e200, 1e-200, np.finfo(np.float64).max / 2, np.nextafter(0.0, 1.0)],
)
def test_extreme_weight_scaling_preserves_tf_ratios_unit_rows_and_profile_evidence(scale):
    data = make_data(
        [
            {},  # Empty rows before/between represented rows stress sparse row scaling.
            {"tv_show_title": "alpha alpha", "tv_show_genre_1": "beta"},
            {},
            {"tv_show_title": "alpha"},
            {"tv_show_genre_3": "beta"},
            {},
        ],
        [[0, 1, 0, 0, 0, 0]],
    )
    # Both feature IDFs are equal. TF*weight gives (4,1), irrespective of s.
    # The largest scale also makes direct TF-IDF*weight multiplication overflow;
    # the smallest scale consists of positive subnormal float64 weights.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        model = ContentBasedModel(
            ngram_range=(1, 1), field_weights={"title": 2 * scale, "genres": scale}
        ).fit(data)
        scores = model.score(0, np.array([1, 3, 4, 2, 1]))
    assert_allclose(
        scores, [1, 4 / np.sqrt(17), 1 / np.sqrt(17), np.nan, 1],
        atol=1e-12, equal_nan=True,
    )
    row_squared_norms = np.asarray(model.item_features_.multiply(model.item_features_).sum(axis=1)).ravel()
    assert_allclose(row_squared_norms, [0, 1, 0, 1, 1, 0], atol=1e-12)
    assert model.diagnostics["items_with_features"] == 3
    assert model.diagnostics["users_with_profile_evidence"] == 1
    assert issparse(model.item_features_)


def test_disjoint_weak_only_rows_survive_full_float64_weight_dynamic_range():
    data = make_data(
        [{"tv_show_genre_1": "beta"}, {"tv_show_title": "alpha"},
         {"tv_show_genre_3": "beta"}, {}],
        [[1, 0, 0, 0]],
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        model = ContentBasedModel(
            ngram_range=(1, 1),
            field_weights={"title": np.finfo(np.float64).max,
                           "genres": np.nextafter(0.0, 1.0)},
        ).fit(data)
        scores = model.score(0, np.arange(4))
    assert_allclose(scores, [1, np.nan, 1, np.nan], equal_nan=True, atol=1e-12)
    assert model.diagnostics["items_with_features"] == 3


def test_missing_values_are_not_literal_null_tokens_and_empty_metadata_is_safe():
    data = make_data(
        [
            {},
            {"tv_show_title": None, "tv_show_category": float("nan")},
            {"tv_show_title": "", "tv_show_genre_1": [None, np.nan, "  "]},
            {"tv_show_title": "!!!", "actors": "disabled credit"},
        ],
        [[1, 2, 3, 0]],
    )
    model = ContentBasedModel().fit(data)
    assert np.isnan(model.score(0, np.arange(4))).all()
    assert model.item_features_.shape == (4, 0)
    assert model.item_features_.nnz == 0
    assert model.vectorizer_ is None
    assert model.diagnostics["n_features"] == 0
    assert model.diagnostics["items_with_features"] == 0
    assert model.diagnostics["users_with_profile_evidence"] == 0


def test_empty_catalog_and_zero_users():
    empty_catalog = make_data([], csr_matrix((2, 0)))
    model = ContentBasedModel().fit(empty_catalog)
    assert model.item_features_.shape == (0, 0)
    assert model.score(0, np.array([], dtype=int)).shape == (0,)
    assert model.score(-1, []).dtype == np.float64
    assert np.isnan(model.score(0, np.array([0, -1]))).all()
    assert model.diagnostics["n_users"] == 2
    assert model.diagnostics["n_items"] == 0
    zero_users = make_data([{"tv_show_title": "news"}], csr_matrix((0, 1)))
    model.fit(zero_users)
    assert np.isnan(model.score(0, np.array([0]))).all()
    assert model.diagnostics["n_users"] == 0
    assert model.diagnostics["items_with_features"] == 1


@pytest.mark.parametrize("user", [-1, 3, 2**63, np.int64(-10)])
def test_unknown_integer_users_never_wrap(user):
    model = ContentBasedModel().fit(basic_data())
    assert np.isnan(model.score(user, np.array([0, 1, 2]))).all()


def test_invalid_candidate_positions_are_nan_without_reordering_or_negative_wrap():
    model = ContentBasedModel(ngram_range=(1, 1)).fit(basic_data())
    assert_allclose(
        model.score(np.int64(0), np.array([2, -1, 5, 0, 2])),
        [4 / np.sqrt(20), np.nan, np.nan, 3 / np.sqrt(10), 4 / np.sqrt(20)],
        equal_nan=True,
        atol=1e-12,
    )
    assert_allclose(
        model.score(0, np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64)),
        [3 / np.sqrt(10), np.nan],
        equal_nan=True,
        atol=1e-12,
    )


@pytest.mark.parametrize("candidates", [np.array([0.0]), np.array([True]), ["0"]])
def test_noninteger_candidates_rejected(candidates):
    with pytest.raises(TypeError, match="integer"):
        ContentBasedModel().fit(basic_data()).score(0, candidates)


@pytest.mark.parametrize("candidates", [np.array(0), np.array([[0, 1]])])
def test_nonvector_candidates_rejected(candidates):
    with pytest.raises(ValueError, match="one-dimensional"):
        ContentBasedModel().fit(basic_data()).score(0, candidates)


@pytest.mark.parametrize("user", [None, 0.0, True, "0"])
def test_noninteger_user_indices_rejected(user):
    with pytest.raises(TypeError, match="integer"):
        ContentBasedModel().fit(basic_data()).score(user, np.array([0]))


def test_score_requires_fit():
    with pytest.raises(RuntimeError, match="fitted"):
        ContentBasedModel().score(0, np.array([0]))


def test_bounded_vocabulary_sparse_storage_cache_and_actual_diagnostics():
    n_items = 30
    data = make_data(
        [{"tv_show_title": f"token{i}"} for i in range(n_items)],
        np.ones((3, n_items)),
    )
    model = ContentBasedModel(max_features=4, profile_cache_size=2).fit(data)
    assert issparse(model.item_features_)
    assert issparse(model._interactions)
    assert model.item_features_.shape == (n_items, 4)
    assert model.item_features_.nnz == 4
    diagnostics = model.diagnostics
    assert diagnostics["n_features"] == 4
    assert diagnostics["item_feature_shape"] == [30, 4]
    assert diagnostics["item_feature_nnz"] == 4
    assert diagnostics["items_with_features"] == 4
    assert diagnostics["users_with_profile_evidence"] == 3
    assert diagnostics["interaction_nnz"] == 90
    assert len(model._profile_cache) == 0  # No global user-feature precomputation.
    for user in range(3):
        assert np.isfinite(model.score(user, np.arange(n_items))).sum() == 4
    assert len(model._profile_cache) == 2
    assert all(issparse(profile) and profile.shape == (1, 4)
               for profile in model._profile_cache.values())


def test_cache_disabled_is_equivalent_and_refit_clears_old_profiles():
    data = basic_data()
    cached = ContentBasedModel(profile_cache_size=1).fit(data)
    uncached = ContentBasedModel(profile_cache_size=0).fit(data)
    for user in range(3):
        assert_allclose(cached.score(user, np.arange(5)), uncached.score(user, np.arange(5)),
                        equal_nan=True)
    assert len(uncached._profile_cache) == 0
    cached.fit(make_data([{}], [[1]]))
    assert not cached._profile_cache
    assert np.isnan(cached.score(0, np.array([0]))).all()


def test_fit_snapshots_training_behavior_and_does_not_mutate_inputs():
    data = basic_data()
    original_counts = data.interactions.copy()
    model = ContentBasedModel(profile_cache_size=0).fit(data)
    before = model.score(0, np.arange(5))
    assert_allclose(data.interactions.toarray(), original_counts.toarray())
    data.interactions.data[:] = 99
    data.metadata[0]["tv_show_title"] = "different title"
    assert_allclose(model.score(0, np.arange(5)), before, equal_nan=True)


def test_heldout_labels_are_not_an_input_and_ignored_fields_cannot_change_features():
    first = basic_data()
    second = basic_data()
    first.provenance["heldout_labels"] = {"user-0": [11]}
    second.provenance["heldout_labels"] = {"user-0": [44, 27]}
    for row in second.metadata:
        row["heldout_count"] = 1_000_000
        row["unknown_text_field"] = "not an approved content field"
    a = ContentBasedModel().fit(first)
    b = ContentBasedModel().fit(second)
    assert list(inspect.signature(ContentBasedModel.fit).parameters) == ["self", "data"]
    assert_allclose(a.item_features_.toarray(), b.item_features_.toarray())
    assert_allclose(a.score(0, np.arange(5)), b.score(0, np.arange(5)), equal_nan=True)
    assert a.vectorizer_.vocabulary_ == b.vectorizer_.vocabulary_


def test_all_disabled_fields_are_empty_and_weight_mapping_is_copied():
    weights = {"title": 0, "category": 0, "genres": 0, "actors": 0, "director": 0}
    model = ContentBasedModel(field_weights=weights)
    weights["title"] = 100
    model.fit(basic_data())
    assert model.diagnostics["n_features"] == 0
    assert np.isnan(model.score(0, np.arange(5))).all()


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"max_features": 0}, ValueError),
        ({"max_features": None}, TypeError),
        ({"max_features": 2.5}, TypeError),
        ({"max_features": True}, TypeError),
        ({"profile_cache_size": -1}, ValueError),
        ({"profile_cache_size": 1.5}, TypeError),
        ({"ngram_range": (0, 1)}, ValueError),
        ({"ngram_range": (2, 1)}, ValueError),
        ({"ngram_range": (1,)}, ValueError),
        ({"ngram_range": (1, 2.0)}, TypeError),
        ({"field_weights": {"unknown": 1}}, ValueError),
        ({"field_weights": {"title": -1}}, ValueError),
        ({"field_weights": {"genres": np.nan}}, ValueError),
        ({"field_weights": {"category": np.inf}}, ValueError),
        ({"field_weights": {"title": "2"}}, TypeError),
        ({"field_weights": {"title": True}}, TypeError),
        ({"field_weights": {"title": 10**400}}, ValueError),
        ({"field_weights": {"title": Fraction(1, 10**400)}}, ValueError),
        ({"field_weights": []}, TypeError),
    ],
)
def test_configuration_validation(kwargs, exception):
    with pytest.raises(exception):
        ContentBasedModel(**kwargs)
