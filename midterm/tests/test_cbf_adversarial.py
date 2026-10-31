"""Association author's independent, mathematical challenges of the CBF.

Assertions describe required behavior, not the observed buggy implementation.
Keep reproduced regressions failing until the CBF author fixes production.
No author-test helpers or production tokenization/profile helpers are oracles.
"""

from __future__ import annotations

import math
import unicodedata

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer

from sweettv.contracts import TrainingData
from sweettv.models.cbf import ContentBasedModel


def data_for(metadata, counts, item_ids=None, provenance=None):
    matrix = csr_matrix(counts, dtype=np.float32)
    if item_ids is None:
        item_ids = np.arange(1, len(metadata) + 1, dtype=np.int64)
    return TrainingData(
        user_ids=[f"adversarial-{i}" for i in range(matrix.shape[0])],
        item_ids=np.asarray(item_ids, dtype=np.int64),
        interactions=matrix,
        metadata=metadata,
        provenance={} if provenance is None else provenance,
    )


@pytest.mark.parametrize("scale", [1.0, 1e200, 1e-200])
def test_common_positive_field_weight_scale_cannot_remove_cosine_evidence(scale):
    """L2 normalization must cancel a common positive field-weight scale."""
    data = data_for(
        [
            {"tv_show_title": "alpha", "tv_show_genre_1": "beta"},
            {"tv_show_title": "alpha"},
            {"tv_show_genre_3": "beta"},
            {},
        ],
        [[1, 0, 0, 0]],
        item_ids=[9001, 73, 8, 222],
    )
    model = ContentBasedModel(
        ngram_range=(1, 1),
        field_weights={
            "title": 2 * scale,
            "genres": scale,
            "category": 0,
            "actors": 0,
            "director": 0,
        },
    ).fit(data)
    # Both features have df=2 in N=4 metadata documents, so their IDFs
    # cancel. Watched X=(2,1)/sqrt(5); candidates are X, e_title,
    # e_genres, and the zero representation, respectively.
    expected = np.array([1, 2 / math.sqrt(5), 1 / math.sqrt(5), np.nan])
    np.testing.assert_allclose(
        model.score(0, np.array([0, 1, 2, 3])), expected, equal_nan=True, atol=1e-12
    )


def test_weak_only_rows_keep_evidence_even_with_large_between_field_dynamic_range():
    data = data_for(
        [{"tv_show_genre_1": "beta"}, {"tv_show_title": "alpha"},
         {"tv_show_genre_3": "beta"}],
        [[1, 0, 0]],
    )
    model = ContentBasedModel(
        ngram_range=(1, 1), field_weights={"title": 1e200, "genres": 1e-200}
    ).fit(data)
    # A nonzero one-coordinate genre row normalizes to e_genres, regardless
    # of weights in a disjoint title-only item. Globally rescaling all field
    # weights is insufficient if it underflows the weak-only rows to zero.
    np.testing.assert_allclose(
        model.score(0, np.array([0, 1, 2])), [1, np.nan, 1], equal_nan=True, atol=1e-12
    )


@pytest.mark.parametrize(
    ("watched", "candidate"),
    [
        ("Straße", "STRASSE"),
        ("\uac00", "\u1100\u1161"),  # Composed/decomposed Hangul.
        ("\u0386", "\u0391\u0301"),  # Greek alpha with tonos.
        # Canonically equivalent mark order. Casefold maps U+0345 to a
        # spacing iota; folding BEFORE canonical ordering changes the word.
        ("\u03b1\u0345\u0300", "\u03b1\u0300\u0345"),
    ],
)
def test_unicode_casefold_and_canonical_equivalence_preserve_identical_content(
    watched, candidate
):
    if "\u0345" in watched:
        assert unicodedata.normalize("NFC", watched) == unicodedata.normalize("NFC", candidate)
    data = data_for(
        [{"tv_show_title": watched}, {"tv_show_title": candidate}], [[1, 0]]
    )
    model = ContentBasedModel(ngram_range=(1, 1)).fit(data)
    # Identical normalized content has cosine 1 regardless of raw spelling.
    np.testing.assert_allclose(model.score(0, np.array([0, 1])), [1, 1], atol=1e-12)


def test_field_namespaces_df_and_raw_term_frequency_against_dense_hand_oracle():
    data = data_for(
        [
            {"tv_show_title": "alpha alpha", "tv_show_category": "beta"},
            {"tv_show_title": "alpha"},
            {"tv_show_category": "beta"},
            {"tv_show_title": "beta"},
        ],
        [[0, 2, 1, 0]],
    )
    model = ContentBasedModel(
        ngram_range=(1, 1), field_weights={"title": 3, "category": 2}
    ).fit(data)
    # N=4: df(title:alpha)=df(category:beta)=2, df(title:beta)=1.
    # The rare title:beta feature is orthogonal to category:beta: identical
    # text in different fields must not become common content by accident.
    common = 1 + math.log(5 / 3)
    rare = 1 + math.log(5 / 2)
    items = np.array(
        [[6 * common, 2 * common, 0], [3 * common, 0, 0],
         [0, 2 * common, 0], [0, 0, 3 * rare]], dtype=float
    )
    items /= np.linalg.norm(items, axis=1)[:, None]
    profile = 2 * items[1] + items[2]
    profile /= np.linalg.norm(profile)
    expected = items @ profile
    expected[expected == 0] = np.nan
    np.testing.assert_allclose(
        model.score(0, np.arange(4)), expected, equal_nan=True, atol=1e-12
    )


def test_genre_slot_permutation_preserves_counts_and_no_cross_value_bigrams():
    data = data_for(
        [
            {"tv_show_genre_1": "space", "tv_show_genre_2": "science"},
            {"tv_show_genre_2": ["science", "space"]},
            {"tv_show_title": "space science"},
            {"tv_show_genre_1": None, "tv_show_genre_2": np.nan},
        ],
        [[1, 0, 0, 0]],
    )
    model = ContentBasedModel(ngram_range=(2, 2)).fit(data)
    # Separate one-word genre values cannot invent a two-word phrase.
    # The only real phrase is in a different item's title; the user has no
    # represented antecedent and must not receive that candidate as backfill.
    assert model.diagnostics["items_with_features"] == 1
    assert model.diagnostics["users_with_profile_evidence"] == 0
    assert np.isnan(model.score(0, np.arange(4))).all()


def test_history_emptied_by_vocabulary_cap_has_nan_not_popularity_backfill():
    data = data_for(
        [{"tv_show_title": "aa"}, {"tv_show_title": "bb bb"},
         {"tv_show_title": "bb"}, {}],
        [[1000, 0, 0, 0], [0, 5, 0, 0]],
    )
    model = ContentBasedModel(max_features=1, ngram_range=(1, 1)).fit(data)
    assert model.vectorizer_.get_feature_names_out().tolist() == ["title:bb"]
    assert model.diagnostics["users_with_profile_evidence"] == 1
    assert np.isnan(model.score(0, np.array([1, 2, 0, 1, 3]))).all()
    np.testing.assert_allclose(
        model.score(1, np.array([1, 2, 0, 1, 3])),
        [1, 1, np.nan, 1, np.nan], equal_nan=True,
    )


@pytest.mark.parametrize("case", ["empty_catalog", "zero_users", "only_missing_metadata"])
def test_empty_and_cold_representations_are_safe_without_synthetic_null_features(case):
    if case == "empty_catalog":
        data = data_for([], csr_matrix((2, 0)))
    elif case == "zero_users":
        data = data_for(
            [{"tv_show_title": "news"}, {"tv_show_title": "sport"}], csr_matrix((0, 2))
        )
    else:
        data = data_for(
            [{"tv_show_title": None}, {"tv_show_category": np.nan},
             {"tv_show_genre_1": [None, 17, "!!!"]}],
            [[1, 20, 300]],
        )
    model = ContentBasedModel().fit(data)
    assert np.isnan(model.score(0, np.array([0, -1, 100, 0]))).all()
    assert model.score(0, []).shape == (0,)
    assert model.diagnostics["users_with_profile_evidence"] == 0
    if case != "zero_users":
        assert model.item_features_.shape[1] == 0


def test_vocabulary_ties_are_repeatable_and_invariant_to_catalog_permutation():
    n_items, cap = 48, 7
    metadata = [{"tv_show_title": f"word{i:02d}"} for i in range(n_items)]
    ids = np.arange(1001, 1001 + n_items)[::-1]
    counts = np.ones((1, n_items))
    a = ContentBasedModel(max_features=cap, ngram_range=(1, 1)).fit(
        data_for(metadata, counts, ids)
    )
    permutation = np.arange(n_items)[::-1]
    b = ContentBasedModel(max_features=cap, ngram_range=(1, 1)).fit(
        data_for([metadata[i] for i in permutation], counts[:, permutation], ids[permutation])
    )
    c = ContentBasedModel(max_features=cap, ngram_range=(1, 1)).fit(
        data_for(metadata, counts, ids)
    )
    assert a.vectorizer_.vocabulary_ == b.vectorizer_.vocabulary_ == c.vectorizer_.vocabulary_
    expected = a.score(0, np.arange(n_items))
    assert np.isfinite(expected).sum() == cap
    # Whichever equal-TF words survive, their coordinate vectors are
    # orthogonal and the equally weighted profile has K equal components.
    np.testing.assert_allclose(expected[np.isfinite(expected)], 1 / math.sqrt(cap))
    np.testing.assert_allclose(
        b.score(0, np.argsort(permutation)), expected, equal_nan=True
    )


def test_scoring_only_densifies_candidate_vectors_and_cache_stays_bounded(monkeypatch):
    n_items, n_users = 31, 7
    metadata = [{"tv_show_title": f"shared token{i}"} for i in range(n_items)]
    counts = csr_matrix(
        (np.arange(1, n_users + 1), (np.arange(n_users), np.arange(n_users))),
        shape=(n_users, n_items),
    )
    calls = []
    original_toarray = csr_matrix.toarray

    def record_dense_shape(matrix, *args, **kwargs):
        calls.append(matrix.shape)
        return original_toarray(matrix, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", record_dense_shape)
    model = ContentBasedModel(max_features=9, profile_cache_size=2).fit(
        data_for(metadata, counts)
    )
    assert calls == []
    for user in [1, 3, 5]:
        scores = model.score(user, np.array([30, 2, 4, 30, -1, 31]))
        assert scores.shape == (6,)
        assert scores[0] == scores[3] > 0
        assert np.isnan(scores[-2:]).all()
    assert calls == [(4, 1)] * 3
    assert issparse(model.item_features_)
    assert issparse(model._interactions)
    assert len(model._profile_cache) == 2
    assert all(issparse(profile) and profile.shape == (1, 9)
               for profile in model._profile_cache.values())


def test_vocabulary_cap_bounds_final_not_transient_sparse_fit_columns(monkeypatch):
    """Document an observed memory limitation, not a contract violation."""
    metadata = [
        {"tv_show_title": " ".join(f"word{i}part{j}" for j in range(16))}
        for i in range(16)
    ]
    observed_shapes = []
    original_limit = TfidfVectorizer._limit_features

    def record_before_limit(vectorizer, matrix, *args, **kwargs):
        observed_shapes.append(matrix.shape)
        assert issparse(matrix)
        return original_limit(vectorizer, matrix, *args, **kwargs)

    monkeypatch.setattr(TfidfVectorizer, "_limit_features", record_before_limit)
    model = ContentBasedModel(max_features=4, ngram_range=(1, 1)).fit(
        data_for(metadata, csr_matrix((1, 16)))
    )
    assert observed_shapes == [(16, 256)]
    assert model.item_features_.shape == (16, 4)
    assert model.item_features_.nnz == 4


def test_other_users_and_heldout_annotations_cannot_change_this_users_profile():
    metadata = [
        {"tv_show_title": "red"}, {"tv_show_title": "blue"},
        {"tv_show_title": "red blue"}, {"tv_show_title": "green"}, {},
    ]
    ids = [9_000_000_000_000_001, 17, 800, 41, 3]
    first = data_for(
        metadata, [[1, 3, 0, 0, 0], [999, 0, 0, 0, 0]], ids,
        provenance={"heldout_labels": {"adversarial-0": [ids[3]]}},
    )
    changed_metadata = [dict(row, heldout_count=1_000_000, channel_title="red") for row in metadata]
    second = data_for(
        changed_metadata, [[1, 3, 0, 0, 0], [0, 0, 0, 999, 0]], ids,
        provenance={"heldout_labels": {"adversarial-0": [ids[2]]}},
    )
    a = ContentBasedModel(ngram_range=(1, 1)).fit(first)
    b = ContentBasedModel(ngram_range=(1, 1)).fit(second)
    candidates = np.array([2, 0, 2, 3, 4, -1, 5, 1, ids[0]])
    # Equal red/blue IDFs cancel. Profile = (1,3)/sqrt(10).
    expected = [4 / math.sqrt(20), 1 / math.sqrt(10), 4 / math.sqrt(20),
                np.nan, np.nan, np.nan, np.nan, 3 / math.sqrt(10), np.nan]
    np.testing.assert_allclose(a.score(0, candidates), expected, equal_nan=True, atol=1e-12)
    np.testing.assert_allclose(b.score(0, candidates), expected, equal_nan=True, atol=1e-12)
    # User ID -1 cannot silently borrow the last (high-popularity) user's row.
    assert np.isnan(a.score(-1, candidates)).all()
    assert np.isnan(a.score(2, candidates)).all()
    # Explicitly test the unsigned overflow boundary separately from raw IDs.
    unsigned = np.array([0, np.iinfo(np.uint64).max, 0], dtype=np.uint64)
    np.testing.assert_allclose(
        a.score(0, unsigned), [1 / math.sqrt(10), np.nan, 1 / math.sqrt(10)], equal_nan=True
    )


def test_static_forecast_metadata_df_is_allowed_and_matches_hand_idf():
    data = data_for(
        [{"tv_show_title": "red blue"}, {"tv_show_title": "red"},
         {"tv_show_title": "green"}],
        [[1, 0, 0]],
    )
    model = ContentBasedModel(ngram_range=(1, 1)).fit(data)
    # All three metadata rows are known; only the first has viewing history.
    # df(red)=2, df(blue)=1 in N=3. Using metadata-only forecast candidates for
    # IDF is explicitly permitted; it is not future behavioral leakage.
    red_idf, blue_idf = 1 + math.log(4 / 3), 1 + math.log(4 / 2)
    expected = red_idf / math.hypot(red_idf, blue_idf)
    np.testing.assert_allclose(model.score(0, np.array([1])), [expected], atol=1e-12)
    assert np.isnan(model.score(0, np.array([2]))[0])


def test_input_snapshot_precedes_first_lazy_profile_and_strict_index_validation():
    data = data_for(
        [{"tv_show_title": "red"}, {"tv_show_title": "blue"}], [[1, 3]]
    )
    model = ContentBasedModel(ngram_range=(1, 1), profile_cache_size=0)
    with pytest.raises(RuntimeError):
        model.score(0, np.array([0]))
    model.fit(data)
    data.interactions.data[:] = 50
    data.metadata[0]["tv_show_title"] = "blue"
    # First profile creation happens after mutation, so a copy must have been
    # made during fit, not merely when the first score was cached.
    np.testing.assert_allclose(
        model.score(0, np.array([1, 0, 1])),
        [3 / math.sqrt(10), 1 / math.sqrt(10), 3 / math.sqrt(10)], atol=1e-12,
    )
    assert model.score(0, []).shape == (0,)
    for invalid in [np.array([0.0]), np.array([True]), np.array(["0"])]:
        with pytest.raises(TypeError):
            model.score(0, invalid)
    for invalid in [np.array(0), np.array([[0]])]:
        with pytest.raises(ValueError):
            model.score(0, invalid)
    for invalid in [True, np.bool_(False), 0.0, None]:
        with pytest.raises(TypeError):
            model.score(invalid, np.array([0]))
