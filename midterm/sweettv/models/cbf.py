"""Field-weighted, Unicode-aware TF-IDF content recommendation.

For each item, namespace word features by metadata field, multiply their
smoothed TF-IDF values by field weights, and L2-normalize the resulting row X_i.
A user's sparse profile is normalize(sum_i training_count[u, i] * X_i).
Scores are cosine similarities with that profile, with zero overlap reported as
NaN. Repeatedly watched items are eligible; this module never backfills.

Vocabulary/IDF fitting uses all supplied static metadata, including forecast
items, but profiles use only TrainingData.interactions. No labels are accepted.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from numbers import Integral, Real
from typing import Any
import unicodedata

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from sweettv.contracts import TrainingData


_FIELDS = {
    "title": ("tv_show_title",),
    "category": ("tv_show_category",),
    "genres": ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"),
    "actors": ("actors",),
    "director": ("director",),
}
_DEFAULT_WEIGHTS = {
    "title": 2.0,
    "category": 1.0,
    "genres": 1.0,
    "actors": 0.0,
    "director": 0.0,
}


def _strings(value: Any) -> Iterator[str]:
    """Accept textual scalars/sequences, not stringified nulls or numeric IDs."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for part in value:
            yield from _strings(part)
    elif isinstance(value, np.ndarray):
        for part in value.flat:
            yield from _strings(part)


def _words(text: str) -> list[str]:
    """Keep Unicode letters/numbers and attached combining marks, including 1-char words."""
    words = []
    current: list[str] = []
    # Canonical ordering must precede case folding: U+0345 folds to a spacing
    # iota, after which a following accent can attach to the wrong letter.
    canonical = unicodedata.normalize("NFC", text)
    for character in unicodedata.normalize("NFC", canonical.casefold()):
        if character.isalnum() or (
            current and unicodedata.category(character).startswith("M")
        ):
            current.append(character)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


class _MetadataAnalyzer:
    """Picklable callable; genre slot positions intentionally share a namespace."""

    def __init__(self, fields: tuple[str, ...], ngram_range: tuple[int, int]):
        self.fields = fields
        self.ngram_range = ngram_range

    def __call__(self, metadata: Mapping[str, Any]) -> Iterator[str]:
        if not isinstance(metadata, Mapping):
            return
        for field in self.fields:
            for source in _FIELDS[field]:
                for text in _strings(metadata.get(source)):
                    words = _words(text)
                    for size in range(
                        self.ngram_range[0], min(self.ngram_range[1], len(words)) + 1
                    ):
                        for start in range(len(words) - size + 1):
                            yield f"{field}:{' '.join(words[start : start + size])}"


class ContentBasedModel:
    """Bounded-vocabulary content model with on-demand sparse user profiles.

    ``field_weights`` partially overrides the defaults for ``title``,
    ``category``, ``genres`` (all three genre slots), ``actors``, and ``director``.
    A zero weight disables a field, including during vocabulary selection.
    Actors/director are optional and disabled by default. Features are selected
    by corpus term frequency, then assigned smoothed IDF log((1+N)/(1+df))+1.
    No stopword list, accent stripping, or English-only tokenization is used.
    Weights must be representable as finite float64 values without rounding a
    positive weight to zero. Per-row log scaling protects nonempty rows from
    absolute weight overflow/underflow; unrepresentably small coordinates or
    final dot products can still underflow in float64.

    Profiles are created lazily and at most ``profile_cache_size`` sparse rows
    are retained; neither a global user-item nor user-feature dense matrix is
    built. Invalid integer candidate indices yield NaN in their own positions.
    Unknown integer users yield all NaNs. Noninteger or non-1D index inputs are
    rejected rather than silently cast. Empty candidate lists are accepted.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        field_weights: Mapping[str, float] | None = None,
        ngram_range: tuple[int, int] = (1, 2),
        profile_cache_size: int = 128,
    ) -> None:
        if isinstance(max_features, bool) or not isinstance(max_features, Integral):
            raise TypeError("max_features must be a positive integer")
        if max_features <= 0:
            raise ValueError("max_features must be positive; the vocabulary is bounded")
        if isinstance(profile_cache_size, bool) or not isinstance(
            profile_cache_size, Integral
        ):
            raise TypeError("profile_cache_size must be a nonnegative integer")
        if profile_cache_size < 0:
            raise ValueError("profile_cache_size must be nonnegative")
        if not isinstance(ngram_range, (tuple, list)) or len(ngram_range) != 2:
            raise ValueError("ngram_range must be a (minimum, maximum) pair")
        if any(isinstance(n, bool) or not isinstance(n, Integral) for n in ngram_range):
            raise TypeError("ngram sizes must be integers")
        if not 1 <= ngram_range[0] <= ngram_range[1]:
            raise ValueError("ngram_range must satisfy 1 <= minimum <= maximum")
        if field_weights is not None and not isinstance(field_weights, Mapping):
            raise TypeError("field_weights must be a mapping")
        weights = dict(_DEFAULT_WEIGHTS)
        if field_weights is not None:
            unknown = set(field_weights) - set(_FIELDS)
            if unknown:
                raise ValueError(f"Unknown metadata field weights: {sorted(unknown)!r}")
            weights.update(field_weights)
        for field, weight in weights.items():
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise TypeError(f"Weight for {field} must be a finite nonnegative number")
            try:
                converted = float(weight)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"Weight for {field} must be representable as float64") from exc
            if (
                not np.isfinite(converted) or weight < 0
                or (weight > 0 and converted == 0)
            ):
                raise ValueError(
                    f"Weight for {field} must be finite, nonnegative, and representable as float64"
                )
            weights[field] = converted
        self.max_features = int(max_features)
        self.field_weights = weights
        self.ngram_range = tuple(map(int, ngram_range))
        self.profile_cache_size = int(profile_cache_size)
        self.diagnostics: dict[str, Any] = {}
        self._fitted = False

    def fit(self, data: TrainingData) -> ContentBasedModel:
        self._fitted = False
        self._profile_cache: OrderedDict[int, csr_matrix] = OrderedDict()
        self._interactions = data.interactions.astype(np.float64, copy=True).tocsr()
        self._n_users, self._n_items = self._interactions.shape
        analyzer = _MetadataAnalyzer(
            tuple(field for field, weight in self.field_weights.items() if weight > 0),
            self.ngram_range,
        )
        self.vectorizer_: TfidfVectorizer | None = None
        # Avoid sklearn's empty-vocabulary error without inventing a null token.
        has_tokens = any(next(analyzer(row), None) is not None for row in data.metadata)
        if has_tokens:
            self.vectorizer_ = TfidfVectorizer(
                analyzer=analyzer,
                max_features=self.max_features,
                norm=None,
                smooth_idf=True,
                sublinear_tf=False,
                stop_words=None,
                strip_accents=None,
                dtype=np.float64,
            )
            features = self.vectorizer_.fit_transform(data.metadata).tocsr()
            column_weights = np.array(
                [
                    self.field_weights[name.partition(":")[0]]
                    for name in self.vectorizer_.get_feature_names_out()
                ],
                dtype=np.float64,
            )
            # A rowwise positive scale cancels in cosine. Work in log space
            # before multiplying weights (which can itself overflow), then
            # give each nonempty row a maximum coordinate of one before L2.
            # Using only that row's coordinates preserves weak-only items even
            # when another field elsewhere has a vastly larger weight.
            np.log(features.data, out=features.data)
            features.data += np.log(column_weights)[features.indices]
            sizes = np.diff(features.indptr)
            nonempty = sizes > 0
            maxima = np.maximum.reduceat(features.data, features.indptr[:-1][nonempty])
            features.data -= np.repeat(maxima, sizes[nonempty])
            with np.errstate(under="ignore"):
                np.exp(features.data, out=features.data)
            self.item_features_ = normalize(features, norm="l2", copy=False).tocsr()
        else:
            self.item_features_ = csr_matrix((self._n_items, 0), dtype=np.float64)
        self.item_features_.eliminate_zeros()
        self.item_features_.sort_indices()
        item_has_features = np.diff(self.item_features_.indptr) > 0
        self._user_has_evidence = np.asarray(
            self._interactions @ item_has_features.astype(np.float64)
        ).ravel() > 0
        self.diagnostics = {
            "backend": "scipy_sparse_csr_sklearn_tfidf",
            "profile_backend": "on_demand_sparse_csr",
            "n_users": self._n_users,
            "n_items": self._n_items,
            "n_features": self.item_features_.shape[1],
            "max_features": self.max_features,
            "item_feature_shape": list(self.item_features_.shape),
            "item_feature_nnz": int(self.item_features_.nnz),
            "interaction_nnz": int(self._interactions.nnz),
            "items_with_features": int(item_has_features.sum()),
            "users_with_profile_evidence": int(self._user_has_evidence.sum()),
            "profile_cache_capacity": self.profile_cache_size,
            "field_weights": dict(self.field_weights),
            "ngram_range": list(self.ngram_range),
            "metadata_scope": "all supplied static metadata; no viewing labels",
        }
        self._fitted = True
        return self

    def _profile(self, user_index: int) -> csr_matrix:
        if user_index in self._profile_cache:
            self._profile_cache.move_to_end(user_index)
            return self._profile_cache[user_index]
        profile = self._interactions.getrow(user_index) @ self.item_features_
        profile = normalize(profile, norm="l2", copy=False).tocsr()
        if self.profile_cache_size:
            self._profile_cache[user_index] = profile
            if len(self._profile_cache) > self.profile_cache_size:
                self._profile_cache.popitem(last=False)
        return profile

    def score(self, user_index: int, candidate_indices: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ContentBasedModel must be fitted before scoring")
        candidates = np.asarray(candidate_indices)
        if candidates.ndim != 1:
            raise ValueError("candidate_indices must be a one-dimensional integer vector")
        if candidates.size and candidates.dtype.kind not in "iu":
            raise TypeError("candidate_indices must contain integer matrix indices")
        if isinstance(user_index, bool) or not isinstance(user_index, Integral):
            raise TypeError("user_index must be an integer matrix index")
        result = np.full(candidates.size, np.nan, dtype=np.float64)
        if (
            not 0 <= user_index < self._n_users
            or not candidates.size
            or not self._user_has_evidence[user_index]
        ):
            return result
        valid = (candidates >= 0) & (candidates < self._n_items)
        if not np.any(valid):
            return result
        # Range-check before casting so huge uint64 indices cannot wrap around.
        indices = candidates[valid].astype(np.intp, copy=False)
        similarities = self.item_features_[indices] @ self._profile(int(user_index)).T
        values = similarities.toarray().ravel()  # Only a candidates x 1 score vector.
        values[~np.isfinite(values) | (values <= 0)] = np.nan
        np.minimum(values, 1.0, out=values)  # Roundoff may put self-cosine just above 1.
        result[valid] = values
        return result
