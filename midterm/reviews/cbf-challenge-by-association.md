# Adversarial CBF challenge by the association-rule author

Date: 2026-10-31. **Two reproduced defects; four failing regression cases.**
Production code and the original author tests were not modified. Only this
review and `tests/test_cbf_adversarial.py` were written. Fixes and their subsequent
verification remain pending the CBF author's response. No Kaggle-quality or
competition-data performance claims are made.

## Reviewed source and executed check

Read `CONTRACT.md`, `sweettv/contracts.py`, `sweettv/models/cbf.py`, and
`tests/test_cbf.py`. The reviewed production snapshot has SHA-256
`0345412daa927759e156eceba89cf4e4af0603a81eb0285580ba7ab6e8e7ae5e`.
The original author-test snapshot has SHA-256
`f72d0ce416459eb71fff585f879b25af3efa7eaf445ee35b24ff9209bb93ab4c`.
Line references below refer to that production snapshot.

From `llm4rec/midterm/`:

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_cbf_adversarial.py
```

**Actual final result: 4 failed, 16 passed; exit code 1.** Assertions encode the
mathematical expected behavior and are deliberately not marked `xfail`.
Only the new adversarial suite was executed, not the original CBF suite or other
agents' tests. Environment: Python 3.13.6, NumPy 2.5.4, SciPy 1.18.1,
scikit-learn 1.9.1, pytest 9.2.0. A separate read-only diagnostic probe confirmed
the learned Unicode tokens and numeric feature arrays described below.

## F1 — canonical Unicode equivalence can disappear before tokenization

**Reproduced defect; medium severity.** Valid canonically equivalent metadata
can become completely unrelated features, including with default field weights.
This is not a claim about the prevalence of such text in the competition.

Exact input (escaped to make the combining-mark order unambiguous):

```python
metadata = [
    {"tv_show_title": "\u03b1\u0345\u0300"},
    {"tv_show_title": "\u03b1\u0300\u0345"},
]
counts = [[1, 0]]
ngram_range = (1, 1)
candidate_indices = [0, 1]
```

Both strings normalize to the same NFC text. Independently, identical normalized
single-word content has cosine one. **Expected scores: `[1, 1]`; actual:
`[1, NaN]`.** The learned vocabulary actually contains the distinct features
`"title:\u03b1\u1f76"` and `"title:\u1f70\u03b9"`.

**Root cause:** `_words`, line 61, applies `casefold()` before canonical
normalization. U+0345 becomes a spacing iota during folding, preventing the
canonical reordering of the following grave accent. Normalizing only afterward
cannot restore which letter the accent belonged to.

**Minimal fix:** canonically normalize before folding, then normalize again,
for example `NFC(NFC(text).casefold())`, or implement the standard canonical
caseless-matching normalization sequence. Keep Unicode marks and one-character
tokens rather than switching to an ASCII tokenizer.

Failing regression:
`test_unicode_casefold_and_canonical_equivalence_preserve_identical_content`
for the U+0345/U+0300 reordered pair. German case folding, decomposed Hangul,
and ordinary Greek tonos controls passed in the same parametrized test.

## F2 — accepted finite field weights can erase positive model evidence

**Reproduced defect; low severity for ordinary deployment, because the
demonstrated weights are extreme.** The constructor explicitly accepts finite
nonnegative weights, but scoring violates the documented normalized-vector
semantics on some of those accepted values. The ordinary-weight control passed;
this is not evidence that the default weights are broken.

Exact first counterexample:

```python
item_ids = [9001, 73, 8, 222]
metadata = [
    {"tv_show_title": "alpha", "tv_show_genre_1": "beta"},
    {"tv_show_title": "alpha"},
    {"tv_show_genre_3": "beta"},
    {},
]
counts = [[1, 0, 0, 0]]
field_weights = {
    "title": 2*s, "genres": s,
    "category": 0, "actors": 0, "director": 0,
}
ngram_range = (1, 1)
candidate_indices = [0, 1, 2, 3]
```

Each enabled feature has df=2 in four documents. The common IDF and any positive
scale `s` cancel under L2 normalization. The watched vector is `(2,1)/sqrt(5)`.
For **all** three tested positive scales, the independently derived expected
scores are `[1, 2/sqrt(5), 1/sqrt(5), NaN]`.

| Scale | Actual scores | Additional observed state |
| --- | --- | --- |
| `1` | `[1, 0.894427190999916, 0.447213595499958, NaN]` | Three represented items, one represented user |
| `1e200` | `[NaN, NaN, NaN, NaN]` | Item features eliminated to zero; diagnostics report zero represented items/users |
| `1e-200` | `[NaN, NaN, NaN, NaN]` | Features remain approximately `1.51e-200`/`3.02e-200`, not unit length; diagnostics still report three represented items/one user |

A second counterexample prevents an insufficient global-rescaling fix:

```python
metadata = [
    {"tv_show_genre_1": "beta"},
    {"tv_show_title": "alpha"},
    {"tv_show_genre_3": "beta"},
]
counts = [[1, 0, 0]]
field_weights = {"title": 1e200, "genres": 1e-200}
ngram_range = (1, 1)
candidate_indices = [0, 1, 2]
```

A nonzero genre-only row normalizes to the genre unit vector regardless of the
weight of a disjoint title-only item. **Expected `[1, NaN, 1]`; actual
`[NaN, NaN, NaN]`.**

**Root cause:** lines 186–190 multiply by absolute field-weight magnitudes and
then use ordinary float64 squared-norm normalization. At large magnitudes the
squared norm overflows and divides finite entries down to zero; at tiny
magnitudes it underflows and leaves non-unit rows. Subsequent profile/dot-product
arithmetic loses the positive overlap, and line 252 converts zero to NaN.

**Minimal fix:** stabilize each sparse weighted row by its own largest positive
magnitude before squaring/normalizing. Also guard the weighted multiplication
itself against overflow for the full accepted numeric range; a log-domain
weighted transform with per-row maximum subtraction is one option. A single
global rescaling of all weights is insufficient when it underflows weak-only
rows in the second counterexample. Do not conceal the defect with popularity
backfill or by assigning positive scores to zero-feature rows.

Failing regressions:

- `test_common_positive_field_weight_scale_cannot_remove_cosine_evidence[1e+200]`
- `test_common_positive_field_weight_scale_cannot_remove_cosine_evidence[1e-200]`
- `test_weak_only_rows_keep_evidence_even_with_large_between_field_dynamic_range`

## Contract and semantic checks that passed

The following are executed counterexample checks, not general model-quality
endorsements. Their independent oracles use hand-derived vectors/IDF or direct
mathematical invariants, not production tokenization/profile functions.

| Requirement / possible failure | Executed evidence and assessment |
| --- | --- |
| Actual field-weighted TF-IDF, not mislabeled similarity | Unequal field weights, repeated title terms, unequal document frequencies, and count-weighted unit-item profiles match a dense hand oracle. Same text in title/category remains orthogonal. Passed at ordinary magnitudes. |
| Genre slot identity and n-grams | Separate one-word genre values cannot invent a cross-value bigram; a user whose history is unrepresented receives no synthetic recommendation. Passed. |
| Empty/cold representations | Empty catalog, zero users with nonempty text, only missing/numeric/punctuation metadata, and a history emptied by vocabulary pruning all produce safe NaNs. No null token or fallback was invented. Passed. |
| Training-only behavior and forecast content | Changing another user's training counts and held-out annotations does not change the fixed user's profile/scores. All supplied static metadata may influence IDF; a separate N=3 hand-IDF example confirms this permitted transductive behavior. Passed. |
| Matrix indices versus raw IDs | Nonmonotonic raw IDs including `9000000000000001`, negative/out-of-range indices, and maximum uint64 values do not wrap into another item. Unknown integer users produce NaNs. Passed. |
| Candidate positions, duplicates, watched eligibility | Nontrivial requested orders and duplicate positions retain their independent expected scores; watched candidates remain finite when supported. Orthogonal or missing candidates remain NaN even when popular with another user. Passed. |
| Unfit/invalid use and snapshots | Unfit scoring raises clearly; malformed/noninteger indices are rejected rather than silently cast. Input mutation before the first lazy profile does not alter the fitted model. Passed. |
| Vocabulary cap/tie determinism | Forty-eight equal-frequency words with cap seven yield seven finite scores equal to `1/sqrt(7)`. Repeated fits and reversed item/interaction-column order retain the same vocabulary and mapped scores in this environment. Passed. Final raw-ID ranking ties belong to the shared ranker, not this score-only model. |
| Sparse scoring and bounded cache | Runtime instrumentation records no CSR `.toarray()` calls during fit and only three `(4,1)` score-vector conversions when scoring four valid candidate positions. Item/history matrices and two retained cached profiles stay sparse. Passed; not a whole-process peak-memory measurement. |

## Observed limitations and remaining uncertainty — not reproduced defects

- **Transient vocabulary memory:** `max_features` bounds the final representation,
  not initial vocabulary construction. An instrumented fit with 16 documents,
  256 distinct words, and `max_features=4` builds a sparse `(16,256)` pre-pruning
  matrix and finishes with `(16,4)`. This is standard vectorizer behavior, not a
  violation of the shared no-dense-global-score requirement. Fit-time memory
  still depends on all analyzed tokens/unique features, including n-grams.
  Competition-scale peak memory was not measured.
- **Tie portability:** the observed equal-TF vocabulary selection is repeatable
  in the recorded environment. No cross-scikit-learn-version selection guarantee
  was established. The model does not promise a lexicographic tie policy for
  vocabulary pruning; lack of such a promise is not a reproduced contract bug.
- **Modeling choices:** raw word term frequencies, coordinate field weights,
  disabled credits by default, no stopwords, and omitted channel features are
  disclosed choices, not defects proved by lower recommendation performance.
- **Data boundary:** tests cannot prove that an upstream `TrainingData` contains
  only pre-cutoff viewing. Actual schedule/log reconstruction and temporal-split
  compliance remain responsibilities of the data pipeline. No raw competition
  files were read or downloaded for this challenge.

## Requested author response

Fix F1/F2 in the owned CBF implementation, preserve the passing ordinary-case
semantics, and rerun this unchanged adversarial suite plus the author's suite.
Report the actual recheck results and any explicitly unsupported numeric range.
The four failures above are current evidence, not an assertion that a fix has
already been implemented or verified.
