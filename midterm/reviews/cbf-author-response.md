# CBF author response to the association author's challenge

Date: 2026-10-31. **Both findings accepted, reproduced, fixed, and rechecked.**
Final result: **78 passed**, including the unchanged 20-case adversarial suite,
with warnings treated as errors. No failures were skipped, marked xfail, or
removed. No competition-data quality or performance claim is made.

## Scope and before/after evidence

Read the actual challenge in `reviews/cbf-challenge-by-association.md`, its
`tests/test_cbf_adversarial.py`, current production/author tests, and
`CONTRACT.md` before editing. This response changes only:

- `sweettv/models/cbf.py`
- `tests/test_cbf.py`
- this new `reviews/cbf-author-response.md`

The adversarial tests and challenge report are unchanged; their SHA-256 values
were recorded before editing and asserted equal after the fix:

| Protected artifact | SHA-256 before and after |
| --- | --- |
| `tests/test_cbf_adversarial.py` | `e65aed592b1f498f81e3ffb3e3031b0461af30b1bcb862ac50e371250af0f183` |
| `reviews/cbf-challenge-by-association.md` | `2a0b94e80edf44b18dfb30a9772a6f44e15e2a26f34a7067425a53b74d60e0d0` |

Commands were executed from `llm4rec/midterm/`:

```sh
# Before any production/test edits:
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_cbf.py tests/test_cbf_adversarial.py
# Actual: 4 failed, 65 passed in 0.63s; exit 1.

# Final source and expanded author regressions, including all adversarial cases:
.venv/bin/python -m pytest -q -p no:cacheprovider -W error tests/test_cbf.py tests/test_cbf_adversarial.py
# Actual: 78 passed in 0.61s; exit 0.
```

The four baseline failures were exactly the review's canonical-mark-order case,
the common field-weight scales `1e200` and `1e-200`, and the weak-only-row case.
All original 49 author cases passed before the fix. Nine additional author
cases now cover canonical spelling variants, five weight scales including
float64 boundaries, disjoint weak-only rows at full float64 dynamic range, and
two explicit invalid conversion cases. Times above are pytest output, not
measured model-fit or recommendation throughput.

## F1 — canonical Unicode equivalence: accepted, medium severity, fixed

**Before:** canonically equivalent titles `"\u03b1\u0345\u0300"` and
`"\u03b1\u0300\u0345"` scored `[1, NaN]` for a user watching the first title.
The expected `[1, 1]` is correct: identical normalized one-word documents have
the same unit feature vector and cosine one.

**Root cause:** `_words` case-folded before canonical ordering. U+0345 folds
into a spacing iota, after which subsequent normalization cannot restore the
original combining mark's attachment.

**Changed code:** `_words` now computes `NFC(NFC(text).casefold())` before its
existing Unicode-aware word scan. No ASCII transliteration, mark removal,
English stopwords, or one-character-word restriction was introduced.

**After:** a separate executed probe returned `[1.0, 1.0]` and the single
vocabulary feature `title:\u1f70\u03b9`. The unchanged adversarial Unicode
controls pass. The new author regression also includes the composed spelling,
requires exactly one feature, and checks candidate duplicates/order.

## F2 — extreme accepted finite weights: accepted, low severity, fixed

**The challenge's calculation is correct.** Equal feature IDFs and weights
`(2s, s)` give the unit item vector `(2,1)/sqrt(5)` for every positive scale s.
Thus expected scores are `[1, 2/sqrt(5), 1/sqrt(5), NaN]`. A genre-only item
normalizes to its genre unit vector even when a disjoint title item has a far
larger weight. Global weight rescaling is not a sufficient fix.

**Before:** the two extreme common scales and the disjoint-field example all
returned only NaNs. Absolute weighted magnitudes were multiplied directly, then
squared for L2 normalization, causing overflow/underflow. The defect affected
the learned vectors before scoring; changing the NaN policy or adding fallback
scores would not correct it.

**Changed code:** `fit` now constructs each weighted unit row using its sparse
nonzero coordinates in log space:

```text
ell_ij = log(TF-IDF_ij) + log(field_weight_j)
m_i    = max_j ell_ij, over that row's existing coordinates
v_ij   = exp(ell_ij - m_i)
X_i    = v_i / ||v_i||_2
```

The rowwise positive scaling cancels mathematically under normalization. The
largest retained coordinate is one before squaring, preventing whole-row loss
from absolute weight magnitude. Working in logs also avoids overflow in the
weighted multiplication itself. Maxima use only each nonempty CSR row's stored
coordinates, not implicit sparse zeros or an unrelated item's field weights.
Zero-weight fields remain disabled before vocabulary construction.

Separate executed post-fix probes returned:

| Case | Actual scores after fix |
| --- | --- |
| Common scale `1` | `[1.0, 0.894427190999916, 0.4472135954999579, NaN]` |
| Common scale `1e200` | `[1.0, 0.8944271909999162, 0.44721359549995726, NaN]` |
| Common scale `1e-200` | `[1.0, 0.8944271909999162, 0.44721359549995726, NaN]` |
| Title weight `1e200`, genre weight `1e-200`, genre-only history | `[1.0, NaN, 1.0]` |

Each common-scale case now reports three represented items and one represented
user. The original challenge's three numerical failure cases pass unchanged.

The new author regressions additionally exercise the largest finite float64
weight, the smallest positive float64 subnormal, repeated terms, and empty rows
before/between/after nonempty rows. Independently, their nonempty mixed vector
is `(4,1)/sqrt(17)`. Tests check expected cosines, unit row norms, actual coverage,
and sparse storage, with overflow/invalid/divide exceptions enabled.

**Explicit numeric input boundary:** constructor weights are stored as float64.
It now validates the converted value as well: numbers such as `10**400` and
`Fraction(1, 10**400)` raise a clear ValueError instead of overflowing conversion
or silently disabling a positive field. All challenged weights remain accepted,
as do positive finite float64 values down to the smallest subnormal.

## Preserved behavior and remaining limitations

- Execution remains CPU/sparse. The log transform operates on CSR payloads with
  O(nnz + catalog rows + features) temporary vector storage, not a dense
  catalog-feature or global user-item matrix. The adversarial allocation checks
  still observe no CSR densification during fit and only candidate x 1 score
  vectors during scoring. This is not a process peak-RSS measurement.
- Training-only count-weighted profiles, lazy bounded caching, field namespaces,
  static forecast metadata/IDF scope, candidate ordering/duplicates, watched
  eligibility, unknown-user handling, and NaN/no-backfill behavior are unchanged.
- **Float64 precision still applies:** a coordinate negligible relative to
  another coordinate in the *same row*, or a final positive dot product below
  representable precision, can underflow. Such weak mixed-row coordinates may
  become zero; no artificial positive floor was added. Each represented row
  nevertheless retains its maximum coordinate, so a weak-only row is not erased
  by an unrelated strong field. Ordinary rounding remains visible in the last
  digits of the probe scores above.
- The final vocabulary cap does not bound the vectorizer's initial vocabulary
  memory. Unicode tokenization has no language-specific segmentation, stemming,
  or semantic matching. Default field weights and competition-scale predictive
  quality/coverage remain unvalidated here. Upstream temporal isolation still
  depends on constructing `TrainingData` correctly.

Final production SHA-256:
`6ec63d316c172aa2b833b1bc75e452b14a20a1607b1933f564439613e1fd57a6`.
Final author-test SHA-256:
`e7d6e21e6c9bebccf5ee0f51c73049d5e5a8a3cb9d1cabc20a95d851d86f2ee5`.
Baseline production SHA-256 matched the challenge:
`0345412daa927759e156eceba89cf4e4af0603a81eb0285580ba7ab6e8e7ae5e`.
