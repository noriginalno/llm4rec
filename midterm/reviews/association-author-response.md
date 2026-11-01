# Association author response to AR-CBF-1

Date: 2026-10-31. **Accepted, reproduced before editing, and fixed.** The reported
defect is low severity: pathological but valid represented counts could change
item-cap membership with user order. Its effect on actual SWEET.TV data is
unknown. No data download, authentication, submission, or additional agent was
used.

Changed only:

- `sweettv/models/association.py`
- `tests/test_association.py`
- this new response

The reviewer-owned `tests/test_association_adversarial.py` and
`reviews/association-challenge-by-cbf.md` were read but not edited.

## Reproduction and confirmation

Run from `llm4rec/midterm/` before and after the change:

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_association.py tests/test_association_adversarial.py
```

| Stage | Actual result |
| --- | --- |
| Before production/test changes | **1 failed, 62 passed; exit 1** |
| After fix and 12 additional author test cases | **75 passed; exit 0** |

The pre-fix failure was exactly the reviewer's
`test_exact_frequency_ties_use_raw_id_and_not_user_row_accumulation_order[huge-count-first]`.
The unchanged adversarial suite now passes both row-order variants, as well as
its directed-rule, cap-policy, state-isolation, int32-product-overflow,
maximum-float32-count, and sparse-memory checks.

Environment: Python 3.13.6, NumPy 2.5.4, SciPy 1.18.1, pytest 9.2.0.

## Exact input, expected/actual result, and root cause

With raw IDs `[1, 99]`, `max_items=1`, and `min_joint_count=1`, the five histories
are, in matrix-index notation:

```python
huge = 2**54
rows = [{0: huge, 1: huge}, {0: 1, 1: 4}, {0: 1}, {0: 1}, {0: 1}]
```

The exact represented frequency of both items is `2**54 + 4`. The expected
retained raw ID is **1**, independent of user order. Before the fix the
huge-first order retained **99**; the huge-last order retained **1**. After the
fix both retain **1**.

The old weighted `np.bincount` performed sequential float64 additions. Its
spacing at `2**54` is four, so four separate additions of one disappeared while
one addition of four survived. The raw-ID tie-break itself was correct, but was
given falsely unequal totals. None of these individual input values was lost
during conversion to the shared float32 representation.

## Minimal behavioral correction and exactness argument

The change is confined to global frequency ordering before item-cap selection:

1. **Keep the vectorized ordinary-count path when provably exact.** With
   canonical CSR, an item has at most one stored count per user. The Python-
   integer bound `U * ceil(max_count) <= 2**53`, together with integral stored
   values, guarantees that every nonnegative partial item sum is exactly
   representable in float64. Integrality is checked in 65,536-entry chunks.
   This path retains `np.bincount` and numeric raw-ID `lexsort`.
2. **Use exact fixed-point accumulation otherwise.** Every finite represented
   float32 value is a multiple of `2**-149`. Accumulate its exact integer number
   of those units in one Python integer per catalog item. Sort by the exact
   totals and then the original int64 raw IDs. This is order-independent and
   preserves arbitrarily small represented differences between large totals.

No accepted count range was narrowed, no integer rounding/truncation of
fractional observations was introduced, and no dense user-item matrix or CSC
copy was added. Merely replacing the old sum with `math.fsum` would fix the
reported tie but still lose distinctions such as `2**54` versus `2**54 + 1`.

New tests use an independent `Fraction` oracle on the actual stored float32
values. They cover exact integral/fractional ties, unequal totals that round to
the same float64 value, the smallest positive float32 subnormal, the largest
finite float32 count, reversed user order, and reversed item columns. Additional
checks verify the ordinary vectorized backend and that fractional frequencies
do not change basket-support A/J semantics or the original-frequency log score.

Basket caps, N/A/J definitions, shrinkage, rule pruning, inference baskets,
watched-item eligibility, NaN handling, and backfill policy were not changed.
Diagnostics expose `frequency_aggregation` and `frequency_ranking_exact`.
Frequency-valued **coverage summaries** remain approximate float64 quantities,
now labeled `frequency_coverage_dtype="float64_approximate"`; those summaries
are not used to choose items or rules. Exactness refers to the represented
`TrainingData` values, not precision already lost upstream when constructing
them or coalescing duplicate float32 entries.

## Bounded processing and performance implication

Let E be input nonzeros, I the catalog size, P the positive-item count, and U the
user count. Ordinary frequency ranking still uses O(E + I + P log P) work, with
two extra linear validation scans. Its check buffers contain at most 65,536
float32 values plus their boolean comparisons, approximately 320 KiB. Existing
bincount and sorting temporaries are separate from those check buffers.

The fallback performs E exact-integer additions and O(P log P) comparisons,
using O(I + P) temporary objects, not transaction pairs. Each nonzero accumulator
needs at most approximately `277 + ceil(log2(U))` bits for float32 input. It is
slower because arithmetic and sorting are in Python. A single huge count, or
any fractional count, can trigger this path; the safe-total bound is
intentionally conservative. The accumulators are not retained after fitting.
Persistent sparse history/rule dimensions and scoring code are unchanged.
Default count blocks remain at most `128 * 4096 * 8 = 4 MiB` of dense int64
counts, plus the already reported sparse operands/product; no process peak-RSS
measurement is claimed.

### Observed synthetic timing — frequency ranking only

A local microbenchmark compared the original weighted-bincount/lexsort
expression with `_rank_training_items`, **not whole-model fitting or scoring**.
The sparse fixture has 50,000 users, 4,000 items, four entries per user, and
200,000 nonzeros. Entry columns are `(7*user + offset) % 4000`, values cycle
through integers 1–20, and raw IDs descend 4000–1. The second fixture replaces
one count with `2**54`. Each callable was warmed once; measurements are medians
of five repetitions of three calls using `timeit.repeat` in the environment
above. Returned item orders matched between old/new code on these timing
fixtures; the separate exact regression fixtures establish correctness.

| Fixture / actual selected backend | Before median | After median |
| --- | ---: | ---: |
| Ordinary integral / `float64_exact_integer_bincount` | 0.314 ms | 0.390 ms |
| One huge count / `python_integer_fixed_point` | 0.312 ms | 35.064 ms |

Thus the normal-path validation overhead was small in absolute terms in this
isolated workload, while the fallback was substantially slower. These are local
synthetic measurements, not competition-scale runtime estimates or quality
measurements. Actual SWEET.TV counts, fallback incidence, full-fit cost, coverage,
and recommendation outcomes remain unmeasured.

## Fixed snapshot

After-fix SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| `sweettv/models/association.py` | `c46ef71c5ad0da4cd3ecf0b4435ead64159c481c2ca8261a023dd05bb3fb1eff` |
| `tests/test_association.py` | `94cfa2d13a3e1f984251b824f4a3b3d4a80b506fb4398da666668813e147c606` |
| Unedited reviewer tests | `f1a25bf2a8913f4f27ea20397256954eb71916564f8bb1a52b6dfb790a4ecfe4` |

The final pytest run verifies the implementation fix locally. It does not imply
an independent reviewer has accepted the response or that a competition result
has been measured.
