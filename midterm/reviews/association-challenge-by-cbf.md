# AssociationRulesModel challenge by the CBF author

## Outcome and scope

**One reproduced low-severity numerical defect; no ordinary-scale mathematical
defect found in the executed checks.** Existing tests: **41 passed**. Independent
adversarial tests: **21 passed, 1 failed**. The failing test is deliberately an
ordinary regression assertion, not an xfail or an assertion of buggy behavior.

Reviewed `CONTRACT.md`, `sweettv/contracts.py`,
`sweettv/models/association.py`, and `tests/test_association.py` on 2026-10-31.
Reviewer-owned additions are this report and
`tests/test_association_adversarial.py`. No production/shared files or existing
tests were edited. No data download, authentication, installation, submission,
publication, or additional agent was used. These are synthetic correctness
checks, **not Kaggle quality measurements**.

## Reproduction commands and observed results

Run from `llm4rec/midterm/`:

```sh
.venv/bin/python -m pytest tests/test_association.py
# 41 passed in 0.21s; exit 0.

.venv/bin/python -m pytest tests/test_association_adversarial.py
# 1 failed, 21 passed in 0.63s; exit 1.
```

The failed node is:

```text
tests/test_association_adversarial.py::test_exact_frequency_ties_use_raw_id_and_not_user_row_accumulation_order[huge-count-first]
```

Environment observed: Python 3.13.6, NumPy 2.5.4, SciPy 1.18.1, pytest 9.2.0.
The elapsed times above are test-runner output, not a model runtime benchmark.

Reviewed production SHA-256:
`1261e105c8dd528c62eab1e51f3b5c53f803462f9d8c0fcb6e2143410860e87e`
(`sweettv/models/association.py`). Existing-test SHA-256:
`2fee587ab40abaeac120cd961553217c557763c293441949ae8bf01b14848813`.

## AR-CBF-1 — Float aggregation violates exact item-cap ties

**Classification:** reproduced implementation defect.
**Severity:** Low / P3. This needs pathological but accepted counts, not realistic
per-user viewing frequencies. Its impact is cap membership and reproducibility,
not a demonstrated recommendation-quality loss.

**Input:** `max_items=1`, `min_joint_count=1`, raw item IDs `[1, 99]`, five users:

| User order | Count for raw ID 1 | Count for raw ID 99 |
| --- | ---: | ---: |
| u0 | 18014398509481984 | 18014398509481984 |
| u1 | 1 | 4 |
| u2 | 1 | 0 |
| u3 | 1 | 0 |
| u4 | 1 | 0 |

The large count is `2**54`. It and every nonzero small count are **exactly
representable by the shared float32 input type**; this is not loss introduced
when constructing `TrainingData`. The exact total for each item is
`18014398509481988 = 2**54 + 4`. The documented numeric raw-ID tie-break must
therefore retain **ID 1**, independently of user order.

**Observed, independently reproduced before adding the regression:**

| Row order | Computed frequency for ID 1 | Computed frequency for ID 99 | Retained ID |
| --- | ---: | ---: | ---: |
| u0, u1, u2, u3, u4 | 18014398509481984 | 18014398509481988 | **99 — wrong** |
| u1, u2, u3, u4, u0 | 18014398509481988 | 18014398509481988 | **1 — correct** |

The new test computes exact totals with `Fraction` and requires retained IDs
`{1}`. The first order fails with actual `{99}`; the second order passes.

**Root cause:** `fit`, `association.py:126-137`, accumulates item frequencies
with `np.bincount(..., weights=interactions.data)`, a sequential float64 sum.
At `2**54`, the float64 spacing is 4: each subsequent addition of 1 is lost,
while adding 4 survives. `np.lexsort` then receives falsely unequal totals; its
raw-ID tie-break is correct but never reached for the intended tie.

**Minimal proposed correction:** sum each item's frequencies with a numerically
stable, order-independent accumulation strategy before ranking. A CSC-column
`math.fsum` implementation fixes this particular counterexample without a dense
user-item matrix. This proposal was **not applied or verified as a production
fix** by the reviewer. For exact rankings over *all* accepted large integral
counts, retain exact integer totals for sorting: even a correctly rounded
float64 sum cannot distinguish every pair of distinct integers above `2**53`.
Alternatively, explicitly validate/document a safe count-total range rather
than silently accepting inputs beyond it. Preserve raw IDs as int64.

**Confirmation criterion for the author:** both parameterized row-order tests
pass without weakening their exact-total assertion; existing and other
adversarial tests remain green. Normal-scale behavior need not change.

## Independent oracle and successful counterexamples

`direct_oracle` in the new test file converts the *represented* training values
to exact `Fraction`s, selects capped items by exact totals/raw IDs, directly
enumerates capped user baskets, and scans every tiny pair for A and J. It uses
the exact rational weight

`max(0, (J*N - A_a*A_b) / (N*(A_a + smoothing)))`

before independent per-antecedent pruning. Query scores use a separate
`math.fsum(log1p(frequency) * weight)` calculation. It imports no production
helpers or existing-test oracle. A dense score reference is used only for tiny
fixtures; the 70,000-user overflow fixture is sparse and checked analytically.

| Attack / expected result | Actual result and classification |
| --- | --- |
| N includes originally positive users emptied by item cap, excludes originally empty users. Fixture N=3, A_a=A_b=J=2, smoothing=3 implies weight 2/15. | Matches; cold post-cap users return NaN. **Pass.** |
| Direction matters: N=6, A_a=4, A_b=2, J=2 gives confidences 1/2 and 1. With smoothing=3, forward/reverse weights are 2/21 and 2/15. | Matches for smoothing 0, 3, 10, and 1,000,000. Scores use original-frequency log weights. **Pass.** |
| Explicit duplicate CSR entries must coalesce as frequencies, not inflate basket A or J. | Shared contract coalesces them; A_a=A_b=J=2, weight=1/6. Scores are log(5)/6 and log(8)/6. **Pass.** |
| Multiple antecedents should add independent rule evidence, not take a maximum or average. | Two weights of 1/7 give candidate score `(log(6)+log(4))/7`. **Pass.** |
| Watched consequents remain eligible; self-rules must not exist. | Watched items score through other antecedents; graph diagonal is zero and self traces are not retained. **Pass.** |
| Training and query baskets must apply identical item and basket caps. | A discarded item owns a learned rule but cannot vote in that user's query. Original-empty and post-cap-empty queries remain NaN. **Pass.** |
| Item, basket, and outgoing-neighbor ties use raw numeric int64 IDs, including IDs above float64's exact range. | Correct on ordinary count totals; column/user permutations preserve raw-ID semantics. **Pass; AR-CBF-1 is the exceptional large-count case.** |
| Four seeded tiny cases stress caps, asymmetric weights, min-joint thresholds, and pruning against exact enumeration. | All baskets, pair counts, retained edges, and candidate-aligned scores match. **Pass.** |
| Independence, negative association, and singleton-only histories must not create tiny roundoff rules, self rules, or popularity fallback. | Zero retained rules and NaN scores where expected. **Pass.** |
| Candidate order/duplicates, unsigned maximum index, unknown users, and metadata-only items. | Position-preserving finite duplicates and NaNs; no negative/unsigned index wrap or backfill. **Pass.** |
| Read-only input arrays, replacement of training arrays, mutation of returned baskets/traces, and refit to empty then nonempty catalogs. | No learned-state aliasing or stale rules observed. **Pass.** |
| Changing static metadata and fake held-out fields in provenance must not affect this behavior-only model. | Diagnostics and scores unchanged. **Pass.** |
| int32-overflow attack: N=70,000, A_a=A_b=J=50,000, so J*N=3,500,000,000 and A_a*A_b=2,500,000,000. One frequency is the largest finite float32. | Int64 binary counts preserve J=50,000; weight matches `10000/35007` with smoothing=10; scores remain finite. **Pass.** |
| Sparse-memory attack: 59 users, 17 actual items, default max_items=4096, huge requested basket cap, chunk size 3, neighbor cap 2. | Observed all six CSR `toarray` calls: at most 3x17 int64, largest 408 bytes. History has 236 entries; graph has at most 34. Reported persistent logical bytes match fitted arrays. **Pass.** |

## Design choices and unverified risks — not additional defects

- **Documented choices:** one user-history basket per originally active user;
  singleton directed rules; capped histories; positive shrunk uplift rather than
  raw confidence; `log1p` frequency aggregation; default minimum J=5; and no
  metadata-only rule evidence. The contract does not prescribe alternative
  versions. Changing these may affect coverage/quality but is not a correctness
  fix supported by this review.
- **Memory evidence is bounded in scope:** the allocation spy establishes the
  observed dense CSR conversion shapes, and logical persistent storage is
  checked. It does not measure process peak RSS, allocator overhead, or every
  SciPy temporary on a competition-scale dataset. The implementation also
  explicitly permits MxM dense counts when the configured chunk covers all M
  selected items; no global dense user-item score matrix was observed.
- **Very large N is unverified:** the executed overflow test exceeds int32
  *products*, not int64 population/count capacity. The Python-integer fallback
  for N above `sqrt(int64_max)` was read but not exercised by a genuine
  multi-billion-user fit. No correctness claim for that scale is made.
- **Performance/coverage remain unmeasured on SWEET.TV data.** No competition
  inputs or labels were used by this reviewer, and no quality conclusion
  follows from passing these fixtures.

The author should respond to AR-CBF-1 or explicitly narrow the accepted numeric
range. No production fix has been made in this challenge phase.
