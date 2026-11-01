# PPR adversarial challenge by the MF author

Date: 2026-10-31. **Verdict: one reproduced low-severity unknown-user contract
edge case; no algorithmic PPR defect found in the executed mathematical checks.**
The author should respond and fix or resolve the edge-case contract before the
two new regression failures are considered closed. Production code is unchanged.

## Scope and reviewed snapshot

Read `CONTRACT.md`, `sweettv/contracts.py`, `sweettv/models/ppr.py`,
`tests/test_ppr.py`, and the author's `reviews/ppr-implementation.md`.
Created only this review and `tests/test_ppr_adversarial.py`.

The following SHA-256 fingerprints were identical before and after this review's
checks; these files were not edited by this reviewer:

| File | SHA-256 |
| --- | --- |
| `sweettv/models/ppr.py` | `f45e4f54876769fc8b1003ee44b34ac119d5e94c002506dead98efc2ae7d81bc` |
| `tests/test_ppr.py` | `343d08e1ac8091d96f4b5568a9480d67710527ceb9cde55ee360dde0501929a4` |
| `CONTRACT.md` | `cf755cf769e56c75224f974bcd76f6249039521521de5c076f451d1cd01afc61` |

The existing `.venv` was ready. Observed NumPy 2.5.4 and SciPy 1.18.1. No
installation, authentication, raw-data download, submission, or publication was
performed. These are synthetic correctness checks, not Kaggle-quality evidence.

## Finding PPR-MF-01: unbounded Python integer users do not get NaN

**Severity: low; implementation/contract edge case, not a PageRank math bug.**

`RecommendationModel` says unknown users return NaNs, and `score` accepts a
Python `int` without documenting an int64-domain restriction. On a fitted
one-user/one-item graph, both a large positive and a large negative Python
integer instead raise an unrelated array-dtype validation error.

### Exact reproduction

```python
import numpy as np
from scipy.sparse import csr_matrix
from sweettv.contracts import TrainingData
from sweettv.models.ppr import PersonalizedPageRankModel

data = TrainingData(["u"], np.array([101]), csr_matrix([[1]]), [{}])
model = PersonalizedPageRankModel().fit(data)
model.score(10**30, np.array([0, -1, 1]))
# Repeat with -(10**30).
```

- **Expected:** a length-three floating array `[NaN, NaN, NaN]`; the scalar is
  an integer but not a fitted user index.
- **Actual:** `ValueError: user_indices must contain matrix indices, not raw IDs or floats`.
- **Root cause:** `score` immediately delegates through `[user_index]`
  (`ppr.py:249-250`). `_indices` calls `np.asarray`, whose dtype is `object` for
  these legitimate but out-of-range Python integers. The integer-dtype guard
  (`ppr.py:131-137`) rejects them before the unknown-user bounds mask executes.
- **Impact:** unknown Python integer inputs outside the NumPy integer domain
  raise instead of following the missing-evidence policy. Ordinary contiguous
  indices, `-1`, and tested `uint64` out-of-range indices do not reproduce this.
  No real-data production failure or ranking-quality consequence was measured.
- **Minimal proposed fix:** in the scalar `score` entry point, validate the
  scalar as an integer (still reject booleans/floats), bounds-check it using
  Python integer arithmetic, and normalize out-of-range integer users to the
  existing `-1` sentinel before constructing the delegated array. Preserve the
  fitted-state check, candidate validation, and no-evidence diagnostics. A
  broader object-array batch policy is optional and should not accidentally
  accept float/string indices.
- **Alternative contract resolution:** explicitly limiting scalar users to
  NumPy-representable integers would require an owner-approved contract change;
  it is not assumed by this review.
- **Regression:** the two parameterizations of
  `test_unknown_python_integer_outside_int64_returns_nan_not_dtype_error` are
  deliberately failing, not skipped or xfailed, pending the author's response.

Focused reproduction command, from `llm4rec/midterm/`:

```sh
.venv/bin/python -m pytest tests/test_ppr_adversarial.py -q -k unknown_python_integer
```

## Independent mathematical audit

The new test-only oracle constructs transition probabilities with scalar loops
over raw fixture counts. It does **not** call the production graph builder or
walk routine to obtain expected values. For each query source `s`, it replaces
zero transition rows with `e_s`, then independently solves

`(I - alpha * P_s.T) p = (1-alpha) * e_s`

using `numpy.linalg.solve`. Production `_iterate_ppr` is only the subject under
test (or a batch-width spy), never the expected-value oracle.

The principal fixture has five users and seven items:

```text
9 1 0 0 0 0 0
0 4 2 0 0 0 0
1 0 7 3 0 0 0
0 0 0 0 5 2 0
0 0 0 0 0 0 0
```

It contains a nontrivial weighted cyclic component, a separate weighted star,
one cold user, one cold item, 34 total training weight, and nonmonotonic raw
program IDs. Alphas `.2`, `.63`, and `.85` are checked. Unknown/cold/disconnected
positions must be NaN; duplicate and reordered inputs must remain aligned.

### Evidence and classification

| Area challenged | Executed evidence | Assessment |
| --- | --- | --- |
| Orientation and normalization | All 12-node transition entries match the independently built **row** transition; full-node solutions and requested item entries match the linear system. Row and column sums are demonstrably different. | Pass; no transpose/column-normalization defect found. |
| Count weighting | Weighted results differ from the binary-edge oracle; multiplying all weights by 13 leaves scores unchanged to numerical tolerance. | Pass; frequency is used as an edge weight and normalized correctly. Raw rather than logged counts is an explicit design choice, not an organizer-mandated transform. |
| Dangling mass | A directed, reachable-sink kernel fixture with sources `[3,0,2]` matches a separate query-specific linear solve and conserves full-node mass. | Pass. Directed sinks are a defensive kernel test; active undirected production components cannot reach isolated cold nodes. |
| Cold start and components | The separate star cannot receive mass from the cyclic component, cold nodes stay absent, and watched programs remain finite. | Pass; no artificial positive bias or watched-item filtering. |
| Full mass versus item mass | Full-node mass is one; active bipartite item mass matches `alpha/(1+alpha)`, not one. Cold-source item mass is zero. | Pass; item scores must **not** be renormalized to sum to one. |
| Per-query convergence/freezing | A directed mixed-rate block with sources `[3,0,5,1,6,3]`, `alpha=.8`, `tol=.05` stops at `[16,0,4,1,0,16]` updates and takes 17 operator applications. Each returned column matches the independent first-acceptable iterate and its measured residual. | Pass; early-converged columns retain the iterate whose residual was recorded. |
| Residual/error reporting | Direct dense `||F(p)-p||_1` agrees with reported residuals; independently solved L1 error is bounded by `residual/(1-alpha)`, with rounding allowance. | Pass; no assumed convergence-rate substitute or off-by-one residual found. |
| Max-iteration boundary/failure | At `alpha=.5`, one update has residual exactly `.5`; tolerance `.5` accepts item vector `[.125,.375]`. The next smaller float tolerance raises `PPRConvergenceError` with one update, residual `.5`, and two operator applications. A subsequent converged query replaces failure diagnostics without altering a retained failed-query snapshot. | Pass; no partial score array returned on failure. |
| Batch width and duplicates | Batch sizes 1, 2, and 3 never propagate more columns than configured; four unique active sources are each solved once despite repeated requested rows. CSR densification is forbidden during fit/score. | Pass for propagation bounds; see output-memory limitation below. |
| Candidate/user order and index safety | Nonmonotonic raw IDs, reordered/duplicate indices, invalid signed indices, and `uint64` values `2**63`/`2**64-1` are tested. Out-of-range unsigned inputs remain absent, not wrapped to known entities; inputs remain unchanged. | Pass for representable integer arrays. PPR-MF-01 concerns scalar Python integers beyond that domain. |
| Mutable aliases and refit | Mutating source counts/IDs or a returned score array does not alter later fitted scores. Different-shape refit resets components/diagnostics. Invalid refit disables scoring; a later empty fit succeeds. | Pass. Invalidating the previous fit rather than transactionally preserving it is a safe fail-closed design choice, not a contract violation. |
| Alpha endpoints | Zero retains restart mass only and emits NaN items. One, NaN, infinity, negatives, and boolean alpha are rejected. The immediately sub-one float remains valid but explicitly fails a short convergence budget. | Pass; no unsupported alpha-one walk is silently accepted. |

An additional observed `.85` weighted-fixture query returned full-node mass
`1.0`, item mass `0.45945945945933386` versus exact aggregate
`0.45945945945945943`, after 178 updates with residual
`4.646699691690515e-13`. This is a correctness trace, **not** a performance or
recommendation-quality measurement.

## Design limitations and untested risks, not reproduced algorithmic bugs

1. **Output allocation is not bounded by `batch_size`.** This is explicitly
   documented by the author. Even with propagation width one, the tested
   eight-user/seven-candidate request allocates all 56 float64 outputs (448
   bytes). General output storage is `8 * R * C` bytes, in addition to bounded
   `O(N * batch_size)` propagation arrays. Requesting every user/item at once
   could therefore violate the workspace's operational no-global-dense-matrix
   constraint. No large allocation/OOM was attempted or claimed. The caller
   must stream bounded requests; an optional output-element/byte guard would
   make that policy harder to misuse.
2. **Very coarse tolerance or alpha near zero can yield zero numerical item
   mass before every reachable node has been visited.** These zeros become NaN
   rather than fallback votes. This follows the chosen approximate stopping and
   absent-evidence policy; the report's residual/error bound, not exact positive
   reachability of the mathematical limit, describes numerical accuracy. No
   default-setting defect is alleged from this observation.
3. **Near-one alpha can exhaust the iteration budget.** Raising a clear error is
   intentional, not a reason to silently backfill or claim convergence. Budget
   selection and call-site error handling remain integration responsibilities.
4. The fixtures do not establish real-data runtime, memory sufficiency, ranking
   performance, or safety under simultaneous mutations of public model
   attributes by another thread. None of those results is claimed.

## Executed commands and actual status

All commands below ran from `llm4rec/midterm/`:

```sh
.venv/bin/python -m pytest tests/test_ppr.py -q
# 47 passed

.venv/bin/python -m pytest tests/test_ppr.py tests/test_ppr_adversarial.py -q
# 61 passed, 2 failed; both failures are PPR-MF-01

.venv/bin/python -m pytest tests/test_ppr_adversarial.py -q -k 'not unknown_python_integer'
# 14 passed, 2 deselected
```

Reviewer-test correction: the first new-suite run additionally failed because
an exact equality assertion compared sparse dangling reinjection with the
mathematically equivalent dense completed transition. The observed difference
was `2.77555756e-17`, not a different frozen iterate. The reviewer replaced exact
equality with absolute tolerance `1e-15`; the freezing test then passed. This was
an overly strict reviewer oracle assertion, **not** a production PPR defect.

No production fix has been applied in this review. Author response and a rerun
of the two deliberately failing regressions are the remaining closure steps.
