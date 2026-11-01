# MF adversarial challenge by the PPR author

Date: 2026-10-31. Review stage: independent author/reviewer role reversal.

## Verdict and ownership

**No training/scoring correctness defect was reproduced against the shared
contract. One low-severity documentation/diagnostic evidence issue needs an
author response:** an enforced BLAS thread cap is asserted even when this
environment provides no controller with which to verify/enforce that cap.
This is not evidence of incorrect recommendations or measured oversubscription.

Only `tests/test_mf_adversarial.py` and this review were added/edited in this
stage. Neither `sweettv/models/mf.py` nor `tests/test_mf.py` was modified. No
installation, authentication, remote download, Kaggle evaluation/submission,
large-model training, or publication was performed.

Reviewed input SHA-256 hashes, unchanged through the review checks:

| Input | SHA-256 |
| --- | --- |
| `sweettv/models/mf.py` | `8dd5e596d3f5fda7a8b677a62b670623f9789e44b696a6c5d1763c85b37255d3` |
| `tests/test_mf.py` | `273348bcfa730c6ce52137ac6ec2033d416627885190b0f5d8a6e20849819080` |
| `CONTRACT.md` | `cf755cf769e56c75224f974bcd76f6249039521521de5c076f451d1cd01afc61` |

## M1: asserted thread cap is not verified on Apple Accelerate

**Severity/type:** low, evidence/documentation; no numerical failure reproduced.

- **Source:** `mf.py:98-100` states that BLAS is limited to one thread during
  fitting and to at most `num_threads` during batch scoring. `mf.py:177-194`
  requests a `threadpoolctl` limit and records detected pools. Diagnostics at
  `mf.py:245-247` record `training_blas_thread_limit = 1` without distinguishing
  a request from an observed/enforced limit. The author's assertion at
  `tests/test_mf.py:243` uses `all(...)`, which passes vacuously for no pools.
- **Expected:** claims of enforced/observed BLAS thread limits are qualified by
  the controller's actual availability; unsupported/unobserved limits are not
  presented as verified performance controls.
- **Actual, reproduced:** NumPy reports BLAS `accelerate`; `threadpool_info()`
  returns `[]` both before/after native fits; every model reports
  `training_blas_pools = []` together with `training_blas_thread_limit = 1`.
  The same empty list occurs with requested native ALS threads 1 and 4.
- **Root cause:** entering `threadpool_limits(...)` does not establish that a
  supported BLAS controller exists. There is no conditional status/qualification
  for an empty controller list. The native ALS `num_threads` argument is a
  separate control; passing it does not prove an Accelerate BLAS cap.
- **Minimal suggested fix:** say that `threadpoolctl` **requests** limits for
  supported pools. Distinguish the requested limit from an observed/verified
  status in diagnostics; with no detected BLAS pool, mark it unverified. Do not
  infer verification from `all([])`. A supported nonempty controller observation
  can verify its own reported thread count. No algorithm/library replacement or
  process-wide environment mutation is requested by this review.
- **Evidence test:**
  `test_missing_blas_controller_is_observed_as_empty_not_a_thread_measurement`
  reproduces the no-controller path without fabricating a pool. It deliberately
  does not invent a new required diagnostics schema or fail otherwise-correct
  scoring. The actual environment reproduction is below.

Minimal reproduction, from `llm4rec/midterm`:

```sh
.venv/bin/python - <<'PY'
import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_info
from sweettv.contracts import TrainingData
from sweettv.models.mf import MatrixFactorizationModel

data = TrainingData(['u'], np.array([11, 22]), csr_matrix([[1, 5]]), [{}, {}])
model = MatrixFactorizationModel(factors=2, iterations=1, num_threads=1).fit(data)
print(np.__config__.show(mode='dicts')['Build Dependencies']['blas']['name'])
print(threadpool_info())
print(model.diagnostics['training_blas_thread_limit'])
print(model.diagnostics['training_blas_pools'])
PY
```

The exact minimal command above was executed and printed `accelerate`, `[]`,
`1`, `[]`. This finding does **not** assert the actual number of Accelerate
execution threads; the available controller cannot establish it.

## Independent numerical and contract challenges

The new fixture is global 6 users x 9 items with compact warm users `[1,4]` and
warm items `[1,5,7]`. Compact qualifying counts are `[[1,5,0],[0,2,8]]`. Cold
entities occur at the boundaries and internally. Raw program IDs are unsorted,
noncontiguous, and unrelated to matrix indices; user IDs include leading zeros
and nonnumeric strings. All metadata categories coincide and held-out-looking
provenance is present, so neither can legitimately create additional warm edges.

| Requirement / attempted counterexample | Independent evidence | Assessment |
| --- | --- | --- |
| Users x items orientation, including nonsquare compact graph | Probe receives 2x3, not 3x2; all 6x9 global scores checked | Pass |
| Cold compaction/global identity | Fixture-derived scalar maps, not production mapping helpers; all global user/item indices checked | Pass |
| Confidence `1 + alpha*log1p(count)` exactly once | Python `math.log1p`, float32 boundary rounding, scales 0/2.3/20; library alpha recorded as 1 | Pass |
| Actual implicit objective, including missing confidence 1 | Independent dense normal-equation solves for one native ALS iteration | Pass |
| Repeats remain eligible | Watched fixture items retain scores -3 and 0 | Pass |
| Legitimate negative and genuine zero latent scores | Fixed nonzero vectors give `[-3,0,1]`, with no clipping or missing-value substitution | Pass |
| Candidate/user order and duplicates | Scalar factor oracle; reordered/repeated user/candidate positions | Pass |
| Unknown/cold indices, no negative wrap | Negative extremes, out-of-range values, scalar `10**100`, and uint64 maximum | Pass |
| Empty fit/refit and changed dimensions | Four empty shapes, warm -> empty -> different warm shape, rebuilt mappings | Pass |
| Mutation and failure isolation | Input/count/index immutability, fitted snapshot, distinct model/old-fit buffers, propagated native-fit exception preserves prior successful state | Pass |
| Bounded requested batch output | Exact boundary accepted; over-cap request rejected before guarded dense allocation, even for unknown-only input | Pass |
| Native CPU and dtype | Actual `implicit.cpu._als` solver, float32 parameters/scores; no mock substituted in native checks | Pass |
| Seed/thread numerical reproducibility | Repeated tiny native fits with requested threads `[1,1,4,4]`; independent factor-dot oracle | Pass on this environment, not a universal guarantee |
| Actual enforced BLAS thread count | No controllable BLAS pools detected for Accelerate | Unverified; M1 |

### Stronger oracle than a score-to-score comparison

The native oracle uses factors=2, one iteration, seed=79, alpha=2.3,
regularization=0.2, and the compact counts above. It reconstructs only the audited
v0.7.3 float32 random initialization sequence, then independently solves

`x_u = solve(Y.T @ diag(c_u) @ Y + lambda*I, Y.T @ (c_u*p_u))`

and the corresponding item equations using `numpy.linalg.solve`. Here observed
`p_ui=1`, missing `p_ui=0`, and missing `c_ui=1`. No production confidence/index
helper, implicit solver, or score method is used to calculate expected factors.
Two latent dimensions with three CG steps make this a small exact-update oracle
within float32 rounding, not a claim of general ALS convergence/global optimality.

In the explicit exploratory reproduction, maximum absolute differences from
the direct-solve oracle were:

- User factors: `2.218308945e-8`.
- Item factors: `2.137313262e-7`.
- Requested factor-product scores: `8.704655807e-8`.

The permanent native test repeats this check on the cold-padded global fixture.
Scoring is also checked separately with scalar float64 products and
`math.fsum`, without calling either production scoring path as its oracle.

## Executed commands and results

From `llm4rec/midterm`, with Python 3.13.6, NumPy 2.5.4, SciPy 1.18.1,
implicit 0.7.3, pytest 9.2.0, macOS 26.6.2 arm64:

| Command / check | Observed result |
| --- | --- |
| `.venv/bin/python -m pytest -q tests/test_mf.py` | 60 passed in 0.27 s |
| `.venv/bin/python -m pytest -q tests/test_mf_adversarial.py` before final no-controller evidence case | 17 passed in 0.21 s |
| `.venv/bin/python -m pytest -q tests/test_mf.py tests/test_mf_adversarial.py` final run including compiled-solver module assertions | 78 passed in 0.30 s, including 18 new adversarial cases |
| Minimal M1 reproduction command above | Printed `accelerate`, `[]`, `1`, `[]` |
| `shasum -a 256 sweettv/models/mf.py tests/test_mf.py CONTRACT.md` | Hashes recorded above; implementation/author tests unchanged |
| Installed CPU ALS `fit`/solver source inspection | Confirms users-by-items input, internal transpose, library alpha multiplication, and compiled native CG dispatch |

An earlier combined run after the no-controller evidence case passed 78 tests
in 0.38 seconds. The final 0.30-second run also includes the compiled native
solver-module assertions; no earlier run is substituted for their verification.

## Small native CPU smoke and memory accounting

No large model was trained. A separate synthetic smoke used NumPy generator seed
20261031, 96 users x 127 items, eight random observed entries per user, integer
weights 1..6, and 752 aggregated nonzeros. One cold item leaves compact shape
96x126. Models used factors=8, four iterations, regularization=.1, confidence
scale=20, and model seed=79. Requested output was 24 users x 64 candidates.
Imports and fixture creation preceded timing; native initialization/cache warmup
can still affect the first fit. This ran in a shared development environment
alongside a brief test command, not as a controlled benchmark.

| Requested ALS threads | Total fit ms | Native training ms | Batch score ms | Max score difference vs first |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14.716 | 6.767 | 0.948 | 0 |
| 1 | 2.427 | 0.771 | 0.740 | 0 |
| 4 | 2.166 | 0.666 | 0.730 | 0 |
| 4 | 2.111 | 0.647 | 0.711 | 0 |

All score arrays were float32. Each requested output occupied 6,144 bytes;
user+item factor arrays occupied 7,104 bytes. Whole-process peak RSS was
62,619,648 macOS bytes, including imports and four retained smoke models; it is
not an isolated estimate of one model's memory. `training_blas_pools` and
post-fit `threadpool_info()` remained empty. These observations do not establish
a thread-count speedup, full-scale runtime, or recommender quality.

For compact warm sizes U/I, F factors, M nonzeros, T iterations, and S inner CG
steps (default three), the usual native ALS work scales approximately as
`O(T*S*((U+I)*F**2 + M*F))`; transient Gram/sparse/solver work is additional to
`O((U+I)*F)` factor storage and global index maps. No dense global UxI score cache
was found. Single-user scoring is O(C*F). Requested R-by-C batch scoring is
O(R*C*F), with result storage O(R*C) plus factor gathers and a temporary product.
`max_batch_scores` caps **output elements**, not total process/workspace memory:
the default 1,000,000 float32 output scores are 4,000,000 bytes, while temporary
arrays can make the actual allocation larger. The existing docstring explicitly
limits this claim to requested output, so this is a tradeoff, not a defect.

## Rejected bug claims / design constraints

- **Frequency confidence is not frequency regression.** Increasing a count
  increases confidence in binary affinity; it does not set the ALS target to
  that count. This matches the declared implicit objective. No claim that this
  is the optimal frequency-ranking model is made without validation data.
- **Cold compaction is not dropped identity.** Compact factors plus complete
  global maps preserve all score positions and correctly return cold NaNs.
- **Float32 MF is not the prior float32-PPR plateau issue.** This native ALS
  implementation has its own dtype/finite-iteration semantics. The PPR
  fixed-point tolerance requirement does not turn float32 MF into a defect.
- **Aliasing:** each model/previous fit/input is isolated by the executed checks.
  The current `model.user_factors` and its own `als_model_.user_factors` do share
  memory (`np.shares_memory` observed true). The public fitted-parameter/backend
  objects are mutable, not an advertised immutable external-mutation API;
  changing them deliberately can invalidate cached evidence masks. No cross-model
  or training-data alias defect was reproduced.
- **Failed refit keeps prior state:** this is explicitly documented, raises the
  failure, and passed the native-failure counterexample. It is not silent reuse.
- **Native determinism:** zero score differences were observed in the tiny
  thread/seed smoke, not guaranteed bitwise identity across platforms/libraries.

## Author response requested

Respond to M1 by qualifying the unsupported BLAS-limit claim/status, or provide
actual supported-controller evidence that establishes the claimed cap. Preserve
the distinction between native ALS thread requests and BLAS thread enforcement.
Rerun the author's and adversarial suites after any change. No algorithmic MF
correction or Kaggle metric claim is requested on the evidence available here.
