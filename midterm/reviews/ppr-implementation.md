# Personalized PageRank implementation record

## Scope and interface

Author-owned code: `sweettv/models/ppr.py` and `tests/test_ppr.py` only.
This note is the optional PPR review handoff. No shared implementation, package
configuration, data, credentials, or other author's model was modified.

`PersonalizedPageRankModel(alpha=0.85, tol=1e-9, max_iter=300, batch_size=16)`
exports `fit(TrainingData)`, `score(user_index, candidate_indices)`, and
`score_batch(user_indices, candidate_indices)`. Inputs are matrix indices;
outputs preserve both input orders and duplicates. Raw IDs remain separate.

The graph is the training-count-weighted, bidirectional user/program graph.
The transition is CSR, row-normalized; all graph construction and walk arithmetic
are float64. `TrainingData` itself stores float32 counts under the shared contract;
conversion cannot recover precision already lost before this model receives them.
Neither metadata nor held-out behavior/provenance supplies edges. Fitting copies
the matrix, and no watched-item filter or internal fallback is applied.

## Convergence and missing evidence

For restart one-hot e_u and row transition P, each update is

`F(p) = alpha * P.T @ p + (1-alpha + alpha*dangling_mass(p)) * e_u`.

At the returned iterate the implementation explicitly measures
`||F(p)-p||_1 <= tol`. Iteration zero is the initial restart; a final residual
evaluation is an extra operator application, not another returned update.
Diagnostics report each queried user's update count, observed fixed-point
residual, residual/(1-alpha) mathematical L1 error bound, total node mass, and
item mass. They do not describe an approximate result as exact/full precision.
For every requested block, failure raises `PPRConvergenceError`, records
`last_query.status = nonconverged`, and returns no partial score array.

Unknown/cold users and unreachable/cold items return NaN. Alpha is validated in
`[0,1)`; at alpha zero all mass is restart mass, so there is no item evidence and
item scores are NaN. Disconnected components do not get artificial positive
bias. Item scores are not normalized to sum to one; at alpha .85 the bipartite
stationary total item mass is alpha/(1+alpha), approximately .45945945946.

## Executed checks

Initial dependency check failed because NumPy was not yet available. After the
coordinator prepared the environment, `.venv/bin/python -m pytest -q
tests/test_ppr.py` passed all 46 then-existing cases in 0.21 seconds. After the
additional near-one alpha edge test and final row-lookup cleanup, the same
command passed **47 tests in 0.20 seconds**.

Environment observed on 2026-10-31: Python 3.13.6, NumPy 2.5.4, SciPy 1.18.1,
pytest 9.2.0, macOS 26.6.2 arm64. Tests include:

- Independent dense construction and `numpy.linalg.solve` oracle for a weighted,
  nonsquare 2-user/4-item fixture, plus full-node probabilities and observed
  residual/error-bound checks; this oracle does not use production iteration.
- Weighted star closed form, row-vs-column normalization orientation, weighted
  vs binary edges, repeat eligibility, and total-node vs item-only mass.
- Directed sink fixture validating each batched query's own dangling restart;
  normal undirected active components cannot reach their isolated dangling nodes.
- Cold/unknown users/items, disconnected components, candidate/user order and
  duplicates, raw-ID separation, empty and zero-row/zero-column matrices.
- Forced nonconvergence at one update, with independently expected last residual
  `2 * .85**2 = 1.445`, and explicit float64/default-tolerance convergence.
- Ignored future metadata/provenance, input immutability and fit snapshot,
  refit reset, parameter/index validation, and bounded batch size. A test forbids
  calling CSR `toarray`/`todense` in production graph construction/scoring.

## Measured synthetic workload (not Kaggle results)

The final in-process CPU run used NumPy generator seed 20261031, 1,600 users,
600 programs, and 12 randomly drawn count entries per user with weights 1..6.
After duplicate aggregation: 19,027 training edges; 2,200 graph nodes; 38,054
transition nonzeros. It requested 1,260 users x 600 candidates at batch size 16.

| Observed quantity | Result |
| --- | ---: |
| Fit time | 0.000968458 seconds |
| Score time | 1.882621375 seconds |
| Propagation blocks / maximum columns | 79 / 16 |
| Updates per query | 131 |
| Maximum fixed-point L1 residual | 9.645578302e-10 |
| Minimum / maximum total node mass | 0.9999999999999998 / 1.0000000000000002 |
| Minimum / maximum item mass | 0.45945945972015056 / 0.4594594597201508 |
| Requested output array bytes | 6,048,000 |
| One dense walk block bytes | 281,600 |
| Whole-process peak RSS (macOS bytes, includes imports/output) | 67,862,528 |

This is a synthetic, warm-import local measurement in a shared development
environment, not a controlled benchmark, real-data extrapolation, or model-quality
metric. The earlier pre-cleanup run took 1.840643250 seconds to score; these two
uncontrolled runs are not evidence that the cleanup improved or worsened speed.
No Kaggle download, authentication, metric, submission, or publication was
performed.

## Complexity and review handoff

Let N be nodes, E directed transition nonzeros, Q unique active requested users,
B <= batch_size the propagation block size, K updates, C requested candidates,
and R requested user rows including duplicates. Fit is O(E+N). Sparse graph
storage is O(E+N); propagation workspace is O(N*B), with a few such arrays alive
per update. The requested score output is O(R*C), not a persistent global
user-item score matrix. Walk time is O((K+1)*(E+N)*Q), plus O(R*C) output handling
and O(R*log(R)) user deduplication. Candidate duplicates are never deduplicated.

The paired adversarial MF review is still pending the coordinator's review stage.
PPR's author should challenge MF without editing its owner's files; the MF author
will challenge PPR. Preserve and reproduce any concrete counterexample, then
modify only these PPR-owned files for a confirmed PPR correction.
