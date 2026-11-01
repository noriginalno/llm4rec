# Independent challenge: EvidenceBlendModel

Reviewed 2026-11-01. Scope: `research_v2/fusion.py:EvidenceBlendModel`, the shared
contract, research protocol/selection rules, its scoring helper, and the relevant
context/evaluation path. Only this review and
`tests/test_research_fusion_challenge.py` were added for this challenge. No
production source, existing tests, configurations, model-selection results, or
submission artifacts were changed. No research outputs, APIs, credentials,
external/public predictions, submissions, commits, pushes, or subagents were used.

## Verdict

**No blocker reproduced for the default RRF60/depth200 path or ordinary-scale
count blending.** The independent arithmetic, evidence, query, fit, and context
checks pass. One **low-severity extreme-value count overflow defect** is
reproduced below; its regression is a strict expected failure, not a passing
integrity check. This review does not establish model quality or recommend MAP
hyperparameters.

### Reviewed source binding

- `research_v2/fusion.py` SHA-256:
  `8d1272cc36cf7f371ae5d5b581b339ffc20e2f00458b14222d9edf244adb7474`.
- `research_v2/contracts.py`:
  `9202448a3b0fb30a8e902112fcbdb825ea78e5901a50744b50e1d5d3aaa9c7e0`.
- `sweettv/evaluation.py`:
  `e676e10af9b5909f91da0282a0b884d20a791e565ad6492ef27a03bc4f352d94`.
- Read `CONTRACT.md`, `research_v2/PROTOCOL.md`, and `research_v2/SELECTION.md`.
  Selection was extended concurrently with the raw-habit grid declaration and
  was reread; this challenge neither inspected its running results nor selected
  configurations. Fusion/context/shared-ranker hashes above stayed unchanged
  across the review's initial and post-test checks.

## Independent oracle and coverage

The new tests compute RRF votes by Python sorting and `Fraction` arithmetic;
count expectations use literal weighted sums and `math.fsum`. They do **not**
import the shared RRF/ranking helpers as an expected-value oracle. Hand-derived
fixtures supplement five deterministic randomized fixtures per mode/tie direction.
Synthetic score providers are mounted as temporary in-memory attributes of a
local research module; no module files are generated. Their source hashes bind
the actual test file. Separate integration tests fit the real locally implemented
`TemporalRepeatModel`, `AiringPriorRepeatModel`, and `EASEModel` on tiny synthetic
data. The EASE count check has an independent two-feature ridge-system derivation.

| Requirement / challenge | Executed evidence | Assessment |
| --- | --- | --- |
| Rank blend is `sum_m w_m/(60+rank_m)`, evidence-only | Hand-derived signed/zero/NaN fixture; scalar oracles under reordered/duplicated queries | Pass |
| Count blend is unscaled additive, not probability/calibrated frequency | Exact `[8, 8.5, -4.5, -2.5, 0, NaN]` fixture and real EASE/repeat derivation | Pass |
| Unsupported positions stay NaN; no popularity-backfill votes | All-NaN/cold profiles; another user's globally popular program stays NaN for the queried user | Pass |
| Negative values and genuine cancellation zeros remain evidence | Negative component scores receive rank votes; literal count sums can be negative or supported zero | Pass |
| Zero-weight components receive no vote or fit | Disabled nonexistent class is never imported; absent evidence is not made finite | Pass |
| Duplicate/permuted/read-only candidate queries | Oracles and duplicate-heavy cutoff queries; repeated IDs do not consume ranks/depth | Pass |
| Default depth is 200 **per component**, not a final global cap | 205 tied candidates; two disjoint top-three lists give six supported fused positions | Pass |
| Reverse ties are explicit diagnostics | Ascending/reversed raw-ID tie fixtures; untied components unchanged; no direction selected using quality results | Pass, with limits below |
| Reject malformed queries and component scores | Invalid scalar users, shapes, float/bool indices, infinity; uint64-sized unknown indices do not wrap | Pass |
| Fit/score input invariance | Full context/CSR/metadata/array snapshots; original constructor configuration and read-only component buffers preserved | Pass |
| Future behavior does not become a feature | Fixed EPG with deleted/reassigned synthetic held-out events changes labels but not fit inputs or any user's scores | Pass for tested real composition |
| Fresh refit and rejected-refit state | Successful refit creates new component objects; component rejection preserves prior predictions/diagnostics | Pass |
| Source binding | Actual class-file digests recorded for three real component classes; simulated before/after source-byte change rejects fit without disk edits | Pass at declared direct-source scope |
| Opposite count overflows must not impersonate missing evidence | Finite weights `1e308,1e308`, finite scores `2,-2` return `[NaN]` | **FUS-1 reproduced**, low severity |

The context checks demonstrate behavior of the tested composition, not that a
generic plugin loader can guarantee arbitrary component honesty. This is a
trusted local-code boundary, not a security sandbox. Direct class-file hashes
are not a claim of recursively hashing every imported dependency; the evaluator
separately binds its shared helpers/requirements. No speculative import/security
escalation is alleged.

## FUS-1: arithmetic NaN bypasses the overflow guard

Location: `research_v2/fusion.py:93–101`.

1. Two supported finite score vectors contain `+2` and `-2` for the same item;
   their positive finite weights are both `1e308` (accepted by the constructor).
2. Each weighted product overflows: `+inf` and `-inf`.
3. Accumulation becomes `NaN`, although exact arithmetic gives supported zero.
4. `supported` is true, but the final guard tests only `np.isinf(values)`.
   `isinf(NaN)` is false, so the model silently returns missing evidence.

**Measured result:** `[nan]`, reproduced with the expected-failure marker
disabled. This is a numerical edge case, not evidence of a defect at normal
qualifying-count scales or of higher/lower MAP.

**Requested author response:** reject nonfinite products/accumulations on
supported positions, or compute a finite correct result. Check before/after
assigning deliberate NaNs to unsupported positions so legitimate missingness is
not rejected. The test accepts either explicit `FloatingPointError` or supported
exact zero; it must not silently become NaN. No production fix was made here.

## Cap, ties, and scale: intended behavior with real consequences

- RRF support is the union of positive-weight components' **finite top-200
  unique query items**. A finite component score at rank 201 contributes nothing;
  if every component truncates the item, the blend returns NaN. This is deliberate
  truncation, not a discovered internal fallback or shape bug.
- The cap can make tie direction change **eligibility**, not just ordering. With
  205 equal-score raw IDs 1000–1204, ascending keeps 1000–1199; reversed keeps
  1005–1204. The tests verify both exact sets and duplicate-query invariance.
- `reverse_score_ties` reverses ties **inside each component's rank list**. It
  does not reverse the final shared ranker's equal-fusion-score ID tie-break,
  and is a no-op in count mode. Label-boundary reversal in `evaluate.py` is a
  separate sensitivity. Do not present one diagnostic as covering all three.
- RRF is listwise/query-relative: score 1 at full-list rank 3 gets `1/63`, but
  querying that item alone gives `1/61`. Candidate-axis chunking is not equivalent
  to a full-catalog RRF call. The inspected evaluator passes the full fold
  candidate vector per user, so this is **not a current execution blocker**.
  Count mode is pointwise and does preserve this subset equivalence.
- Positive rescaling preserves component RRF order, support, and votes when
  finite values/ties are preserved. It does not preserve count blending: repeat
  evidence 100 versus discovery evidence 1 favors repeat; multiplying only the
  discovery scores by 101 reverses the count outcome. The tests verify both
  modes. The word `count` must not imply compatible units for raw/log/normalized
  reconstructions, probability calibration, or a scale-independent tradeoff.

## Repeat versus discovery: mathematical conditions, not tuning advice

Let `C=60`, `D=200`, `w_R` be the repeat weight, and `A_i` the sum of auxiliary
RRF votes for item `i`. Missing/truncated components contribute no term.

For a repeat-supported item `p` at repeat rank `r<=D` and a repeat-unsupported
discovery item `d`:

```text
F(p) = w_R/(C+r) + A_p
F(d) = A_d
d strictly outranks p  iff  A_d - A_p > w_R/(C+r).
```

Equality is decided by the final numeric raw-ID ascending tie-break. If auxiliary
weights sum to `W_A`, then `0 <= A_i <= W_A/(C+1)`. Therefore a sufficient
uniform condition for repeat's first `k<=D` items to outrank every nonrepeat is:

```text
w_R/(C+k) > W_A/(C+1).
For k=5: w_R > (65/61) * W_A.
For every retained repeat through rank 200: w_R > (260/61) * W_A.
```

These are sufficient worst-case bounds, not necessary conditions. Repeat can
dominate with weaker weights when auxiliary votes also support repeats or do not
agree on discoveries. A discovery absent from all component top-D sets cannot
receive a fused evidence rank at all; shared final fallback remains a separate
operation when fewer than five evidence items exist.

Counterexamples executed:

- Repeat weight 65 versus auxiliary weight 61 gives an exact tie between the
  repeat's fifth item and the auxiliary's first discovery: both score 1. A lower
  raw discovery ID wins. Repeat weight 66 strictly preserves all five repeats;
  weight 64 does not. These are algebra fixtures, not suggested trial weights.
- Repeat weight 1 plus four auxiliary weights 0.25 does **not** prohibit
  discovery: unanimous first-place discovery votes give `1/61`, exceeding a
  fifth repeat's unassisted `1/65`. Thus neither “repeat guarantees all five” nor
  “having four discovery models guarantees discovery” follows from that design.

For count blending, write `F(i)=w_R*R_i+A_i`, with missing component terms omitted
and all-unsupported items still NaN. For supported items, discovery beats repeat
exactly when `A_d-A_p > w_R*R_p` (assuming `R_d` absent). There is **no universal
weight-only guarantee** with unbounded signed auxiliary scores. If each auxiliary
score is bounded in absolute value by `M_m`, then the conservative bound
`A_d-A_p <= 2*sum_m(w_m*M_m)` gives a sufficient repeat-dominance condition;
for scores constrained to `[0,M_m]`, the factor 2 can be removed. Missingness is
not an implicit supported zero that lets an unsupported item beat negative
evidence. None of these conditions is a MAP recommendation.

## Executed checks and reproduction

Environment: existing `.venv`, Python 3.13.6, NumPy 2.5.4, SciPy 1.18.1,
pytest 9.2.0. No installs. Bytecode and pytest cache writes were disabled.
Run from `llm4rec/midterm`:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_research_fusion.py tests/test_research_fusion_challenge.py
```

Final combined result: **57 passed, 1 xfailed** in 0.24 seconds, comprising
**49 passing new challenge cases**, the documented FUS-1 xfail, and the original
**8/8 passing author cases**. The three production source hashes above were
rechecked after the final run and were unchanged. Challenge test SHA-256:
`a2987ba171e264cee2a934e1cb97f6df62e9a40e7d14f0462ddc9644bc1b91b5`.

To expose FUS-1 as an ordinary failure instead of the documented strict xfail:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider --runxfail tests/test_research_fusion_challenge.py -k opposite_count_overflows
```

Observed failure: `AssertionError: supported cancellation returned [nan]`.
The strict marker should be removed after an author fix passes this same input;
an unexpected pass deliberately fails the marked suite until reviewed.

## Follow-up: FUS-1 closure and independent batch challenge

Follow-up on 2026-11-01; this section supersedes the earlier open-FUS-1 status.
The parent fixed production code and removed the FUS-1 expected-failure marker.
This reviewer only **appended** cases to the owned challenge test and this review;
no parent test or production source was edited.

**Current result: FUS-1 closed. User-axis batching passes. One additional nested
duplicate-query contract defect, FUS-2, is open.** No failure was reproduced for
the planned unique-candidate fold path. This is not a blanket all-cases clearance:
the duplicate-input API counterexample below is real and was reported before
the parent fit, rather than treated as a passing check.

### Actual source checks

Read the updated `_blend` and `score_batch` implementation, plus the real
Legacy/PPR delegation path. Hashes were taken before the follow-up and again
**after** the final test commands completed. The following matched at both ends:

- `research_v2/fusion.py`:
  `5b3d2d4db214b45c9cd7cbf7e277418dd5173297ea1a6b44709cdaa630ef923d`.
- `research_v2/legacy.py`:
  `50a8abbf9cbaa49c8a28df7ef473b360044d891a27cebed5116f114081c9abba`.
- `sweettv/models/ppr.py`:
  `eccc19a05966433925d422a3a09a19edc6ad2dbd2c5bedfa94270f3414b92bc5`.
- `research_v2/contracts.py` and `sweettv/evaluation.py`: unchanged from their
  full hashes in the original review above.

**Not all source was frozen.** The separately owned EASE source changed from
`a39bb66d253a11038b4171ded98da75e48879550371d4f59bd78ed138f049e5a`
to `640183b589a3172548c51890ff1a3d5cd6c5e54e20960f3a507aa2f716b02242`
during this follow-up, consistent with the announced independent EASE work.
Its old tiny integration case passed in the suite, but that does not certify an
all-source freeze or its separately changing index edge case. The new Legacy/
repeat batch integration does not use EASE. No MAP experiment, scoreboard,
prediction, credential, or API files were inspected or used.

### FUS-1: independently closed

- The original unmarked exact-arithmetic cancellation regression passes.
- A new scalar-and-batch regression confirms that weights `1e308,1e308` and
  scores `2,-2` now raise `FloatingPointError`; they do not return misleading NaN.
- The same large-weight composition still returns deliberate NaNs for genuinely
  unsupported/cold/unknown rows without an exception.
- Cause/fix verified in source: arithmetic uses `np.errstate(over="raise",
  invalid="raise")`, followed by a finite check on supported positions before
  unsupported positions are assigned NaN (`fusion.py:98–107`).

### Batch checks that passed

Added 37 passing follow-up cases plus the FUS-2 expected-failure regression.

| Check | Independent evidence | Result |
| --- | --- | --- |
| Native batch and scalar-only components compose correctly | Scalar/Fraction oracle for signed/zero/NaN tables, three seeds, both modes and tie directions; scalar API and user-partitioned calls agree | Pass |
| Whole candidate vector reaches each component | Recording native fixture receives one full valid vector per call, including duplicate positions, never candidate chunks; both inputs are read-only | Pass |
| RRF remains listwise with depth200 | Full 205-item tied catalogs match the scalar oracle; exact 200-ID retained sets and duplicate-query behavior verified for pointwise table components | Pass |
| Missing/negative/zero evidence and zero weights | Independent oracle includes all-unsupported columns, signed counts and cancellation; disabled nonexistent component never loads or votes | Pass |
| Batch alignment/invariance | Duplicate/permuted/unknown users, read-only input preservation, full context snapshots, no result/cache alias; user-axis partitions reproduce full batches | Pass |
| Future behavior excluded | Deleted synthetic held-out viewing changes labels but not either real composition's batch inputs or scores, with fixed EPG | Pass |
| Empty/invalid input handling | Empty axes, huge Python/uint64 unknown indices, invalid scalar/matrix/bool/float/string users/candidates; unknown-only requests do not invoke components | Pass |
| One-million requested-cell bound | Exactly 1000×1000 allowed; both 1001×1000 and 1000×1001 rejected, even when all rows/columns are unsupported | Pass |
| Malformed native outputs | Wrong batch dimensions and infinity in a later row reject the call | Pass |
| Real Legacy/PPR path, unique candidates | Both outer modes match independently reconstructed inner and outer RRF/count sums; batch, scalar, and user-partitioned output are exactly equal on the fixture | Pass |

The real integration uses the unchanged Legacy models, not a replacement for
PPR/ALS. Immediately after its outer batch, PPR diagnostics show four requested
known rows, two distinct active users propagated together in one internal batch,
and the full candidate count. This verifies delegation, not a measured speedup.
The expected nested fusion is computed from each constituent's evidence using
the independent scalar oracle, rather than calling the shared RRF helper or
Legacy's own fused output as the oracle.

### FUS-2: duplicate queries consume nested Legacy depth before outer deduplication

This is a preexisting nested-listwise issue exposed by the expanded real-model
challenge, **not** a scalar/batch parity regression or a PPR defect.

Reproduced configuration: ordinary RRF60/depth200, temporal unweighted repeat
weight1 plus `LegacyHybrid` weight0.5, on the existing tiny synthetic fixture.
User Alice has training counts 101→2 and 307→1; another user watches 9007.

| Query | Score101 | Score307 | Score9007 |
| --- | --- | --- | --- |
| Unique catalog `[18,101,307,9007,60001]` | `1.5/61` | `1.5/62 = 0.024193548387096774` | `0.5/63 = 0.007936507936507936` |
| Same catalog plus 200 additional positions for101 | `1.5/61` | `1/62 = 0.016129032258064516` | `NaN` |

The duplicate positions consume the inner Legacy top200 ranks, so the later
items lose their Legacy evidence. The outer blend's deduplication cannot restore
evidence already truncated by its component. Scalar and batch calls produce
the **same wrong duplicate-dependent output**. The independent nested oracle
reconstructs the unique-item inner votes, and correctly expects simple expansion
of the unique-catalog output to duplicate positions.

Root path: `fusion.py:80–82,132–142` forwards duplicated valid indices to a
component; `legacy.py:23–34` assigns inner RRF ranks to positions without
deduplication; only later does `fusion.py:92–96` deduplicate the outer rank list.

**Scope/action:** medium API-integrity edge case, not evidence that the planned
research catalog is duplicated. The inspected production fold builds its
candidate vector from a sorted **set** (`sweettv/data.py:905–906`), so the declared
unique-candidate raw/legacy/repeat grid avoids this counterexample. Either
canonicalize unique candidates before invoking listwise components and expand
after fusion, or repair Legacy's inner ranking. Until repaired, do not claim
duplicate-query invariance for arbitrary nested Legacy compositions. No source
fix or grid change was made by this reviewer.

### Executed results and reproduction

Same existing environment and no-cache/no-bytecode command as above:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_research_fusion.py tests/test_research_fusion_challenge.py
```

Final result: **97 passed, 1 xfailed** in 1.08 seconds: 87 passing challenge cases,
10 passing parent cases, and the strict FUS-2 xfail. **FUS-1 has no xfail marker.**
The baseline before the appended cases was independently rerun: **60 passed**.
The original FUS-1 test alone also passed independently.

Forced FUS-2 reproduction:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider --runxfail tests/test_research_fusion_challenge.py -k nested_legacy_duplicate
```

Observed: **1 failed, 87 deselected**, with `nan location mismatch` and the
explicit message `duplicate positions consumed the nested LegacyHybrid depth
budget`. This is a reproduced failure, not an unexecuted proposed test.

Follow-up challenge-file SHA-256:
`dae9a8e98ac80bc6ace74e76d339de04a4800fe8756cce6830e07cdc8d4fecd5`.
No model quality estimates, tuning advice, submission authorization, or general
source-freeze claims follow from these checks.

## Final closure: FUS-2 repaired; no open fusion blocker found

Independently rechecked 2026-11-01 after the parent's Legacy fix and removal of
the FUS-2 xfail marker. **FUS-1 and FUS-2 are now closed.** This section supersedes
the earlier open-defect statuses. Only this review was appended in this closure
round: no `.py`, tests, source configurations, selection artifacts, or result
files were edited. No confirmation/primary12 outputs were opened.

### Exact counterexample and nested oracle

- Original FUS-2 case, run with `--runxfail`: **1 passed, 87 deselected**. It is
  genuinely unmarked, not a retained xfail hiding a failure. The catalog plus
  200 extra positions for program101 now reproduces simple expansion of the
  unique-catalog scores: program307 retains `1.5/62`; program9007 retains
  `0.5/63`. Scalar and batch calls agree.
- FUS-1, its scalar/batch closure case, FUS-2, and both real nested Legacy/PPR
  oracle modes together: **5 passed, 83 deselected**. Expected nested votes are
  still computed by the independent scalar/Fraction oracle, not the production
  RRF helper.
- Full fusion author + challenge files: **98 passed, zero xfails**, in 0.99
  seconds. This independently includes existing invalid-query, cell-bound,
  evidence, input-invariance, and full-candidate listwise checks. The parent's
  additional four selection cases were not rerun here; its reported combined
  102-case result is not substituted for this reviewer's observed 98-case run.

Commands, from `llm4rec/midterm`, with the existing environment and bytecode/cache
writes disabled:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider --runxfail tests/test_research_fusion_challenge.py -k nested_legacy_duplicate
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_research_fusion.py tests/test_research_fusion_challenge.py
```

### What changed, and the numerical-equivalence check

Inspected `legacy.py:23–37`: candidates are canonicalized with
`np.unique(..., return_inverse=True)` **before** every constituent call and inner
RRF. Fused results are expanded with `[:, inverse]`. Empty user requests return
the correctly shaped zero-row float64 array. All constituent names, fit defaults,
weights, and the RRF60/depth200 definition remain unchanged in the inspected
path; `fusion.py` and the shared RRF source retain their prior hashes.

An additional read-only in-memory check, without adding or editing a test file,
compared both Legacy tie directions over four unique-item queries: full catalog,
reversed catalog, permutation `[3,0,4,2,1]`, and subset `[2,1]`. Users were ordered
`other, alice, cold, other` on the synthetic fixture.

- **8 exact matches** against a reconstruction of the pre-fix wrapper's behavior
  on unique inputs. That comparator uses the unchanged shared RRF only to assess
  old/new wrapper equivalence, not as the independent mathematical oracle.
- **8 independent scalar/Fraction oracle comparisons passed.**
- **8 duplicate-expansion/scalar/batch checks matched exactly**, including 200
  repeated warm-item positions, both tie directions, and all empty-axis shapes.
- Read-only queries and full context snapshots were unchanged.

Thus no ordinary scoring-definition change was found, and unique-query scores
were exactly preserved on these executed fixtures. This is not a replay of all
development results or a claim of universal bitwise equivalence for every
possible native numerical backend. The intended changes are duplicate-query
semantics and the previously unsupported empty-user shape.

### Current source binding

The six named production files and two test files were hashed at the start and
after checks; all eight matched across this closure round. This is a scoped hash
check, not a claim about every file in the workspace.

- Fixed `research_v2/legacy.py`:
  `077d84e2e40c467324c00e06ae0c2ab32a67ce95182f5cceba602752a7af1f33`.
- `research_v2/fusion.py`:
  `5b3d2d4db214b45c9cd7cbf7e277418dd5173297ea1a6b44709cdaa630ef923d`.
- `research_v2/contracts.py`, `sweettv/evaluation.py`, and
  `sweettv/models/ppr.py`: unchanged from the full hashes above.
- `research_v2/collaborative.py`: remained
  `640183b589a3172548c51890ff1a3d5cd6c5e54e20960f3a507aa2f716b02242`
  throughout this closure round, unlike the earlier changing-source round.
- Unedited challenge test:
  `00e32cc39433ffdb7fe0eb0995b7629f23d7fbfe068efb7f245293db470ed54d`.
- Unedited parent fusion test:
  `a861b0bfbb276c269bd5041106955169220e236f3b5e76c32126913fed2cffd0`.

The current Legacy hash differs from the development-era wrapper; archived
results must retain their actual historical source bindings rather than being
relabeled as runs of the repaired source.

### Frozen shortlist and remaining gates

Read only the permitted development-selection artifacts
`results/research-v2/shortlist-001/selection.json` and `grid.json`. They name
exactly two new candidates, `raw_day_h56_p1` and `raw_legacy_w05`, with
`development_only=true`, `confirmation_seen=false` at selection, and an explicit
no-retuning decision rule. No new candidate or weight is proposed here.

The grid structurally contains R5's nested tie control: both the outer blend
and its inner Legacy component have `reverse_score_ties=true`, plus a separate
reversed Legacy reference. This verifies the configuration, **not** its outcome.
Artifact hashes read here:

- `selection.json`: `f5a5ad0fd73bd34a33c12a63c611f6cc16bec7d8eb321ec1c611f5d6e656e63b`.
- `grid.json`: `4e3fca45526b28023c05ad49c87d8385ac32427c1a25781117e740d9c8843795`.

Still outside this closure: the parent's full-source-hashed suite completion;
the frozen confirmation and primary12 comparisons on both strict-five and
all-template cohorts; and R5 nested/reversed-label tie sensitivity not reversing
the conclusion. These remain for the parent to evaluate without retuning.
Reused historical windows are not untouched test data. No confirmation success,
MAP improvement, delivery completion, or live-submission authorization follows
from this code review; the selection artifact explicitly authorizes no new
Kaggle submission.
