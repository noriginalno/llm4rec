# Round-two collaborative research: EASE author record

Date: 2026-11-01. Implementation and focused checks are complete; independent
challenge and SWEET.TV MAP@5 evaluation are **not** claimed here. No ItemKNN
class was added, to keep the independent implementation/review surface small.

Owned files only:

- `research_v2/collaborative.py`
- `tests/test_research_collaborative.py`
- `reviews/research-v2-collaborative.md`

No existing model, data, contract, runner, or scored artifact was edited. No
Kaggle access, credential access, submission, install, push, or further agent
delegation was performed.

## Method and verified primary source

Harald Steck. 2019. **Embarrassingly Shallow Autoencoders for Sparse Data.**
*The World Wide Web Conference (WWW '19)*, 3251–3257.
[DOI: 10.1145/3308558.3313710](https://doi.org/10.1145/3308558.3313710).

Author/title/year/venue/pages/DOI were checked against
[Crossref's DOI record](https://api.crossref.org/works/10.1145/3308558.3313710).
The primary text was read at [arXiv:1905.03375v1](https://arxiv.org/html/1905.03375).
Relevant passages: Section 2/Eq. 1 (scores), Eqs. 2–3 (objective), Section
3.1/Eqs. 4 and 8 (closed form), and Section 3.4 (item-item memory/cubic solve).
Footnote 2 explicitly permits nonbinary input. The implementation is independently
written from the mathematics; no answer files or public prediction outputs were
imported. Published ranking gains on other datasets are not evidence of a gain
on SWEET.TV.

For a sparse effective training matrix `C`, the feature construction is:

1. With no half-life, `C` is the original training qualifying-broadcast count.
   Otherwise `C_ui = sum_e 2^(-(train_end-end_e)/(86400*h))` over that user's
   pre-cutoff qualifying events for the item. `h` is in days. Source timestamps
   are not shifted or claimed to be observed UTC.
2. Only positive-support training items can be active. Select at most the
   configured cap by **raw original training count descending**, resolving ties
   by numeric raw program ID ascending. Decay does not redefine the selection
   ranking; complete float64 temporal underflow removes an item from eligibility.
   Compact columns are then ordered by global matrix index.
3. Apply `raw`, `log` (`log1p`), or `binary` (`1[C_ui>0]`). Binary is a control
   that deliberately discards repeat magnitude and positive decay magnitude.
4. Optional IDF multiplies each column by
   `1 + log((1+N)/(1+df_i))`, fitted solely from effective training support.
   `N` counts effective warm training users **before truncation**, not forecast
   users, retained-only users, or query-batch users. Optional row L2 follows IDF;
   stable maximum-first scaling avoids squared-norm underflow.

The learned objective is `||X-XB||_F^2 + lambda*||B||_F^2`, with `diag(B)=0`.
The loss is a **sum**, not an average over users. Solve
`(X.T@X + lambda*I) P = I`, then `B_ij=-P_ij/P_jj` for `i != j` and zero the
diagonal. Both inputs and reconstruction targets are in the selected feature
space. Raw features have qualifying-broadcast-count units; log, binary, IDF,
and row normalization alter those units/scales and the useful lambda scale.
There is no inverse feature transform. The score is `X_u@B`, not a calibrated
future count, probability, schedule-adjusted forecast, or guaranteed frequency
regression improvement. Signed coefficients/predictions are retained.

## API and configuration

```python
from research_v2.collaborative import EASEModel

model = EASEModel(
    regularization=1000.0,
    transform="raw",            # "raw", "log", "binary"
    half_life_days=None,         # None or finite positive days
    row_l2=False,
    item_weighting="none",      # "none", "idf"
    max_active_items=4096,       # integer 1..4096, hard ceiling
    num_threads=1,              # requested BLAS limit, not observed activity
    max_batch_scores=1_000_000,  # float64 requested-output cells, 8 MB default
).fit(context)
scores = model.score(user_index, candidate_indices)
batch = model.score_batch(user_indices, candidate_indices)
```

All defaults are explicit engineering choices, not fitted recommendations.
The parent owns the full development/confirmation grid and any repeat blend.

Learned attributes: `training_matrix_` (compact CSR), `coefficients_`,
`item_weights_`, `active_user_indices_`, `active_item_indices_`,
`active_item_ids_`, `user_index_to_row_`, and `item_index_to_column_`.
Maps contain `-1` for absent entities; bounds are checked **before** casts or
indexing. Orders and duplicates in both query axes are preserved. Arbitrarily
large Python integers and uint64 maxima remain unknown, never wrapped indices.

NaN means cold/unknown/truncated entities or no nonzero coefficient from any
retained history feature to that target. Evidence is exact `B_ij != 0`, without
an epsilon threshold. Signed predictions and cancellation zeros with coefficient
evidence remain finite. Disconnected warm components and all-zero coefficient
matrices do not become popularity scores. No watched-item filter or internal
fallback is applied. **A sole-history item cannot recommend itself because
`B_ii=0`; its self score is NaN.** The parent must add temporal-repeat evidence
separately rather than changing EASE's diagonal or hiding a fallback.

## Requirement / evidence matrix

| Requirement | Implemented evidence | Assessment |
| --- | --- | --- |
| Training-only counts; source-clock cutoff checked | `_training_snapshot` rechecks event cutoff/ranges, relevant provenance clocks, and exact event/count agreement before either feature-source mode | Verified in focused tests |
| No forecast behavior/labels, schedule-based fitting, or file access | Score core retains only copied learned arrays; metadata and unrelated provenance are ignored; permitted schedules/candidates do not fit CF | Horizon, schedule, poison-payload, and catalog-growth tests pass; model has no file/network calls |
| Correct asymmetric closed form | Hard-coded rational precision/coefficient/score fixture; independent Python 3x3 adjugate oracle for all feature transforms | Verified |
| Count-sensitive rather than binary-affinity targets | Raw and binary independent fixtures produce different coefficients/scores | Verified; MAP hypothesis unevaluated |
| Deterministic bounded active items and global mapping | Training-only raw-count cap, raw-ID tie fixture, column permutation and interleaved cold indices | Verified |
| Honest evidence and repeat eligibility | Negative and cancellation-zero scores, disconnected graphs, diagonal-only/all-zero/one-item tests; watched pairs remain eligible | Verified; sole-history limitation explicit |
| Bounded memory/output | Hard item ceiling; sparse profile storage; only item Gram matrix is densified; output-cell guard precedes allocation, even for cold/duplicate requests | Verified on fixtures; maximum-cap peak memory not measured |
| Refit/mutation safety | Fresh snapshots; warm→empty→singleton→warm refits; failed cutoff/native fits preserve previous state | Verified |
| Native BLAS request versus observation | Requested limit and reported pools/status stored separately; no-pool and Accelerate reports stay unverified | Verified classification; actual pool report empty here |
| No numerical substitution | Positive finite ridge/config checks; solve warnings/failures and invalid precision matrices raise, without pseudoinverse or changed lambda | Verified, including named fault injections |

Diagnostics include original/effective/active warm dimensions, original and
retained raw count mass, nnz/item/user coverage, truncated item count, items lost
to temporal underflow, transformed feature range, coefficient counts/bytes,
sole-history users, cutoff/observed event maximum, runtime, and BLAS control
request/report. Coverage is **training retention**, not a claim of test accuracy
or candidate-level recommendation coverage. Per-query coverage follows the
returned NaNs and is for the parent evaluator to report.

Dense coefficient storage is `8*K*K` bytes plus `K*K` support-mask bytes. At
4096 active items those are 128 MiB and 16 MiB. Gram, identity, solver work, and
possibly a writable precision copy add further bounded item-item allocations;
these numbers are not a peak-RSS measurement. No dense global user-item cache
is retained. Batch output is bounded, and intermediate scoring uses one compact
item vector per user. Threadpoolctl reports supported controller configuration,
not actual active worker counts; Apple Accelerate remains unverified.

## Actual executed checks

Command from `midterm/`, using the pre-existing environment, with no install:

```text
.venv/bin/python -m pytest -q -p no:cacheprovider --tb=short tests/test_research_collaborative.py
98 passed in 0.36s
```

Environment observed: Python 3.13.6, NumPy 2.5.4, SciPy 1.18.1,
macOS 26.6.2 arm64. This is the focused author suite only, not the parent's full
suite or an independent cross-review. Synthetic fixtures invoke actual SciPy
CPU solves except explicitly identified fault-injection/controller tests.

Post-check source SHA-256 fingerprints (both also passed an AST parse):

- `research_v2/collaborative.py`:
  `418c5791b27372eb12daf97716fa0e25e08a56aa00113f5b16ceb4031f1b2adb`
- `tests/test_research_collaborative.py`:
  `651a5d6e345612918a3cba66fe6ae17ee4b23d9456759e20453f2713781de7c8`

Observed defect and fix: the first suite run gave **65 failed / 29 passed**;
nonempty fits failed while trying to normalize a read-only precision array.
A standalone native 3x3 reproduction showed `solve(..., overwrite_b=True)`
returning `writeable=False`, `owndata=False`, while sharing the writable RHS.
The implementation now copies a read-only result before in-place normalization.
The same mathematical fixture passes, and a forced-read-only regression test
also passes. This was an actual local author defect, not an invented failure.

### One bounded native-matrix smoke

One additional **synthetic** run used `np.random.default_rng(20261101)`,
256 users, 96 items, eight distinct randomly selected items per user, integer
counts in `[1,5]`, and event ages sampled from integer days `[0,13]`. Raw IDs
were `1001 + 13*rng.permutation(96)`. No real-data labels or metrics were read.
Configuration: `regularization=50`, `transform="log"`, `half_life_days=7`,
`row_l2=False`, `item_weighting="none"`, `max_active_items=64`,
`num_threads=1`, `max_batch_scores=4096`.

| Observed quantity | Result |
| --- | ---: |
| Raw shape / nnz / qualifying events | 256×96 / 2048 / 6121 |
| Compact feature shape / nnz | 256×64 / 1476 |
| Truncated items / retained raw count mass | 32 / 0.7371344551543866 |
| Coefficient bytes | 32768 |
| Requested output / bytes | 32×96 / 24576 |
| Finite / negative scores | 2048 / 194 |
| Maximum off-diagonal normal-equation residual | 7.105427357601002e-15 |
| Fit / native-training / scoring wall seconds | 0.014207917 / 0.013204500 / 0.001262958 |
| BLAS limit requested / reported pools | 1 / `[]` |
| BLAS control status | `unverified_no_reported_pools` |

The independent algebra check formed
`R=(X.T@X+lambda*I)@B-X.T@X`, required every off-diagonal residual below
`1e-10`, and checked zero diagonal and NaNs for excluded columns. These checks
passed. Timings are one local observation (concurrent with focused tests), not
a benchmark or a throughput promise; no runtime thread count was measured.

## Remaining risks and independent challenge

- No SWEET.TV MAP@5 trial was performed by this author. Reconstruction of
  historical frequencies is not future-frequency supervision; counts may be
  activity-dominated, and ridge/feature scales need time-consistent evaluation.
- Raw-count item truncation can exclude rare or recently active relevant items
  and turn some training-warm users into model-cold users. Inspect the recorded
  count/nnz/user retention; do not silently change the common candidate policy.
- Zero diagonal deliberately drops a direct personal-repeat signal, especially
  for sole retained history. Evaluate the standalone model before any explicit,
  evidence-only temporal-repeat blend. No popularity backfill vote belongs in
  CF evidence or fusion.
- Exact coefficient evidence can include very small floating-point coefficients;
  no empirical pruning threshold was tuned. Very small positive ridge values
  can still be numerically inadequate and cause an explicit failure.
- The 4096 ceiling bounds, but does not eliminate, quadratic memory/cubic CPU
  cost. Max-cap RSS and complete-grid runtime were not measured here. BLAS
  control on the observed Apple stack is unverified.
- Independent critic review is pending. This author has not represented these
  self-checks as an independent challenge or replaced the frozen original
  submission/reference. Additive responses should cite a reproduced input,
  assessed cause, correction if warranted, and the actual recheck result.

## Reverse-role challenge: validation/context and parent runner

Date: 2026-11-01. The EASE author independently challenged the context author's
contract/tests/review and the new `research_v2/evaluate.py` /
`tools/run_research.py` execution path. This is a **separate reverse review**,
not an independent endorsement of this author's EASE implementation.

Only this section was appended and `tests/test_research_validation_reverse.py`
was created. EASE and its original tests remained byte-identical; no parent
source, other review/tests, original scored output, or prepared-cache payload
was edited. No actual model fitting, parameter search, archived prediction/metric
file ingestion, credential/API access, upload, install, push, or delegation was
performed. The executed CLI integration uses two named **synthetic probes**,
not two candidate recommendation models or a quality search.

### Decision and actionable findings

**No ordinary source-clock boundary, shared-cohort, one-time-fallback, or AP
normalization blocker was demonstrated on the normalized `make_context` path.**
The current context suite passes all 55 checks, including the parent's C1–C4
repairs. `history_start` is treated as the already-documented opportunity bound,
not a behavioral truncation API; that known characterization is not reopened.

However, four new integrity/diagnostic shortcomings were reproduced, and a fifth
promotion-interpretation pitfall was demonstrated. Fix R1/R2/R4 before treating
the experiment's input/source guard as complete; add R5 before making claims of
prediction-tie robustness. R3 is an API consistency problem, not evidence that
the already-audited duplicate-free official cache has incorrect labels.

| ID / priority | Executed counterexample and observed result | Action for the owning author |
| --- | --- | --- |
| R1 / P1 integrity claim | `context_fingerprint`, `evaluate.py:63–66`, hashes event-array bytes without dtype/shape headers. A probe replaces the four-element Unicode channel vector by its `(1,4)` reshape, or its 1-D `int32` view. Bytes stay identical but channel representation/semantics change. **Both evaluator runs complete rather than raising** the expected mutation error. | Hash a length-delimited header containing field name, dtype, shape and byte length before every context-array payload, as `training_fingerprint` already does for CSR arrays. Keep the existing byte-change checks. |
| R2 / P1 integrity claim | `evaluate.py:67–69` hashes historical then forecast airings without collection tags/lengths. Moving the last historical airing to the **front** of the forecast tuple leaves the concatenated payload identical. The resulting forecast tuple contains an airing ending at the training cutoff, yet **evaluation completes**. | Bind distinct field names and lengths for both schedule collections, and retain chronology validation at construction. The fingerprint must distinguish the collections even if concatenation is unchanged. |
| R3 / P2 diagnostic correctness | Duplicate a held-out `(user,airing)` row twice. The repaired `make_context`/shared fold records two ignored duplicates and retains labels `[11,21,31,41,51]`; `label_frequencies` at `evaluate.py:51–56` reports program61 frequency **3**, versus the independent unique-broadcast count **1**. Reverse-label and marginalized diagnostics therefore use a different truth reconstruction from the official labels. | Derive frequencies from the same canonical broadcast identity/count path, or deduplicate by `(user,airing)` with conflicting-signature rejection. Do not change shared labels to match duplicate-inflated diagnostics. This is a new helper mismatch, not a reopening of repaired C2 training selection. |
| R4 / P1 provenance coverage | The returned source-binding manifest omits **`sweettv/pipeline.py`, `sweettv/submission.py`, `sweettv/contracts.py`, and `tools/run_research.py`**. These implement fingerprint/export/data/entry behavior on the reviewed path. The expected coverage assertion fails while every digest that *is* recorded matches its file. | Freeze/hash the executed local source dependency closure, including entry/export/fingerprint code; ideally use a project source manifest rather than another incomplete handpicked list. Transitive model implementations also need binding when adapters execute them. |
| R5 / P1 interpretation gate | With **unique held-out frequencies** `(6,5,4,3,2,1)` and equal finite prediction scores on seven shared candidates, all three reported strict diagnostics (`map_at_5`, `reverse_tie_map_at_5`, `tie_marginalized_map_at_5`) equal **1**. Independently reversing prediction-score ties yields exact toy AP **43/150**. Existing reverse/marginalized fields vary **labels only**, so they cannot establish prediction-order robustness. | Clearly name the current field as reverse-*label*-tie sensitivity. Independently vary prediction ties, including any constituent RRF/top-depth tie handling, while preserving ascending-ID official evaluation. Do not select a favorable ID direction. |

R1/R2 deliberately bypass frozen attributes using `object.__setattr__` in a
named fault-injection probe. No real model was observed to mutate these inputs,
and this is not an alleged security breach. The result establishes missing
semantic coverage in a guard that otherwise detects the exercised mutations;
it does not establish contamination of a completed official-data trial.

### Positive independent checks

- A direct broadcast-identity `Counter` agrees with training counts, labels,
  candidates, and unweighted temporal counts on a fixture with a **nonzero
  train-to-forecast gap**. Events ending exactly at the training cutoff remain
  training; whole forecast airings ending exactly at forecast end remain labels.
  Gap events and end-crossing behavior are excluded from both. End-crossing
  schedules remain candidates, later-starting catalog IDs and held-out-only
  users cannot expand this fold's fit inputs.
- Across evidence-present and all-NaN probes, template order, the one-user
  strict-five cohort, three-user all-template cohort, and seven candidates stay
  fixed. Read-only permuted candidate queries preserve alignment. Two negative
  evidence scores precede one-time fallback; fallback positions are exactly
  **13 = 3+5+5** versus **15** for all-absent evidence.
- The disk score cache retains only the two finite original evidence values and
  the original NaNs, with **no popularity backfill votes**. Original scored
  predictions are not involved.
- Ordinary count, metadata, and event-byte mutations during **either fit or
  scoring** raise before `result.json` is produced (six passing fault probes).
- Five independent exact-`Fraction` subset-enumeration cases agree with label
  tie-marginalized AP, including short/empty truth and dependent same-boundary
  relevance. No mathematics defect in that expectation formula was reproduced.
- A patched-input CLI integration executes the real runner/evaluator on two
  synthetic nonmodel probes. It confirms one shared context/catalog/cohort and
  unchanged template ordering. All emitted artifacts are temporary and labeled
  synthetic; there is no raw/cache payload read in this integration.
- A separate **read-only metadata/signature check** compared the actual frozen
  cache's manifest to `preparation_signature(Path("data/raw"))`, without loading
  its Parquet payloads or fitting/scoring a model. Both signatures were
  `0202a5476a3e3dde382c07783adf72282ab98fd215336ea2c339b4af2373055e`.
  Thus a hypothesized default-runner cache-signature mismatch was **not**
  demonstrated. This does not newly authenticate raw data or recompute metrics.

### Executed test results, including unmasked failures

Using the existing Python environment, `-B`, `-p no:cacheprovider`, and fresh
unique `--basetemp` paths under the approved workspace runtime directory:

```text
pytest -q --tb=short -rx tests/test_research_validation_reverse.py \
    tests/test_research_context.py tests/test_research_evaluation.py
76 passed, 5 xfailed in 0.44s

pytest -q --tb=short --runxfail tests/test_research_validation_reverse.py
5 failed, 17 passed in 0.38s
```

The five strict xfails are **new R1 (two cases), R2, R3 and R4**, not C1–C4.
They are not counted as passed requirements. The unmasked run confirms the
first three fail specifically with **`DID NOT RAISE RuntimeError`**, R3 fails
with `Counter({61:3,...}) != Counter({61:1,...})`, and R4 fails with the four
missing source paths listed above. No unrelated fixture exception was hidden
by xfail. Remove each marker only after the owning fix is reproduced; a fix
must make its correctness assertion pass. R5 has a passing *characterization*
test, not an assertion that prediction-tie robustness is already implemented.

Temporary artifacts, relative to the `LLM4Rec` workspace root:

- `.opencode/runtime/tmp/opencode/validation-reverse-9536dafe50c447cfa4e20731badf1a1f/`
- `.opencode/runtime/tmp/opencode/validation-reverse-unmasked-9dfa5c8a73e4479c91a27eb865655cab/`

### Reviewed source snapshots and preservation

Before/after SHA-256 checks over all 16 listed source/test/review/lock files
reported **`SOURCE_FILES_CHANGED_DURING_CHECKS=[]`**. These are the reviewed
versions, not a claim that later parent edits are already verified:

```text
research_v2/contracts.py
  9202448a3b0fb30a8e902112fcbdb825ea78e5901a50744b50e1d5d3aaa9c7e0
research_v2/evaluate.py
  b0166c6980b15319c1e6667aaa7c616bb7c5ed9a0c69b3d2e20b85c83644812b
tools/run_research.py
  bac1ddd232c251c328922db3e985980ae93cbeb1fa0525621ec96e4bb2cfc5df
tests/test_research_context.py
  06d8d1583a362bd9fe20d88d6cf4ab6cbc0b8bd4130117640de9d16d37184ee0
tests/test_research_evaluation.py
  4668197f88b0e0e13fe86b2bfdaa9f5b226a255b0dc8ecdb21512625f62fd2a2
reviews/research-v2-validation-challenge.md
  11a696c4b2d9a206cb521795189c6c12564f7979eb547e57568f88d1b5fb1b51
research_v2/PROTOCOL.md
  d92047c7ac0bb96cf47c2587a8fd212c9f1406eb1d589ebffbedbe4cc606201d
sweettv/data.py
  4a573981a1adbb9fe9df57a443689deabc17b98e488c4dde9cc695c875f84e32
sweettv/evaluation.py
  e676e10af9b5909f91da0282a0b884d20a791e565ad6492ef27a03bc4f352d94
sweettv/pipeline.py
  0e4c82417d29a7c3bb0fa0dea5f66b57a49c1b797aa7795ccc9cc698ec401f83
sweettv/submission.py
  34dbee94ec50eee8b8efb7c0d0826d6f7253df5f609a33105a6ee4207905da69
sweettv/contracts.py
  d242dec34cbbe277f1729f9d0185076c4040681a6a2d995128a2f8b16d6ba527
requirements.lock
  55a7c72c648feaa59dc05769b67f5c45a05fb7619bcbcee672af48a82c299317
research_v2/collaborative.py
  418c5791b27372eb12daf97716fa0e25e08a56aa00113f5b16ceb4031f1b2adb
tests/test_research_collaborative.py
  651a5d6e345612918a3cba66fe6ae17ee4b23d9456759e20453f2713781de7c8
tests/test_research_validation_reverse.py
  f297c68845092c7ad9691c837f64bc37f56672b8eb69e229caa32d706f5df03d
```

EASE's independent critic response remains separate and pending; its source
was deliberately left untouched during this challenge and concurrent fits.
The original scored `.41001` artifacts were not opened, rewritten, replaced,
or used as supervision by this reverse review.

## Author response to the independent EASE challenge: E1/E2 closed

Date: 2026-11-01. The parent explicitly paused grids and authorized changes to
`research_v2/collaborative.py`, `tests/test_research_collaborative.py`, and an
append-only response here. Only those three files were edited in this stage.
The independent critic's files and all parent code/reverse tests were left
untouched. No archived quality results, official-data fits, APIs, credentials,
uploads, installs, pushes, or additional agents were involved.

**Assessment:** E1 and E2 are valid API consistency defects, with the critic's
stated limited scope. They do not establish wrong canonical `int64` predictions
or an EASE algebra defect. The independent rational-inverse and separate-target
ridge oracles discriminate the relevant mathematics; I found no demonstrated
defect in those fixtures. E3 is the same parent-owned duplicate-frequency helper
problem as reverse finding R3, not an EASE defect; it is now parent-fixed.

### Concrete reproduction → cause → correction → confirmation

| Finding | Before, independently reproduced | Correction and after result |
| --- | --- | --- |
| E1: mixed Boolean coordinates | Untouched critic tests with `--runxfail` fail twice with `DID NOT RAISE ValueError`: `[True,0]` is silently treated as `[1,0]` on either query axis. Scalar Boolean rejection did not extend to mixed lists. | `_indices` now preserves original Python sequence elements in an **object array before numeric dtype inference**, then checks `Integral` and excludes Python/NumPy Booleans. Mixed lists/tuples/object arrays reject; explicitly typed integer arrays retain their existing fast path. Both independent critic assertions now pass normally when markers are disabled. |
| E2: valid unknown integers depend on inferred dtype | Untouched critic tests fail twice because `[-1,2**63]` becomes float64 under NumPy's inference, then raises the integer-array error. Explicit object arrays instead return NaNs. | Exact object conversion also preserves mixed signed/unsigned/arbitrary-size integers through the existing bounds check. Only in-bounds indices are cast for mapping. Both axes return correctly shaped NaNs for unknown coordinates and preserve known scores/order/duplicates. Actual float/string/Boolean inputs remain invalid. |
| Huge scalar/inference-overflow defense | On this environment, the five probed huge scalar users already returned NaNs before the fix; a native scalar overflow was **not** reproduced. The old helper nevertheless depends on numeric inference, and an explicitly labeled injected inference `OverflowError` reached `score(2**100,...)`. | The helper chooses object conversion **before** inference rather than catching an exception afterward. Unknown scalar boundary tests, including `10**400`, pass; the inference-overflow fault probe now returns NaNs while preserving finite known positions. This is a defensive regression check, not a newly alleged native failure. |

Already-coerced typed integer arrays cannot reveal whether the caller previously
converted a Boolean; they remain valid integer coordinate arrays. Conversely,
the preservation change does not start accepting actual floating-point arrays
or mixed noninteger sequences. A zero-dimensional array embedded in an untyped
sequence is rejected rather than silently becoming an integer scalar.

The **only production change** is `_indices`' choice of lossless conversion for
untyped sequences, plus its explanatory comments. No fit math, feature order,
training selection, ridge, normalization, coefficient/evidence rules, scoring
algebra, bounds, defaults, or diagnostics/model parameters changed.

### Actual before/after checks

Before source modification, on the critic's existing file:

```text
.venv/bin/python -B -m pytest -q -p no:cacheprovider --tb=short --runxfail \
    tests/test_research_collaborative_challenge.py
4 failed, 34 passed in 0.33s
```

All four failures are E1/E2's documented assertions; parent-fixed E3 already
passes without its old marker. After adding author regressions but **before**
the source correction, the selected Boolean/integer/noninteger/overflow subset
gave **21 failed / 14 passed / 104 deselected**. This includes the explicitly
identified inference-overflow fault injection, not 21 independent native bugs.

After the correction, the untouched independent challenge with `--runxfail`
gave **38 passed in 0.30s**. The final combined post-fix run was:

```text
.venv/bin/python -B -m pytest -q -p no:cacheprovider --tb=short --runxfail \
    tests/test_research_collaborative.py \
    tests/test_research_collaborative_challenge.py \
    tests/test_research_context.py tests/test_research_evaluation.py \
    tests/test_research_validation_reverse.py
259 passed in 0.68s
```

That consists of **140 author regressions, 38 independent challenge cases, and
81 context/evaluation/reverse cases**. The author suite has 42 new boundary
checks beyond its original 98. Parent-fixed R1/R2/R3/R4 pass in the reverse suite;
none of its repaired markers were reintroduced. These are author-executed
rechecks of independent tests, not a substitute for the critic's next recheck.

**Marker handoff:** a normal, untouched critic-file run currently reports
**34 passed / 4 failed in 0.28s**, with every reported failure explicitly
`[XPASS(strict)]`: E1 on users/candidates and E2 on users/candidates. There are
**no remaining expected correctness failures** in the unmasked challenge.
The E3 marker is already absent and its assertion passes normally. The parent
or critic should remove the four E1/E2 markers after independent confirmation;
this author did not remove or modify them. An earlier commentary count of five
remaining markers is superseded by this executed final marker inspection.

### Numerical-compatibility evidence, without replaying quality experiments

The entire `EASEModel` class AST is unchanged before/after, SHA-256:
`df9d8c1553c5dd98788c6881c356b63e2da551f3b4ddffef5fc5362cafc08e47`.
The changed module-level helper's typed-integer path still makes the identical
coordinate mappings. The exact rational coefficient/score oracles continue to
pass, as do the independent unequal-IDF/decay/row-L2 target-regression fixtures.

A separate fixed before/after check used
`X=[[2,1,0],[0,1,3],[1,0,1]]`, integer users `[-1,2,0,2]`, and integer
candidates `[2,1,0,2,1000,-1]`. It hashed, in order, `coefficients_.tobytes()`,
`training_matrix_.data.tobytes()`, and the requested score array's bytes.
Each fingerprint is **exactly identical** before/after on this environment:

| Fixed configuration | Before = after SHA-256 |
| --- | --- |
| ridge2, raw; other defaults | `c7c96d5e4f802b9b83f4d1124b2b34d6c5335f88a17247750c9eaa77641865e6` |
| ridge2, log, half-life2, IDF, row-L2 | `915ee5fe2e4ebb4d347e2f483608bf91a020ed0a1e5768d9eb5978d097c3ac12` |
| ridge2, binary; other defaults | `9ffb222f421bc3c29bb9baf8ab2d7c8e69e34c042a596e19eaea6ba33e9e7b82` |

This supports unchanged canonical numerical behavior, not a newly measured MAP
claim or a cross-platform bitwise guarantee. Existing runs retain their original
source hashes and must not be relabeled as new-source experiments. No original
scored prediction file was opened or altered. R5's separate prediction-tie
promotion gate remains a parent concern, not part of this input-validation fix.

### Stable files and final handoff

The final combined checks and numeric comparisons verified no changes during
execution across ten tracked author/critic/parent review-source files:
`SOURCE_FILES_CHANGED_DURING_CHECKS=[]`.

```text
Updated author source:
research_v2/collaborative.py
  640183b589a3172548c51890ff1a3d5cd6c5e54e20960f3a507aa2f716b02242
tests/test_research_collaborative.py
  634481b73c6c07ed0b8db648af8d2b60352ec0b3ea0b3c9ac07680470b879d9b

Untouched critic files, matching this stage's pre-edit snapshot:
tests/test_research_collaborative_challenge.py
  fffd6349cf79f4156a6f3380473b6e9b6afdb4155cdc8300b4e2bc78d2f9cb31
reviews/research-v2-validation-challenge.md
  81c31fdef95c0fa50072c274a1b87fc09f25806c7ed0c24b4553351f05af3949

Untouched parent repair snapshot:
research_v2/evaluate.py
  fa5d43eae0269be9f2b635e9ad28c195225395a417cf455bb9292fad130d274c
tests/test_research_validation_reverse.py
  312f5b8e38b790c0b92b0d83a2459843bddb14a5c7b3ebe9446203bf4ec2f117
```

The snapshot containing the final synthetic evaluation artifacts is
`.opencode/runtime/tmp/opencode/ease-response-dba0113ed62041848ee7b5df35403f94/`
relative to the `LLM4Rec` workspace. EASE is stable for independent recheck;
any grid restart or additional quality experiment remains the parent's decision.
