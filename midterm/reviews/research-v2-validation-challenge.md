# Round-two temporal context and validation challenge

Reviewed 2026-11-01. Independent challenge role; **no new model-quality trial,
Kaggle query, upload, network-data access, authentication, commit, push, or
subagent**. Only this review and `tests/test_research_context.py` were created.
Production, contracts, models, cache, and scored submission files were not edited.

## Decision before model trials

**Proceed with bounded local experiments on the frozen normalized cache, not
with a new live submission or a claim of independent test improvement.** No
future-behavior leakage was reproduced in `make_context`. Four narrower API
limitations were reproduced below; none is demonstrated to corrupt the frozen
official-cache folds. Model implementation and prediction checks remain required.

The strongest experimental hypothesis is a restrained recency/count baseline,
followed by separately ablated schedule opportunity and collaborative signals.
More methods or a better observed primary score do not establish generalization.
The unchanged reference is historical primary hybrid MAP@5
**0.4003947734222964 / 1,199 strict-five users**; the already-returned best private
score is **0.41001, ref57851385**. Neither score was recomputed or queried here.

### Priority risks and promotion gates

| Priority | Risk / evidence | Required gate |
| --- | --- | --- |
| P0, interpretation | Every proposed window is previously observed history. Original primary weeks 19–30 contain both later development weeks 23–26 and confirmation weeks 27–30. These are overlapping robustness checks, not independent replications. | Freeze exact configurations and selection rule before confirmation. Preserve every failed trial. Call results reused-history validation, never a fresh test estimate. Do not optimize against the three known private scores. |
| P0, representation | Three proposed forecasts are four weeks; deployment and original primary are twelve. History grows from 4/8/12/16 historical weeks to 20 at deployment. Short-horizon decay can favor transient habits. | Retain the original twelve-week primary as a non-retuned long-horizon guardrail; require gains over the unchanged hybrid in multiple windows, including that guardrail, before recommending promotion. |
| P0, cohort shift | Measured later-blocked cohort is only **677/1,260** strict-five, versus original primary **1,199/1,260**; it has **356 empty-label** users. The cause was not established. | Publish strict-five and all-1,260 MAP with short/empty-label counts, paired user deltas, and training-only activity strata. Do not compare raw MAP levels across different cohorts or select by the favorable active subset alone. |
| P0, tie artifacts | Frequency labels and repeat-score ties both use ascending raw IDs; fusion can inherit constituent rank ties. | Vary label boundary-tie and score/rank-tie policies **independently**, including inside RRF/top-depth selection, not just the final sort. Retain ascending-ID MAP as the unchanged main metric; do not select an ID direction. Report reverse-ID and tie-marginalized sensitivity. |
| P1, feature semantics | Airing counts are channel/program availability proxies, not logged personal exposures, recommendations, or causal effects. Weighted counts divided by unweighted history can produce inconsistent effective horizons. | Specify count/rate units, deduplication, quarantine, boundary and smoothing policies. Ablate decay, availability adjustment, channel/time compatibility, and their combination separately. |
| P1, shared-state integrity | Frozen context fields/read-only event arrays are useful safeguards, but nested `TrainingData` remains mutable. Original production pipeline fingerprints `TrainingData`; no round-two runner was reviewed. | Fingerprint **all** context arrays, schedules, boundaries, raw data, and metadata before/after each fit and score. Fit receives context only; labels stay evaluator-owned. Mutation/failure must stop a run, not silently fall back. |
| P1, model evidence | No new EASE/collaborative implementation was present at the initial inspection. Opportunity/count models can emit zero or cold-item pseudo-evidence and accidentally alter fallback behavior. | Complete the independent formula, candidate-order, cold-user/item, NaN, and no-internal-backfill checks described below before considering its score. |
| P2, delivery | A local win is not an improved private score or authorization to query one. | Freeze predictions/configuration/source hashes, preserve the original scored CSV, and obtain new explicit submission approval. Existing personal-verification/screenshot/package requirements remain separate. |

## What was actually checked

Sources read: `research_v2/PROTOCOL.md`, `research_v2/contracts.py`,
`CONTRACT.md`, shared `sweettv/{contracts,data,cache,evaluation,pipeline,baselines}.py`,
existing data/cache tests, audited protocol, prior primary summary, verification
record, submission manifest, and the existing source-only public-solution review.
No public notebook outputs or answer files were read. Ideas originating in the
public solution remain attributed in `reviews/research-v2-public-solution.md`;
the forecast/history ratio is our extension, not that notebook's final formula.
No new scientific-source research was needed for this contract audit.

Reviewed SHA-256 snapshots:

```text
research_v2/contracts.py  cb598e3834cc892a6fd8be5b49ce62a889502a180c3a3fa1d4c652d1d61899f4
research_v2/PROTOCOL.md   d92047c7ac0bb96cf47c2587a8fd212c9f1406eb1d589ebffbedbe4cc606201d
sweettv/data.py          4a573981a1adbb9fe9df57a443689deabc17b98e488c4dde9cc695c875f84e32
cache/manifest.json      6bfdabdca49cf2874e60f2f452d250e7fd81c64b6cb24713d741454c478df219
cache/events.parquet     9ce9b9c8b38311a1f00a676cc92fd1984e05ae423e404a89a034ac89a98bf918
cache/schedules.parquet  c637326529ef3f7edf9be6b77ef989769128230d044843f92243ea43a84066c4
```

Here `cache/` abbreviates `data/processed/official-v1/`. The cache's recorded
archived preprocessor digest is
`0202a5476a3e3dde382c07783adf72282ab98fd215336ea2c339b4af2373055e`,
**not the current `data.py` digest**. The read-only load explicitly pinned that
archived snapshot digest and validated all payload hashes/bytes. This establishes
snapshot integrity, not current-preprocessor identity or newly authenticated raw
source provenance; no raw rebuild or validation bypass was performed.

### Executed synthetic tests

```sh
.venv/bin/python -B -m pytest -q -p no:cacheprovider -rx tests/test_research_context.py
```

Result: **51 passed, 4 xfailed**, 0 unexpected failures, 0.19 seconds. Environment:
Python 3.13.6, NumPy 2.5.4, SciPy 1.18.1. Strict expected failures are explicit
reproductions of C1–C4, not passed checks; fixing them should produce XPASS failures
until the parent removes their markers. The full existing suite was not run here.

Passing checks include:

- Delete/reassign held-out events and introduce held-out-only users: labels
  actually change while **every exposed fit input** remains identical.
- Change only future coverage/order; add cutoff-crossing behavior or a later-than-
  forecast catalog item: no fitted behavioral/candidate input changes.
- Training metadata and supplied forecast metadata remain usable; changing
  permitted forecast metadata may change features without being leakage.
- Independent user/program event `Counter`, shared raw-count CSR, and unweighted
  `temporal_matrix` agree. The raw `TrainingData` counts remain float32 integers.
- Half-life one day gives `[1/8, 1/2, 1, 1/2]` for event-end ages `[3,1,0,1]`
  days. The anchor is the fold cutoff, with 86,400 seconds/day, not event starts,
  milliseconds, a future maximum timestamp, or a user-specific last event.
- Exactly one-day-old events survive a one-day lookback; invalid/nonfinite
  half-life/lookback values fail. Empty events retain a valid sparse shape.
- Event/candidate arrays are copied and read-only, Airings are frozen, exact
  string user/channel identifiers survive, and numeric matrix indices are kept
  distinct from raw program IDs. Query helpers preserve read-only reordered/
  duplicated candidate requests and reject scalar/noninteger representations.
- Future event ends, invalid chronology, out-of-range indices and invalid
  historical/forecast schedule boundaries fail the exercised constructors.

### Executed frozen-cache checks, no model scoring

A read-only Python run used `load_prepared` with the pinned archived digest,
`make_context` for all four windows, and an independently aggregated
`Counter((user_id,tv_show_id))` of pre-cutoff events. Every nonzero raw CSR entry,
event count, and unweighted temporal matrix agreed. Counts were **611,760 events,
381,892 unique schedules, 1,260 template users**, with no duplicate user/airing
events or duplicate schedule identities. Context arrays were read-only.

| Fold | Train cutoff | Forecast end | Train events | Historical / contained forecast airings | Candidates | Strict-five / all | Empty labels | Candidate IDs lacking a contained forecast airing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11–14 → 15–18 | 1586131200 | 1588550400 | 125,285 | 45,025 / 46,531 | 3,539 | 1,039 / 1,260 | 58 | 14 |
| 11–22 → 23–26 | 1590969600 | 1593388800 | 381,609 | 138,766 / 48,616 | 3,911 | 677 / 1,260 | 356 | 11 |
| 11–26 → 27–30 | 1593388800 | 1595808000 | 461,409 | 187,447 / 49,259 | 3,957 | 1,046 / 1,260 | 44 | 12 |
| Original 11–18 → 19–30 | 1588550400 | 1595808000 | 237,136 | 91,615 / 145,102 | 7,045 | 1,199 / 1,260 | 20 | 5 |

Integer boundaries are the existing common naive-clock coordinates, not proof of
the source's timezone. Load took 9.804 seconds; the complete read-only run took
43.544 seconds. The reviewed source and every cache payload had identical
before/after hashes. Primary `TrainingData` fingerprint exactly reproduced the
archived value `b60d4527c10a279714f732f4164cc985e9a8ecffc6872962e05fb0c6c7ed240c`.

## Reproduced limitations: input → expected → actual

These are bounded input/API findings, **not invented defects in fitted models**.
The official normalized cache does not contain the duplicate inputs in C1/C2;
`make_context` supplies proper one-dimensional integer arrays in C3/C4.

| ID / severity | Reproduction and expected behavior | Actual result / location | Action |
| --- | --- | --- | --- |
| C1 / P2 correctness outside normalized-cache assumption | Duplicate one historical and one forecast schedule row with the same `(channel,start,end,program)`. Shared fold deduplicates; context opportunity counts should not increase. | Minimal fixture: context historical/forecast counts become **2/2**, versus **1/1** unique airings. `make_context` iterates raw `prepared.schedules` at lines 123–128, whereas `build_fold` normalizes them. | Either enforce/document already-canonical unique prepared schedules at this boundary or use one shared canonical schedule view; avoid raw-row opportunity counts. Strict xfail test preserves the reproduction. |
| C2 / P2 compatibility, fail-closed | Duplicate one training `(user,airing)` event. Shared fold accepts it and counts once; context should use the same deduplication or reject it with an explicit boundary precondition. | Shared count remains 1 in the minimal fixture, but `make_context` raises **`ValueError: Context event counts disagree with the unchanged production fold`**. It selects raw events at line 114. | The existing check safely stops count disagreement; do not remove it. Align canonical event selection or document/enforce normalized uniqueness. No inflated production score was observed. |
| C3 / P3 constructor validation | Pass `event_channels` shaped `(n_events,1)` to `TemporalContext`; expected one channel scalar per event and a clear shape error. | Accepted; only leading length is checked, unlike the other arrays. A scalar channel array instead reaches Python's `len` error rather than the intended shape validation. | Apply the same one-dimensional check to channels. The matrix-shape reproduction is a strict xfail. |
| C4 / P3 constructor validation | Pass fractional candidate indices `[0.9]` directly to `TemporalContext`; expected rejection, as with `candidate_array`. | Accepted and silently converted to `[0]` before validation at line 44. This is index coercion, not future leakage. | Validate original integer dtype before casting, or clearly make direct construction an internal-only validated path. Strict xfail records the fractional-index case. |

Two **documented semantic hazards, not bugs** also have passing characterization
tests: (1) candidates may include end-crossing airings while forecast features
contain only fully contained airings; the measured unmatched candidate counts
above must not cause model-specific catalog pruning. (2) changing `history_start`
filters historical schedules but **does not truncate behavioral counts/events**.
A recent behavior window must be requested through the weighting/lookback path,
with an aligned historical opportunity denominator.

`TemporalContext` is not deeply immutable: `data.item_ids`, CSR arrays, lists and
metadata remain mutable. The read-only array tests do not establish resistance to
deliberate flag changes or hostile code. An actual model mutation was not observed.

## Hypotheses: falsifiable critiques before trials

| Hypothesis | Main concern | Minimal discriminating comparison / check |
| --- | --- | --- |
| Recency-weighted personal counts | Habit drift may improve short forecasts but hurt a twelve-week frequency target. Decay also breaks old score ties, which can masquerade as a substantive win. | Preserve no-decay repeat. Use a small frozen half-life grid, then test one bounded lookback separately rather than a full cross-product. Compare paired deltas and tie sensitivity on identical cohorts. Keep event-end cutoff semantics and raw counts unchanged. |
| Forecast/historical airing adjustment | `C_ui * (F_i+a)/(H_i+a)` is a proxy-based reranker, not an exposure-corrected probability. Sparse historical items can explode; changing history length changes the effect of a fixed pseudocount. | Ablate no adjustment, forecast count alone, then one shrunk ratio. Inspect zero/one historical airing, schedule duplication, missing-contained-airing candidates, and increased/decreased-availability strata. State caps/powers in advance, not after seeing primary labels. |
| Effective horizon / smoothing | Decayed `C_ui` divided by all-history unweighted `H_i` compares different effective windows. Four- versus twelve-week forecasts change the interpretation of counts and smoothing. | If using a rate interpretation, align the kernel/window of `C_ui` and `H_i`; a possible explicit score is `F_i * (C_ui + a*p0_ui)/(H_i + a)`, with a training-only prior and `a` in effective-airing units. This remains an availability proxy, not a causal or measured-exposure estimate. Test horizon scaling and cold limits analytically. |
| Channel/time backoff | Context has completed airings, not all raw viewing time or measured non-exposure. A completed-airing profile cannot be described as overall availability. Exact weekday/hour bins are sparse and timezone-sensitive. | Back off from user/channel/time to training-only broader profiles with shrinkage. Require incremental value over future-airing count alone. Fix source-clock bins; never fit slots from future viewing. Count an airing once or normalize any multi-slot allocation. |
| Cold-program metadata | Cold future IDs can have legitimate static metadata, but title variants do not justify replacing program identity. Earliest canonical metadata may be unrepresentative. | Keep raw IDs, candidates and labels unchanged; separate new-to-training IDs from merely new-to-user IDs. Measure discovery hits, fallback and losses to strong repeats. Do not hard-drop movie/long-duration categories based on the public notebook. |
| Count-linear CF / EASE | Squared-error reconstruction of historical counts is not a guarantee of future Top-5 membership. Zero-diagonal EASE removes direct self-repeat; a one-program user cannot recover that same item from its own feature alone. | Compare standalone scores with raw repeat and the unchanged hybrid before fusion; preserve a separate repeat signal. Ablate count versus binary/log-count inputs without selecting across an unbounded grid. Report negative scores, cold rows/columns, memory, regularization and conditioning. |

### EASE/collaborative review handoff — pending author completion

No author model file was available at the initial directory inspection. I did not
poll for new files or claim this future review passed. After the author supplies
the stable implementation, use the following independent checks before trials:

1. For a nonsymmetric-by-column example such as
   `X=[[3,1,0],[0,2,1],[4,0,1]]`, independently solve
   `P=(X.T@X + lambda*I)^-1`, `B_ij=-P_ij/P_jj`, `B_jj=0`.
   Verify `X@B`; column versus row normalization is not interchangeable even
   though `P` is symmetric. Use a direct small solve as oracle, not the author's
   factorization helper.
2. Exercise readonly candidate requests `[2,0,2]`, empty requests, NumPy integer
   users, invalid scalar/bool users, unknown users, and raw IDs unlike indices.
   Output must preserve query order/duplicates, not sort or interpret IDs as indices.
3. Test all-zero users, all-zero/cold item columns, one-item histories, repeated
   counts versus binary counts, nearly collinear columns, invalid regularization,
   and legitimate negative scores. Define when zero means absent evidence and
   emit NaN accordingly; never inject internal popularity backfill.
4. Fit-input/prediction invariance under valid future-behavior mutation; verify
   no metadata/normalization/co-occurrence statistic consumes labels. Check
   context fingerprints before/after fit and score.
5. Record selected training-item dimensions and dense Gram/solve memory rather
   than claiming that sparse input makes a dense inverse sparse. Do not allocate
   a dense global user-by-item prediction matrix.

The EASE author should also challenge these context tests and this protocol
analysis, record any counterexample and response, and recheck after fixes. That
role-swap review is a separate pending gate, not replaced by this report.

## Less adaptive selection with no untouched history remaining

1. **Before scores**, save exact fold boundaries, candidate/cohort lists, source
   hashes, finite configurations, ablations, runtime limits, tie policies, and a
   rule for rejecting rather than continually tuning an unconvincing result.
   Select neither matching thresholds nor evaluation users to improve a model.
2. Use the two declared development windows for a small prespecified search.
   Compare per-fold paired deltas to the immutable hybrid/repeat controls, fitted
   separately on that fold's training history with frozen original parameters.
   Reuse archived primary predictions only when training, catalog, cohort and
   prediction provenance match; never use primary predictions as a later-fold
   baseline. Prefer a simpler stable plateau over a narrow best point. A possible frozen selection
   rule is mean all-template paired delta with a minimum-fold non-regression
   constraint; specify any tolerance **before** scores. Preserve strict-five MAP.
3. Freeze a genuinely small shortlist and all fusion weights, including the
   fallback policy, before running weeks 27–30 and replaying the original primary.
   A model changed after those results is a new exploratory revision; it has not
   passed the prior confirmation. Do not keep retrying the same named gate.
4. Keep the original twelve-week primary for horizon relevance, while explicitly
   acknowledging its already-known labels and overlap with both later windows.
   Optional additional fixed twelve-week replays (for example 11–16 → 17–28)
   can diagnose horizon mismatch, but are overlapping sensitivities, not new tests;
   agree to them before inspecting their model scores.
5. Separate label boundary-tie reversal from score-tie reversal. For label
   marginalization, choose the needed `q` items uniformly without replacement
   from the `m` programs tied at the fifth-frequency boundary. Compute expected
   AP, not optimistic best-case membership. In particular, two programs in the
   same tie group have joint relevance probability `q*(q-1)/(m*(m-1))`, not
   `(q/m)^2`; preserve the usual AP denominator. Prediction-tie randomization must
   likewise be label-independent and shared across methods/seeds.
6. For uncertainty, bootstrap paired user deltas and report wins/losses as well as
   means. If summarizing overlapping folds, resample the same user's entire
   multi-fold vector together; do not treat overlapping user-window rows as
   independent observations. Such intervals do not undo adaptive model selection
   or limited calendar coverage.
7. Evaluate constituents before one small evidence-only fusion. Reject duplicated
   fallback votes or a blend whose weights mathematically preclude meaningful
   discovery. Promotion needs observed cross-window benefit, no integrity failure,
   and acceptable coverage/runtime. A new private score requires fresh approval;
   there is no basis here to promise one above 0.41001.

---

## Addendum: completed independent EASE challenge and contract recheck

Reviewed 2026-11-01 after explicit author-completion handoff. This addendum
**supersedes the earlier pending-EASE status and the original context-test
51-pass/4-xfail snapshot**. That earlier record remains intact as history.
The parent fixed C1–C4 and removed their expected-failure markers; I did not edit
the parent's contracts or those existing tests.

**Outcome:** the exercised EASE algebra, training-only feature construction,
signed/evidence behavior, compact mappings and bounds are correct against the
independent checks. No numerical-orientation, future-behavior leakage, internal
fallback, or global dense scoring defect was reproduced. Two batch-input policy
inconsistencies and one optional runner-helper inconsistency were reproduced
below. They do **not** demonstrate corruption of canonical `int64` queries or the
previously verified duplicate-free official cache. Author response and reverse
review of my fixtures/protocol remain pending.

### Scope, independence, and actual commands

Only NEW `tests/test_research_collaborative_challenge.py` was created and this
report was appended. No author source/tests, contracts, existing context tests,
runner, model parameters, production files, cache, or original submissions were
edited. No APIs, authentication, credentials, uploads, pushes, external data,
subagents, or new model-quality/MAP results were accessed. All challenge inputs
are tiny synthetic matrices or schedules; no official-data quality experiment or
parameter search was performed in this addendum.

The new challenge does not import author test helpers or private model helpers:

- A **four-item exact `Fraction` Gauss–Jordan inverse** independently constructs
  `P`, then column-normalized `B_ij=-P_ij/P_jj`, and exact rational `X B` scores.
  Three fixed ridge values exercise the formula; the fixture explicitly produces
  a different answer under incorrect row normalization. No NumPy/SciPy algebra is
  used in this rational oracle.
- Six fixed feature-space cases independently sum event-end decay, apply
  raw/log1p/binary transforms, unequal training-fitted IDF, and optional row L2.
  Their coefficients are checked with **separate target-wise ridge regressions**
  excluding each target column, rather than reusing the precision-normalization
  implementation. IDF includes an effective training user lost to truncation and
  excludes a cold user. These are correctness fixtures, not a hyperparameter grid.
- Independent fixtures exercise negative predictions, finite cancellation zeros,
  structural no-path NaNs, isolated items, sole-history self NaNs, watched targets
  supported by other history, cold/truncated entities, exact raw IDs above 2^53,
  raw-count/ID cap ties, row/column permutations, one-time shared fallback,
  readonly reordered/duplicated queries, empty requests, huge/unsigned indices,
  and output bounds **before allocation**, including all-cold requests.
- Valid future event/label mutation and additional cold forecast catalog entries
  leave warm fitted coefficients and raw-ID-aligned predictions unchanged.
  Changing forecast horizon, candidates, schedule/channel metadata and unrelated
  provenance does not become a CF statistic. Context snapshots are unchanged by
  fit/score, and fitted predictions survive subsequent caller-data mutation.
  A deliberately bypassed constructor with a future event end is rejected on
  refit; that is a labeled fault-injection defense, not an observed leakage bug.
- After native-library/controller initialization, filesystem/network entry-point
  tripwires cover a fresh fit and score. No guarded entry point was used. This is
  a bounded runtime check, not a sandbox/security proof or an assertion about
  all library initialization. Instrumented sparse densification permits only the
  compact item Gram matrix, not global user-item training/score densification.
- A separate exhaustive enumeration of all 15 admissible boundary-tie label sets
  agrees with `tie_marginalized_ap`; its without-replacement joint term is correct
  on that fixture. This is synthetic metric arithmetic, not a SWEET.TV MAP result.

Executed from `midterm/`, with bytecode/cache writes disabled:

```sh
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_research_context.py
# 55 passed in 0.16s; C1–C4 now pass normally, with no xfails.

.venv/bin/python -B -m pytest -q -p no:cacheprovider -rx tests/test_research_context.py tests/test_research_collaborative_challenge.py
# 88 passed, 5 xfailed in 0.40s.

.venv/bin/python -B -m pytest -q -p no:cacheprovider -rx tests/test_research_context.py tests/test_research_collaborative_challenge.py tests/test_research_collaborative.py
# 186 passed, 5 xfailed in 0.59s.
```

The last run comprises **55 context passes, 33 NEW independent challenge passes,
and 98 author-test passes**, plus five strict expected failures for E1–E3. The
author's 98 checks are regression evidence, not substitutes for the new oracles.
The five xfails are **not passed checks**: two positions each for E1/E2, plus E3.
Fixes should produce strict XPASS failures until their markers are removed.
No full course-wide suite or official-cache fold reconstruction was rerun in this
addendum. Environment remained Python 3.13.6 / NumPy 2.5.4 / SciPy 1.18.1 /
pytest 9.2.0.

### Genuine reproductions and appropriate severity

| ID / scope | Minimal input and expected result | Actual result and cause | Action / operational impact |
| --- | --- | --- | --- |
| E1 / P3 batch type-policy consistency | Fit a two-user/two-item positive count matrix. `score(True,[0])` rejects a Boolean scalar; batch lists `[True,0]` should retain the same rejection policy for user or candidate coordinates. | `score_batch([True,0],[0])` silently scores users **1 and 0**; `score_batch([0],[True,0])` silently scores columns **1 and 0**. `_indices` at `collaborative.py:56–65` first calls `np.asarray`, which converts the mixed list to int64 before Boolean identity can be checked. Pure `[True]` is rejected, making the behavior input-composition dependent. | Validate original sequence element types before coercion, preserving the fast path for already-typed integer arrays. Already-explicit integer arrays cannot reveal previously discarded Boolean identity. No issue was reproduced for the runner's canonical integer arrays; this is not an EASE algebra defect. |
| E2 / P3 unknown-integer API consistency | For user or candidate requests `[-1,2**63]`, both coordinates are unknown integers; expected correctly shaped NaNs, as for scalar unknown users and explicit object-array requests. | Raises **`ValueError: Candidate indices must be a one-dimensional integer array`** for either position. NumPy infers float64 for that mixed signed/large-integer Python list; `_indices` then rejects its inferred dtype. Explicit `np.array([-1,2**63],dtype=object)` returns NaNs as intended. No wraparound or incorrect finite score occurred. | Preserve integer sequences before NumPy's lossy dtype inference; do not start accepting actual floats. Normal bounded `int64` queries remain unaffected. This is safe rejection of a documented unknown-index case, not corruption. |
| E3 / P2 optional runner-helper consistency outside unique-cache assumption | Seven forecast programs have one qualifying airing each for `u0`. Duplicate the same held-out `(user,airing_id)` record for program 901. Shared fold ignores it, so evaluator frequency counts should also remain one per airing. | `make_context` returns unchanged labels and `duplicate_events_ignored=1`, but `label_frequencies(...)["u0"][901]` changes **1 → 2**. `evaluate.py:51–55` counts raw event rows without shared deduplication. A tied program can therefore become certain under the tie-marginalization diagnostic despite unchanged canonical relevance. | Share canonical held-out frequency construction or deduplicate with the same identity/signature policy before diagnostics. No duplicate records were found in the prior official-cache audit; no production metric corruption is asserted for that cache. This finding belongs to the runner helper, not EASE. |

The negative/cancellation fixture is independently derived:
`X=[[2,1,0],[0,1,1],[0,5,1],[0,0,1]]`, ridge 3. For target column 0, the other
two coefficients are exactly `+1/12` and `-1/12`; user 1's prediction cancels to
zero with evidence, and user 3's prediction is `-1/12`. Both remain finite, while
user 3's sole-feature self target is NaN. The implementation passes these checks;
neither a negative score nor failure to repeat from the diagonal alone is a bug.

### Runner semantic follow-up, without a file-writing run or edits

The parent-owned `context_fingerprint` now includes context arrays, boundaries,
historical/forecast schedules and raw training fingerprint. `evaluate_model`
binds model/shared source hashes and checks before/after fit/scoring. These close
the specific *unreviewed runner* gap recorded above; this review did not exercise
its file-writing evaluation path or claim that every runner defense was tested.

One earlier sensitivity gate remains incomplete in this snapshot: `reverse` at
`evaluate.py:135` reverses **held-out label frequency ties only**; predictions
were already ranked with unchanged ascending-ID ties at line 123. This is a
valid label-only diagnostic, not the independent label-versus-score/rank tie
matrix proposed above. Name it accordingly and retain separate prediction-tie
sensitivity before promotion; do not tune an ID direction or call this an EASE
scoring defect. The exhaustive toy oracle supports the current marginalization
formula, subject to canonical frequencies as in E3.

### Stable snapshot and response handoff

SHA-256 values were recorded before tests and checked unchanged afterward for
all reviewed parent/author source/tests and the common dependencies. The new
challenge's completed hash is also recorded:

```text
research_v2/collaborative.py                  418c5791b27372eb12daf97716fa0e25e08a56aa00113f5b16ceb4031f1b2adb
research_v2/contracts.py                     9202448a3b0fb30a8e902112fcbdb825ea78e5901a50744b50e1d5d3aaa9c7e0
research_v2/evaluate.py                      b0166c6980b15319c1e6667aaa7c616bb7c5ed9a0c69b3d2e20b85c83644812b
tests/test_research_collaborative.py          651a5d6e345612918a3cba66fe6ae17ee4b23d9456759e20453f2713781de7c8
tests/test_research_context.py                06d8d1583a362bd9fe20d88d6cf4ab6cbc0b8bd4130117640de9d16d37184ee0
tests/test_research_collaborative_challenge.py 49fe8b78996198f3e569957c1cb812e0146d0007a699ba4f4256b16462d581ba
requirements.lock                           55a7c72c648feaa59dc05769b67f5c45a05fb7619bcbcee672af48a82c299317
```

The unchanged shared `sweettv/contracts.py`, `data.py`, and `evaluation.py` hashes
were also checked against the earlier baseline in the tool record. This handoff
does not authorize a source change or live submission. Next, the EASE author
should independently reproduce/respond to E1/E2, challenge the new oracles and
the protocol (including my severity/scope judgments), and record the result;
the parent can decide E3 and remaining runner sensitivities. Model quality and
private-score improvement remain unestablished by these synthetic checks.

---

## Closure addendum: reciprocal fixes independently rechecked

Reviewed 2026-11-01 after the EASE author response and parent repairs. This
section supersedes the preceding **33-pass/5-xfail** challenge snapshot and its
pending-author-response status; the earlier records are retained as history.
**E1/E2/E3 and reciprocal R1/R2/R3/R4 are closed on the exercised inputs.** No new
EASE mathematical defect or remaining unexpected test failure was reproduced.
R3 and E3 identify the same held-out-frequency mismatch, not two distinct bugs.

Only this report was appended in this stage. No additional tests were necessary:
the parent had removed the repaired markers, and the original assertions now
pass normally. No model, evaluator, runner, source, parameter, or test file was
edited. The frozen fusion grid, its configurations and its results were not
polled or read. There was no quality experiment, official-cache reload, API,
authentication, credential access, submission, commit, push, or subagent.
Synthetic evaluator/CLI fixtures wrote only to the approved runtime directories
listed below; production outputs and the parent's running grid were untouched.

### Adequacy of the author response and reciprocal critique

I reviewed `reviews/research-v2-collaborative.md:362–495` and the repaired helper.
The response is **adequate for E1/E2**: it independently reproduced the original
assertions before correction, explained the pre-coercion loss of Boolean/integer
identity, preserved original sequence elements before numeric inference, and
added boundary regressions without changing EASE defaults or fitting algebra.
It appropriately calls the additional inference-overflow probe fault injection,
not a newly observed native overflow. It also correctly assigns E3 to the parent
and distinguishes the old strict-XPASS marker handoff from an unresolved defect.
That marker handoff is now superseded by the parent's removals and my normal run.

The author's actual reverse challenge exposed gaps in my earlier runner review:
hashing unchanged **bytes** was insufficient without array interpretation headers,
historical/forecast collection boundaries, and the full local source closure.
I accept R1/R2/R4 rather than dismissing them because their probes deliberately
bypass frozen attributes. They test claimed mutation defenses, not an observed
attack or an actual model's behavior. The repaired fingerprint/source manifest
and the author's reverse assertions now pass under my independent execution:

| Finding | Independently rechecked repair / status |
| --- | --- |
| E1 / E2 | Mixed Boolean coordinates now raise for both batch axes; mixed signed/huge integer requests retain exact identity and return correctly shaped NaNs. Typed integer queries and all ordinary mathematical oracles still pass. **Closed.** |
| E3 = R3 | Held-out `(user,airing)` duplicates are ignored consistently with shared labels; conflicting signatures are rejected by the helper. The original critic and reverse duplicate-count assertions pass. **Closed for the exercised valid-duplicate cases.** |
| R1 | Context-array fingerprints include field name, dtype, shape, byte count and a length-delimited header. Same-byte reshape/dtype-view probes now raise the expected mutation error. **Closed.** |
| R2 | Historical and forecast schedule collections have separate named, length-delimited headers. The same-concatenation partition-move probe now raises. **Closed.** |
| R4 | Source bindings cover all local `sweettv/**/*.py` and `research_v2/**/*.py`, plus runner, preparation entry point and lockfile. Reverse coverage/recorded-digest checks pass; before/after closure checks are unchanged. **Closed for the reviewed local-source coverage claim.** |
| R5 | Reverse-*label*-tie naming is now explicit, and the label-marginalization arithmetic remains correct. Neither diagnostic varies prediction-score tie order. **Still a promotion-interpretation gate, not an EASE algebra bug.** |

I independently computed the current `EASEModel` class AST SHA-256 as
`df9d8c1553c5dd98788c6881c356b63e2da551f3b4ddffef5fc5362cafc08e47`,
matching the author's recorded before/after class digest. The new full-module
digest below changes because `_indices` changed. My rational-inverse and separate-
target regression tests passed again. I did not replay old quality experiments
or independently re-execute the author's old-code bitwise comparison; that
comparison remains author-attributed evidence, not a new private/MAP result.

### Executed post-fix checks

From `midterm/`, the direct independent command was:

```sh
.venv/bin/python -B -m pytest -q -p no:cacheprovider --tb=short --runxfail tests/test_research_collaborative_challenge.py
# 38 passed in 0.31s; no skipped/expected-failure cases.
```

Two additional `.venv/bin/python -B` wrappers invoked `pytest.main` with these
argument lists; each hashed source/review inputs before and after and asserted
they were unchanged:

```text
-q -p no:cacheprovider --tb=short --runxfail --basetemp <reverse-runtime-directory>
tests/test_research_context.py tests/test_research_evaluation.py
tests/test_research_validation_reverse.py
=> 81 passed in 0.26s = 55 context + 4 evaluator + 22 reverse cases.

-q -p no:cacheprovider --tb=short --basetemp <normal-runtime-directory>
tests/test_research_collaborative.py tests/test_research_collaborative_challenge.py
tests/test_research_context.py tests/test_research_evaluation.py
tests/test_research_validation_reverse.py
=> 259 passed in 0.73s = 140 author + 38 independent + 81 context/evaluator/reverse.
```

The 259-case run is a **normal run without `--runxfail`**, with zero failures,
errors, skips, xfails or XPASSes. AST inspection also found zero `xfail`
attributes in the two repaired challenge files. Thus the result does not depend
on suppressing marker behavior. Some historical docstrings still mention xfails;
they are not active decorators or current failure states. I did not edit them.
Environment remained Python 3.13.6 / NumPy 2.5.4 / SciPy 1.18.1 / pytest 9.2.0.

Synthetic runtime directories, relative to the workspace:

```text
.opencode/runtime/tmp/opencode/validation-critic-recheck-b657a9e29fff4eb782585ddb5b512c75/
.opencode/runtime/tmp/opencode/validation-critic-normal-50cb511d5ba9497f907bb8039448a4d6/
```

The evaluator/CLI fixtures use synthetic `ScoreProbe` outputs and mock the data
load; they neither load official observations nor inspect the frozen research
grid. Their `result.json` files are test artifacts, not recommendation-quality
evidence. The source-coverage tests verify bindings and guards on their specified
probes, not a universal security or filesystem-atomicity guarantee.

### Current stable hashes

All 28 files in the local execution source manifest, the five test files, and
the author response retained their before/after hashes during the rechecks.
The SHA-256 of the sorted-key JSON mapping of **root-relative source paths to
digests** was
`7d4ab15e5afccb2a1ff97559dd406298e27db9649aa77882e90f1b9cdd4f1027`.
Key inputs to this closure record:

```text
research_v2/collaborative.py                  640183b589a3172548c51890ff1a3d5cd6c5e54e20960f3a507aa2f716b02242
research_v2/contracts.py                     9202448a3b0fb30a8e902112fcbdb825ea78e5901a50744b50e1d5d3aaa9c7e0
research_v2/evaluate.py                      fa5d43eae0269be9f2b635e9ad28c195225395a417cf455bb9292fad130d274c
tests/test_research_collaborative.py          634481b73c6c07ed0b8db648af8d2b60352ec0b3ea0b3c9ac07680470b879d9b
tests/test_research_collaborative_challenge.py d73e5e06522a739c5ba898ffe469b8974a51f1a2a3bf52a948a90fef26b18846
tests/test_research_context.py                06d8d1583a362bd9fe20d88d6cf4ab6cbc0b8bd4130117640de9d16d37184ee0
tests/test_research_evaluation.py             4668197f88b0e0e13fe86b2bfdaa9f5b226a255b0dc8ecdb21512625f62fd2a2
tests/test_research_validation_reverse.py     312f5b8e38b790c0b92b0d83a2459843bddb14a5c7b3ebe9446203bf4ec2f117
reviews/research-v2-collaborative.md          e7a945774ea151d0550bc8902af00ec5b194529a755482d33a3da4366e05baaa
tools/run_research.py                        bac1ddd232c251c328922db3e985980ae93cbeb1fa0525621ec96e4bb2cfc5df
tools/prepare_official.py                    7cc9ef4fcc58c5a4844225f676813508766a0278e28f239bc526afb152c638a6
requirements.lock                           55a7c72c648feaa59dc05769b67f5c45a05fb7619bcbcee672af48a82c299317
```

### Actual remaining promotion gate

The reciprocal implementation/review loop is now closed for the repaired
findings; an additional author response is not needed merely to restate these
passes. **R5 remains:** independently inspect prediction-score ties (including
constituent RRF/top-depth tie handling), separately from label-boundary ties,
without retuning or expanding the frozen grid. Keep the official ascending-ID
metric unchanged and report the sensitivity rather than selecting a favorable
ID direction.

I independently recomputed the author's synthetic sentinel with exact rational
AP: unique held-out frequencies `(6,5,4,3,2,1)`, seven shared candidates, and equal
prediction scores give ascending-score-tie AP **1** but reverse-score-tie AP
**43/150**. Label-only marginalization on the original ascending prediction
remains **1**. This demonstrates the missing interpretation gate with no label
ties; it is not a new SWEET.TV MAP observation or a defect in the correctly named
label-only diagnostic.

After that sensitivity gate, the already-stated frozen-shortlist/multiple-window
and twelve-week-horizon checks still determine promotion. Previously observed,
overlapping history cannot become an untouched test set. I have not inspected
the running grid and make no quality recommendation from it. Any new private
submission still requires fresh explicit approval; synthetic correctness and
closed reciprocal review do not establish an improvement over 0.41001.
