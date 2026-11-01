# Independent coordinator-pipeline challenge

Date: 2026-10-31. Verifier-owned changes in this stage are limited to
`tests/test_pipeline_adversarial.py` and this review. No production modules,
earlier tests, model/data-author files, or saved result artifacts were changed.

## Outcome

**25 adversarial cases passed; 7 failed, covering five findings.** The failures
are coordinator-local and reproduce with deterministic doubles, not with an
all-suite run against concurrently changing model/data implementations.

| ID | Priority / classification | Finding | Failing cases |
| --- | --- | --- | --- |
| P1 | P2 / confirmed preflight-order defect | Missing protocol confirmation is checked after full raw loading/reconstruction | 1 |
| P2 | P2 / confirmed provenance overclaim on unverified input | `official_kaggle_data=True` follows solely from the default `synthetic=False` | 1 |
| P3 | P2 / confirmed no-overwrite hole on destination collision | The selected final CSV copy overwrites a destination created during final fitting | 1 |
| P4 | P2 / fault-injection robustness gap with demonstrated silent misalignment | A model can mutate shared candidate/user query arrays and still produce a successful export | 2 |
| P5 | P3 / fingerprint coverage gap | Changing provenance or sparse-matrix shape leaves the mutation fingerprint unchanged | 2 |

P4/P5 are missing defenses demonstrated by injected faults. **No actual model
author is alleged to mutate these inputs**, and these findings do not invalidate
the existing nominal synthetic predictions by themselves. P3 requires a
destination collision; a normal fresh-directory run was not shown to overwrite
anything. No official-data quality or Kaggle result is claimed.

The independent standard-library recount of the saved **synthetic** primary
repeat method agrees with the stored aggregate metrics and all **72 per-user
records**. Its MAP@5 is `0.6742592592592592`; precision and recall are both
`0.7611111111111111`. The saved final CSV is byte-identical to the selected
repeat method's final-fit CSV. This is integration/arithmetic evidence only.

## Reviewed source and test isolation

Sources: `CONTRACT.md`, `README.md`, `sweettv/pipeline.py`, `sweettv/cli.py`,
`sweettv/submission.py`, and the shared evaluation/baseline interfaces. Important
bases are the common candidates/cohort, evidence-only fusion, exact five raw
IDs/template order, no output overwrites, training-only selection, explicit
protocol confirmation, and honest evidence status. README explicitly identifies
primary MAP as **model-selection validation**, not an untouched test score.

The reviewed coordinator files had these SHA-256 values, unchanged before and
after the adversarial test run:

```text
sweettv/pipeline.py
a938d5c2059f109a119a22cce96e9f62bd287ff6fc0abf5952d979409d58a735
sweettv/cli.py
5ca24b4772fb4f444fa54ba70bb7cfba3fa836b922810625722ef543f3495b3c
sweettv/submission.py
bf8f9956c811ac11cdcd755609ec166c2e84f375bf392028cc079f44d56b13a9
```

All model constructors are replaced by test doubles. Most cases also replace
`build_fold` with a literal normalized fixture. One bounded integration check
uses the public fold builder to check source zero/missing IDs versus cold
candidate nodes. No real model fit, raw competition access, all-suite run,
authentication, download, installation, or delegation occurred in this stage.

### Grounded tiny oracle

The central fixture has raw item columns:

```text
999, 60, 10, 20, 30, 40, 50, 70
```

Training rows, fixed before any coordinator invocation:

```text
u0:         100, 0, 4, 0, 0, 0, 0, 1
u1:           0, 0, 0, 3, 1, 0, 0, 0
u2:           0, 0, 0, 0, 0, 2, 0, 0
train-only:   0, 0, 0, 0, 0, 0, 0, 9
```

Template order is `u2,u0,u1`; candidate matrix indices are `1,2,3,4,5,6`, i.e.
raw IDs `60,10,20,30,40,50`. Raw `999` is globally most popular but is not a
candidate. Training popularity within the candidate catalog is
`10:4,20:3,40:2,30:1,50:0,60:0`, so the expected fallback list is
`[10,20,40,30,50]`.

Labels are:

```text
u0:         [10,20,30,40,50]
u1:         []
u2:         [60,10,20,30,40]
train-only: [10,20,30,40,50]
```

Thus the primary cohort is exactly `u2,u0`, the all-template diagnostic is
`u2,u0,u1`, and the sparse-history diagnostic is `u2,u1`. The non-template
training user must not enter any of these cohorts. All constituent/fusion
methods use the same candidates and each corresponding cohort.

The CBF double has only raw `60:7,10:6` evidence for `u0`, only raw `50:9` for
`u2`, and no evidence for `u1`. Expected CBF predictions are:

```text
u2: [50,10,20,40,30]
u0: [60,10,20,40,30]
u1: [10,20,40,30,50]
```

MF has only NaNs. Fusion weights `cbf:1,mf:100` **must not** convert MF's
fallback list into 100x popularity votes. Its predictions should equal CBF's:
the supported `u0` programs receive RRF evidence `1/61` and `1/62`; all other
positions are backfilled once. Expected fallback-position totals across three
users are CBF/fusion `12`, MF `15`, repeat `11`, popularity `3`. All were observed.
Both primary CBF users hit ranks 2,3,4,5, so:

```text
AP = (1/2 + 2/3 + 3/4 + 4/5) / 5 = 163/300
all-template MAP = (163/300 + 163/300 + 0) / 3 = 163/450
```

These values are independently stated, not computed using a production metric
as its own oracle.

## P1 — confirmation is checked after expensive input processing

**Reproducer:**
`test_protocol_confirmation_is_checked_before_expensive_raw_loading` supplies
CLI arguments equivalent to:

```sh
.venv/bin/python -m sweettv.cli run --raw-dir raw --protocol not-read.json --out out
```

There is no `--confirm-protocol`. A load spy records calls and returns a tiny
`PreparedData` object, so no actual files or expensive work are needed to expose
the ordering.

**Expected:** argparse exit 2 about missing confirmation **before any call** to
`load`, raw parsing/reconstruction, parameter loading, or model work.

**Actual:** argparse eventually exits 2 with the right message, but the load spy
has already recorded one call. In production that call invokes `load_raw` and
reconstructs the raw source before rejecting the command.

**Cause:** `cli.main` calls `prepared = load(args)` at line 104; the confirmation
guard is at lines 117–118 in the reviewed source.

**Minimum fix:** check a `run` command's confirmation immediately after argument
parsing, before the raw-data load. Keep `audit` usable without this flag. This is
a costly-work/preflight ordering issue, **not** a claim of unauthorized network
access or that the flag authenticates/approves a Kaggle submission.

## P2 — unverified input receives a positive official-source claim

**Reproducer:**
`test_unverified_input_is_not_positively_claimed_as_official_kaggle_data` passes
the handmade normalized fixture with audit
`{"fixture":"unverified","schema_verified_on_actual_source":false}` to
`run_fold` without a `synthetic` argument.

**Expected:** unknown/unverified source status must not be positively asserted
as official Kaggle data. A null/unverified status or an explicit provenance
requirement is acceptable. The test does not require calling all unknown data
synthetic.

**Actual:** successful summary has `official_kaggle_data: true`.

**Cause:** `pipeline.run_fold` defaults `synthetic=False` and writes
`"official_kaggle_data": not synthetic` at line 225. CLI likewise distinguishes
synthetic from other inputs only by existence of a `SYNTHETIC.json` marker.
Absence of that marker/flag is not source verification.

**Minimum fix:** record explicit source status such as synthetic, unverified,
or verified official, based on declared/validated source provenance rather than
negation of a flag. Keep unknown status honest and preserve source manifests.
Do not treat the raw-schema verification boolean alone as authentication of a
download's origin; schema inspection and source identity are different facts.

The existing `synthetic-integration-v1` outputs correctly say synthetic and
unsubmitted. This test does **not** demonstrate an official-quality overclaim in
that saved run, only that an unmarked/unverified input can be mislabeled.

## P3 — selected final copy bypasses the no-overwrite guard

**Reproducer:**
`test_final_selected_copy_does_not_overwrite_a_destination_created_during_final_fit`
uses a three-stage `run_fold` double. It creates a valid final
`submission_cbf.csv`, candidate IDs, and a destination `final/submission.csv`
containing exactly:

```text
existing final artifact\n
```

The new run root itself did not exist beforehand. This deterministically models
a destination created during the final-fit interval; it is not an assertion that
the actual final fold normally creates that file.

**Expected:** `FileExistsError` and unchanged destination bytes; no successful
selection publication after a collision.

**Actual:** the sentinel is replaced with:

```csv
user_id,tv_show_id
u2,60 50 40 30 20
u0,60 50 40 30 20
u1,60 50 40 30 20
```

Validation succeeds and selection is published.

**Cause:** `cli.full_run`, line 47, uses `shutil.copyfile(source,destination)`,
which replaces an existing destination. Exclusive run-root creation protects
normal reruns but not a destination collision after the run starts. By contrast,
the production submission writer already publishes atomically without replacing
an existing destination when `force=False`.

**Minimum fix:** preserve exact source bytes but publish the copy exclusively,
preferably copying to a temporary file in the final directory and using the
writer's no-replace hard-link pattern. An exclusive `xb` copy is also a
no-overwrite improvement, though not atomic publication. Do not regenerate the
CSV and accidentally change its exact selected bytes.

## P4 — mutable query arrays can silently invalidate score alignment

**Reproducer:** two parameter cases in
`test_mutable_query_arrays_cannot_silently_change_candidate_or_user_alignment`.
The CBF double reverses one passed NumPy argument in place and returns correctly
shaped scores for the **now-mutated** inputs. All values remain finite or NaN.

| Injected action | Expected | Actual successful export |
| --- | --- | --- |
| `candidate_indices[:] = candidate_indices[::-1]` | Mutation error or immediate read-only error; original raw catalog/order remains authoritative | Popularity `u2` becomes `[40,30,10,20,50]` instead of `[10,20,40,30,50]`; earlier score vectors are interpreted using the changed candidate order |
| `user_indices[:] = user_indices[::-1]` | Mutation error or immediate read-only error; row-to-user association remains authoritative | CBF `u2` becomes `[10,20,40,30,50]` instead of `[50,10,20,40,30]`; reversed returned rows are attached to unchanged template users |

Both runs publish a success summary and CSVs. The candidate mutation leaves CBF
itself correct in this fixture while corrupting an **earlier** model's retained
vector; this is why validating only the mutating model's shape is insufficient.

**Cause:** `run_fold`, lines 152–169, passes the same mutable user/candidate
arrays to constituent models, retains earlier score vectors, and later consumes
them using shared arrays/template order. The training fingerprint includes
neither query array. A permutation preserves shape/finite-value checks.

**Minimum fix:** make authoritative query inputs immutable to models or pass
separate per-call copies and check them against immutable snapshots before
accepting returned scores. Preserve the authoritative candidate ordering for
ranking and the user ordering for assignment. Detect changes before accepting
another model's results/export. Consider copying returned score arrays as well
if a model can return shared mutable buffers.

This is a demonstrated coordinator robustness gap under injected model faults,
not evidence that current author models perform these mutations.

## P5 — the mutation fingerprint omits provenance and shape

**Reproducer:**
`test_training_fingerprint_covers_provenance_and_sparse_matrix_shape` starts
from the fixed 4x8 training fixture. Separate cases perform either:

```python
data.provenance["train_end"] = 999
data.interactions.resize((4, 9))
```

**Expected:** fingerprint changes, or an explicit structural validation rejects
the changed shape. The shape mutation intentionally violates the item/column
contract while keeping the same sparse buffers.

**Actual:** both mutations retain the same fingerprint:

```text
bb17971ca79bd06b33d1eb44d10cfc449edf4190170857c6ee104bf1bbbd4377
```

**Cause:** `training_fingerprint`, lines 44–50, hashes item IDs, CSR buffers,
users and metadata but not CSR shape or `TrainingData.provenance`.

**Minimum fix:** include explicit structural descriptors (shape and dtypes,
unambiguously framed), plus canonical provenance if the guard claims all shared
training state. Alternatively freeze provenance separately and validate the
users/items/matrix shape invariant after model calls. At minimum document a
narrower fingerprint scope accurately. Query arrays need their own protection
as in P4.

This is lower-priority coverage hardening. No hash collision in nominal valid
training data or silent successful end-to-end run with the resized matrix was
demonstrated. Count-buffer and metadata mutations **are** detected by existing
guards in separate passing tests.

## Passing coordinator contracts and non-bug design distinctions

| Challenge | Observed result |
| --- | --- |
| Common cohorts/candidates | Identical primary/all-template/sparse cohorts for every method, including zero-label users only in the applicable diagnostic; no evidence-based per-method user dropping. Extra non-template training user is not evaluated. |
| Batch/scalar parity | Manual expected predictions, cohorts and CSVs agree for batch sizes 1,2,100 and a scalar-only model API. Batches preserve requested candidate matrix-index order. |
| NaN versus invalid score values | NaN-only methods legitimately backfill; 1-D, transposed, short-candidate and extra-user batches, positive/negative infinity all abort with an explicit model-named error. |
| Model errors | Injected fit/score exceptions propagate; no successful summary or submission CSVs are published. Partial newly created diagnostic output is not confused with a completed run. |
| Known training mutations | Fit-time count/metadata mutations and score-time count mutation abort. |
| Shared fallback / no RRF fallback votes | Exact literal fallback counts and evidence-only fusion predictions agree, including an all-NaN constituent with weight 100. |
| Per-method CSV exactness | Every generated method's CSV has the exact header, template string order and IDs from its saved prediction JSON; exactly five distinct candidate IDs per user. |
| Source schedule zeros/cold nodes | A bounded real-fold fixture drops zero/missing program rows, retains six positive candidates, reports one seen/five cold candidates, and does not filter a cold candidate just because it lacks training counts. |
| Existing output guards | Existing fold directories, full-run roots and JSON paths are rejected before model fitting; sentinel bytes stay unchanged. This does not close P3's later final-destination collision. |
| Training-only example users | Activity `[1,4,6,8,10,50]` gives middle pool `b,c,d`; min/max top shares select `b,c`, independent of template order. In the main fixture `u1,u2` remain selected when all held-out labels are removed, and predictions remain unchanged. |
| Development versus primary selection | Development's overall repeat winner is not reused blindly: development selects only `hybrid_balanced` fusion weights; those same frozen weights/constructor parameters go to primary and final, while primary selects `cbf` in the mock counterexample. |
| Empty evaluation cohort | Empty development or primary MAP values cause an explicit `ValueError`, no final fit and no selected output. The coordinator does **not** silently choose a baseline. This is appropriate abstention, not an implementation bug. |
| Exact selected copy / no official score | In a collision-free mock run the selected source and final copy are byte-identical with the same SHA-256. Synthetic flags, `submitted=False` and `kaggle_score=None` are preserved. |

### Design gaps that are not mislabeled as current bugs

- Primary MAP selects the final method; this makes it model-selection validation,
  not an untouched performance estimate. README/summary warnings say so. The
  primary-selected winner's primary MAP is selection-biased, but no independent
  test claim was observed.
- Cross-fold protocol chronology is caller-confirmed rather than enforced by a
  coordinator preflight. Per-fold ordering is checked downstream. Adding checks
  that development labels precede primary evaluation, validating all three
  protocols before fitting, and recording parameter-file hashes would harden
  configuration management. No overlap/leakage in the saved protocol was
  demonstrated: its development end equals primary train cutoff, and primary
  end equals final train cutoff.
- Empty-cohort rejection occurs after the applicable fold is computed. Earlier
  cheap cohort preflight could save work, but current rejection is explicit and
  does not invent scores.
- The same two examples are fixed **across methods within a fold**. Training
  history changes across folds, so different selected examples across folds are
  not contrary to the documented policy.
- Winner selection is constrained by predefined fusion grids and the observed
  temporal windows; it does not establish official recommendation quality.
  Official CSVs remain authentication-blocked, and a returned Kaggle score,
  personal verification and final delivery evidence are still absent.

## Independent saved-artifact recount

Executed the existing verifier in isolated, no-site-packages mode, importing no
production parser, metric or ranker:

```sh
.venv/bin/python -I -S tools/independent_verify.py \
  --submission results/synthetic-integration-v1/primary/submission_repeat.csv \
  --template results/synthetic-integration-v1-raw/submission.csv \
  --allowed-ids results/synthetic-integration-v1/primary/candidate_ids.json \
  --predictions-json results/synthetic-integration-v1/primary/predictions_repeat.json \
  --labels-json results/synthetic-integration-v1/primary/labels.json \
  --users-json results/synthetic-integration-v1/primary/evaluation_users.json
```

A standard-library-only wrapper compared the returned MAP/precision/recall and
every per-user AP/precision/recall/hit/relevance count with
`primary/metrics.json`, with absolute float tolerance `1e-15`. All 72 records
and aggregates agreed. The independently checked primary CSV has 72 users,
five recommendations each and 34 distinct recommended IDs; SHA-256:

```text
861408a53e7289ce12610e99b022da4fa56a56f693c2b935f9ee5967f7c7c4f7
```

The wrapper also compared `final/submission_repeat.csv` with
`final/submission.csv` byte-for-byte and checked the recorded selection hash:

```text
794dc337d33715956c8d331e27485370bdd0b3a10389a799866400d78f41c254
```

This independently repeats **one primary method and the selected final copy**,
not all 22 checks reported by the coordinator. It does not retrain the saved
models or validate against organizer-private labels.

## Execution record and handoff

Environment: Python **3.13.6**, NumPy **2.5.4**, SciPy **1.18.1**, pytest **9.2.0**.
From `midterm/`:

```sh
.venv/bin/python -m pytest -q --tb=short tests/test_pipeline_adversarial.py -p no:cacheprovider
```

Observed: **7 failed, 25 passed in 0.26s**, exit 1. The actual invocation also
set `PYTHONDONTWRITEBYTECODE=1` and used a new `--basetemp` under the approved
workspace runtime temporary directory; fixtures were removed afterward. The
stdlib artifact recount ran independently in parallel and completed
successfully. No all-suite result is inferred from these runs.

No failing expectation was skipped, xfailed or weakened, and no production fix
was made by this verifier. Coordinator response and independent post-fix
recheck remain pending. Address P1–P4 as scoped, choose/document P5's fingerprint
scope, and rerun this file without changing unrelated author implementations.
