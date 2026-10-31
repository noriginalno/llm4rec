# Independent pipeline post-fix verification

Date: 2026-10-31. Scope: independent inspection and re-execution of the original
pipeline challenge after the coordinator's response. **Only this new report was
created.** Original challenge tests/report, production, other tests, and saved
results were not edited.

## Verdict

**All seven original failing cases are independently verified closed.**

- Unchanged `tests/test_pipeline_adversarial.py`: **32 passed in 0.21s**.
- Full repository suite: **561 passed in 2.56s**.
- Additional in-memory probes confirmed protocol preflight, query-mutation
  after-checks, batch-return ownership, and recorded/source-drift hashes.
- **One additional scalar-return-buffer ownership gap remains**, outside those
  pytest cases. A scalar model reusing a row buffer produces an incorrect but
  successful mock export. See S1; do not interpret the green suite as coverage of
  that case or an unconditional robustness sign-off.

These are verifier-executed results, not a repetition of the parent's reported
159 passes. No official competition files, returned Kaggle score, recommendation
quality, personal verification, or completed submission is claimed. The earlier
72-user synthetic artifacts were not regenerated in this stage.

## Commands and integrity

Working directory: `llm4rec/midterm/`. Environment actually observed: Python
**3.13.6**, NumPy **2.5.4**, SciPy **1.18.1**, pytest **9.2.0**.

The executed pytest commands, omitting only generated temporary paths, were:

```sh
.venv/bin/python -m pytest -q --tb=short tests/test_pipeline_adversarial.py -p no:cacheprovider
.venv/bin/python -m pytest -q --tb=short -p no:cacheprovider
```

Both exited **0**. Each invocation also received `PYTHONDONTWRITEBYTECODE=1`
and a separate new `--basetemp` directory under the approved workspace runtime
temporary directory. These independent processes ran in parallel. Their
temporary fixtures were removed afterward. The full suite includes synthetic
all-model integration, but is not an official-data experiment.

SHA-256 snapshots taken before/after execution agreed. The full-suite snapshot
covered all **14 production Python files**, all test files, and the original
challenge report; the supplemental probes checked the production tree and the
original reviewer files again. Relevant exact hashes:

| File | SHA-256 |
| --- | --- |
| `sweettv/pipeline.py` | `c67d6d4a9a01fb847198128961ae6b29d4f9f1292771cc2a80a0c4a5abff85c9` |
| `sweettv/cli.py` | `f3ced3b7e7e259583b96c260b21442b0df9432f6c7cd0d02df0a0e68f289f9e0` |
| `sweettv/submission.py` | `34dbee94ec50eee8b8efb7c0d0826d6f7253df5f609a33105a6ee4207905da69` |
| `tests/test_pipeline_adversarial.py` | `7fa871a3da247bbf706f80d0b935a47f9e9250d9d44ccd8f2cb3751731bd1349` |
| `reviews/pipeline-challenge-by-verifier.md` | `dd915fac9ff0767c8ef05f3631fb0ad8f5942c7f27c7f796796ba2dbcd42f4d9` |

The challenge's literal expected ranks, cohorts, fallback totals and AP values
were reread and not weakened. The original report remains a historical record of
the pre-fix results; this additive report records the recheck.

## Closure of the original findings

Line references below identify the reviewed post-fix files, not the old report's
line numbers.

| Original finding | Inspected correction | Independently observed result / status |
| --- | --- | --- |
| P1: late confirmation | `cli.py:107–130` rejects unconfirmed `run` immediately after parsing and validates the protocol before `load`; `full_run` also validates before creating output. | Original missing-confirmation case passes with zero load calls. Three extra invalid confirmed protocols also fail before loading or creating output. **Closed.** |
| P2: positive official-source inference | `pipeline.py:260–264` writes `official_kaggle_data=None` and `data_source_status="unverified_source_identity"` when not marked synthetic; synthetic inputs retain false/synthetic status. | Original handmade unverified input no longer receives a positive official claim. **Closed.** This is honest uncertainty, not source authentication. |
| P3: final destination overwrite | `cli.py:65` calls `copy_submission_exclusively`; `submission.py:101–115` copies bytes into a temporary file in the destination directory and publishes using `os.link`, cleaning the temporary file in `finally`. | Original collision case now raises `FileExistsError` with sentinel bytes intact; collision-free exact-copy/hash case still passes. **Closed.** |
| P4: mutable query alignment | `pipeline.py:183–196` gives each model read-only copies of user/candidate arrays, verifies them against authoritative arrays after the call, and copies batch return arrays. | Both original direct-mutation cases fail closed rather than exporting misaligned results. Extra write-flag-bypass probes are also detected. **Original cases closed**; scalar return ownership remains separately incomplete in S1. |
| P5: incomplete fingerprint | `pipeline.py:44–58` includes matrix shape, canonical users/metadata/provenance, and named array descriptors containing dtype, shape and byte length before each buffer. | Both original provenance/shape counterexamples now change the fingerprint. Existing count/metadata mutation regressions remain green. **Closed for the reported coverage gaps.** |

### Correct classification of the original observations

- **P1 and P2:** real operational/provenance behaviors reproduced through normal
  API/CLI invocation, not hypothetical attacks. They were not proof of network
  activity or an official-score claim in the correctly marked synthetic run.
- **P3:** a real overwrite operation under a controlled destination collision.
  It was not evidence that a nominal fresh run had already destroyed user work.
- **P4:** injected model-input mutations exposing absent defensive boundaries.
  No current model author was observed mutating query inputs. The fix is useful
  contract hardening, not evidence of a prior security breach.
- **P5:** fingerprint-scope hardening. The shape probe deliberately made the
  normalized data structurally invalid; no nominal-data cryptographic collision
  or prior official-data corruption was demonstrated.

## Additional probes actually executed

These ran through `.venv/bin/python -B -` with a Python stdin diagnostic,
`runpy.run_path("tests/test_pipeline_adversarial.py")`, temporary fixture outputs,
and `pytest.MonkeyPatch.context()` for in-memory doubles. No additional test file
or production source was written. There were eight satisfied supplemental checks
and the S1 ownership mismatch; these are **not added to the pytest totals**. The
diagnostic printed the mismatch rather than exiting as a failing pytest test.

### Protocol preflight: three checks passed

Starting from the original valid protocol, separate confirmed CLI calls used:

1. `development.train_end=True` → `development: cutoffs must be integer epoch seconds`.
2. `development.forecast_end=21`, while primary training ends at `20` →
   `Protocol leakage: development labels extend beyond primary training cutoff`.
3. Omitted final stage →
   `final: require exactly ['forecast_end', 'forecast_start', 'train_end']`.

All raised `ValueError` before the load spy was called; no output directory was
created. This closes the former cross-stage chronology/preflight design gap for
these inputs. It does not verify calendar-week interpretation against raw data.

### Query guards: two checks passed beyond read-only flags

For each query array separately, a CBF double called `setflags(write=True)`,
reversed the copy, and returned correctly shaped scores. Both attempts raised:

```text
cbf mutated query user/candidate alignment
```

No success summary was published. Thus detection is not dependent solely on
NumPy's reversible write flag. These checks still are not a sandbox against
arbitrary malicious Python or a model deliberately returning false scores.

### Batch return buffer: one check passed

A CBF batch double returned an array that a later association double overwrote
with NaNs. The coordinator retained the independently expected original CBF
predictions for all three fixture users. The explicit batch `copy=True` prevents
that cross-model alias from corrupting earlier scores.

### Source hashes: two checks passed

1. A nominal mock fold's `implementation_sha256` map matched an independent
   `hashlib.sha256(path.read_bytes())` scan of all 14 production `.py` files.
   The SHA-256 of `json.dumps(map, sort_keys=True).encode()` was:

   ```text
   f30aebb82bb1e95f1110ce70a449f31a0ea629a1872d83157504df4a49ac0de0
   ```

2. An in-memory hash-function double returned the correct map at entry, then
   changed only the reported `pipeline.py` digest on its second call. The fold
   raised:

   ```text
   Implementation source changed during this run; rerun a stable version
   ```

   No summary or per-model submission CSV was published. This verifies the
   comparison/error path without violating the prohibition on production edits.
   It is **not** represented as a live edit of a source file during execution.

## S1 — scalar score-return reuse still aliases rows

**Priority/classification:** P2 robustness/ownership gap, adjacent to P4 but not
one of the original seven failing cases. This is a demonstrated incorrect export
with a controlled scalar-only double using a reusable scratch buffer. There is
no evidence that a current production model uses that return-buffer pattern or
that existing saved synthetic outputs are affected.

**Cause:** the batch path copies each returned batch immediately, but the scalar
path at `pipeline.py:191` first collects all returned row references:

```python
scores = np.vstack([model.score(int(i), query_candidates) for i in query_users])
```

If every call returns the same scratch array, subsequent calls replace earlier
rows before `np.vstack` copies anything. This is output ownership, not a mutation
of training data or query inputs, so the current guards do not detect it. The
model interface does not explicitly require a newly owned return array per call.

### Exact bounded reproducer

From `midterm/`, the following reproduces the supplemental observation without
editing any existing file:

```sh
.venv/bin/python -B - <<'PY'
import json
from pathlib import Path
import runpy
import tempfile
import pytest
from sweettv import pipeline

fx = runpy.run_path("tests/test_pipeline_adversarial.py")
root = Path("../../.opencode/runtime/tmp/opencode").resolve()
with tempfile.TemporaryDirectory(prefix="scalar-ownership-probe-", dir=root) as tmp:
    with pytest.MonkeyPatch.context() as mp:
        prepared, calls = fx["_install"](mp)
        original_factory = pipeline.create_model

        class ReusedScalar(fx["_ScalarModel"]):
            def score(self, user, candidates):
                value = super().score(user, candidates)
                if not hasattr(self, "buffer"):
                    self.buffer = value.copy()
                else:
                    self.buffer[:] = value
                return self.buffer

        def factory(name, parameters=None):
            return (ReusedScalar(name, calls) if name == "cbf"
                    else original_factory(name, parameters))

        mp.setattr(pipeline, "create_model", factory)
        output = Path(tmp) / "fold"
        summary = pipeline.run_fold(prepared, 10, 10, 20, output, synthetic=True)
        actual = json.loads((output / "predictions_cbf.json").read_text())
        print(json.dumps({"expected": fx["CBF_PREDICTIONS"], "actual": actual,
                          "support": summary["support"]["cbf"]}, indent=2))
PY
```

Expected versus observed:

| User | Expected CBF IDs | Actual CBF IDs |
| --- | --- | --- |
| `u2` | `[50,10,20,40,30]` | `[10,20,40,30,50]` |
| `u0` | `[60,10,20,40,30]` | `[10,20,40,30,50]` |
| `u1` | `[10,20,40,30,50]` | `[10,20,40,30,50]` |

The last user has no evidence, so all retained row references become NaNs.
Observed fallback support is **15 positions / 3 no-evidence users**, instead of
the literal oracle's **12 positions / 1 no-evidence user**. A success summary
and incorrect CBF CSV were published in the disposable fixture directory.

**Minimum follow-up:** copy each scalar result immediately, before requesting
the next user's row, for example:

```python
scores = np.vstack([
    np.array(model.score(int(i), query_candidates), dtype=float, copy=True)
    for i in query_users
])
```

Add a coordinator-owned regression for reusable scalar buffers. No fix or test
change was made by this verifier; S1 remains open at this source snapshot.

## Remaining limits and handoff

- Read-only copies, fingerprints and source comparisons are consistency defenses,
  not isolation from arbitrary in-process code. A model can still compute wrong
  finite scores without violating shape/value checks.
- Source hashes cover production `.py` files at fold entry and after scoring,
  not all dependencies, configuration files, loaded-bytecode identity, or every
  possible edit during later export/between folds. No mixed-version run was
  observed here; this is a scope limit, not a demonstrated new source-drift bug.
- Exclusive hard-link publication preserves the tested destination and exact
  bytes on this filesystem. It is not a general crash-durability/fsync guarantee
  or a claim of hard-link support on every platform.
- Empty-cohort handling, frozen development fusion configuration, primary-based
  method selection, common candidates/cohorts and evidence-only fallback remain
  covered by the passing unchanged cases. Primary MAP is still model-selection
  validation, not an untouched generalization estimate.
- Official files, source/date/week reconciliation, organizer reconstruction
  fidelity and external submission evidence remain outside this synthetic/local
  verification. No authentication, downloads, installation, network submission,
  publication or subagents were used.

**Handoff:** accept closure of the original seven regressions at the recorded
hashes; address or explicitly constrain scalar return ownership (S1), then
independently rerun that reproducer and the unchanged suites. Do not rewrite the
original challenge history or treat this report as an official-quality result.
