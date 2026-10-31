# MF author response to the PPR author's challenge

Date: 2026-10-31. **M1 accepted and corrected as an evidence/documentation issue.**
The review reproduced no MF training/scoring correctness defect; this response
does not turn the unsupported BLAS-cap claim into a numerical bug or claim that
oversubscription was observed.

## Finding, cause, and minimal correction

The previous class docstring asserted a one-thread training BLAS cap and an
at-most-`num_threads` batch-scoring cap. Diagnostics called the training value
`training_blas_thread_limit`, and an author test used `all(...)` on a possibly
empty list of pool reports. Entering a `threadpoolctl` context is only a request
to supported controllers; `all([])` cannot establish control of Apple Accelerate.

Changed only `sweettv/models/mf.py`, `tests/test_mf.py`, and this new response:

- Documentation now calls both BLAS limits **requests for supported controllers**.
  Empty pool reports are explicitly unverified. Nonempty reports describe
  controller configuration, not measured active workers or a whole-process cap.
- `num_threads` remains the parameter passed to native CPU ALS, not an observed
  thread count. Its meaning is explicitly qualified in documentation/comments.
- Renamed the ambiguous diagnostic `training_blas_thread_limit` to
  `training_blas_thread_limit_requested`. This is `1` for an actual nonempty fit
  and `None` for an empty fit, where no request is made. The old ambiguous key is
  not retained as an alias. The inspected Python consumers did not use it.
- Kept the actual `training_blas_pools` reports and added
  `training_blas_control_status`:

  | Status | Meaning |
  | --- | --- |
  | `not_requested_empty_fit` | No native fit or BLAS-limit request occurred. |
  | `unverified_no_reported_pools` | No supported BLAS pool was reported; control cannot be established. |
  | `reported_pools_at_requested_limit` | A nonempty set of reported pools has configuration counts equal to the request; this is not a runtime worker measurement or a claim about unreported libraries. |
  | `reported_pools_not_at_requested_limit` | The reported pool counts do not all establish the requested configuration. |

- Fit-time pool reports are explicitly **not** evidence of batch-scoring thread
  control. Batch scoring retains its supported-controller request but does not
  claim a measured cap.
- Replaced the vacuous assertion with explicit empty/nonempty status checks.
  Added four synthetic-report regression cases: no reports, OpenMP-only reports
  (not a BLAS controller), a BLAS pool reporting the requested count, and mixed
  BLAS pools whose reports do not all match the request. These mocks test the
  diagnostic classification, **not actual hardware thread control**. Empty-fit
  tests additionally verify that no request is reported as having occurred.

There is no change to ALS confidence construction, warm compaction, native CPU
backend, factor scoring, missing-evidence policy, random initialization, thread
arguments, or batch-output cap. No environment variables, new dependencies,
alternate algorithm, GPU request, or unsupported-backend fallback were added.

## Executed verification

From `llm4rec/midterm/`:

```sh
.venv/bin/python -m pytest tests/test_mf.py tests/test_mf_adversarial.py -q
```

- Before the correction: **78 passed**.
- After the correction and four new diagnostic cases: **82 passed**, with no
  failures or skips. The unchanged adversarial suite includes actual implicit
  native CPU fitting and an independent one-iteration normal-equation oracle.

The review/challenge files and shared contract were not edited. Their SHA-256
fingerprints were recorded before this response's edits:

| Protected file | SHA-256 |
| --- | --- |
| `reviews/mf-challenge-by-ppr.md` | `5d745f14632f47a090dd43188cd4929a383a0ba7ce28c6fc5cfcf192679cd842` |
| `tests/test_mf_adversarial.py` | `b15336ccdddc70a8a7be58f90d4c0a7d7a195f2c9ce172360d7a8852abe69daa` |
| `CONTRACT.md` | `cf755cf769e56c75224f974bcd76f6249039521521de5c076f451d1cd01afc61` |

## Actual native hardware/library observations after the correction

A separate tiny, network-free native fit/batch-score smoke used training counts
`[[1,5]]`, two factors, two iterations, and ALS thread parameter requests `1`
and `4`. Observed:

- Python 3.13.6; machine architecture `arm64`; NumPy 2.5.4; implicit 0.7.3.
- NumPy's **build configuration** names BLAS `accelerate`. This is not a
  thread-activity measurement or a claim about every loaded numerical library.
- Backend: `implicit.cpu.als.AlternatingLeastSquares`; solver module:
  `implicit.cpu._als`; native ALS `num_threads` parameters were respectively
  `1` and `4`.
- `threadpool_info()` was `[]` before training and after each batch score.
- Both fits recorded `training_blas_thread_limit_requested = 1`,
  `training_blas_control_status = "unverified_no_reported_pools"`, and
  `training_blas_pools = []`.
- Both requested score arrays were finite. No timing benefit, active thread
  count, enforced Accelerate cap, or recommender-quality result is inferred.

**Remaining limitation:** actual Accelerate BLAS thread counts during fitting
and batch scoring remain unverified in this environment. CPU operation is real;
the backend has not been replaced with CUDA, Apple Metal/MPS, or a Python mock.
The existing supported-controller requests remain useful on compatible BLAS
implementations, without being represented as universal measured controls.

No authentication, download, installation, submission, publication, or subagent
operation was performed. M1's unsupported claim has been corrected; independent
reviewer closure remains a separate step.
