# PPR author response to the MF author's adversarial review

Date: 2026-10-31. Finding: **PPR-MF-01 accepted, reproduced, and fixed.**

The review correctly identified an integer-domain contract defect, not a
PageRank math defect. The response changes only `sweettv/models/ppr.py`, adds
author-owned regressions in `tests/test_ppr.py`, and creates this note.
`tests/test_ppr_adversarial.py` and `reviews/ppr-challenge-by-mf.md` are unchanged.

## Before: actual reproduction

I read the challenge, adversarial tests, current implementation, and shared
contract before editing. From `llm4rec/midterm`, I executed:

```sh
.venv/bin/python -m pytest -q tests/test_ppr.py tests/test_ppr_adversarial.py
```

**Observed before the fix:** 61 passed, 2 failed in 0.28 seconds. Both failures
were the unchanged adversarial parameterizations for `10**30` and `-(10**30)`.
On a one-user/one-item fitted graph and candidates `[0,-1,1]`, both raised

```text
ValueError: user_indices must contain matrix indices, not raw IDs or floats
```

**Expected:** a float array `[NaN,NaN,NaN]`, because these are valid integer
inputs but unknown users.

## Cause and minimal fix

Previously `score(user_index, candidates)` immediately delegated through the
list `[user_index]`. NumPy inferred object dtype for integers outside its native
integer domain. `_indices` rejected that dtype before the unknown-user bounds
mask could execute. This accidentally imposed an int64-representability
restriction on the scalar Python-integer interface.

The fix is localized to index handling:

1. `score` first preserves the fitted-state check, rejects scalar booleans, and
   validates the integer index protocol with `operator.index`. Floats, strings,
   and malformed scalar types still raise `ValueError`.
2. It bounds-checks the resulting Python integer **before** constructing a NumPy
   array. Any negative/out-of-range integer is normalized to the existing `-1`
   unknown sentinel. It then delegates to `score_batch` as before.
3. Delegation preserves candidate type/shape validation and creates ordinary
   no-evidence diagnostics, rather than returning early with stale diagnostics
   or bypassing malformed candidate checks. Unknown queries perform zero walk
   operator applications and do not claim convergence.
4. `_indices` now bounds-checks unsigned values against int64's maximum before
   narrowing. Values outside that domain become `-1`; they never wrap during
   casting. Existing unsigned public-score tests already passed before this
   change; this is defensive conversion hardening, not a second reproduced
   scoring defect.

No modulo operation, clamping to a valid user, lossy float conversion, or
conversion-before-bounds-check is used. In particular, `2**64`, `2**64+1`, and
`-(2**64)+1` cannot become real users 0 or 1.

The broader policy for object-dtype batch/candidate arrays is unchanged: they
remain rejected. The review explicitly treated extending that policy as
optional. No contract change was made.

## Added author regressions

Eighteen additional cases cover:

- Nine unknown scalar integer values: positive/negative `10**30`, the int64/
  uint64 boundaries, values that would become 0/1 under incorrect modulo
  narrowing, and `np.uint64` maximum. A two-warm-user graph detects accidental
  wrap into actual indices. Every result is float64 NaN in candidate order,
  with `no_evidence`, zero operator applications, and no solved-user entries.
- Seven malformed scalar types, including Python/NumPy floats and booleans,
  remain rejected rather than silently accepted as integer indices.
- An unknown huge integer still requires a fitted model and still validates
  malformed float/bool/string/multidimensional candidate arrays.
- Unsigned users/candidates normalize safely before narrowing, preserve inputs
  and duplicate positions, and cannot produce evidence for unknown indices.

## After: executed verification

| Executed command, from `llm4rec/midterm` | Actual result |
| --- | --- |
| `.venv/bin/python -m pytest -q tests/test_ppr.py tests/test_ppr_adversarial.py` | **81 passed in 0.30 seconds**: 65 author cases and all 16 unchanged adversarial cases |
| `.venv/bin/python -m pytest -q tests/test_ppr_adversarial.py -k unknown_python_integer` | **2 passed, 14 deselected in 0.14 seconds** |
| `shasum -a 256 sweettv/models/ppr.py tests/test_ppr.py tests/test_ppr_adversarial.py reviews/ppr-challenge-by-mf.md` | Source/test changes and unchanged reviewer artifacts fingerprinted below |

The original 63 combined cases now pass, including the two exact reviewer
failures; the 18 added author cases also pass. Neither reviewer test was edited,
skipped, or xfailed. The independent linear-system, weighted graph, dangling,
convergence, nonconvergence, batching, and mass-conservation checks all remain
included in the passing suites. Graph construction and PPR iteration arithmetic
were not changed.

Observed environment remains Python 3.13.6, NumPy 2.5.4, SciPy 1.18.1, pytest
9.2.0 on macOS arm64. These are synthetic correctness checks, not Kaggle metrics
or a new full-workload performance measurement.

## Artifact fingerprints

| Artifact | Before SHA-256 | After SHA-256 |
| --- | --- | --- |
| `sweettv/models/ppr.py` | `f45e4f54876769fc8b1003ee44b34ac119d5e94c002506dead98efc2ae7d81bc` | `eccc19a05966433925d422a3a09a19edc6ad2dbd2c5bedfa94270f3414b92bc5` |
| `tests/test_ppr.py` | `343d08e1ac8091d96f4b5568a9480d67710527ceb9cde55ee360dde0501929a4` | `a20d3b559fdef57c6b9947efc1a7a066abe73c5c4460dba416deab247663bd83` |
| `tests/test_ppr_adversarial.py` | `ea44e6a5332ca6be3b68407bc9632078dbeac8f5223f10161258606dc7e5a86a` | unchanged |
| `reviews/ppr-challenge-by-mf.md` | `7b68e7eb770100ba1746196f9f696c15cb4bdf885704f19f9088af46d213091d` | unchanged |

## Other review observations

The review's documented output-allocation limitation, approximate-zero evidence
policy, and near-one-alpha iteration-budget behavior are not treated as fixed
bugs here. This minimal response does not add an output-size guard or alter
convergence semantics. Callers must still issue bounded requested score outputs.

**Closure:** PPR-MF-01 has a reproduced-before / passing-after implementation
fix and unchanged adversarial regressions. No shared/other-model edits, installs,
authentication, downloads, subagents, submission, or publication occurred in this
response stage.
