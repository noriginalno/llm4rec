# Data author response to independent challenge D1

Date: 2026-10-31. **D1 accepted and fixed.** The default raw adapter now matches
the documented log `start_time`/`end_time` and EPG `start_time` interface. Actual
CSV bytes remain unavailable; documented-source compatibility is not actual-file
verification or organizer reconstruction equivalence.

## Scope and source basis

Read `CONTRACT.md`, `sweettv/contracts.py`,
`reviews/data-challenge-by-verifier.md`, and the unchanged
`tests/test_data_adversarial.py` before editing. The current author-response
request confirms that the endpoint spellings had been verified in official
competition content before implementation. This response relies on that
supplied provenance; it did not retrieve the page again or inspect a downloaded
CSV. The original blanket uncertainty about header names incorrectly conflated
known documented names with unknown actual export bytes/version.

Only these files were changed:

- `sweettv/data.py`
- `tests/test_data.py`
- `reviews/data-contract.md`
- this new `reviews/data-author-response.md`

The independent tests/review and shared tests were not edited. No CLI, model,
credentials, installation, authentication, download, submission, publication or
delegation actions were performed.

## Reproduction → cause → fix → confirmation

| Item | Author-observed evidence / response |
| --- | --- |
| Documented default fixture | `test_documented_raw_csv_headers_work_with_default_schema` failed because EPG validation required `start` despite an observed `start_time` header. |
| Isolated log default fixture | `test_documented_log_headers_work_when_only_epg_adapter_is_configured` failed because log validation required `start,end` despite `start_time,end_time`. |
| Root cause | The author chose generic defaults even though the documented interface was known. Initial author fixtures mirrored those defaults, so 44 passing author cases did not cover documented-default compatibility. This was an author interface/integration defect, not justified by unavailable actual CSVs. |
| Minimum production correction | `RawSchema.log_start="start_time"`, `log_end="end_time"`, `schedule_start="start_time"`. The documented EPG has duration but no known end column, so `schedule_end=None`; scheduled end remains start plus duration in seconds. |
| Exact alternate adapters | Caller-supplied fields still use exact-name validation. Legacy `start,end` inputs work through `RawSchema(log_start="start",log_end="end",schedule_start="start",schedule_end="end")`; they fail under the documented default rather than being fuzzily renamed. |
| Author fixture correction | Default-path fixtures now contain literal documented names, not names read from `RawSchema`. A separate nonempty legacy-adapter test proves default rejection and explicit-adapter acceptance with two exact expected events. The EPG end-mismatch test explicitly opts into `schedule_end="end_time"`. |
| Evidence correction | Module/contract documentation and audit `schema_default_basis` distinguish known documented defaults from actual-file verification. `schema_verified_on_actual_source` remains **false**. |
| Unchanged semantics | No fragment union, coverage threshold, conflict quarantine, identifier parsing, cohort, catalog, candidate, label-ranking or split-boundary policy changed. Public API signatures are unchanged. Per the handoff, the parent CLI already configures documented names explicitly; it needed no edit and was not executed here. |
| Confirmation | Both original failing cases pass inside the unchanged 41-case adversarial suite. All updated author and unchanged shared tests pass as listed below. |

## Executed before/after checks

Environment observed: Python **3.13.6**, NumPy **2.5.4**, SciPy **1.18.1**,
pytest **9.2.0**. Commands ran from `midterm/` with
`PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`; fixtures were synthetic and
created under the approved workspace runtime temporary directory. Before editing,
the two named failures were first rerun directly (**2 failed, 39 deselected in
0.16s**), then the full adversarial and author suites were rerun.

| Suite | Before this correction, executed by author | After correction, executed by author |
| --- | --- | --- |
| `tests/test_data_adversarial.py` — unchanged | **2 failed, 39 passed in 0.21s**, exit 1 | **41 passed in 0.19s**, exit 0 |
| `tests/test_data.py` — author fixtures updated; one legacy case added | **44 passed in 0.17s**, exit 0 | **45 passed in 0.19s**, exit 0 |
| `tests/test_shared.py` — unchanged | Not rerun by author before this correction; the independent report records its prior 120-pass run | **120 passed in 0.37s**, exit 0 |

Full-suite commands:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_data_adversarial.py --tb=short -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_data.py --tb=short -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_shared.py --tb=short -p no:cacheprovider
```

The post-fix suites ran independently in parallel. Counts and reported elapsed
times above are observed outputs, not estimates. These are **author rechecks**;
the verifier's separate post-fix acceptance remains pending.

SHA-256 was checked before/after the correction:

| File | Before | After |
| --- | --- | --- |
| `sweettv/data.py` | `f11d5eb935b83c45960e7b86634b001dfda2115893ec34257ef57c4b796481e8` | `c28121a2165849b4eb71cc1891415cc7200f734062f9e609c77ac8bf09158025` |
| `tests/test_data.py` | `d1d6f13c5fc3432ffdb8242e606a214a002df6bb5725cfec4b34eacfe1b74c1d` | `14b8ffb86dd95e7751586d83f606d050d059126f6e725037ee605c9394aaf290` |
| `tests/test_data_adversarial.py` | `d6726efbe399fa74d428b9de890a6741ed1a3b88d10d33c1ade63daddaa2252a` | Same; asserted unchanged |
| `tests/test_shared.py` | `7e9d02b64bc5a3bb15d2d206f7395d6e8cdded12390ce40accede6dc08dceb0f` | Same; asserted unchanged |

## Remaining actual-data unknowns and limitations

- Actual CSV header bytes/version and conformance to the documented interface,
  encoding/delimiters, timestamp representations/scale/timezone and DST cases,
  date/week semantics, row counts and raw log-duration units still require source
  inspection. No default UTC assumption, unit rescaling or lossy parsing was
  introduced to make the fixtures pass.
- The organizer's hidden fragment, duplicate, conflict and airing-boundary
  policies remain unknown. Full-component overlap quarantine intentionally loses
  otherwise safe coverage; synthetic consistency does not establish hidden-label
  fidelity or its loss rate on real data.
- `build_fold` still has an upper training cutoff but no lower training bound;
  callers needing a sliding lower bound must supply an appropriately bounded
  source. Actual week/date boundaries remain unverified.
- Runtime/memory complexity is unchanged: reconstruction uses per-channel
  sorting/binary search and real overlaps, not a Cartesian join, but materializes
  grouped log intervals plus schedule/event lists in **O(L+S+E)** memory. The
  earlier synthetic timing is not a real-data throughput or feasibility claim;
  no scale benchmark was rerun for this header-only correction.
- No score, real-data correctness result, personal verification, new independent
  acceptance, or submission confirmation is claimed.
