# Independent data-pipeline challenge

Date: 2026-10-31. Scope: `sweettv/data.py`, challenged through public APIs by a
verifier who did not author that module. Changes in this stage are limited to
`tests/test_data_adversarial.py` and this review. No production code, author
tests, earlier shared tests, raw files, or model files were changed.

## Outcome and evidence limits

- **39 adversarial cases passed; 2 failed.** Both failures reproduce one
  default-schema/documented-source-interface mismatch, isolated for EPG and logs.
- **120 unchanged shared cases passed**, including the earlier duplicate raw-ID
  catalog and strict BOM-header regressions.
- No future-behavior leakage, incorrect broadcast union/counting, ID precision
  loss, or null-year casting failure was observed in the exercised cases.
  This is bounded synthetic evidence, not a proof for arbitrary inputs.
- **Actual Kaggle data remain unavailable and unverified.** No authentication,
  downloads, installation, submission, publication, or delegation occurred.
  No actual CSV version, encoding, date range, timestamp unit/time zone, row
  count, throughput, or organizer reconstruction equivalence is claimed.

Reviewed `data.py` SHA-256, unchanged before/after the adversarial run:

```text
f11d5eb935b83c45960e7b86634b001dfda2115893ec34257ef57c4b796481e8
```

The rechecked `tests/test_shared.py` SHA-256 was:

```text
7e9d02b64bc5a3bb15d2d206f7395d6e8cdded12390ce40accede6dc08dceb0f
```

## Sources and independence

| Basis | What it establishes | What it does not establish |
| --- | --- | --- |
| Official-documentation facts supplied in the current independent-review request | Log columns `start_time`/`end_time`; EPG `start_time` | This verifier did not retrieve the original page or inspect current CSV bytes; this is user-carried source provenance, not a new source audit |
| `CONTRACT.md`, lines 19–30 | Channel matching, half-open intervals, EPG duration in seconds, positive program IDs, per-airing union and >=80% threshold, training isolation, schedule candidates, frequency-based labels | Organizer details for conflicts, duplicates, fragment union and boundary airings remain explicit engineering assumptions |
| `sweettv/contracts.py`, lines 17–46 | String user IDs, separate raw item IDs, aligned metadata, sparse positive count interface | No claim that every possible upstream export fits the signed-int64 item interface |
| `reviews/data-contract.md`, especially parsing/reconstruction/fold sections | Author's declared fail-closed parsing, quarantine, fully-contained label and fixed-cohort policies | A declared generic adapter default is not evidence that it matches the documented source |
| Executed `tests/test_data_adversarial.py` | Literal expected matrices, labels, candidate IDs, event counts and metadata compared with production | No production helper is used to calculate an expected count, relevance list or qualifying broadcast |

Fixture constructors only assemble records. Expected results are explicit
constants and hand arithmetic, not calls to `reconstruct_events`, private union
helpers, label counters, or the ranker. Metamorphic invariance is grounded in
nonempty manually specified training matrices, so two equally empty outputs
cannot satisfy the central leakage check accidentally.

## D1 — confirmed default-interface mismatch (high operational priority)

**Classification:** hard, reproducible failure against the documented interface;
**not** verified failure on a downloaded CSV and **not** silent label corruption.
It is understandable that the author chose an explicit generic adapter rather
than fuzzy guessing. The problem is the choice of defaults, not fail-closed
validation itself.

### Exact reproducer inputs

`_documented_raw_fixture` in the new test file writes all five expected files.
Use `B = 1700000000`; all timestamp values below are deliberately synthetic
integer seconds. Delimiter and unit are explicitly supplied to eliminate
sniffing and timestamp-unit inference as alternate causes.

Template `submission.csv`:

```csv
user_id,tv_show_id
0007,ignored
9223372036854775807,0
9007199254740993,NaN
0000,
```

Logs `dataset11-30week.csv`:

```csv
user_id,vsetv_id,start_time,end_time,duration,tv_show_id
0007,0007,1700000000,1700000100,100000,999
9223372036854775807,7.0,1700000000,1700000100,100,999
9007199254740993.0,7,1700000000,1700000100,100,999
0007,7,1700000100,1700000200,100,999
8,7,1700000100,1700000200,100,999
```

Each EPG file has this exact header:

```csv
channel_id,tv_show_id,start_time,duration,tv_show_title,year,tv_show_genre_1
```

Its single row is:

| File | channel_id | tv_show_id | start_time | duration | tv_show_title | year | tv_show_genre_1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `export_arh_10-19.csv` | `7` | `9007199254740993` | `1700000000` | `100` | History | NaN | None |
| `export_arh_20-30.csv` | `7` | `9223372036854775807` | `1700000100` | `100` | Forecast | NULL | None |
| `export_arh_31-42.csv` | `7` | `77` | `1700000200` | `100` | After horizon | 2026 | None |

The log/template program-ID fields are decoys, not relevance inputs. The numeric
raw ID `999` must never enter the schedule catalog or labels.

### Expected versus actual

1. `load_raw(directory, delimiter=",", timestamp_unit="seconds")` should accept
   those documented names. Instead,
   `test_documented_raw_csv_headers_work_with_default_schema` raises:

   ```text
   export_arh_10-19.csv: export_arh_10-19.csv: missing exact columns ['start']; observed ['channel_id', 'duration', 'start_time', 'tv_show_genre_1', 'tv_show_id', 'tv_show_title', 'year']. Set RawSchema to the inspected source names; no fuzzy renaming is used.
   ```

2. Setting **only** `RawSchema(schedule_start="start_time")` isolates the second
   default mismatch. Instead of accepting the documented log endpoints,
   `test_documented_log_headers_work_when_only_epg_adapter_is_configured` raises:

   ```text
   dataset11-30week.csv: dataset11-30week.csv: missing exact columns ['end', 'start']; observed ['duration', 'end_time', 'start_time', 'tv_show_id', 'user_id', 'vsetv_id']. Set RawSchema to the inspected source names; no fuzzy renaming is used.
   ```

3. **Positive control, observed passing:**

   ```python
   schema = RawSchema(
       log_start="start_time", log_end="end_time", schedule_start="start_time"
   )
   prepared = load_raw(directory, schema=schema, delimiter=",", timestamp_unit="seconds")
   data, candidates, labels, audit = build_fold(
       prepared, 1700000100, 1700000100, 1700000200
   )
   ```

   The expected and actual results agree:
   - Template order remains `0007`, `9223372036854775807`,
     `9007199254740993`, `0000`.
   - Matrix users are `0000`, `0007`, `9007199254740993`,
     `9223372036854775807` (sorted **strings**).
   - Raw matrix item IDs are `[9007199254740993, 9223372036854775807]`.
   - Count rows are `[[0,0],[1,0],[1,0],[1,0]]`; the forecast candidate is
     `[9223372036854775807]`.
   - Only `0007` has a cohort label: `[9223372036854775807]`; the other three
     labels are empty. Held-out-only user `8` does not expand the matrix cohort.
   - Years/genre null tokens become `None`; raw string/int64 identity is exact.

**Cause:** `RawSchema.log_start`, `log_end`, and `schedule_start` default to
`start`, `end`, and `start` (`data.py`, lines 88–94 in the reviewed version).
Required-column validation correctly checks those configured names before rows
are processed. The EPG error appears first because schedules are loaded first.

**Minimum fix requested from the data author:** use the documented defaults
`start_time`, `end_time`, and `start_time`, or make a named documented-source
adapter the default entry-point configuration. Preserve explicit alternate
adapters and exact-name fail-closed validation. Update the module/data-contract
documentation to distinguish documented names from names verified on actual
bytes. Do not set `schema_verified_on_actual_source=True` on this evidence.
The verifier did not apply this production fix. Independent recheck after the
author's response is pending.

## Semantic challenges that passed

| Challenge / source requirement | Independent expected result and observed confirmation | Status |
| --- | --- | --- |
| All held-out behavior mutated or removed; training must be isolated | Users `0007`, `9007199254740993`, `template-only`; item columns `101,202,303,404,505`; rows `[[2,0,0,0,0],[0,1,0,0,0],[0,0,0,0,0]]`. Replacing **all** held-out events changes labels but leaves CSR arrays, raw mappings, candidates, metadata and training provenance unchanged. Removing all held-out events also preserves that state. | Passed |
| Raw future seconds in a log spanning the cutoff | Original `[0,200)` becomes `[0,100)` and all forecast observations move to other users. The ended `[0,100)` airing remains one training count; future-only/unreachable users do not enter its row mapping. | Passed |
| Metadata/candidate nodes without any behavior | With zero events, historical item `90` and cold forecast items `2,10` exist in the matrix catalog; candidates are `2,10`; a schedule starting exactly at the horizon end is absent. | Passed |
| Unreachable and held-out-only users | Neither an unscheduled-channel viewer nor an otherwise new held-out viewer expands training users; adding them explicitly to the template creates zero-count rows and allows that template user's forecast labels. | Passed |
| Exact window boundaries | Airing end equal to train cutoff is training; end equal to forecast end is label-eligible. For bounds `(15,25,45)` over consecutive ten-second airings, only raw `10` is training, only `40` is a label, `40,50` are candidates, and three crossing events are excluded. | Passed |
| >=80% watched before cutoff, airing not yet ended | `[0,80)` qualifies 80% of scheduled `[0,100)`, but the retained airing end `100` prevents training at cutoff `80`; the denominator is not shortened. | Passed |
| Fragment union per broadcast, not lifetime | One user's four repeat airings have exactly `80,70,80,40` watched seconds: only the first/third qualify. A separate user with `60+60` seconds in two airings gets **zero** events, not one combined completion. Duplicate viewing fragments do not inflate union coverage. | Passed |
| Exact threshold, no rounding | A 101-second broadcast needs 81 seconds. 79/80 seconds fail; 81 passes. | Passed |
| Duplicate EPG metadata versus broadcast identity | Reordered `A,A,Z` metadata rows for the same airing produce one training event, no false overlap conflict and deterministic canonical `A` metadata. A later airing is a separate held-out event. | Passed |
| Channel identity | Same item/timestamps on two different channels yield two distinct broadcast completions, not a duplicate or same-channel conflict. | Passed under the declared identity policy |
| Conservative conflict quarantine | Two transitive overlap components quarantine five airings, suppress labels/counts, retain known program/candidate nodes, and report affected logs/users. Detailed policy distinction below. | Passed under the declared quarantine policy |
| Frequency labels, numeric ties, top five | Forecast counts `3:2`, `100:2`, and seven one-count items produce `[3,100,2,4,5]`. A single long program watched for 80 seconds contributes **one broadcast**, not eight ten-second votes. | Passed |
| String and 64-bit identity through CSV -> events -> CSR/labels | Leading-zero user `0007`, user/program IDs above 2^53 and signed-int64 maximum stay distinct; decimal-style integral IDs normalize without binary-float rounding. | Passed |
| Null/nonintegral year metadata and raw ID exclusions | Empty/NaN/NULL/N/A/None years become `None`; `2026`, `2026.0`, and `unknown` remain metadata strings. Null/zero/negative program rows are dropped and counted without virtual zero programs. | Passed |
| Schema integrity / explicit failure | Missing each required documented log role and a wrong-header empty EPG file raise schema errors, not silently empty folds. | Passed |

## Intentional fail-closed behavior versus unresolved source policies

### Conservative quarantine is not reported as a counting bug

The exact same-channel schedules in the challenge are:

```text
item 11 [0,100), 22 [90,110), 33 [105,200), 44 [200,300),
item 55 [300,400), 66 [390,410), 77 [410,510)
```

User `u` watches `[0,80)`, `[200,280)`, `[300,380)`, `[410,490)`;
user `unreachable` watches `[90,110)`. Bounds are `(300,300,510)`.
Manual outcomes under the **declared full-component policy** are:

- First component: `11,22,33`; second: `55,66`.
- Only raw `44` is a training count and `77` a label. The otherwise sufficient
  80-second spans for `11` and `55` are lost even though those watched spans do
  not intersect the ambiguous overlap itself.
- Candidates remain `[55,66,77]`; all three are training-cold. The known catalog
  retains all seven IDs. The unreachable non-template viewer has no matrix row.
- Audits report two conflict components, five quarantined unique airings,
  three affected log rows and two affected users.

All outcomes were observed. This quantifies conservative information loss and
guards against accidental behavior-dependent candidate removal. It does **not**
establish that the organizer uses full-component quarantine. Real-data overlap
prevalence and the resulting relevance loss cannot yet be measured; disclose
this assumption and inspect actual overlap diagnostics before claiming fidelity.

### Other deliberate guards, not demonstrated defects

- Program IDs above signed-int64 maximum, fractional IDs and scientific-notation
  ID strings raise `DataSchemaError` with file context instead of silently
  casting. Pre-coerced binary floats, NaN and bool IDs at the normalized API
  boundary raise `ValueError`. CSV null tokens, by contrast, are dropped and
  audited as missing IDs. These are separate, intentionally tested paths.
- Raw log duration `100000` with 100 elapsed endpoint seconds is audited as a
  mismatch; it does not rescale, repair or override the viewed interval.
- Fully-contained forecast labels, exclusion of unfinished training broadcasts,
  earliest-schedule metadata selection and the numeric top-five frequency tie
  break are deterministic engineering choices. Hidden organizer choices are
  still unresolved. A deterministic passing fixture is not source confirmation.
- `build_fold` has an upper training cutoff, not a configurable lower training
  bound. These tests do not certify arbitrary sliding lower-bound windows;
  callers must provide an appropriately bounded source when such a window is
  required. Actual week/date semantics remain unverified.
- This review did not stress-test scale, model fitting, malformed externally
  forged `PreparedData.events` beyond the stated normalized interface, or every
  encoding/time-zone case. It did not repeat the author's entire data suite.

## Executed checks and handoff

Environment observed: Python **3.13.6**, NumPy **2.5.4**, SciPy **1.18.1**,
pytest **9.2.0**. From `midterm/`, the test commands were:

```sh
.venv/bin/python -m pytest -q tests/test_data_adversarial.py -p no:cacheprovider
.venv/bin/python -m pytest -q tests/test_shared.py -p no:cacheprovider
```

The actual invocations also set `PYTHONDONTWRITEBYTECODE=1` and used separate
new `--basetemp` directories under the approved workspace runtime temporary
directory. Temporary fixtures were removed after execution. The suites ran
independently in parallel:

```text
tests/test_data_adversarial.py: 2 failed, 39 passed in 0.20s (exit 1)
tests/test_shared.py:           120 passed in 0.35s (exit 0)
```

No failure was hidden with a skip, xfail, changed prior expectation or production
edit. The two default-schema tests remain failing regression fixtures for the
data author's response. Requested handoff: address D1, retain explicit source
uncertainty and the passing semantic invariants, then rerun both commands. This
review does not yet claim that an author response or independent post-fix
recheck occurred.
