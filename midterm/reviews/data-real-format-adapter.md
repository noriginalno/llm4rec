# Explicit adapter for observed source datetime formats

Date: 2026-11-01. The additive datetime-format API is implemented and checked
against synthetic fixtures and the existing bounded source-sample JSON. **Full
ingestion and actual-source event reconstruction were not run in this stage.**

## Updated source evidence and limits

The parent reports that official CLI downloads succeeded, original CSV bytes
were preserved, and hashes were recorded in `data/raw/source-manifest.json`.
This supersedes the earlier unavailable-download status. This author did not
authenticate, download, rehash the full raw files, or modify that manifest.

Read `results/official-source-sample.json`, which explicitly records
`bounded_sample_only=true` and `reconstruction_run=false`. Its observed headers
and examples establish the following bounded facts:

- Log columns are exactly `user_id,vsetv_id,start_time,stop_time,duraton`; the
  misspelling `duraton` is present in source, not a normalization choice.
- The first three log examples use naive ISO datetime strings. Endpoint elapsed
  values equal their raw `duraton` values: **111, 204, 81 seconds**.
- Two sampled user IDs, `17226860011138219284` and `18417531283109304442`, exceed
  signed-int64 range and must remain exact strings. Program/matrix ID constraints
  are separate and were not changed.
- EPG start examples use `%d.%m.%Y %H:%M:%S`; duration is seconds. The first
  sampled clocks are `09.03.2020 06:00:00`, `18.05.2020 06:30:00`, and
  `27.07.2020 06:30:00`, corresponding to calendar ISO weeks 11, 21 and 31.
  These are sample observations, not global date-range/coverage measurements.
- **Recorded timezone remains unknown.** No Kyiv zone, DST regime or offset is
  selected from channel names, date patterns, geography or filenames.

The parent's explicit `assumed_timezone="UTC"` is supported as a shared
naive-clock coordinate anchor. It is **not a claim that the source was recorded
in UTC**. Full-source format consistency, chronology, duration mismatches,
overlap prevalence, memory/runtime feasibility and organizer label equivalence
remain for the parent's full audit and independent source profiling.

## Additive API and usage

```python
parse_timestamp(
    "09.03.2020 06:00:00",
    timestamp_format="%d.%m.%Y %H:%M:%S",
    assumed_timezone="UTC",  # explicit coordinate anchor, not a source fact
)  # 1583733600 under that anchor

prepared = load_raw(
    raw_dir,
    schema=RawSchema(log_end="stop_time", log_duration="duraton"),
    file_names=source_files,  # existing explicit logical-to-actual basename map
    timestamp_formats={
        logical: "%d.%m.%Y %H:%M:%S" for logical in SCHEDULE_FILES
    },
    assumed_timezone="UTC",  # caller-chosen common naive-clock anchor
)
```

- `parse_timestamp(..., timestamp_format: str | None = None)` uses
  `datetime.strptime` **only** when a nonempty format is explicitly supplied.
  That format is authoritative: a mismatch raises instead of falling back to
  ISO or numeric UNIX parsing. Numeric objects are not coerced to formatted text.
- `load_raw(..., timestamp_formats: str | Mapping[str,str] | None = None)`
  accepts a global format or exact **logical filename** keys, like the other
  per-file options. Omitted/None roles retain the ISO/UNIX parser; therefore the
  example above leaves logs on their existing ISO path. Actual alias names are
  rejected as unknown format keys rather than guessed. Blank/non-string format
  configurations fail closed.
- A global day-first pattern would govern the ISO logs too and correctly fail
  on them; callers with mixed file formats must use the logical-role mapping.
- Existing `RawSchema` defaults remain `log_end="end_time"` and
  `log_duration="duration"` for compatibility. Actual `stop_time`/`duraton`
  support is explicit, not fuzzy renaming or another silent default change.
- `strptime` calendar and field matching govern an explicitly chosen pattern.
  No date-order guessing occurs. A syntactically matching but semantically wrong
  pattern, such as swapped day/month for an ambiguous date, cannot be corrected
  without source evidence; choosing the pattern remains the caller's decision.

## Shared normalization and audit

Both datetime parsers now call `_normalize_datetime`, which retains the exact
integer-second and explicit-localization rules. Naive input requires a supplied
IANA zone/anchor; DST gaps/folds under a caller-selected nonfixed zone still fail
strictly. Explicit offsets remain authoritative. Fractional seconds or fractional
offsets are rejected rather than rounded, truncated, or canceled into a whole
second. The original ISO text is checked for sub-microsecond precision after ISO
validation; dotted day-first dates never go through that ISO-only fraction check.

The configuration and counting fields are deliberately separate:

| Audit field | Meaning |
| --- | --- |
| `timestamp_format_config` | Copied caller format string/mapping, or None |
| `timestamp_format_policy` | Explicit formats only; no mismatch fallback |
| `files[logical].timestamp_format` | Effective format for that role, or None |
| `files[logical].timestamp_format_basis` | Caller-supplied strptime format versus existing ISO/UNIX path |
| `timestamp_formats` and per-file `timestamp_formats` | Existing successful-parse counters, now including `explicit_format_naive_explicit_zone` / `explicit_format_with_offset`; these are counts, not format configuration |
| `assumed_timezone` / `assumed_timezone_basis` | Caller-chosen anchor and explicit warning that it is not recorded source-timezone evidence |

Only successfully normalized timestamps increment parse counters; excluded
program/ID rows and parse errors do not become successful format observations.
The raw typo remains in `files[LOG_FILE].raw_columns` / `.columns`, and its
explicit semantic binding is retained in `schema.log_duration="duraton"`.
Raw duration auditing still uses endpoint differences without rescaling.
Source hashes still cover opened bytes. `schema_verified_on_actual_source`
remains false; configured parsing and bounded samples do not certify the full
source. Event reconstruction, candidate policies, split boundaries and sparse
count semantics were not changed.

## Executed checks

Environment: Python **3.13.6**, NumPy **2.5.4**, SciPy **1.18.1**, pytest
**9.2.0**. From `midterm/`:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_data.py --tb=short -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_data_adversarial.py --tb=short -p no:cacheprovider
```

- **101 author cases passed in 0.28s**: all previous 74 cases plus 27 new format
  cases, including exact 2020 epochs, zone requirements, DST gaps/folds, wrong
  formats, fractional precision, global/per-file formats through aliases,
  unsigned source user IDs through events/CSR, source-typo audits, unchanged
  defaults and byte-preserved synthetic fixtures.
- **41 unchanged adversarial cases passed in 0.28s**. No independent test was
  edited, skipped or xfailed.
- A separate bounded probe read only `official-source-sample.json`, then parsed
  **3 log rows and 9 EPG timestamps** with the new public helper. It checked the
  three elapsed durations above, exact retention of both unsigned user IDs and
  first sampled EPG seconds/weeks under the explicit UTC anchor:

  | Sample | Anchored seconds | ISO week |
  | --- | ---: | ---: |
  | `09.03.2020 06:00:00` | 1583733600 | 11 |
  | `18.05.2020 06:30:00` | 1589783400 | 21 |
  | `27.07.2020 06:30:00` | 1595831400 | 31 |

  The sample JSON was asserted byte-unchanged. Its SHA-256 is
  `955376e07895eb7a47d939bbf4f25bb35ef70a5087d66b88b3581239e87f9441`
  (**sample artifact**, not a raw CSV hash). No real event count or training
  matrix was produced by this probe.

Recorded implementation hashes:

- `sweettv/data.py`:
  `4a573981a1adbb9fe9df57a443689deabc17b98e488c4dde9cc695c875f84e32`
- `tests/test_data.py`:
  `799b77e3c6b6cdf55d462e0cc6e4f112d72d6200f784bf97c2928ad83c073577`
- Unchanged `tests/test_data_adversarial.py`:
  `d6726efbe399fa74d428b9de890a6741ed1a3b88d10d33c1ade63daddaa2252a`

All reconstructed events/counts in tests come from tiny synthetic schedules;
some log spellings/values were copied from the bounded source examples. Their
matching schedules were deliberately constructed and are **not** observed source
completions. Only `data.py`, its author tests, this new note and an append to
`data-contract.md` were changed. No original CSV, manifest, source sample, CLI,
configuration, authentication, download or other agent work was modified.
