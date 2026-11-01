# Data adapter and temporal reconstruction contract

## Evidence status

Official competition content documents log endpoints `start_time`/`end_time`
and EPG `start_time` plus duration in seconds. The current author-response request
confirms those names had been verified in that content before implementation;
this response did not retrieve the source again. `RawSchema` now follows those
documented names. The initial generic `start`/`end` defaults were a confirmed
documentation/interface mismatch, not a justified source uncertainty; see
`data-author-response.md` and the unchanged independent challenge.

Actual competition CSV contents remain unavailable. Matching documented names
does **not** verify actual CSV header bytes/version, encodings, timestamp
units/timezones, row counts, date/week semantics or log duration units. The
implementation must not be described as verified on real SWEET.TV data, and
`schema_verified_on_actual_source` remains false. Unknown headers fail with the
observed names; callers must configure exact alternate names. No fuzzy aliases,
date/week inference, downloads, authentication or raw-file changes are performed.

| Requirement | Implementation / verification |
| --- | --- |
| Preserve 64-bit user identity | CSV strings and exact decimal parsing; no binary-float conversion |
| Known channel relation | Explicit log `vsetv_id` -> normalized `channel_id`; EPG `channel_id` |
| Documented default interface | Log `start_time,end_time`; EPG `start_time,duration`; explicit legacy adapter remains supported |
| Whole-broadcast >=80% | Half-open overlap; union fragments; integer `5*covered >= 4*duration` |
| Repeat broadcasts count separately | Identity `(channel,start,end,tv_show_id)`, one event/user/airing |
| Leakage-free folds | Train `end <= train_end`; labels fully inside forecast window |
| No label-based candidate filter | All positive known schedule IDs starting in forecast window |
| Sparse counts | CSR; users/items have separate sorted raw-ID mappings |
| Known unknowns and exclusions | Parsing, duration, duplicate, conflict, cutoff and fold audits |

## Explicit parsing policies

Expected filenames are `dataset11-30week.csv`, `export_arh_10-19.csv`,
`export_arh_20-30.csv`, `export_arh_31-42.csv`, and `submission.csv`.
Default exact columns are logs `user_id,vsetv_id,start_time,end_time` plus optional
`duration`; EPG `channel_id,tv_show_id,start_time,duration`; template `user_id`.
No EPG end column is documented, so `schedule_end=None` by default and the end
is derived from start plus duration. A source with an inspected additional end
column can explicitly configure, for example, `RawSchema(schedule_end="end_time")`.
Pass `RawSchema(...)` for different exact source names; legacy headers remain
supported without fuzzy fallback:

```python
legacy = RawSchema(
    log_start="start", log_end="end", schedule_start="start", schedule_end="end"
)
prepared = load_raw(raw_dir, schema=legacy)
```

EPG duration is positive integral seconds. Explicitly configured EPG end disagreement drops
and counts the row; no correction is invented. Log end is required: unknown log
duration units cannot justify reconstructing it. Unused EPG fields become
metadata without fuzzy renaming; no viewing statistics enter metadata.

Only header BOM and outer whitespace are normalized. Delimiters use
`csv.Sniffer` over at most 65,536 decoded characters, restricted to comma,
semicolon, tab and pipe; failure requires an explicit delimiter. UTF-8 is strict
by default. CP1251 is allowed only by explicit argument, recorded per file; no
invalid character replacement or implicit encoding fallback occurs. Delimiter,
encoding and UNIX unit may be configured per exact filename. Each file audit
records its SHA-256, byte count, row count and original/normalized columns.
Single-column CSVs may require an explicit delimiter when Sniffer cannot infer
one. Required headers are validated even for a zero-data-row CSV. Per-file
timestamp-format counts and normalized interval ranges, plus per-role ID-format
and canonicalization counts, make the parsing choices inspectable.

Null tokens are explicitly enumerated in the module/audit. IDs accept only
integral decimal input; decimal-style IDs normalize through `Decimal`, never
`float`. Plain digit user IDs retain leading zeros. Channels and program IDs
use canonical integer spelling. Binary floats are rejected; upstream CSVs that
have already rounded identifiers cannot be detected/repaired reliably. Zero,
missing and negative program IDs are dropped and counted; null channels cannot
match. Program IDs must fit the shared signed-int64 model interface.

ISO timestamps require an explicit UTC offset, or the caller must supply an
IANA `assumed_timezone` (including an explicitly chosen `"UTC"`). That choice
is audited. DST gaps and folds in local datetimes are rejected, not guessed.
Automatic numeric scale inference considers seconds and milliseconds only,
and succeeds only when exactly one falls in [2000-01-01,2100-01-01) UTC. This
range is an engineering guard, not an observed dataset date range. Other epoch
values require explicit `timestamp_unit="seconds"` or `"milliseconds"`.
Subsecond precision is rejected rather than rounded/truncated. Already normalized
`reconstruct_events` inputs use integer UTC epoch seconds directly, even near 0.

## Reconstruction and conflict policies

Broadcast identity excludes metadata. Identical identity rows are deduplicated;
metadata variants choose the lexicographically smallest canonical JSON row and
airing-ID variants choose the smallest string, independent of behavior/order.
IDs generated by ingestion are SHA-256 hashes of the broadcast identity.
Different channels are independent. Distinct identities with positive-length
same-channel overlap create connected conflict components; **all** their
airings are quarantined from events, even if a viewer only watched an apparently
unambiguous subinterval. Adjacent airings do not conflict. Audits state raw
affected schedule rows, unique airings/components, and affected valid log
rows/users after optional cutoff clipping. This is conservative loss, not a
claim of matching the organizer's hidden reconstruction. Known quarantined
schedules stay flagged in `PreparedData.schedules`, and remain available as
nonbehavioral candidate/metadata evidence.

Invalid endpoint intervals are dropped/countable. Raw numeric log duration is
compared to endpoint elapsed seconds under an explicitly unverified seconds
hypothesis; missing, invalid and mismatched values are counted. It never rescales,
repairs, or overrides intervals. Per-user/channel viewing intervals are unioned
first, then intersected with nonconflicting sorted airings. Different airings
never share a coverage denominator. Every qualifying event retains the original
scheduled start/end, not the observed fragment limits.

`cutoff` clips only viewing to `[start,min(end,cutoff))`, retaining full scheduled
duration. A crossing airing can qualify if >=80% was seen before cutoff; its
future end is still marked, and `build_fold` excludes it from training. Because
every training airing ends before/equal to the boundary, building folds from all
reconstructed events is equivalent for training coverage: future seconds cannot
overlap an ended airing. This avoids using future viewing to complete training
events. Forecast labels require entire scheduled airings inside the window.

## Fold policies and interface

`build_fold(prepared,train_end,forecast_start,forecast_end)` requires
`train_end <= forecast_start < forecast_end`; returns `TrainingData`, a NumPy
array of **candidate matrix indices**, a user->list-of-raw-program-IDs label
dictionary, and an audit. Matrix users/label keys are all training users plus
all template users, sorted as strings; held-out-only non-template users do not
enlarge this cohort and are counted separately. Empty template labels remain.
The original template list preserves submission order.

The catalog is all historical/forecast schedule IDs with airing start strictly
before `forecast_end`; later schedules never enter it. Candidates have starts
in `[forecast_start,forecast_end)`, including crossing airings, watched IDs and
cold items. Labels additionally require `end <= forecast_end`. Catalog IDs are
sorted numerically. One canonical metadata row per ID is the earliest known
airing, then end/channel/airing-ID/metadata JSON tie-breaks; neither popularity
nor held-out viewing selects metadata. Counts are qualifying broadcast
frequencies, not binary preferences or seconds. Top-five labels rank frequency
descending then raw numeric ID ascending. Frequency ties, top-five boundary
ties, short/empty labels, train/held-out counts, cold candidates, boundaries and
excluded users are audited. Training provenance contains boundaries and only
training behavior counts, never held-out label statistics.

## Complexity and limits

For L valid logs, S input schedule rows, E qualifying events and K actual overlaps,
preparation/reconstruction uses O(S log S + L log L + L log S + K + E log E)
worst-case time (sorting within groups is bounded by L log L). Memory is
O(L+S+E), including interval groups and event lists, plus viewer/group dictionaries.
Strict text decoding, parsing and input SHA-256 hashing also cost O(raw bytes).
Each lookup uses per-channel binary search, never an L*S Cartesian join.
Conflict interval lookups have the same logarithmic form. Repeated broadcasts
and users still generate real output-sized work. Very large data may need an
out-of-core adapter; no data-size-based feasibility is asserted before files
are inspected. Folds use O(S log S + E + P log P + U log U + I log I) time plus
per-user held-out ranking, where P is nonzero user/program pairs, and sparse
O(S+E+P+U+I) memory. The shared interface coerces counts to float32, so counts
above 2^24 are explicitly rejected rather than silently losing integer precision.

Checks are synthetic, including an independent discrete-time interval oracle;
see `tests/test_data.py`. Real-source reconciliation remains pending.

## Observed development regressions

The first executed module check produced 32 passes and three failures:

1. Metadata-variant duplicate counts depended on row ordering. Counting variants
   against the final canonical metadata, rather than the current streaming
   choice, fixes the permutation regression.
2. Decimal division used the default precision context and rounded an explicitly
   scaled 41-digit epoch. Exact integer `divmod` after an integral-value check
   preserves its value; automatic scale comparison also avoids division.
3. Required headers were checked only on data rows, allowing an empty EPG CSV
   with unrelated headers. Header validation now runs immediately after reading
   the header, even when no data rows follow.

The same suite then passed all 35 cases. Additional tests cover subsequent
edge-case hardening, including ISO fractional digits beyond Python datetime's
microsecond precision; these digits are checked before parsing so that they
cannot be silently truncated. These are author-executed synthetic checks, not
personal verification, real-data results, or evidence of organizer equivalence.

## Initial author-executed checks (2026-10-31)

- `.venv/bin/python -m pytest tests/test_data.py -q`: **44 passed in 0.17s**.
  Environment: Python 3.13, NumPy 2.5.4, SciPy 1.18.1, pytest 9.2.0.
- A bounded in-memory synthetic run with 20,000 adjacent airings on 10 channels
  and 60,000 log rows produced 60,000 events with no EPG conflicts. Reconstruction
  took 0.2640 seconds; fold construction took 0.2613 seconds in this single run,
  excluding input generation. The fold had 30,000 training and 30,000 held-out
  events, four matrix users, 137 candidates, shape (4,137), and 411 nonzeros.
  Assertions checked event counts and CSR count sum (30,000). These measurements
  are nondeterministic synthetic workload evidence, not real-data feasibility
  estimates or organizer-quality results.

## Independent challenge and author recheck (2026-10-31)

The independent challenge D1 correctly identified that documented `start_time` /
`end_time` names were incompatible with the initial generic defaults. The author
reproduced both reported failures before editing: the full adversarial suite had
**2 failed, 39 passed in 0.21s**, while the original author suite had **44 passed
in 0.17s**. That author suite used the same generic headers as the implementation
and therefore did not test the documented default interface.

Defaults now match the documented endpoints, `schedule_end=None` reflects the
documented duration-only EPG, and literal documented-header author fixtures are
separate from an explicit legacy-header adapter regression. No fuzzy renaming,
timestamp assumptions or reconstruction changes were introduced. Author reruns:

- `tests/test_data.py`: **45 passed in 0.19s**.
- Unchanged `tests/test_data_adversarial.py`: **41 passed in 0.19s**.
- Unchanged `tests/test_shared.py`: **120 passed in 0.37s**.

See `data-author-response.md` for commands, before/after hashes, accepted root
cause and remaining actual-data unknowns. These are author-executed post-fix
checks, not a new independent reviewer acceptance or real-source verification.

## Verified listing and explicit filename aliases (2026-10-31)

The parent session's authenticated official CLI listing for
`sweettv-tv-program-recommender` establishes different actual filenames from the
earlier content description. The earlier parsing section's expected filenames
are now explicitly **library logical defaults**, retained for compatibility,
not a claim about the current listing:

| Logical default | Actual listed basename | Listed bytes |
| --- | --- | ---: |
| `dataset11-30week.csv` | `dataset11-30.csv` | 396216555 |
| `export_arh_10-19.csv` | `export_arh_11-20-final.csv` | 153322643 |
| `export_arh_20-30.csv` | `export_arh_21-30-final.csv` | 154366907 |
| `export_arh_31-42.csv` | `export_arh_31-42-final.csv` | 185952991 |
| `submission.csv` | `submission.csv` | 38348 |

These listing facts were supplied in the current handoff; this author did not
authenticate or run the CLI. Downloads remain reported as 403 pending the user's
rule acceptance. Actual CSV contents/headers, encoding, timestamp units/timezone,
date/week semantics and row counts remain unknown. File sizes do not establish
any of those properties or source hashes.

The additive `load_raw(..., file_names: Mapping[str,str] | None = None)` keyword
accepts an explicit logical-to-actual basename map. Omitted roles keep their
defaults. Unknown keys, invalid/nonrelative basenames and duplicate effective
names fail before file access. There is no automatic discovery, renaming,
copying or alias fallback. Missing and parsing errors identify actual source
names. Existing per-file parsing-option mappings stay keyed by **logical names**
even with aliases; this clarifies the earlier "per exact filename" wording.

Audits retain logical file keys, record all effective bindings in `file_names`,
and add per-file `logical_filename`/actual `source_filename`. SHA-256 and byte
counts come from actual opened bytes. No actual-source verification flag or
schema/time-unit assumption changed. Author checks: **74 author cases passed in
0.21s** and **41 unchanged adversarial cases passed in 0.19s**. Configuration,
source-preservation tests, commands, hashes and remaining limits are documented
in `data-filename-adapter.md`; parent CLI/configuration/download tools were not
edited.

## Downloaded-source samples and explicit datetime formats (2026-11-01)

This update supersedes the earlier historical unavailable-download status: the
parent reports successful official CLI downloads, preserved raw bytes and hashes
in `data/raw/source-manifest.json`. This author read only the bounded
`results/official-source-sample.json`; full ingestion and actual-source event
reconstruction were **not** run in this stage.

Observed log headers are `user_id,vsetv_id,start_time,stop_time,duraton` (source
typo retained). The sampled logs are naive ISO strings and their raw duration
values match endpoint differences of 111, 204 and 81 seconds. Sampled users
`17226860011138219284` and `18417531283109304442` exceed signed-int64 range and
remain exact strings. EPG examples use `%d.%m.%Y %H:%M:%S` plus seconds duration;
first sampled dates are 2020-03-09, 2020-05-18 and 2020-07-27 (calendar ISO weeks
11, 21 and 31). These are bounded observations, not full date-range coverage.

New optional APIs are `parse_timestamp(..., timestamp_format: str | None = None)`
and `load_raw(..., timestamp_formats: str | Mapping[str,str] | None = None)`.
Only an explicit format enables strptime; format mismatches never fall back to
ISO/UNIX parsing. Global formats are supported, but mixed logs/EPG should use
**logical filename** keys for EPG patterns and omit logs to retain ISO parsing.
Unknown format-map keys fail closed. Library schema defaults remain unchanged;
the actual log export requires the explicit
`RawSchema(log_end="stop_time",log_duration="duraton")` adapter.

Both parsers share exact integer-second normalization and strict DST validation.
**Recorded timezone is still unknown.** The parent will explicitly choose
`assumed_timezone="UTC"` as a common naive-clock coordinate anchor; neither the
library nor these checks claim recorded UTC or infer a Kyiv/DST offset. Audits
record the configured pattern, caller-supplied basis, successful explicit-format
counts, raw typo and clock-anchor provenance separately. The full-source schema
verification flag remains false; source-wide consistency is still pending audit.

Author-executed checks: **101 author cases passed in 0.28s**, **41 unchanged
adversarial cases passed in 0.28s**, and a separate bounded sample probe parsed
3 log rows / 9 EPG start values with exact ID/duration/date checks. No full
ingestion, actual event reconstruction, authentication, downloads or unrelated
edits occurred. See `data-real-format-adapter.md` for API examples, audit keys,
hashes, executed evidence and remaining full-source limitations.
