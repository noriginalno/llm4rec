# Independent official-source profile

Generated: 2026-11-01T00:25:32.101468+03:00. Input identity is attributed to the existing acquisition manifest.

## Scope and interpretation

Full source scans used DuckDB with four threads, a 4GB engine memory limit, and disposable spill/database files under project `.cache/`. No production parser, viewing reconstruction, model training, authentication, or download was used.

**Times below are timezone-unknown source wall clocks, not UTC.** Logs use `start_time`, `stop_time`, and `duraton` (source typo); EPG starts use `%d.%m.%Y %H:%M:%S`, with whole-second duration diagnostics. No timezone offset was inferred or applied.

All raw columns were VARCHAR and every CSV completed strict UTF-8 decoding. Users remain exact strings, including values above signed-int64 maximum. Existing manifest SHA-256 values are copied with attribution; CSV bytes were **not rehashed**. Byte sizes and before/after stat stability were checked.

## Viewing logs and template

- Logs: **5,821,987 rows**, **4,914 distinct nonmissing users**, **202 channels**.
- Wall-clock starts: `2020-03-09 00:00:02` through `2020-07-26 23:58:42`; maximum stop: `2020-07-26 23:59:58`.
- Invalid intervals: **0**; missing users/channels: **0/0**.
- Raw `duraton` negative/zero: **0/0**; endpoint mismatch rows: **117**; invalid/nonintegral duration rows: **0**.
- Template: **1,260 rows**, **0 duplicate IDs**, **1,260 distinct users present in logs**, **0 absent**.
- Distinct user IDs above 2^63-1 and within uint64: logs **2,483**, template **655**. Never cast these identifiers to signed int64 or binary floats.

## EPG files

Positive IDs are exact integral values >0. Missing means blank or an explicitly listed null token. Duplicate/overlap counts below concern valid positive-ID airing intervals, not zero placeholders.

| Source | Rows | Positive / zero / missing ID rows | Positive IDs | Channels | First start / last end (wall clock) | Extra positive duplicate rows | Positive conflicted unique airings |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| `export_arh_11-20-final.csv` | 1,074,235 | 115,613 / 958,622 / 0 | 5,847 | 400 | 2020-03-08 23:00:00 / 2020-05-18 12:00:00 | 0 | 0 |
| `export_arh_21-30-final.csv` | 1,074,394 | 121,505 / 952,889 / 0 | 6,363 | 403 | 2020-05-18 00:00:00 / 2020-07-27 12:00:00 | 0 | 0 |
| `export_arh_31-42-final.csv` | 1,295,378 | 144,774 / 1,150,604 / 0 | 6,636 | 404 | 2020-07-27 00:00:00 / 2020-10-19 12:00:00 | 0 | 8 |

EPG duration/interval exclusions, in source order:

- `export_arh_11-20-final.csv`: invalid/nonintegral duration **0**, zero **1**, negative **0**; unparsed starts **0**; invalid positive-ID airings **0**.
- `export_arh_21-30-final.csv`: invalid/nonintegral duration **0**, zero **0**, negative **0**; unparsed starts **0**; invalid positive-ID airings **0**.
- `export_arh_31-42-final.csv`: invalid/nonintegral duration **0**, zero **0**, negative **0**; unparsed starts **0**; invalid positive-ID airings **0**.

Per-file malformed/nonintegral durations, nonpositive durations, parsed timestamp failures, cross-midnight counts, and source schemas are recorded in the JSON. The actual file ranges, not legacy logical filenames, must drive protocol reconciliation.

## Catalog and metadata

- Historical EPG program IDs: **8,234**; future IDs: **6,636**; future IDs already present in historical schedules: **5,199**; future-only IDs: **1,437**. These are schedule-presence counts, not viewing-derived seen/cold users.

Availability means at least one nonmissing metadata value for a positive ID within that source. JSON also records whether it is present on every row for that ID.

| Source | Positive IDs | Any title | Any category | Any genre slot |
| --- | ---: | ---: | ---: | ---: |
| `export_arh_11-20-final.csv` | 5,847 | 5,847 | 5,847 | 5,720 |
| `export_arh_21-30-final.csv` | 6,363 | 6,363 | 6,363 | 6,227 |
| `export_arh_31-42-final.csv` | 6,636 | 6,636 | 6,636 | 6,486 |

Across all positive IDs, **4,351** have multiple exact nonmissing titles and **0** multiple categories. Display-title variants do not establish episode-level identity; do not replace raw-ID aggregation with guessed episode identities.

- ID `10002000`: 1209 exact title variants, 1 category variants. Examples: ["Т/с 'След', 1-4 с.", "Т/с 'След'. '12 месяцев зимы'."].
- ID `12001682`: 936 exact title variants, 1 category variants. Examples: ["М/с 'Губка Боб Квадратные Штаны'.", "М/с 'Губка Боб Квадратные Штаны', 183 с. 'Морской ёж / Защита Сквидварда'."].
- ID `12001712`: 761 exact title variants, 1 category variants. Examples: ["М/с 'Симпсоны'.", "М/с 'Симпсоны', 13 сезон, 15 с. 'Во всем виновата Лиза'."].
- ID `5800327`: 650 exact title variants, 1 category variants. Examples: ["'Гадалка'. '9 дней до смерти'.", "'Гадалка'. 'Аромат из прошлого'."].
- ID `10001336`: 527 exact title variants, 1 category variants. Examples: ["Т/с 'Кости', 5 сезон, 13 с. 'Дантист в траншее'.", "Т/с 'Кости', 5 сезон, 3 с. 'Простое в потрясающем'."].

Year metadata is missing on **3,473** positive-ID rows and has non-digit text on **169,713** rows (examples include open-ended/range years in JSON). This field is not safely modeled by a blanket integer cast.

## Duplicates and overlap diagnostics

Across all EPG files after positive-ID filtering: **0** extra exact identity rows; **4** positive-overlap components containing **8** distinct airings.
Exact identity is `(channel_id,start,end,program_id)`, excluding metadata. Exact duplicates are removed before the running-maximum interval sweep; adjacency is not overlap. Components include containment and transitive chains. These are **not pairwise overlap counts**, and the profile does not quarantine or relabel anything.
Including zero-ID rows would additionally place **1** otherwise nonconflicting positive airings into conflict components. Zero placeholders must be excluded before computing positive-program conflict policy. Positive-only conflicts above remain and require a documented decision.

Bounded per-file/global component examples, duplicate multiplicities, dates and raw IDs are in the JSON. Cross-midnight/duplicate-source entries can contribute; observed interval overlap alone does not identify the organizer's intended reconstruction.

## Channel/time availability and policy implications

- Log rows without any matching raw-string EPG channel: **113,579** across **5** channels; log rows lacking a positive-ID EPG channel: **1,103,935**.
- Valid log intervals overlapping a positive-ID channel's outer schedule envelope: **4,175,018**. This is an envelope check, **not actual airing coverage**, and assumes only a common wall-clock coordinate for the comparison.
- Use the actual `stop_time`/`duraton` names and day-first EPG parsing. Preserve nonnumeric/range year metadata as text. Reconcile the observed source date ranges before choosing split cutoffs.
- A shared timezone/UTC offset remains **unknown**. The source has no zone; overlap/envelope statistics cannot justify converting either source to UTC.
- No production parsing policy was changed. Counts are source diagnostics, not recommendation quality, Kaggle scores, qualifying viewing events, or personal verification.

## Reproduction and evidence

```sh
.venv/bin/python -B tools/profile_official.py
```

Observed total elapsed: **10.147 seconds**; DuckDB **1.5.6**; **101** timed SQL statements. This is one profiling run, not a model/runtime feasibility benchmark.
`results/official-independent-profile.json` contains the executed SQL/parameters/timings, complete aggregates, exact bounded examples, raw schemas, declared hashes and their verification limits. Output files must be new; reruns require explicitly different output paths.

## Bounded anomaly examples

The **5 stored log mismatch examples** have `(stop-start)-duraton` differences of `[3600]` seconds. This does not describe all mismatches or establish a source timezone/DST rule; do not automatically shift or repair timestamps.

| Channel | Component wall-clock span | Positive program IDs |
| --- | --- | --- |
| `1060` | 2020-10-04 22:20:00 / 2020-10-05 01:50:00 | 90081561, 90081851 |
| `1117` | 2020-10-05 05:00:00 / 2020-10-05 07:50:00 | 90079423, 90059058 |
| `1520` | 2020-10-05 03:55:00 / 2020-10-05 07:50:00 | 9003571, 90039050 |
| `1521` | 2020-10-05 05:25:00 / 2020-10-05 09:30:00 | 90072474, 90059006 |

All stored component examples start after the observed viewing-log end. They are source overlap diagnostics, not reconstructed user events.
