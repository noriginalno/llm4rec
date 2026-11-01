# Independent raw-data recount of two primary example users

Generated: 2026-11-01T00:38:26.638909+03:00.

**Raw/cache agreement: True.** This compares an independent reconstruction with the existing normalized cache, not organizer-private targets or model quality.

## Independence and clock contract

No production parsing/counting/ranking helper was imported. The selection rule was read as a contract from `select_example_users`; the two IDs were independently derived from typed cached events and the raw template. All classification and broadcast counts for those users were then recomputed from actual CSVs.
Both sources are treated as one timezone-unknown naive wall-clock coordinate. Numeric seconds subtract naive 1970-01-01; they are not a claim that source clocks were UTC. Raw `duraton` is audited but never repairs endpoints. Primary training ends at 2020-05-04 00:00:00 (1588550400); labels require whole airings inside [2020-05-04,2020-07-27).

No cache, raw, production, model, existing report or test was edited. The running benchmark's output directory was not read or polled. Temporary DuckDB spill files were confined to the approved runtime temporary directory, not a data/cache directory.

## Training-only user selection

Raw template users: **1260**; active before cutoff: **1239**. Middle activity slice `[309, 929]` contains **620** users. Ordering uses total training broadcast count then the exact user string, followed by min/max top-program share and string ties. Shares are compared as exact fractions.

| Selected user | Training broadcasts (selection cache) | Distinct programs | Max program count / total |
| --- | ---: | ---: | --- |
| `18417531283109304442` | 32 | 30 | 2/32 = 1/16 |
| `5364378838289882469` | 29 | 5 | 24/29 = 24/29 |

Selection depends on the cached training counts for the full template. This audit does not independently reconstruct all users to prove that ordering; its raw-data verification is restricted to the chosen two.

## Independent per-airing reconstruction

Positive raw EPG rows are parsed with `%d.%m.%Y %H:%M:%S` and whole-second duration; zero IDs are excluded first. Exact broadcast identity is `(channel,start,end,program_id)`. Historical scope includes broadcasts crossing the July 27 boundary without truncating their scheduled denominator.
The applicable positive-EPG assertion found **0** overlaps and **0** extra exact rows before assuming nonconflicting airings.
Each raw log is intersected with each applicable same-channel airing. Its clipped start contributes +1 and clipped end -1. Simultaneous endpoints are grouped, active counts are cumulatively summed **within each user/airing**, and disjoint segment lengths count only when active>0. Eligibility uses exact `5*covered >= 4*scheduled_seconds`, once per user/broadcast. No global log-interval union or production sweep is reused.

| User | Raw log rows | All qualifying events raw/cache | Training events raw/cache | Fully contained held-out events raw/cache |
| --- | ---: | --- | --- | --- |
| `18417531283109304442` | 1407 | 154/154 | 32/32 | 122/122 |
| `5364378838289882469` | 1925 | 49/49 | 29/29 | 20/20 |

Full canonical event-set differences: raw-only **0**, cache-only **0**. Cache airing IDs were mapped to cached schedule channels and event/schedule fields checked, rather than reproducing the production ID hash.
Coverage differences above 1e-12: **0**; maximum absolute difference: `0.0`. Exact integer thresholding was used for the raw oracle; tolerance applies only to comparison with cached floating coverage.

## Primary histories and labels

The JSON contains complete raw and cache training/held-out frequency histograms for each selected user. Top-five labels use broadcast frequency descending then numeric raw program ID ascending, not seconds or display titles.

- `18417531283109304442`: training histogram equal **True**; held-out histogram equal **True**; top five `[240081, 10001261, 12002107, 10001336, 10002253]`; cache top-five equal **True**.
- `5364378838289882469`: training histogram equal **True**; held-out histogram equal **True**; top five `[10001336, 10001085, 12002856, 5200303, 12002070]`; cache top-five equal **True**.

## Actual bounded hand-verification candidate

This is evidence prepared for a later human check, **not a claim of user/personal verification**.

User `18417531283109304442`, channel `1519`, raw program ID `90014551`; title: Х/ф 'Джуниор'..
Source EPG: `['export_arh_21-30-final.csv']`; broadcast `2020-07-26 09:55:00` to `2020-07-26 11:45:00`, **6600 seconds**.

| Raw log start | Raw log stop | Raw duraton | Clipped start | Clipped end | Clipped seconds |
| --- | --- | ---: | --- | --- | ---: |
| 2020-07-26 06:08:10 | 2020-07-26 11:18:22 | 18612 | 2020-07-26 09:55:00 | 2020-07-26 11:18:22 | 5002 |
| 2020-07-26 10:44:21 | 2020-07-26 10:49:14 | 293 | 2020-07-26 10:44:21 | 2020-07-26 10:49:14 | 293 |
| 2020-07-26 10:49:14 | 2020-07-26 11:30:23 | 2469 | 2020-07-26 10:49:14 | 2020-07-26 11:30:23 | 2469 |

Union coverage is **5723 / 6600 = 5723/6600** (0.86712121). Exact comparison: `28615 >= 26400`. Endpoint segments/active counts are in JSON so overlaps are not double-counted.

## Checks, timing and limitations

The manual SQL self-check passed (expected coverage 80,79,60,60,80 seconds; only first/last qualify). Actual raw recount used **1666** clipped fragments across **943** matched user-airings, with **0** endpoint-balance errors and **0** impossible coverage totals.
Executed **55** timed SQL statements in **1.612 seconds** overall; DuckDB 1.5.6, four threads / 4GB engine buffer limit. SQL and parameters are recorded in JSON.

```sh
.venv/bin/python -B tools/recount_official_users.py
```

Inputs are anchored by manifests/audit metadata, schemas and before/after size/mtime checks. Existing large-file hashes are attributed, not freshly rehashed. Only the new profiler script and output files are created. Source timezone, hidden organizer fragment/conflict/tie policies and target fidelity remain unknown. Agreement establishes consistency with this declared reconstruction policy for two users, not official quality or a returned Kaggle score.
