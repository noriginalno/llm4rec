# Temporal repeat and schedule-opportunity research

Implementation review, 2026-11-01. Scope is only `research_v2/temporal.py`,
`tests/test_research_temporal.py`, and this review. This author did not edit
scored models, data, shared contracts/helpers, runner, or submission artifacts.
No Kaggle access, credentials, installs, submissions, publication, or subagents.

## Requirements and API

| Requirement / source | Implementation / independent check | Status |
| --- | --- | --- |
| Training-only context; original counts and identifiers unchanged (`PROTOCOL.md`, `sweettv/contracts.py`) | Both fits take only `TemporalContext`; original CSR remains float32; feature CSR is separate float64; exact string user IDs and int64 raw program IDs are copied | Verified by fixtures and one bounded actual-data feature check |
| End-age decay and optional inclusive lookback; no future events (task, `research_v2/contracts.py`) | Shared `temporal_matrix` and unchanged `temporal_weights`; fixture weights 1, 1/2, 1/4 and one-second-outside boundary | Verified |
| Same schedule/event weighting, explicit smoothing units, bounded adjustment (task) | Historical weights use the same helper, forecast counts use fully contained context metadata; common-horizon formula below | Verified |
| NaN unsupported; no backfill/watched-item exclusion; candidate order/duplicates retained (shared contract) | Sparse row extraction and candidate-aligned schedule factors; invalid indices NaN; invalid types rejected | Verified |
| Power-zero exact ablation, sparse/empty/refit safety (task) | Power zero returns the base result directly; successful fits atomically replace state | Verified |
| Local validation before any performance recommendation (`PROTOCOL.md`) | Parent owns the common runner and model-selection/confirmation folds | Not run here |

```python
from research_v2.temporal import TemporalRepeatModel, ScheduleAdjustedRepeatModel

TemporalRepeatModel(half_life_days=28.0, lookback_days=None)
ScheduleAdjustedRepeatModel(
    half_life_days=28.0, lookback_days=None,
    schedule_power=0.5, smoothing_airings=10.0, ratio_cap=10.0,
)
# Both: fit(context) -> self; score(user_matrix_index, candidate_indices) -> float64 vector.
```

Half-life/lookback must be `None` or positive finite numbers. The exponent is
in `[0,1]`, smoothing is positive finite, and the ratio cap is finite and at
least 1. Unknown integer user/candidate indices yield NaN, never wraparound;
noninteger/bool indices are rejected. Empty vectors/catalogs are supported.
Scoring before fit raises `RuntimeError`. A failed refit leaves the previous
successfully fitted scoring state intact. An extreme parameter combination
whose `R*factor` cannot be represented in float64 is rejected during fit, before
commit, rather than silently saturating the formula or emitting infinity.

## Exact formula and units

All timestamps retain the existing numeric source-clock anchor. Let
`a=(train_end-end)/86400` in days. Both event and historical-airing features
require **`history_start < end <= train_end`**. A qualifying-event or historical-airing
weight is `w(a)=2**(-a/h)`, or 1 without decay, and zero for `a>L` when a
lookback is enabled. Starts do not determine age, and `a=L` is retained.

`R_ui=sum_event w(a)` is a float64 CSR. `H_i=sum_historical_airing w(a)` uses
unique `(item index, channel, start, end)` airings, not metadata-row frequency.
`F_i` counts unique fully contained forecast airings supplied by the context.
Historical quarantined airings are forbidden by the shared context; forecast
quarantine flags are permitted metadata and are **not secretly filtered**.
The latter are counted and disclosed separately in diagnostics.

Let `D=min((train_end-history_start)/86400,L)` days (omit `L` if absent),
`E=D` without decay or `E=h/ln(2)*(1-2**(-D/h))` weighted days with decay,
and `Q=(forecast_end-forecast_start)/86400` days. Historical opportunity counts
projected to the forecast horizon are `B_i=H_i*Q/E` scheduled airings. Then

```text
ratio_i = clip((F_i+s)/(B_i+s), 1/cap, cap)
score_ui = R_ui * ratio_i**schedule_power
```

The prior `s=10` means **10 scheduled airings per forecast horizon**, equally
added to numerator and denominator; it is not 10 days or 10 historical airings.
For unchanged broadcast intensity `lambda_i` airings/day, `H_i=lambda_i*E`
and `F_i=lambda_i*Q` produce ratio 1 independent of item frequency. Actual
discrete schedules can differ from this continuous-rate reference. Both ratio
tails are bounded; for the default exponent, score multipliers lie between
`sqrt(0.1)` and `sqrt(10)`. Log-domain arithmetic avoids overflow in `H*Q/E`.

`R_ui=0` is always NaN. With positive power, `F_i=0` is also NaN: a candidate
whose only occurrence crosses the forecast end stays in the original candidate
vector as NaN, not removed/reindexed. **Power zero bypasses every schedule
adjustment, including availability masking**, exactly reproducing temporal
repeat even for boundary-only candidates. This explicit exception preserves a
clean ablation; the temporal baseline itself has no forecast-availability mask.

This is an available-airing proxy, **not observed user exposure, a calibrated
completion probability, or a causal effect**. It supplies no cold-program
evidence; channel/time and collaborative experts remain separate.

## Diagnostics and computational bounds

Public fitted arrays: `repeat_counts_` (users × items CSR), `historical_counts_`,
`forecast_counts_`, `availability_ratios_`, and `schedule_factors_` (item vectors
for the adjusted expert), plus exact `item_ids_` and `user_ids_`.

Diagnostics report actual events, retained/positive-weight events, sparse pairs,
active users/items, total temporal count, cutoff/history/lookback/forecast span,
effective weighted history days, candidate support, unique schedule counts,
quarantined forecast count, ratio/factor summaries and clipping counts. Coverage
denominators use unique supplied candidate indices, not duplicate positions.
No dense global user-item matrix, all-user score cache, labels, or filesystem
access is used in fit. Memory is O(events + airings + CSR nnz + users + items),
with schedule-identity sets and transient sparse candidate-column extraction;
query memory is O(requested candidates + one sparse row).

## Executed checks

Only this author's test file was executed:

```sh
.venv/bin/python -m pytest tests/test_research_temporal.py -q -p no:cacheprovider
```

**120 passed in 0.21s** in the final fingerprint-guarded check: 117
temporal/schedule-adjustment tests and three
independent role-reversal channel/time challenge tests. Environment: Python
3.13.6, NumPy 2.5.4, SciPy 1.18.1, pytest 9.2.0. The checks include file/network
and sparse-densification guards around fit. No other author's test suite ran.

One real numerical edge defect was reproduced during development: `R=2`, no
historical opportunities, one forecast airing, exponent 1, prior equal to the
smallest positive float64, and cap `1e308` would allow an infinite score despite
finite individual parameters. The regression initially failed (`DID NOT RAISE
ValueError`). Fit now checks representable score limits before committing its
state; the same regression passes. A separate robustness audit
also changed effective-duration evaluation to avoid `h/ln(2)` overflow at the
largest finite half-life, without changing the defined integral.

### One bounded actual-data feature check (not MAP@5)

Read **512 pre-cutoff qualifying events only** through an Arrow filter, plus
historical/static forecast EPG for those programs, from
`data/processed/official-v1/`. The cache manifest SHA-256 was
`6bfdabdca49cf2874e60f2f452d250e7fd81c64b6cb24713d741454c478df219`;
this is a sample feature check, not a separate full cache-integrity audit.

- Fixed numeric boundaries: history start `1583712000`, cutoff `1586131200`,
  forecast end `1588550400`; no time-zone conversion.
- Sample dimensions: **19 users, 130 programs, 93 candidates**; **12,111**
  historical and **12,024** fully contained forecast airings for sampled IDs.
- Independent per-event `math.exp2` aggregation differed from the fitted CSR by
  at most **1.7763568394002505e-15**.
- Power-zero output was exactly equal to temporal repeat for every sample user;
  the original count CSR arrays remained byte/value unchanged.
- Repeat supported **185/1,767** user-candidate pairs; the positive-power
  schedule expert supported **184/1,767**. Fully contained occurrence support was
  **92/93** candidates, exposing rather than concealing one availability-only
  support difference.
- No held-out event rows/labels, evaluator, prediction files, or metrics were
  requested. The sample used the default configuration, not a parameter search.

The sample preceded the defensive overflow guard and history-boundary fix;
neither changes the sample's default-range scores because all sampled event and
historical-airing ends are strictly after the source history start. No local
MAP@5, real-data recommendation
quality claim, hyperparameter search, or Kaggle improvement is asserted. The
historical private 0.41001 is an immutable reference, not a training target/label
and not a result of this implementation.

## Role reversal: independent channel/time challenge

Reviewed `research_v2/schedule.py` and its design notes, then added independent
checks to **this author's own** `tests/test_research_temporal.py` (not the channel
author's tests or fixtures). The production channel/time module was not edited.

| Challenge | Independently derived expected result | Observed |
| --- | --- | --- |
| Cold-program transfer, unmatched channel, order/duplicates, unknown/cold users | With one completion on channel c among two historical c airings and prior 2, c-rate is 1/4; two forecast c airings score 1/2. Channel d rate is 1/3. Unseen channel and cold/unknown users are NaN. Duplicate EPG rows cannot change counts. | Passed |
| Exact weekhour hierarchy and unseen-day activity gate | One observed airing, prior 2: channel 1/3; hour 11/27; matching weekday/hour 103/243. Same hour on a never-observed weekday is NaN. | Passed |
| Quarantined forecast support, without candidate removal | A flagged future airing on a trained channel is NaN under explicit exclusion and 1/3 under explicit inclusion. Candidate indices/order remain unchanged; score irrelevance must not be claimed. | Passed |

No channel/time correctness defect was reproduced in these checks. Its
start-slot grouping and zero-preserving weekday gates can miss neighboring-hour
habits, and its cap bounds only personal-repeat extrapolation, not all scheduled
channel-frequency influence. Those are documented modeling limitations, not
implementation defects. The channel model's default quarantine **exclusion**
differs from this expert's explicit metadata **inclusion**; the parent should
preserve/disclose that feature-policy difference rather than globally deleting
program IDs or attributing a policy ablation to parameter quality.

### Author response to the channel author's T1 challenge

**Accepted and fixed.** The independent report in
`reviews/research-v2-schedule.md` correctly identified that a shortened
`history_start` affected effective-duration normalization without trimming old
behavior. The prior implementation could retain `R=2`, `H=1`, `E=2 days`,
`F=1`, `s=1`, giving score `8/3` instead of the matched-window `4/3`.

Our own regression independently reproduced `R=2` rather than 1 with no
lookback, and a second regression reproduced `R=H=3` rather than 1 when both
events and supplied historical schedules included a pre-history end and an end
exactly at `history_start`. Both regressions failed before the fix.

The feature window now explicitly requires **`history_start < end <= train_end`**
for both behaviors and opportunities. The model passes a separate trimmed
in-memory context view to the **unchanged shared `temporal_matrix`**, and the
same shared weight helper still supplies history/event half-life and inclusive
lookback weights. It does not change the original context, `data.interactions`,
candidate catalog, ground truth, helper implementation, or clock anchor. The
same base repeat state keeps power-zero output exact.

After the fix, both reproduced cases score `4/3`, original CSR counts remain 2
and 3 respectively, and exclusion counts are exposed in diagnostics. Two more
fixtures cover entirely pre-history feature-cold users. The no-I/O/no-dense-fit
guard also exercises trimming. All **120** tests pass. The half-life fixture's
nominal history was moved from 56 to 84 days so its known two-half-life event is
inside the declared observation window; its expected 1, 1/2, 1/4 arithmetic did
not change. This distinguishes a strict source-history bound from the separately
inclusive lookback boundary.

Independent author recheck of this correction is still pending; this response
does not relabel an author-executed test as an independent confirmation.

### Final source fingerprint

The final 120-test run checked these hashes before and after execution and
confirmed they were unchanged during the run:

| File | SHA-256 |
| --- | --- |
| `research_v2/temporal.py` | `f85788ea10aaf127b55fb6b9e06fb1e9caf4205bda07a95b24805202d20a3771` |
| `tests/test_research_temporal.py` | `6321b7c86ad17dd89c333388fac2330febca5bc23dc915ed89ff186aea2248e9` |
| Shared `research_v2/contracts.py` (read-only dependency) | `9202448a3b0fb30a8e902112fcbdb825ea78e5901a50744b50e1d5d3aaa9c7e0` |
| Challenged `research_v2/schedule.py` (read-only dependency) | `615d01626e6d99828f417943740795c9942b30178eee817af20cfe5f24d6fca6` |

## Weaknesses / integration notes

- Schedule-intensity stationarity, continuous effective-duration normalization,
  and a common 10-airing prior are engineering assumptions, not tuned results.
- Personal repeats cannot discover programs with zero weighted user history;
  short-history/user-cold coverage must be reported by the parent runner.
- Very stale events may numerically underflow to zero (recorded as unsupported),
  especially with extremely small half-lives.
- Context history coverage must actually begin at `history_start`; missing EPG
  periods can bias the forecast/history ratio. Train/forecast horizon mismatch
  and source-clock uncertainty remain inherited limitations.
- Supplied quarantined forecast intervals are counted as available metadata;
  they are not evidence that a usable resolved historical broadcast exists.
- Shared candidate policy remains unchanged. Positive-power opportunity support
  can differ from pure repeat on boundary-only or absent-contained-EPG programs;
  power zero intentionally restores the pure baseline's support.
- No parameter choice or new live submission is recommended until the common
  local runner demonstrates improvements on more than one prescribed window.
