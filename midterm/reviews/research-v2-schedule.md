# ChannelScheduleModel: design and verification

## Design fixed before experiments (2026-11-01)

Scope: only `research_v2/schedule.py`, `tests/test_research_schedule.py`, and
this review. No changes to the scored solution, shared contracts, candidates,
matching policy, labels, or evaluation. No parameter search is authorized here.

### Requirements and choices

| Requirement | Implementation choice / check |
| --- | --- |
| Future completed-broadcast frequency, not novelty affinity | Sum personal channel/slot rates over supplied forecast airings; keep repeats eligible. |
| Training-only behavior | Use only context qualifying-event arrays, matched to historical airings; never inspect raw logs or files. |
| Same opportunities/count weighting | Same end-based decay and inclusive lookback boundary for both; require `history_start < end <= train_end`. |
| Sparse bounded aggregation | CSR user × channel/time-group counts and sparse group × forecast-item airing counts; only a single query's score vector is dense. |
| Unsupported evidence | Zero-preserving hierarchy; no positive prior for an unseen channel or never-observed slot. Cold/unknown users and unsupported items return NaN. |
| Rare/newly overscheduled repeats | Smoothed personal-program rate plus a capped positive repeat extrapolation term. |
| Quarantined forecast metadata | Default: exclude flagged airings from opportunity features, **not** their program IDs from the candidate catalog. Optional explicit `include` policy. |
| Naive source clock | Integer epoch-anchor arithmetic for hour and Monday-based weekhour; no machine-local clock, time-zone conversion, or DST inference. |

### Exact formulas

All observations are distinct completed broadcasts, not viewing fragments. For
each retained historical airing ending at `e`, define

`w(e) = 2 ** (-(train_end - e) / (86400 * half_life_days))`.

Use weight 1 if the half-life is `None`; set weight to zero outside
`history_start < e <= train_end`, or when age exceeds `lookback_days` (if set).
End exactly at the lookback boundary is included. The identical weight/window
applies to completed events and scheduled opportunities. Exact duplicate
airings/events are counted once. An event without its historical airing is
diagnosed and excluded rather than assigned a fabricated opportunity.

Let `C[u,g]` be weighted qualifying completions and `O[g]` the weighted number
of historical scheduled airings in group `g`. Opportunities are not multiplied
by user count. Let `a = prior_strength > 0`:

- `r[u,c] = C[u,c] / (O[c] + a)` (zero prior; no global popularity).
- `A[u,h] = C[u,h] / (C[u,h] + a)`.
- `r[u,c,h] = (C[u,c,h] + a*r[u,c]*A[u,h]) / (O[c,h] + a)`.
- `A[u,wh] = C[u,wh] / (C[u,wh] + a)`.
- `r[u,c,wh] = (C[u,c,wh] + a*r[u,c,h]*A[u,wh]) / (O[c,wh] + a)`.

`wh = weekday*24 + hour`. Groups use airing **start** slots, while weights use
airing **end**. `slot_mode` selects the channel, channel/hour, or full hierarchy.
The activity gates deliberately suppress prior transfer into never-observed
hours/weekhours. A group with no direct completions can still receive modest
backoff when both its channel and activity slot have personal evidence elsewhere.
Rates lie in [0,1] because each matched completion is one of its opportunities;
they are **not calibrated completion probabilities or measured exposure rates**.

Let `F[i]` be retained forecast airing count, `H[i]` weighted historical airing
count, and `B[u,i] = sum_{forecast airings of i} r[u,selected_group]`.
For personally completed programs only, blend a smoothed repeat rate using
`b = programme_blend`, `m = max_repeat_multiplier`:

`S[u,i] = (1 - b*H[i]/(H[i]+a))*B[u,i]`
`          + b*C[u,i]*min(F[i]/(H[i]+a), m)`.

Without the cap, this is the convex blend of `B` with
`F*(C + a*(B/F))/(H+a)`: the forecast per-airing channel/slot rate is the
personal-program prior. The cap only limits the repeat-count extrapolation,
not the channel/slot forecast-frequency signal. Each single weighted completion
can contribute at most `b*m` through the repeat term. Cold programs use `B`
directly. No retained forecast airing means no schedule evidence (NaN), not a
global eligibility decision. Positive programme history may still support a
repeat when its forecast slot/channel lacks channel/slot evidence.

Constructor defaults: `half_life_days=28.0`, `lookback_days=None`,
`slot_mode="weekhour"`, `prior_strength=4.0`, `programme_blend=0.5`,
`max_repeat_multiplier=3.0`, `quarantined_future_policy="exclude"`.
These are engineering choices, not selected hyperparameters or organizer rules.
`fit(TemporalContext) -> self` and `score(user_index, candidate_indices)` return
candidate-aligned float64 scores. There is no batch API or all-user score cache.
Parameter edits take effect only on refit; a failed fit invalidates scoring.

## Limits and quarantine interpretation

- Scheduled airings are only an opportunity proxy: no measured personal exposure,
  recommendation availability, causal, engagement, or retention claim follows.
- The behavioral source contains only >=80% qualifying completions. Partial
  viewing and non-viewing are unobserved; activity gates cannot distinguish
  genuine inactivity from partial viewing or incomplete logging. Never silently
  read raw logs (especially future logs) to repair this limitation.
- Start-hour grouping ignores within-airing cross-hour activity, duration, and
  simultaneous channel competition. Fine weekhours can be sparse; a missed slot
  has no transferred evidence, even when a broader habit might generalize.
- A large legitimate forecast frequency can still dominate the channel backoff;
  the cap controls the personal-repeat uplift, not all forecast-airing counts.
- The parent reports eight quarantined future flags in the source audit. This
  model must count actual flags from each supplied context (never hard-code eight)
  and report whether exclusion changes the aggregated opportunity matrix. This
  note does not independently verify the real-data count or claim irrelevance:
  full empirical flag irrelevance requires an include/exclude score comparison
  for that context. The diagnostics also give a conservative sufficient proof:
  if no flagged airing has any personally completed channel **or** program in
  training, those flagged additions cannot change scores under either policy.
- Metadata may include future-supplied descriptions. The model intentionally does
  not use titles/categories or infer channels from canonical metadata: actual
  historical/forecast airing channels are authoritative.

## Executed checks

Command from `midterm/`, using the preexisting environment (no installs):

```sh
.venv/bin/python -m pytest tests/test_research_schedule.py -q
```

Final result: **60 passed in 0.19s**, NumPy 2.5.4, SciPy 1.18.1, pytest 9.2.0.
The first run had 56 passes and one incorrect fixture expectation: four
completions with `a=4` give hour rate `(4+4*(1/2)*(1/2))/8 = 5/8`, not 5/12.
The oracle was corrected with that derivation; this was not a production defect.
Subsequent fixtures also verify parameter snapshots and a shortened history
start. No MAP@5 comparison, real-data fit, parameter search, or Kaggle operation
has been run by this author. Fixture runtimes are not production benchmarks.

| Check | Independent evidence |
| --- | --- |
| Cold program on preferred versus unrelated channel | All three slot modes; channel scores exactly 8/10 versus 1/10. |
| No unsupported slot overconfidence | Dense historical opportunities at 20:00, no personal completions at 03:00; only 20:00 has evidence. Monday versus Tuesday and midnight-crossing start slots checked separately. |
| Exact hierarchy | Channel 1/3, channel/hour 4/15, channel/weekhour 53/180, hand-derived in fixtures. |
| Positive but modest cross-slot backoff | Channel and hour observed separately: 1/9 for hour, 1/27 for weekhour; a never-observed channel remains NaN. |
| Programme blend and rare cap | Exact 51/40 blend and 41/40 capped result; one personal completion contributes at most 2 for forecast multiplicities 20, 100, and 1000 in an isolated repeat fixture. |
| Identical event/opportunity time weights | Ages 0 and exactly 2 days contribute 1 + 1/2 to both sides; one second outside and `end=history_start` excluded. Separate shortened-history/no-lookback check. |
| Missing/empty data and refits | Empty history, empty forecast, unmatched event, zero users/items, unknown/cold users, changed catalog/channel dimensions; no stale fitted state. |
| Alignment and identifiers | Reordered/duplicate candidates, signed and uint64 unknown indices, raw int64 ID above 2^53, malformed inputs. |
| Quarantine | Eight **synthetic**, score-irrelevant flagged airings change feature counts but not scores; same program's unflagged preferred-channel airing remains usable. Supported flags are correctly not declared irrelevant. |
| Shared state and I/O | Arrays/CSR/metadata/schedules unchanged; monkeypatched filesystem/socket access and sparse `.toarray()` forbid reads/network/densification during fit. |

Verified source hashes for this check:

- `research_v2/schedule.py`: `615d01626e6d99828f417943740795c9942b30178eee817af20cfe5f24d6fca6`.
- `tests/test_research_schedule.py`: `498005c96545adeb18c9afca7d13425609ab55b3ad6717364e0648f6731c4819`.

## Cross-review

### Independent challenge to TemporalRepeat / ScheduleAdjustedRepeat

Snapshot reviewed: `research_v2/temporal.py`, SHA-256
`9d7bc6ed85984b70d48b813ddb371197054cd868753d344b9fa7c4425f008cbe`.
The implementation correctly uses end-age decay, common-horizon opportunity
units, bounded ratios, sparse scores, explicit future quarantine inclusion,
and a genuine power-zero score ablation. No held-out behavior access was found.

**T1 / medium, conditional window defect, reproduced:** `history_start` bounds
the effective-history normalization but does not mask event weights. A valid
`TemporalContext` may contain earlier events; `make_context` itself selects all
pre-cutoff events while its historical opportunities start strictly after
`history_start`. Consequently a nondefault shortened `history_start` can retain
old completions without their matching opportunities. This is not a claim that
the current real-data/default folds contain such old events.

Executed a standalone synthetic reproduction with:

```python
old = airing(0, "known", 0)
recent = airing(0, "known", 3)
ctx = context(
    [recent], [(0, old), (0, recent)], [airing(0, "known", 4)],
    history_start=MONDAY + 2*DAY, train_end=MONDAY + 4*DAY,
    forecast_end=MONDAY + 5*DAY,
)
model = ScheduleAdjustedRepeatModel(
    half_life_days=None, schedule_power=1, smoothing_airings=1,
).fit(ctx)
```

The fixture helpers above are defined in `tests/test_research_schedule.py`;
the reproduction was run with `tests` added to `sys.path`, not with raw data.
Observed `R=2`, `H=1`, `E=2 days`, `F=1`, factor `4/3`, score **8/3**. A matched
two-day history has `R=1` and score **4/3**. The pre-history completion doubled
the score although its scheduled opportunity could not enter the denominator.

Requested resolution: apply the same `history_start < end` mask to both
behavioral and opportunity weights (preserving power-zero equality by using the
same repeat state), or explicitly reject inconsistent contexts and document a
stronger history-bound contract. Do not edit shared production counts or labels.
The schedule implementation's own shortened-history fixture already excludes
the old completion and reports it as outside the feature window.

**Status:** challenge recorded for the parent/temporal author; author response
and independent recheck pending. No foreign-owned source/test files were edited.

### Response to the temporal author's challenge

No challenge has been received yet. Await the author's concrete fixture/findings;
do not label the paired author-response stage complete until it is actually run.

## Closed-loop role-swap recheck (2026-11-01; supersedes pending statuses above)

The temporal author supplied the corrected source and three channel challenges
in `reviews/research-v2-temporal.md`. This follow-up owns **only** the new
`tests/test_research_temporal_challenge.py` and this appended review section.
Neither model source, either original author test file, shared contracts, nor
any running experiment was edited. The parent reported an active 14-configuration
EASE/schedule experiment on two development folds; its sources remained frozen.

### T1: independent confirmation of the temporal author's fix — closed

Read the new implementation, not just its author response. `_repeat_state`
now constructs a separate aligned in-memory event view with
`event.end > history_start`, preserving the original `TrainingData` object.
`ScheduleAdjustedRepeatModel.fit` independently applies the same strict bound
to historical opportunities. Both then use the unchanged shared end-age decay
and inclusive-lookback helper. Power zero still uses the same repeat state.

The new regression file does **not** import either author's test helpers or
production weight/ranking helpers for its oracle. It builds fresh synthetic
contexts, independently scans observations with scalar `math.exp2`, and uses
direct horizon arithmetic. Tiny dense expected arrays exist only in this
test-only oracle, not in either production model.

| Independent recheck | Expected | Observed / disposition |
| --- | --- | --- |
| Original shortened-history failure: one old and one recent completion, only recent historical opportunity | `R=H=F=1`, effective duration 2 days, factor/score `4/3`; original CSR count stays 2 | Reproduced fixed result `4/3` (not former `8/3`); one old event diagnosed as excluded; original arrays/CSR/metadata unchanged. |
| Old, exact-history-boundary, and recent ends supplied on both sides | Only recent event/airing retained; score `4/3`; original CSR count stays 3 | Confirmed with no lookback and with a 2-day lookback. `end=history_start` remains excluded even though that end also meets the separately inclusive lookback boundary. |
| End-age decay and one-second lookback boundary | Identical weighted `R`/`H`, independent effective duration and adjustment; exact 1-day age included, 1 day + 1 second excluded | All three arithmetic cases passed: no decay/1-day window, 2-day half-life/no extra window, and 2-day half-life/1-day window. These are synthetic unit cases, not model-selection trials. |
| Entirely pre-history or exact-boundary user | Feature-cold NaNs for repeat, power-zero, and positive-power models; original count remains 1 | Confirmed for both boundary placements; no fallback introduced. |
| Power-zero after trimming, including personally watched program with no forecast airing | Exactly equal to temporal repeat, including duplicate candidate positions and NaN locations | Confirmed for active, cold, and unknown users. Positive power separately leaves the no-forecast program as NaN. |
| Refit with shortened history | Both completion and opportunity counts rebuild from 2 to 1; score becomes `4/3` | Confirmed; shared context and original CSR count 2 remain unchanged. |
| Trimming fit under I/O/densification guards | No filesystem/socket access or sparse `.toarray()` during either fit | Confirmed with monkeypatched guards, then scalar score checks after restoring normal scoring operations. |

**Resolution:** T1 is accepted, fixed by its author, and now independently
rechecked by the original challenger. No change to labels, raw completion-count
CSR, candidates, helper implementation, or clock representation was needed.
The defect-to-fix evidence is now closed; the earlier pending recheck status is
historical, not the current status.

### Channel author response to all three temporal challenges — closed

Read the author's three actual fixtures, then constructed separate synthetic
fixtures in the new test file. Expected values are independent counts and exact
`Fraction` arithmetic, not calls to channel aggregation/slot helpers. The three
behaviors are reproduced, not merely accepted based on the author's 120-pass
claim. Numeric comparisons below use absolute tolerance at most `1e-15`.

| Author challenge / response | Independently expected | Reproduced actual result and assessment |
| --- | --- | --- |
| R1: cold-program channel transfer, unknown channel, order/duplicates, cold/unknown users | `r_c=1/(2+2)=1/4`, `r_d=1/(1+2)=1/3`; two future c airings give `1/2`. Query `[5,4,3,4,-1,6]` gives `[NaN,1/3,1/2,1/3,NaN,NaN]`. | Matched. Adding duplicate historical/forecast airings and reversing candidate order preserved aligned values. Unknown/cold users and raw program IDs supplied as out-of-range indices remained NaN. Cold programs have no personal-program counts. **Accepted; no defect/fix.** |
| R2: exact weekhour hierarchy and zero-preserving unseen weekday | One completed airing with prior 2: channel `1/3`; hour `(1+2*(1/3)*(1/3))/3=11/27`; weekhour `(1+2*(11/27)*(1/3))/3=103/243`. | Matched separately in channel, hour, and weekhour modes. The next weekday retains the broad channel/hour scores but is NaN in weekhour mode for this cold program. Duplicate candidates align; maximum group rate stays <=1. **Accepted; no defect/fix.** |
| R3: quarantine support changes without candidate-ID removal | A flagged airing on a trained channel gives NaN under exclusion and `1/3` under inclusion; candidate vector `[3,4]` unchanged; no score-irrelevance proof. | Matched, including duplicate query `[3,3,4]`, opportunity count 0 versus 1, zero candidate IDs removed, and `quarantine_score_irrelevance_proven=False` under both policies. **Accepted; no defect/fix.** |

No counterexample to these three reported outcomes was found in the independent
reproductions. No channel-source correction is requested, so the parent's
running schedule experiment does not need a source change for this review.

The author's limitations are also accepted: start-slot bins do not generalize
to neighboring bins automatically, and the cap bounds the personal-repeat term,
not total channel/slot forecast-frequency influence. These are explicit modeling
choices, not newly reproduced correctness defects. Unseen-slot suppression in
R2 concerns cold-program channel/slot evidence; documented personal-program
history can independently support a repeat in a new forecast slot.

Quarantine **exclusion** in this channel model versus explicit **inclusion** in
the temporal availability expert remains a disclosed feature-policy difference,
not candidate removal and not an implementation defect. These fixtures do not
independently establish whether the eight real-data future flags affect scores.

### Executed tests and frozen fingerprints

Using the preexisting environment, from `midterm/`:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_research_temporal_challenge.py -q -p no:cacheprovider
# 16 passed in 0.14s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/test_research_temporal_challenge.py tests/test_research_temporal.py tests/test_research_schedule.py -q -p no:cacheprovider
# 196 passed in 0.29s = 16 new independent + 120 temporal-author + 60 channel-author
```

The second command independently reran the existing author suites without
editing them. A module-scoped guard verified all five frozen source/dependency
files below were unchanged across each run; their hashes also matched the
explicit pre-run snapshot. Python 3.13.6, NumPy 2.5.4, SciPy 1.18.1, pytest 9.2.0.

| File | Verified SHA-256 |
| --- | --- |
| `research_v2/temporal.py` (corrected T1 source) | `f85788ea10aaf127b55fb6b9e06fb1e9caf4205bda07a95b24805202d20a3771` |
| `research_v2/schedule.py` (unchanged frozen channel source) | `615d01626e6d99828f417943740795c9942b30178eee817af20cfe5f24d6fca6` |
| `research_v2/contracts.py` (parent-owned read-only dependency) | `9202448a3b0fb30a8e902112fcbdb825ea78e5901a50744b50e1d5d3aaa9c7e0` |
| `tests/test_research_temporal.py` | `6321b7c86ad17dd89c333388fac2330febca5bc23dc915ed89ff186aea2248e9` |
| `tests/test_research_schedule.py` | `498005c96545adeb18c9afca7d13425609ab55b3ad6717364e0648f6731c4819` |
| `tests/test_research_temporal_challenge.py` (new independent tests) | `652432bc950791dc5a065d6afee4c3d7a7f87a0056a4c049a75b67d8629e3991` |

### Remaining gates / handoff

- **Paired correctness review closed:** temporal T1 author fix independently
  confirmed; channel R1–R3 independently reproduced and answered. No pending
  channel correctness patch or experiment interruption follows from this review.
- The parent still owns completion/inspection of its two-fold experiment,
  shortlist freeze, prescribed confirmation folds, fixed strict-five and
  all-user diagnostics, tie sensitivity, source/configuration fingerprints,
  and any evidence-based comparison with the unchanged reference. No MAP@5 or
  model-quality improvement was computed or inferred in this recheck.
- The parent reported C1–C4 context fixes with 55 passing tests. That suite was
  **not** rerun here; the corrected context module was a frozen dependency of
  the independently executed 196 tests above, not an authorship transfer.
- Real-data quarantine relevance and missing-history/exposure assumptions must
  remain explicit in parent diagnostics. The fixtures use only completed-event
  arrays and permitted synthetic forecast EPG, never raw partial/future viewing.
- No full parameter search, actual-data fit, API/credential access, Kaggle
  submission, installation, push, or subagent was used for this recheck. Existing
  approval and delivery/evidence gates remain unchanged.
