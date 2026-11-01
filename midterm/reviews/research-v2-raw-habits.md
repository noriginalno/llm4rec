# Raw-watch channel/clock occupancy: implementation record

Date: 2026-11-01. Scope: independent feature-model implementation, synthetic
unit checks and bounded offline cache preparation/validation. **No evaluation,
parameter search, MAP claim, submission, network operation or publication.**
The original matching/labels, `TemporalContext`, protocol, evaluator and scored
artifacts are unchanged. Independent challenge is still a separate next step.

## API and integration

Run commands from `midterm/` with its existing `.venv`; nothing was installed.

```python
from research_v2.raw_habits import RawScheduleModel, build_cache

# Offline once; validates and reuses an existing pair without overwriting it.
build_cache(
    source_path="data/raw/dataset11-30.csv",
    cache_path=".cache/research-v2/raw-watch-union.parquet",
    manifest_path="data/raw/source-manifest.json",
    temp_dir=None,  # defaults to cache parent / "tmp"
)

model = RawScheduleModel(
    raw_cache_path=".cache/research-v2/raw-watch-union.parquet",
    half_life_days=28.0,   # None disables decay
    lookback_days=None,    # positive; duration floored to whole seconds
    slot_minutes=30,       # positive integer divisor of 1440; e.g. 15/30/60
    coverage_power=1.0,    # positive; 2 is a heuristic alternative
    slot_mode="week",     # Monday-first week bins, or pooled "day" bins
    score_block_size=4096,
    quarantined_future_policy="exclude",  # or "include"
)
model.fit(context)                         # returns self; TemporalContext unchanged
scores = model.score(user_index, candidates)  # matrix indices, not raw IDs
```

`score` preserves candidate order and duplicates. Unknown/out-of-range users or
candidates, no history, no retained airing and zero supported overlap return NaN.
Malformed index types raise `ValueError`; scoring before fit raises `RuntimeError`.
No popularity, completion-count blending, watched exclusion or context-candidate
mask is applied internally. Previously watched and cold program IDs can score
from permitted forecast channel/time metadata. Fits snapshot inputs/parameters;
parameter edits apply only on refit, and a failed refit disables stale scoring.

CLI (offline only):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m research_v2.raw_habits
```

CLI options: `--source`, `--cache`, `--manifest`, `--temp-dir`. There is deliberately
no automatic download, authentication, rule acceptance, upload or overwrite flag.
The parent runner must register/benchmark this model separately; no parent-owned
file was edited to introduce it.

## Construction and temporal isolation

- DuckDB 1.5.6, four threads, `memory_limit=4GB`, spill directory
  `.cache/research-v2/tmp`. The limit is DuckDB's, not a measured process-RSS cap.
- Read CSV with `all_varchar=true`; user and channel strings are never routed
  through float, signed int64 or numeric normalization. Leading zeros survive.
- Parse only ISO-naive second-resolution endpoints. DuckDB naive `TIMESTAMP`
  epoch arithmetic provides the existing common wall-clock numeric anchor,
  **not evidence of a physical UTC timezone**. Reject offset/subsecond/invalid
  timestamps, missing IDs and nonpositive intervals. `duraton` is audit-only.
- Union every raw user's intervals separately per channel using running maximum
  prior end and connected-component islands. Merge adjacency, duplicates, nested
  intervals and overlapping device fragments. Do not sum overlapping seconds.
- Store Parquet columns `user_id:string, channel_id:string, start:int64, end:int64`;
  never pickle. Sidecar is
  `.cache/research-v2/raw-watch-union.parquet.meta.json`.
- The existing raw manifest supplies the SHA, which is actually recomputed at
  build time. Record raw size/mtime_ns, manifest SHA, cache SHA/schema/size/mtime,
  construction-code SHA and timings. Every fit verifies cache SHA/schema and raw
  size/mtime plus manifest binding. Raw SHA is not rescanned on every fit. This
  stat guard does not defeat deliberate same-size, same-mtime source replacement.
- Fit selects **context user IDs** from the all-user cache, then filters and clips
  to `[max(history_start, train_end - floored_lookback_seconds), train_end)`
  **before** channel discovery, expansion, weighting or aggregation. No raw future
  channel/interval/end statistics are used. `data.interactions` supplies dimensions
  only; completed-event arrays, historical schedules and descriptive metadata are
  unused. Permitted forecast schedules supply candidate opportunity metadata only.
- Diagnostics include effective history/source cutoff, actual clipped endpoint
  range, source/cache hashes and training-feature SHA. Duplicate forecast identity
  is `(item_index, channel_id, start, end)`; quarantine flags are OR-combined before
  applying the selected policy. No program ID is removed globally.

**Why a global union is safe:** for raw intervals `I_j` and a history window `W`,
`(union_j I_j) intersect W = union_j (I_j intersect W)`. Mutating only viewing
outside `W` therefore cannot change the observed set. Weights depend on clipped
calendar bins, not an untrimmed union's end, so future interval extensions cannot
silently alter recency. This is tested with a crossing-cutoff interval that merges
into a 50-day future extension and new future-only user/channel evidence. Source
hashes change as expected, but training features and scores do not. Moving future
viewing backward into the history window would change the premise, not prove
invariance.

## Exact feature and score definitions

For absolute source-clock bin `b` of width `slot_minutes * 60`, let `k(b)` be its
Monday-first weekly (or daily) slot and `w_b` its midpoint decay weight:

`w_b = 2 ** (-max(0, (train_end - full_bin_midpoint) / 86400) / half_life_days)`.

With decay disabled, `w_b=1`. Let `V_uc` be the clipped union for user/channel and
`W` the effective history window:

`D_k = sum_{b: k(b)=k} w_b * seconds(W intersect b)`

`O_u,c,k = sum_{b: k(b)=k} w_b * seconds(V_uc intersect b) / D_k`.

Only positive supported cells are stored; zero-denominator slots have no cells.
The **same full calendar-bin midpoint weight** is used in numerator/denominator,
even when the window clips a bin. Thus union coverage ensures `0 <= O <= 1`;
only a <=1e-10 floating-point tolerance is permitted before rounding to one.
Daily pooling changes both numerator and denominator consistently.

For retained forecast airing `a` with full duration `L_a`:

`mean_u,a = sum_b O_u,channel(a),k(b) * seconds(a intersect b) / L_a`

`score_u,program = sum_{a: program(a)=program} mean_u,a ** coverage_power`.

Power one is exact linear overlap integration, not start-slot assignment. A
10-minute and 60-minute airing under the same occupancy have equal contributions;
two supported airings contribute twice, matching the target's **airing-frequency**
unit rather than total minutes. Other powers apply **after each airing's mean**,
not to each bin or the final program aggregate.

The occupancy matrix is sparse users x channel-slots. Forecast features are sparse
channel-slots x programs for power one, or x airings for other powers. Score uses
one sparse user row and bounded output blocks; nonlinear scoring additionally
keeps per-query airing indices/owners, never a dense users x all-airings matrix.
No file or network access occurs during score.

## Executed checks and actual cache

| Requirement | Executed evidence | Result |
| --- | --- | --- |
| Union, opaque IDs and cache binding | Nested/duplicate/device fixtures; string IDs above signed int64 and leading zeros; checksum/manifest/stat/schema corruption; raw-byte preservation | Passed |
| Time semantics and leakage | Independent per-second datetime oracle; Sunday/Monday, midnight/hour and partial-bin boundaries; future mutation; lookback; decay | Passed |
| Correct frequency scale | Hand-derived 10-vs-60-minute, second-resolution cross-bin overlap and nonlinear per-airing examples | Passed |
| API, sparsity and isolation | Unknown/empty inputs, ordering/duplicates, input mutation/refit, file/network guards, output-block shape guards | Passed |

Verification command (the executed run additionally set `--basetemp` to the
approved OpenCode runtime temporary directory, run `raw-habits-tests-003`):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_research_raw_habits.py -p no:cacheprovider -q
```

**63 passed in 1.23 seconds**. Existing dependencies: DuckDB 1.5.6, PyArrow 24.0.0,
NumPy 2.5.4, SciPy 1.18.1. Only this owned unit suite was run, not a full benchmark.

Actual offline build (original raw file preserved):

| Quantity | Observed |
| --- | ---: |
| Raw rows | 5,821,987 |
| Union intervals | 5,441,773 |
| Raw users / channels | 4,914 / 202 |
| Ununioned seconds / union seconds | 8,619,636,424 / 8,503,411,121 |
| Invalid source rows / duration mismatch rows | 0 / 117 |
| Cache bytes | 37,866,265 |
| Source hash / parse+audit / union+write+audit / cache hash seconds | 0.187724 / 0.980891 / 1.455071 / 0.017718 |
| Build total seconds | 2.666259 |

Source SHA-256:
`8c9e5f058157041e5c33f13e64dc1625a5177cc6d5e434ba18cd8a69afafb53b`.
Cache SHA-256:
`0a0533cdf5c1ceda6c2c6f07bb387b8e6a54d231d6a820876114fb1a5d0682b0`.

Additional **executed cache validation**, without labels or predictions:

1. Independent stdlib CSV scan of all **5,821,987** raw rows. Select the first
   template user whose ID has 20 characters (3,927 raw rows across all dates),
   clip endpoints to `[1583712000, 1586131200)` before an independent
   endpoint-depth sweep. All **1,163 union intervals / 843,477 seconds** agree
   exactly with the global-union Parquet clipped afterward. Elapsed 6.272510s.
2. Full-cache neighbor scan found **zero invalid intervals** and **zero remaining
   overlap/adjacency pairs** per user/channel. All **1,260/1,260** raw template users
   appear. Reusing `build_cache()` validated the existing pair without overwriting;
   a fresh full raw SHA-256 recheck passed.

## Limitations and next handoff

- This estimates observed viewing occupancy over a common calendar history.
  Active-user availability, device/person identity and actual channel exposure
  are unknown. Long history before a user's arrival can dilute occupancy.
- Mean coverage, and especially its power, is **not** the joint probability of
  >=80% completion. There is no calibration or causal/exposure interpretation.
- Channel/time habits miss program-specific preference, shifting routines and
  new channels. Exact weekly bins can be sparse; daily pooling is an explicit
  alternative, not an automatically selected improvement. Simultaneous channels
  are unioned separately; channel occupancies need not sum to one.
- The forecast is supplied metadata, with organizer-unknown schedule/conflict
  semantics unchanged. Quarantined airing inclusion is explicit, not tuned here.
- Four threads/4GB constrain DuckDB only; no process peak-memory measurement or
  full real-cohort fit/score timing is claimed. Python sparse forecast storage is
  proportional to retained overlap nonzeros; nonlinearity retains per-airing data.
- No held-out metrics or quality gain are claimed. Parent-owned evaluation must
  compare the fixed initial default (and a preplanned small alternative shortlist)
  under the unchanged protocol, followed by independent adversarial challenge.

## Reverse-role review: raw-habits author challenges the temporal expert

2026-11-01. **No new correctness blocker was demonstrated.** This is an
independent synthetic challenge of the frozen `TemporalRepeatModel` and
`ScheduleAdjustedRepeatModel`, not a response to the separate critique of raw
habits. That author response is deferred to the next step. No source/cache fix
was made or proposed for immediate execution during the parent's frozen grid.

Reviewed `research_v2/temporal.py`, `research_v2/contracts.py`, and
`reviews/research-v2-temporal.md`. Added only
`tests/test_research_temporal_reverse_raw.py` and this appended section. Model
constructors and the context/data record types are the only production model
interfaces used. Expected values come from hand arithmetic and direct 2,048-panel
Simpson integration of the decay kernel, not production weighting/feature helpers
or another author's fixtures. Test scenarios are fixed mathematical cases, not
a hyperparameter search. No raw data, cache contents, labels, predictions,
parent-run metrics, API, credentials, network, submission, push or subagent was
used in this review.

### Independent oracle checks

All values below are **synthetic expected values confirmed by executed assertions**,
not recommendation-quality measurements.

| Challenge | Independent expectation | Observed |
| --- | --- | --- |
| Opportunity ratio and smoothing units | `H=2`, history `D=4 days`, `F=3`, prior `s=2 forecast-horizon airings`: `B=H*Q/D`; ratios for `Q=1/2/4 days` are `2`, `5/3`, `5/4` | Passed |
| Forecast gap vs horizon | Moving the same 2-day forecast 9 days after cutoff does not turn its horizon into 11 days; `R=H=F=s=1`, `D=4` gives score `4/3` both times | Passed |
| Frequency-neutral intensity reference | `D=10`, `Q=2`, rare `H=5,F=1` and frequent `H=20,F=4` both give ratio 1; repeats remain `2` and `3` | Passed |
| Effective weighted history | Numerically integrate `2**(-age/3)` in **days**, over `min(history span, lookback)`; tested spans 6 and 3 days. Independent projected counts and final scores agree within `2e-13` relative tolerance | Passed |
| Matched event/opportunity windows | Half-life 2 days, lookback 2: end ages 0 and exactly 2 give `R=H=1.5`; 2 days + 1 second is excluded. `F=2`, `E=1/ln(2)` days, `Q=3`, `s=1` give score `4.5/(4.5*ln(2)+1)` | Passed |
| Unique broadcast identity | Exact schedule duplicates do not add opportunities. Same program/time on distinct channels remains **two** airings: `H=F=2,D=4,Q=2,s=1` gives `1.5` | Passed |
| Previously fixed T1 | Pre-history and exact-`history_start` ends are excluded from both feature sides. With original count 3, retained `R=H=1`, `D=2,Q=F=s=1`, score is `4/3`; original CSR still contains 3 | Independently reconfirmed |
| Source-clock boundaries | On the common naive anchor, 2020-03-28 to 2020-03-30 is 172,800 seconds, regardless of the intervening DST date. One-day end decay is exactly `1/2`; one second outside lookback is excluded | Passed |
| Caps and power-zero ablation | Raw ratios `2/101` and `51/3`, cap 3, become `1/3` and `3`; powers 0, 1/2, 1 match hand-derived multipliers | Passed |
| Zero evidence / power zero | A prior cannot invent a completion. Positive power gives NaN for `F=0` or `R=0`; zero power restores pure repeat, including missing-forecast candidates, duplicate positions and finite-value bit patterns | Passed |
| Full horizon containment | One-day and one-second contained airings each count once, including end exactly at horizon end; missing-contained-airing candidate stays NaN in its original position. Invalid crossing-horizon metadata/future behavior are rejected by context | Passed |
| Explicit quarantine policy | Flagged and unflagged forecast metadata both contribute under temporal's declared **inclusion** policy; fixture scores are `4/3` in both cases | Passed; not a raw-policy bug |
| Cold-axis alignment and coverage | Opaque 20-digit/leading-zero user IDs and noncontiguous raw item IDs stay separate from matrix indices. One supported pair among `4 users * 3 unique candidates` means coverage `1/12`; duplicate query positions do not inflate it | Passed |
| Sparse, nonmutating fit | Trimmed fitting succeeds with filesystem/network and sparse densification forbidden; shared CSR, immutable arrays and metadata remain unchanged | Passed |

Temporal's forecast quarantine **inclusion** differs from raw habits' default
**exclusion**, but both are disclosed feature policies. It is not evidence of a
correctness defect, candidate-catalog drift or a reason to delete program IDs.
The normalized schedule ratio remains an opportunity proxy, not measured user
exposure or a calibrated completion probability. These unit checks do not
establish schedule stationarity or predictive improvement.

### Execution and evidence fingerprints

Executed only the newly owned test file, through `pytest.main` with before/after
SHA-256 checks and an approved temporary directory (`temporal-reverse-raw-002`).
Equivalent test command:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_research_temporal_reverse_raw.py -q -p no:cacheprovider
```

**23 passed in 0.14 seconds; exit code 0.** No skips/xfails. Both an autouse fixture
and the execution wrapper verified frozen dependency hashes before/after the
run. No other author's tests or parent benchmark were executed.

| File | SHA-256, unchanged during execution |
| --- | --- |
| `research_v2/temporal.py` | `f85788ea10aaf127b55fb6b9e06fb1e9caf4205bda07a95b24805202d20a3771` |
| `research_v2/contracts.py` | `9202448a3b0fb30a8e902112fcbdb825ea78e5901a50744b50e1d5d3aaa9c7e0` |
| `sweettv/contracts.py` | `d242dec34cbbe277f1729f9d0185076c4040681a6a2d995128a2f8b16d6ba527` |
| `research_v2/raw_habits.py` | `c10952b6654e90cdebe64f4af5924e709d114adacd8176508b19b96c2b0be5ad` |
| New `tests/test_research_temporal_reverse_raw.py` | `0e514920f10570887916df8d5207b18a0f7b7742e22336bbffcebb541a795d08` |

The preexisting 11,364-byte review prefix had SHA-256
`fc83e89f84f722d4fb7d9b6235c078ea6899c64d4ba2096f7faf8d83f0fec1e2`;
it is retained unchanged. This closes only this independent temporal challenge,
not the upcoming raw-model author response or the parent's quality evaluation.

## Final author response to the independent raw-habits challenge

2026-11-01. Read `reviews/research-v2-raw-habits-challenge.md` and
`tests/test_research_raw_habits_challenge.py`. **Accepted: no implementation
defect was reproduced, and no source/cache/test correction is requested.**
The independent implementation/challenge/author-response loop is now closed for
this scoped model verification. The parent's frozen fusion grid and source
closure remain untouched; this closeout does not evaluate or select a model.

### Attribution and actual confirmation

- **Independent reviewer execution:** the separate reviewer reports **36 passed
  in 0.84s**, using independently authored endpoint-depth, literal-second,
  calendar and hand-fraction oracles. Its real-cache inspection was read-only;
  it did not rerun the author's full raw-data scan or target-ID coverage scan.
  These remain the reviewer's checks, not independently executed checks by me.
- **Author-run combined confirmation:** I executed the unchanged **63 author
  tests plus 36 independently authored challenge tests together: 99 passed in
  1.91s, exit code 0**, with no failures, skips or xfails. Per-file result counts
  were recorded directly from pytest reports. Rerunning the independent test code
  is an author reproduction, not a second independent review.
- **Reverse role already closed:** the 23 independently authored temporal-model
  checks passed in the preceding section. They were not rerun here and are not
  included in the 99-test total.

Combined command from `midterm/` (executed through a fingerprint-guarded
`pytest.main` wrapper with bytecode/cache-provider writes disabled):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_research_raw_habits.py \
  tests/test_research_raw_habits_challenge.py -q -p no:cacheprovider
```

Synthetic fixture caches and integrity-negative controls were confined to the
approved temporary directory, suffix
`raw-habits-author-closeout-861f7a81e5224e7b90cd327e9ff97f83`.
The wrapper checked exact SHA-256, byte size and nanosecond mtime before/after
for all 20 top-level Python source files in `research_v2/` and `sweettv/`, both
raw test files, the reverse-review test, independent critique, existing author
review, production union cache and sidecar. All matched; no source file was
added/removed. The only subsequent persistent change is this appended response.

### Limits explicitly retained

1. The denominator is weighted calendar history, **not observed active-user
   availability or channel exposure**. A sum of powered duration-normalized
   airing mean coverages is a frequency proxy, **not** a calibrated probability
   of joint >=80% completion or a causal recommendation effect.
2. The union/clip identity requires equal within-window viewing sets. The reviewer
   independently verified that premise and exact fitted invariance; arbitrary
   edits that move future viewing into training do not satisfy it.
3. Four DuckDB threads and its 4GB memory limit are not a measured process-RSS
   limit. Sparse storage and bounded dense score blocks do not establish full
   production latency/peak memory; forecast overlap storage and nonlinear
   query-local airing arrays still scale with their respective sparse workload.
4. The raw source's build-time SHA is followed by a size/mtime trust guard during
   fits. Deliberate same-size, same-mtime replacement of raw data is outside that
   stated guard; synthetic corruption tests do not allege production corruption.
5. Preserve both provenance snapshots: cache construction code SHA
   `63e08f42ed4f63f246dd0e9b252e3e89d05eebad0b6efedf20b0b5cfb005bef9`
   and currently reviewed source SHA below are different identifiers, not a
   reason to rebuild/relabel the unchanged cache during a source-frozen grid.

No new real-data feature fit, raw-row/target-ID scan, prediction output, held-out
metric or MAP result was requested in this closeout. No parent-grid results were
read. **No recommendation-quality or Kaggle improvement claim follows from the
99 passing checks.** No API, credential, submission, push or subagent was used.

### Closed-loop evidence hashes

All following hashes were stable across the combined check:

| Artifact | SHA-256 |
| --- | --- |
| Reviewed `research_v2/raw_habits.py` | `c10952b6654e90cdebe64f4af5924e709d114adacd8176508b19b96c2b0be5ad` |
| Author `tests/test_research_raw_habits.py` | `922a8e583c757291aab68852db6fdeda8b760afa7d90fefe688405f3a2ca3d7c` |
| Independent `tests/test_research_raw_habits_challenge.py` | `bf312cb60b43b4bfec8bfeb9b7a81b248abbd3f9442ee2738a217eee91204ed8` |
| Independent `reviews/research-v2-raw-habits-challenge.md` | `b9ebfc7fa672b3e6ba556f9c439346d6397496d7abb1e6c66cc9afc179b34024` |
| Production `.cache/research-v2/raw-watch-union.parquet` | `0a0533cdf5c1ceda6c2c6f07bb387b8e6a54d231d6a820876114fb1a5d0682b0` |
| Production `.cache/research-v2/raw-watch-union.parquet.meta.json` | `ed63837ec1105bdba981c3e45d72e28de29cf4642681ca4edf421594d9cbd492` |
| Already-closed `tests/test_research_temporal_reverse_raw.py` | `0e514920f10570887916df8d5207b18a0f7b7742e22336bbffcebb541a795d08` |

The 20-file source-closure fingerprint is
`6761772d8cdb7a0747af24f2bb5340a80d6c3cdb28b344bf01723145db26b028`:
SHA-256 of the relative-path-to-file-SHA mapping serialized with
`json.dumps(mapping, sort_keys=True, separators=(',', ':'))`, UTF-8 encoded.
The entire pre-response review prefix is preserved: **17,614 bytes**, SHA-256
`31bed30bbe7b3c7fa47531706a2845e48ded9f5fe742223136499b02c657921b`.
