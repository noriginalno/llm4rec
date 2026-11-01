# Independent challenge: raw-watch schedule occupancy

2026-11-01. Review scope: `research_v2/raw_habits.py:RawScheduleModel` and
`build_cache`. This author owns only this new report and
`tests/test_research_raw_habits_challenge.py`. No implementation, real cache,
parent runner, scored artifact, credential, or old review/test was edited.
No API/network operation, installation, upload, publication, subagent, held-out
label/prediction access, parameter search, or performance evaluation is involved.

## Independent verification plan

The numerical checks do not import the implementation author's test fixtures or
oracles. They use an endpoint-depth sweep, literal sets of seconds, naive
`datetime.weekday()`/calendar arithmetic, and hand-derived fractions. All test
source CSVs, manifests, cache files, and deliberate integrity-negative controls
are **synthetic artifacts in the approved pytest temporary directory only**.
Production raw files and caches remain read-only.

| Question | Independent check | Status |
| --- | --- | --- |
| Are nested, adjacent, duplicate and overlapping device intervals unioned once? | Endpoint-depth sweep versus built Parquet; misleading/NaN reported durations remain audit-only | Passed |
| Are user/channel IDs opaque and all raw users retained? | Distinct uint64 neighbors, `0009`/`9`, `1e3`/`1000`, null-like strings, `006`/`6`, and a noncontext raw user | Passed |
| Can future/prehistoric mutations affect fitted features? | Independently establish equal clipped sets, then compare global-union cache clipping, exact CSR arrays, denominators, score vectors, channel discovery and feature fingerprints | Passed |
| Are calendar crossings handled without month/day reset errors? | Second-level independent oracle across leap day, month end, year end, Sunday/Monday; weekly and daily pooling; linear/nonlinear powers; single airings spanning seven/eight days | Passed |
| Does recency match in numerator and denominator, including partial bins? | Three repeated Tuesday bins with different observed/watched lengths, clipped first/last bins, and explicit full-bin-midpoint weights | Passed |
| Is duration normalization before airing power and program aggregation? | Occupancy `[1,1/3]`: hour mean `2/3`, 45-minute mean `7/9`, two airing contributions `1+(1/3)^p` | Passed |
| Is completion evidence ignored, with no artificial backfill/filter? | Change count CSR, completion arrays and descriptive metadata; known-channel cold programs can score; unsupported paths remain NaN | Passed |
| Are candidate bounds/order/duplicates and CPU memory boundaries respected? | Signed/uint64 bounds, raw-ID/index separation, 1,024 sparse users, 73 programs/219 airings, block size 7; fit densification/network and score I/O guards | Passed |
| Are source/cache identities checked rather than silently reused? | Reuse leaves all fixture bytes/stats unchanged; wrong source SHA, different-source binding, changed manifest/stat and same-stat cache-byte mutation are rejected | Passed |

## Actual cache inspection (read-only, executed)

The author's report, actual sidecar, Parquet footer/schema, file size, cache SHA,
source stat, and existing manifest binding were inspected. No real viewing rows,
labels, or predictions were decoded for this check; there was no real-data fit.

- Actual Parquet: **5,441,773 intervals**, **45 row groups**, **37,866,265 bytes**
  (37.866 MB decimal, not a process-memory measurement).
- Schema is exactly string `user_id`, string `channel_id`, int64 `start`, int64
  `end`. Cache SHA is
  `0a0533cdf5c1ceda6c2c6f07bb387b8e6a54d231d6a820876114fb1a5d0682b0` and
  matched a fresh read-only hash.
- Source stat and source-manifest SHA/entry binding matched. Source bytes are
  396,216,555; the recorded raw SHA is
  `8c9e5f058157041e5c33f13e64dc1625a5177cc6d5e434ba18cd8a69afafb53b`.
  The 396 MB raw file was **not rescanned** by this reviewer.
- Sidecar audit, not a newly rerun source scan: **5,821,987 raw rows**, **4,914
  users**, **202 channels**, **8,619,636,424 ununioned seconds** and
  **8,503,411,121 union seconds**, **0 invalid rows**, **117 duration mismatches**.
- All 1,260 target users are present according to the user and the author's
  executed full-ID scan. That scan was not repeated here. It does **not** mean
  the cache is target-only: its recorded population is all 4,914 raw users, and
  the synthetic challenge explicitly checks preservation of noncontext users.
- Stored construction source SHA is
  `63e08f42ed4f63f246dd0e9b252e3e89d05eebad0b6efedf20b0b5cfb005bef9`;
  source inspected at the start of this challenge is
  `c10952b6654e90cdebe64f4af5924e709d114adacd8176508b19b96c2b0be5ad`.
  These identify different snapshots, not a checksum failure: the cached data
  retain their original builder provenance, while a later implementation may
  read the unchanged format. The parent must retain both fingerprints.

## Findings and executed test results

**No implementation defect reproduced.** All **36 independent challenge tests
passed in 0.84 seconds**. Only the new owned challenge file ran; the author's
63-test suite and the parent's development grid were not executed by this reviewer.

Command from `midterm/` (the actual run used a new UUID-suffixed `--basetemp`
inside the approved OpenCode runtime temporary directory):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_research_raw_habits_challenge.py -q -p no:cacheprovider
```

Final fixture directory suffix: `raw-habits-challenge-cc15a518860c48b5bdc195e3f6e989ee`.
Environment: Python 3.13.6, DuckDB 1.5.6, PyArrow 24.0.0, NumPy 2.5.4,
SciPy 1.18.1, pytest 9.2.0. No dependencies were installed.

### Concrete expected/actual evidence

| Fixture | Expected | Actual executed result |
| --- | --- | --- |
| Nested/device/adjacent raw union | Long enclosing `[0,180)` plus adjacent `[180,240)` becomes `[0,240)`; a one-second gap remains separate; totals agree with an endpoint-depth sweep | Exact row/second agreement |
| Opaque user IDs | Separate occupancy for the uint64 maximum and its predecessor, `0009` and `9`, `1e3` and `1000`, `NA` and `null`; channel `006` differs from `6` | Exact IDs retained; each profile matches its independent watched-seconds/3,600 fraction |
| Future/past mutation commutation | All three raw variants have the same independently clipped union: 100 covered seconds in the 120-second history window. Only channel `c` may enter learned channels. | Exact clipped unions, CSR arrays, denominator arrays, scores and feature SHA; source SHA changes; behavior endpoints stay at the fixed history bounds |
| Recency in a partially observed repeating Tuesday hour | `(300*w_old + 900*w_middle + 600)/(600*w_old + 3600*w_middle + 1200)`; remove old terms with the fixed 10-day lookback. Current partial-bin weight is 1 because its full midpoint is after cutoff. | Both hand-derived ratios and explicit denominator match |
| Per-airing normalization/power | Hour `mean=2/3`; 45-minute `mean=7/9`; two separate airings give `1+(1/3)^p`, not `(4/3)^p` | All three fixed powers (1, 2, 0.5) match |
| Forecast longer than a whole period | Eight-day weekly mean `1/96`, daily mean `1/168`; seven-day mean `1/168` in either mode; two separate hour airings aggregate after power | Both modes and both fixed powers match |
| Bounded CPU scores | No sparse densification in fit; every dense score block is one user by at most seven cells; absent users remain NaN | Passed for linear and nonlinear storage, with 1,024 context users and 219 forecast airings |
| Integrity negative controls | Reject changed fixture source stat/manifest, same-size same-mtime mutated cache bytes, wrong source SHA, and unrelated-source reuse | All rejected; these are explicit synthetic negative controls, not observed production corruption |

The fixed power/slot variants above are **mathematical fixture coverage**, not
validation parameter selection or a quality search. No real-user recommendations,
held-out metrics, or prediction files were generated.

### Read-only/source fingerprint guard

The final run hashed the implementation, production cache and sidecar before and
after pytest, checked their byte sizes and nanosecond mtimes, and asserted exact
equality. All remained unchanged. Final fingerprints:

| Artifact | SHA-256 |
| --- | --- |
| New `tests/test_research_raw_habits_challenge.py` | `bf312cb60b43b4bfec8bfeb9b7a81b248abbd3f9442ee2738a217eee91204ed8` |
| Reviewed `research_v2/raw_habits.py` | `c10952b6654e90cdebe64f4af5924e709d114adacd8176508b19b96c2b0be5ad` |
| Existing production cache | `0a0533cdf5c1ceda6c2c6f07bb387b8e6a54d231d6a820876114fb1a5d0682b0` |
| Existing production sidecar | `ed63837ec1105bdba981c3e45d72e28de29cf4642681ca4edf421594d9cbd492` |

### Handoff for the implementation author's response

No correction or pause of the parent's source-frozen grid is requested: there is
no reproduced defect to fix. The author can acknowledge the 36 passing independent
checks and retain the construction/current-source fingerprint distinction and
limitations below. An author acknowledgment is still separate from this
independently executed challenge; this reviewer did not edit the author's report.

## Interpretation and limits

- Union/clip commutation is a set identity, but the tests must independently
  verify the premise (equal within-window coverage) and exact fitted invariance.
  Extending an untrimmed union end must not change recency; midpoint weights
  must depend only on the fixed history calendar.
- The denominator is weighted calendar time, not measured active-user exposure.
  The score sums powered duration-normalized mean-coverage proxies; it is not a
  probability of a jointly >=80%-completed broadcast or a causal recommendation
  effect. Sparse/unsupported paths must not receive implicit popularity votes.
- Four DuckDB threads and a 4GB DuckDB memory limit do not prove a process RSS
  bound. This challenge checks sparse storage and bounded score densification,
  not production peak-memory or latency. Python overlap storage still scales
  with retained forecast pieces; nonlinear scoring holds query-local airing
  indices/owners in addition to bounded dense blocks.
- Source size/mtime is deliberately trusted after the full build-time SHA.
  Same-size, same-mtime deliberate raw replacement is outside that stated guard;
  it is not misreported here as an unexpected discovered exploit. Synthetic
  integrity-negative tests are not claims that production data were corrupted.
- The original private **0.41001** is unchanged. This review does not execute or
  interpret the parent's planned two-development-window grid, select parameters,
  or establish any local/Kaggle quality improvement.
