# Best verified SWEET.TV solution: raw habits + legacy hybrid

**Private MAP@5 0.43462 / public 0.44743**, submission **57853036**, account
`dkiad4`. The exact scored file is
[`../submissions/research-v2/submission_raw_legacy_w05.csv`](../submissions/research-v2/submission_raw_legacy_w05.csv):
75,534 bytes, SHA-256
`7af8fe8fc6ce156d7783d621058a0657d9ab02aa95fa071bfc3df44764d19477`.
The [saved upload receipt](../submissions/research-v2/kaggle-test-001/kaggle-manifest.json)
records equality with the actual server-returned bytes. Its delivery flags describe
the upload-time checkpoint, not the later report state. No new upload is performed
by publishing or running this code.

## Model

1. Union **all raw pre-cutoff viewing intervals** per user/channel, including
   partial viewing; preserve opaque user IDs as strings. Fit only after clipping
   behavior to the training window.
2. Pool source-clock occupancy into **30-minute daily bins**, using a **56-day
   half-life** in both covered seconds and calendar-seconds denominators. There
   is no finite lookback. Source clocks are naive; no time-zone inference is made.
3. Integrate occupancy over each fully contained forecast airing, divide by its
   full duration, then sum by program ID (power1). Forecast EPG metadata are
   permitted; future viewing is not. Occupancy is an opportunity proxy, not a
   calibrated probability of 80% completion or personal exposure.
4. Use evidence-only RRF with **constant 60 / depth 200**, weights **raw 1 : legacy 0.5**.
   The unchanged legacy expert itself fuses repeat at 1 plus CBF/rules/implicit-MF/PPR
   at 0.5 each. No backfilled popularity votes enter either fusion. With five raw
   supported items, `0.5/61 < 1/65` prevents auxiliary-only discovery: this blend
   reranks raw evidence. One shared popularity/raw-ID backfill fills missing slots.

Previously watched programs remain eligible. All methods use the same temporal
windows, forecast candidates, users and unchanged >=80% union-coverage labels.
Standard AP@5 uses relevant-set membership and denominator `min(5, relevant_count)`.
The final file has 1,260 users in exact template order and five unique positive
eligible IDs per row; 655 target IDs above signed int64 remain exact strings.

## Recorded local comparison

| Train weeks / forecast weeks | Strict users | Original hybrid | Raw habits | Selected blend |
|---|---:|---:|---:|---:|
| 11–14 / 15–18 | 1,039 | 0.428017 | 0.441578 | **0.456333** |
| 11–22 / 23–26 | 677 | 0.372157 | 0.403619 | **0.407602** |
| 11–26 / 27–30 | 1,046 | 0.354822 | 0.372279 | **0.380940** |
| 11–18 / 19–30 | 1,199 | 0.400395 | 0.426931 | **0.436152** |

Primary all-1,260-user MAP is 0.386756→0.420731. Primary paired improvement is
+0.035758; the descriptive 5,000-user-bootstrap 95% interval is
[+0.028987,+0.042566]. These overlapping, adaptively reused historical windows
are **not untouched tests or independent replications**; the interval is not
selection-adjusted or calendar-level uncertainty. Local0.456333 is not a Kaggle
score. The shortlist was frozen before confirmation; label/nested-rank tie
sensitivities preserve gains over matched controls. Historical selection and
audit records are in `PROTOCOL.md`, `SELECTION.md`, and `../reviews/research-v2*.md`;
their referenced full local research artifacts are not all bundled publicly.

Final saved CPU timings on Apple M3 Pro: fit 4.47 s, score 34.94 s, shared rank 7.31 s.
These are single-run batch stages including constituent work, not online latency;
ETL/export are excluded and in-call lazy initialization is included. No GPU was
needed. Independent RRF rebuilds matched saved scores; separate raw refits had
different feature byte hashes of undiagnosed cause, so **the published scored CSV,
not a universal byte-identical-refit claim, fixes the actual result**.

## Reproduce locally

Obtain the official files under the competition's rules, without committing raw
data or credentials. Follow the root README's environment/download instructions.
Run the following from `midterm/` with its pinned environment:

```sh
# Validate official source files and build the normalized completion cache.
.venv/bin/python tools/prepare_official.py --cache data/processed/official-v1 --out results/preparation-new

# Build the separately source-bound raw-interval union and required .meta.json.
.venv/bin/python -m research_v2.raw_habits

# One exact frozen candidate; every output directory must be unused.
.venv/bin/python tools/run_research.py --grid configs/best-solution-grid.json --folds final --out results/best-final-new

# Optional historical replay; these are already-observed periods, not new tests.
.venv/bin/python tools/run_research.py --grid configs/best-solution-grid.json --folds confirmation primary12 --out results/best-validation-new
.venv/bin/python -I -S tools/verify_research.py results/best-validation-new --out results/best-validation-independent-new.json

# Original methods plus completed round-two regression tests; no raw data needed.
.venv/bin/python -B -m pytest -q
```

New final predictions are written to
`results/best-final-new/final/raw_legacy_w05/submission.csv`. Inference-only metric
fields have no held-out labels and are **not quality measurements**. Fits guard
source/cache bindings; RRF batches users with complete candidate lists, never
candidate-axis chunks. The canonical already-scored CSV is published separately.

`contracts.py`, `raw_habits.py`, `legacy.py`, `fusion.py`, `evaluate.py` and
`../tools/run_research.py` form the selected path. Other completed round-two
modules and synthetic tests retain the tested selection context; **later rounds
and unfinished fusion research are excluded**. Model sources are unchanged from
the scored source snapshot. Publication checks are in
[`../results/published/best/verification.json`](../results/published/best/verification.json).

## Attribution and delivery

The public [Gornyaki notebook](https://www.kaggle.com/code/kinahem/gornyaki-final-solution)
informed recency/schedule-frequency hypotheses only. No notebook code, predictions
or prejoined data were copied or executed; its time profiles were unused by its
own final ranker. The occupancy/forecast integration is independently implemented.
Source-only provenance is in `../reviews/research-v2-public-solution.md`; EASE's
completed comparison uses Steck (2019), DOI 10.1145/3308558.3313710.

The original 0.41001 scored CSV remains unchanged. Exactly four approved uploads
occurred overall; no further Kaggle submission or notebook publication is
authorized by these instructions. Reports, screenshots and native sessions stay
private. Genuine personal verification and oversized-log delivery agreement are
separate course-submission gates; GitHub publication does not imply LMS delivery.
