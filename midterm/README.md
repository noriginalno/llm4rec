# SWEET.TV: four independent Python recommendation models

**Best verified solution:** `raw_legacy_w05`, submission **57853036**, returned
**private MAP@5 0.43462 / public 0.44743** for all **1,260 target users**.
The exact scored CSV is
[`submissions/research-v2/submission_raw_legacy_w05.csv`](submissions/research-v2/submission_raw_legacy_w05.csv),
SHA-256 `7af8fe8fc6ce156d7783d621058a0657d9ab02aa95fa071bfc3df44764d19477`.
See the [best-solution implementation and reproduction guide](research_v2/README.md)
and [compact result record](results/published/best/summary.json).

The selected model combines raw channel/time occupancy with the original hybrid
using evidence-only RRF60, depth 200 and weights 1:0.5. Four approved uploads occurred:
three original candidates plus one separately approved fixed file. Publishing
this implementation makes no new Kaggle submission. The original scored CSV
[`submissions/submission.csv`](submissions/submission.csv), private 0.41001, is
preserved rather than overwritten. Raw data, private AI logs and report drafts
are excluded. The private course package still needs genuine personal
verification and agreement on oversized-log delivery; a real score screenshot
has been supplied separately.

The four independent implementations are content-based filtering, directed
association rules, implicit-feedback ALS, and Personalized PageRank. Popularity
and personal-repeat frequency are reference methods. Their authors cross-review
each other and switch author/challenger roles; reports and reproductions live in
`reviews/` and `tests/test_*_adversarial.py`.

## Results and files

| Method | Primary validation MAP@5 |
|---|---:|
| Popularity | 0.323150 |
| Personal-repeat reference | 0.386950 |
| CBF | 0.317320 |
| Association rules | 0.278913 |
| Implicit MF | 0.201760 |
| Personalized PageRank | 0.395179 |
| Original hybrid | 0.400395 |
| Raw channel/time habits | 0.426931 |
| Selected raw-aware hybrid | **0.436152** |

All methods use the same **1,199** strict-five-label validation users; diagnostics
also include all 1,260 template users. The descriptive paired-user 95% interval
for hybrid minus PPR is **[-0.002994,+0.013320]**: the point estimate is higher,
but this does not establish a statistically decisive or independent-test gain.
Reversing the unspecified fifth-label ID tie-break leaves the hybrid highest;
absolute scores change. This original comparison predates the raw-aware extension.
The extension's local historical selection and limitations are documented in
[`research_v2/README.md`](research_v2/README.md); local MAP is not a Kaggle score.

| Approved Kaggle candidate | Submission | Public | Private |
|---|---:|---:|---:|
| Hybrid | 57851385 | 0.40590 | **0.41001** |
| PPR | 57851387 | 0.39637 | 0.38258 |
| Personal repeat | 57851386 | 0.39756 | 0.38950 |
| Selected raw-aware hybrid | 57853036 | 0.44743 | **0.43462** |

The original three and the later fixed-file fourth upload were separately
authorized. Choosing among returned scores is not an unbiased estimate or evidence of
business impact. Full local runs are under `results/official-run-001/`; the
public snapshot contains small summaries and exact final-fit CSVs in
`submissions/`, not restricted raw data or private AI-session exports.

The [aggregate summary](results/published/summary.json) and
[two-user recommendations](results/published/two_user_recommendations.csv) are
included in the published snapshot.

Individual required-method files: `submission_cbf.csv`,
`submission_association.csv`, `submission_mf.csv`, and `submission_ppr.csv`
inside `submissions/`. Baseline/hybrid files and the scored-file manifest are
retained alongside them.

## Environment

From the `llm4rec` repository root:

```sh
python3.13 -m venv midterm/.venv
midterm/.venv/bin/python -m pip install -r midterm/requirements.lock
```

Run the remaining commands **inside `llm4rec/midterm/`**. The lock records the
installed environment; it is a version lock, not a cross-platform hash lock.
Native ALS requires a matching `implicit` CPU wheel/runtime. CPU is the default
on Apple Silicon; its CUDA backend does not use the integrated Apple GPU.
No GPU training or acceleration is claimed. If resource measurements justify an
MPS implementation later, it requires its own numerical verification.

Actual preparation reconstructed **611,760** qualifying events from **5,821,987**
logs and **381,892** positive-ID airings in 119.46 seconds, peaking near 3.9GiB
on the M3 Pro. Validated Parquet reload took 9.87 seconds. GPU was unnecessary.

## Authenticate and obtain the official files

Use your existing account and inspect the consent screen. Do not paste secrets
into a chat or repository. This wrapper isolates credential storage in the
workspace's protected runtime HOME (or ignored `.cache/kaggle-home` in a
standalone clone) and suppresses authorization-code HTTP logs:

```sh
.venv/bin/python tools/kaggle_local.py auth login
.venv/bin/python tools/kaggle_local.py competitions files sweettv-tv-program-recommender
.venv/bin/python tools/download_data.py
```

If browser downloads work but the API is denied, an official browser-downloaded
ZIP can be imported without another API call:

```sh
.venv/bin/python tools/download_data.py --archive /absolute/path/to/sweettv-tv-program-recommender.zip
```

This records browser-archive acquisition separately from a CLI download and
keeps the source ZIP untouched. Never submit a dummy CSV just to obtain data.

Accept competition rules yourself in the browser if required. The script does
not accept rules, authenticate automatically, or submit predictions. The OpenCode
profile also includes the official remote Kaggle MCP; `/mcps` handles its separate
OAuth consent. CLI and MCP credentials are not assumed interchangeable.

Raw files and credentials must not be committed. `download_data.py` preserves
the ZIP, does not overwrite existing CSVs, extracts only expected filenames,
and writes a size/SHA-256 source manifest under ignored `data/raw/`.

The authenticated file listing exposed a discrepancy with the old competition
description: live files are `dataset11-30.csv`, `export_arh_11-20-final.csv`,
`export_arh_21-30-final.csv`, `export_arh_31-42-final.csv`, and `submission.csv`.
`configs/source-files.json` records that exact explicit mapping; source files
are not renamed. The CLI uses it by default; `--file-map` can select another
explicit mapping. Library adapters retain their logical historical role names.
File ranges are not a substitute for inspecting actual timestamps.

Inspect bounded samples before choosing adapters or time zones:

```sh
.venv/bin/python tools/inspect_data.py --raw-dir data/raw --out results/raw-sample.json
.venv/bin/python -m sweettv.cli audit --raw-dir data/raw --out results/raw-audit.json --log-end stop_time --log-duration duraton --timestamp-formats configs/timestamp-formats.json --assume-timezone UTC
```

Actual logs use `start_time`, `stop_time`, and the typo `duraton`; the old
description's `end_time`/`duration` names do not match. Actual EPG date text uses
`%d.%m.%Y %H:%M:%S`, while logs use ISO-style text. These are explicit adapters,
not fuzzy column/date guessing. Strict UTF-8 is default; use `--encoding cp1251`
only when justified by source inspection. Both sources have naive timestamps:
the explicit UTC assumption above is a **common wall-clock numeric anchor**, not
evidence of an original UTC time zone; no guessed Kyiv/DST conversion is applied.

## Data and evaluation contract

See `CONTRACT.md` and `reviews/data-contract.md` for detailed semantics.

- Match viewing channel to EPG channel and intersect half-open intervals.
- Drop zero/missing program IDs. Union fragments within each airing; qualify at
  exact coverage >=80%, count once, then sum repeat airings by user/program.
- Audit schedule conflicts and exclusions. Organizer fragment/conflict/ID/tie
  details not documented in full remain explicit proxy assumptions.
- Train from fully completed airings before the global cutoff. Held-out labels
  use fully contained forecast airings. Keep watched programs eligible.
- Use the same forecast-schedule candidate catalog and template-user cohorts for
  every method; provided forecast metadata is permitted, future viewing is not.
- Primary evaluation conditions on at least five distinct held-out qualifying
  programs to resemble the hidden strict-five target. Report all template users
  and sparse-history users separately, including cold starts and empty labels.
- Ground truth is the five most frequent held-out program IDs, with ascending
  raw-ID tie-break as an engineering assumption. MAP@5 is membership-based AP,
  normalized by min(5, relevant-label count), never recommendation hits.
- Rank finite evidence by score then numeric raw ID. NaN means absent evidence;
  backfill once by training popularity, then raw ID. Record fallback positions.
- MF learns binary affinity with count-weighted confidence, not viewing-count
  regression. Rule support describes bounded user-history baskets, not airings.

## Run the complete benchmark

Create a JSON protocol only **after** reconciling actual timestamps and filename
week semantics. It must contain `development`, `primary`, and `final`, each with
integer epoch-second `train_end`, `forecast_start`, and `forecast_end` fields.
The proposed windows are 11-14 -> 15-18, 11-18 -> 19-30, and 11-30 -> 31-42;
they are not automatically inferred from a presumed calendar year.

```sh
.venv/bin/python -m sweettv.cli run --raw-dir data/raw --protocol configs/audited-protocol.json --confirm-protocol --out results/official-run-001 --log-end stop_time --log-duration duraton --timestamp-formats configs/timestamp-formats.json --assume-timezone UTC
```

Each output directory must be new; runs do not overwrite existing evidence.

For the exact one-time preparation/validated Parquet-cache workflow used here:

```sh
.venv/bin/python tools/prepare_official.py --cache data/processed/official-v1 --out results/preparation-new
.venv/bin/python tools/prepare_official.py --cache data/processed/official-v1 --out results/cache-load-new --protocol configs/audited-protocol.json --run-out results/official-run-new
```

The first command verifies source bytes against the acquisition manifest and
binds the cache to preprocessing/configuration/source-stat state. The second
validates every cached payload before fitting. Use fresh output/cache names if
an earlier artifact already exists or preprocessing changes. Full local source
and two-user recounts are optional additional evidence; inspect their `--help`.
Outside this course workspace, pass `--temp-parent .cache` to the raw recount
after creating that local temporary directory.

Optional `--parameters` supplies model constructor parameters as a JSON object
keyed by `cbf`, `association`, `mf`, `ppr`, `repeat`, or `popularity`. Freeze those
parameters before primary comparison. Two predefined evidence-only fusion weights
are compared on development; primary MAP selects the final predictor, with exact
ties preferring repeat then popularity. This is model selection, **not** an
untouched test, statistical significance claim, or returned Kaggle score.

Each fold writes:

- `submission_cbf.csv`, `submission_association.csv`, `submission_mf.csv`,
  `submission_ppr.csv`, both reference CSVs, and applicable hybrid CSVs.
- `recommendations.csv`: ranks, IDs, titles, categories, evidence/backfill source,
  scores, and held-out relevance annotations attached only after ranking.
- `metrics.json`, per-method predictions, labels and exact evaluation users.
- Candidate IDs, source/split fingerprints, dimensions, timings, and PPR
  convergence diagnostics; any model error fails rather than silently replacing it.
- The same two training-selected example users across every method, with their
  training histories and the selection rule.

`final/submission.csv` is an exact copy of the primary-selected method's final-fit
CSV. Every target user is preserved in template order as a string and receives
exactly five unique positive candidate IDs. No automatic upload is performed.

## Verification and synthetic demonstration

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m sweettv.cli demo --out results/synthetic-demo-new --users 72
.venv/bin/python tools/record_verification.py --out results/verification-new
```

The demo generates its own clearly marked raw data in a separate sibling
directory. It exercises reconstruction, all four models, both references,
development selection, primary comparison, final fitting, and CSV validation.
Its measurements establish integration behavior only—not competition quality.

The independent stdlib oracle imports no production parser, metric, or ranker:

```sh
python3 -I -S tools/independent_verify.py --submission results/official-run-001/final/submission.csv --template data/raw/submission.csv --allowed-ids results/official-run-001/final/candidate_ids.json
```

It can also independently recompute metrics using `--predictions-json`,
`--labels-json`, and `--users-json`; supply all three together. Mathematical
fixtures include the presentation's AP examples, exact threshold boundaries,
PPR linear-system solutions, independent ALS dot products, and rule recounts.

To independently verify all per-model CSVs/metrics in a complete run and the
selected file's byte identity:

```sh
python3 -I -S tools/verify_run.py --run results/official-run-001 --template data/raw/submission.csv
```

The first synthetic run incorrectly charged all model-module import time to
popularity's fit. Current timing separates initialization/import from fitting
and scoring. Versioned historical results are preserved rather than relabeled.

## Delivery boundaries

The course requires a PDF (<=5 pages excluding references), GitHub code and the
exact scored CSV, the actual highest returned private-score screenshot for the
designated final submission, and an original unmodified native AI-session export.
A genuine user-performed check must be documented, not invented by an agent.
The current best code and scored CSV are included here; the report, screenshot,
and native transcripts remain in the private local delivery package. No LMS
submission or completed personal verification is claimed.

Kaggle uploads, Git commits/pushes, and publication need separate approval.
Use your real Russian full name and Kaggle name in the submission description,
verify remaining quota and the deadline time zone, and retain the file hash and
submission ID. Do not substitute a public score for the required private score.

## Sources

- [Midterm Quiz](https://docs.google.com/presentation/d/1xf9tJ0nU0L9GeSGxLQVT_-GzZ160rDSXKRVH5TD6MSE/htmlpresent).
- [SWEET.TV competition](https://www.kaggle.com/competitions/sweettv-tv-program-recommender):
  data description and MAP@5 evaluation retrieved through its official MCP.
- [Official implicit CPU ALS API](https://benfred.github.io/implicit/api/models/cpu/als.html).
- [scikit-learn feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html).
- Official Kaggle CLI skill pinned at commit
  `2a007074fa8a0a191fbcc95ed222e1a4a2d442ba`; official Kaggle MCP, not a third-party
  credential-handling server. The project-local skill records safeguards.
