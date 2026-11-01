# Public solution review: Gornyaki_final_solution

Reviewed 2026-11-01. **Static source/metadata review only:** no notebook cells,
downloaded scripts, installs, outputs, prediction files, labels, or credentials
were executed or accessed. No performance experiment or Kaggle submission was
performed. The source is third-party, untrusted research material.

## Bottom line

The published final ranker is **filtered personal-repeat frequency × decayed
completion rating × historical EPG frequency**, with user-based cosine CF only
for short-list completion. It is not an ALS/PPR/association/content ensemble.
The most useful additions to our current count-repeat/RRF baseline are
**recency-aware repeat ranking** and **explicit schedule-opportunity features**.
The notebook does **not** establish a score above our user-reported best private
MAP@5 of **0.41001**. It contains no numeric score claim in cell source; its
validation code does not evaluate its final scoring formula.

## Provenance, version, and license

- Public reference: <https://www.kaggle.com/code/kinahem/gornyaki-final-solution>.
- Downloaded metadata: `id=kinahem/gornyaki-final-solution`, `id_no=12955147`,
  `title=Gornyaki_final_solution`, public Python notebook, competition source
  `sweettv-tv-program-recommender` (`kernel-metadata.json:2–8,17–19`).
- Parent-provided official competition-filtered listing: display author
  `MyhayloTurkin`, last run `2021-01-25`, 10 votes. These listing fields are not
  repeated in the downloaded metadata and were not independently refreshed here.
  One result in that listing is not proof that no other public solutions exist.
- Reviewed source:
  `midterm/.cache/research-v2/public-kernels/gornyaki/gornyaki-final-solution.ipynb`.
  SHA-256: `082b83ed9fc5eb81ec161b2e5fead818cb1b0a116648b8894867f2dfd66bfc1c`.
- Metadata SHA-256:
  `820705cabd164c5b87fd46431d363fa0b19cac9605184592227f496bc98c28e0`.
- Neither downloaded JSON exposes a notebook version number/session ID or
  license. `id_no` is the notebook ID, not a version. The notebook reports
  Python `3.6.4`, nbformat `4.4`; metadata pins a Kaggle Docker image digest,
  not a notebook revision. Treat this content hash as the exact review snapshot.
- **License not established.** Public visibility does not establish reuse rights.
  Prefer an independently written, attributed implementation of the ideas; verify
  the actual license before copying code. No source code or predictions are
  imported into our implementation by this review.

### Citation convention

The `.ipynb` is a single JSON line. `C46:L10–12` below means one-based cell 46,
lines 10–12 of its decoded `source` string. All 48 cell sources and notebook
metadata were inspected; cell outputs were not inspected. This is reproducible
by JSON parsing and numbering `cells[*].source`, without executing any source.

## Exact published pipeline

| Stage | Actual source behavior | Evidence |
| --- | --- | --- |
| Inputs | Reads a separate `/kaggle/input/sweet-data/big_data.csv` prejoined viewing/EPG table; reads the three official EPG archives and `submission.csv`. An exploratory branch separately reads `crop_data_1.csv`. It does not read or reconstruct official raw `dataset11-30.csv`. | C19:L1–8; C29:L1 |
| Catalog restriction | Drops schedule ID zero. Removes historical programs absent from the weeks 31–42 EPG from both the prejoined behavior table and historical EPG. No watched-item exclusion. Cold forecast-only programs have no personal/CF evidence. | C19:L10–16 |
| Time representation | Explicit naive timestamp parsing: prejoined timestamps `%Y-%m-%d %H:%M:%S`, EPG starts `%d.%m.%Y %H:%M:%S`. Duration is `(stop_time_tv-start_time_tv).dt.seconds`. | C20:L5–13 |
| Coverage | Calculates and clips each input row's `interception_time/duration` to `[0,1]`. Then sums intersection seconds by `(airing start, channel, user)` and keeps one row. Crucially, it does **not** recompute coverage after combining fragments. | C20:L14–17; C10:L3–11 |
| Metadata | Joins distinct `(program ID, category)` pairs from historical EPG. No title, genre, actor, or director similarity is used. | C20:L19–20 |
| Final event filter | Retains rows with `view_pct >= 0.8`, duration **strictly below 10,000 seconds** (2h 46m 40s), and category in `Развлечения`, `Инфо`, `Сериалы`, `Детям`, `Другое`, `Познавательное`. Thus excludes `Фильмы` and every other category, including missing categories. | C41:L1–7 |
| Final completion rating | `q(v)=5` when `v==1`; otherwise `q(v)=2+2v²`. Multiplies by `0.8**(age_days/60)`, where age is whole `.dt.days` since the latest airing start in the full historical `train_df`. | C40:L1–7 |
| Final personal score | For each `(user, program)`, computes count `n_ui` and **mean** decayed rating. Score is `n_ui * mean_rating_ui * log10(H_i)`, where `H_i` is the number of rows for that program in the filtered historical EPG. Equivalently, `log10(H_i) * sum_e[q(v_e)*0.8**(age_days_e/60)]`. It boosts frequent historical broadcasts, rather than dividing by availability. | C45:L1–5; C46:L1–12 |
| Unused future counts | Computes/merges `test_count` from weeks 31–42 EPG but never uses it in the final score. Despite C44's heading, there is **no train/test schedule ratio**. | C44:L1; C46:L2–12 |
| Final ranking/ties | Sorts by personal score descending, then `global_cnt` descending. Here `global_cnt` counts distinct user/program rows, i.e., distinct users for each retained program. No explicit tertiary raw-ID tie-break is supplied. | C46:L14–20 |
| Cold-user fallback | Global top five retained programs by row count. This fills a completely missing profile before CF, so a cold user normally receives popularity, not neighbors. | C12:L2–4; C43:L1; C47:L3–5 |
| Short-profile CF | Builds user × program **event-count** pivot; normalizes each program column by `(x-min)/(max-min)` across users; dense cosine similarities, diagonal zero. Takes the 10 largest-similarity users, the top 10 normalized program counts per neighbor, and sums these values by program to return up to 10 programs. Similarity values select neighbors but do **not** weight their contributions. | C15:L2–13; C16:L2–9 |
| List completion | Preserves the user's ranked repeats, appends CF programs, removes duplicates in first-occurrence order, truncates to five. No separate popularity backstop after CF; only a final length assertion. | C17:L3–7; C47:L8–9 |
| Serialization | Joins program IDs with spaces and writes `my_submission.csv`. This review did not execute that code or access its output. | C48:L1–2 |

### Validation and unused features are materially different

- Local split: July-started viewing rows are validation; **all non-July rows**
  are training (`C26:L1–2`). No explicit year, global cutoff, or airing-end check.
  Labels are the five most frequent `>=0.8` programs, but users with fewer than
  five distinct labels are discarded (`C26:L4–5`).
- Validation compares category-filtered frequency with/without duration filtering
  (`C34–38`). Its sorted-frequency variant uses personal count, then **global
  event count**, unlike final ranking's distinct-user tie-break (`C9:L5–14`).
- Validation computes a different rating: `5 if v==1 else 2+9v²`, decay
  `0.4**(age_days/60)` (`C33`). The tested count rankers do not consume that
  rating at all. The final rating, schedule multiplier, and CF completion are
  **not validated in the shown evaluation path**.
- Final decay's half-life is approximately **186.38 days**, not 60 days;
  validation decay's is **45.39 days**. These are arithmetic derivations from
  source constants, not measured results. The validation completion rating also
  jumps downward from nearly 11 to 5 at full completion; final rating jumps
  upward from nearly 4 to 5.
- `prepare_time_df` builds per-user weekday/hour count profiles and min-max
  scales each user's profile; `prepare_show_df` similarly scales program counts
  per user (`C27–31`). Neither output is consumed by validation or final ranking.
  There is no demonstrated schedule/time-of-week preference matcher.
- Source contains both a call to imported `ml_metrics.mapk` and a nonstandard
  custom metric (`C4–6`). The custom one compares **same-position equality**,
  then averages all prefix equality rates: it is not standard AP@5. A permutation
  of all five relevant IDs can receive a penalty despite standard AP@5 being 1.
  No numeric validation or leaderboard result is claimed in the source text.

## Risks: do not reproduce these as improvements

1. **Unknown matching provenance.** `big_data.csv` supplies already-computed
   intersections; no source here establishes its generation, date coverage,
   half-open interval semantics, overlap deduplication, or raw-log units. Metadata
   `dataset_sources` contains only an empty string. `crop_data_1.csv` is likewise
   unexplained. Do not download these intermediates or treat them as official raw
   data. Reconstruct from our official `dataset11-30.csv` (`stop_time`, `duraton`)
   and the actual EPG archives, preserving provided naive clocks and ID identity.
2. **Fragment bug and duplicate sensitivity.** `full_intercept` sums rather than
   unions fragments; grouping omits program ID/end time. Retained `view_pct` is
   the original first fragment's value. Two disjoint fragments covering 60% and
   25% can incorrectly fail 80%; overlapping rows can inflate summed seconds.
   Our union-per-airing reconstruction is the method to preserve, not replace.
3. **Duration/category/count artifacts.** `.dt.seconds` wraps whole days; it is
   not `total_seconds()`. A program with multiple categories can produce duplicate
   joined event rows. EPG row counts need not be unique broadcast opportunities.
   `log10(H_i)` gives zero to a program scheduled once. Hard category/duration
   filters can delete genuine future top-five programs.
4. **Temporal validity not established.** No explicit future viewing is read in
   the visible final path, and using supplied future **EPG metadata** is allowed.
   Nevertheless, an unexplained prejoined table cannot establish behavior-only
   weeks 11–30. `month != 7` would include later months/other years if present.
   Full-data aggregation before the split and row-start-based assignment do not
   enforce our airing-end cutoff. These are risks, not proof that the notebook
   actually used future behavior.
5. **Validation leakage/mismatch risks.** Final-forecast IDs are used to prune
   behavior before the July validation split. This is not hidden-label access,
   but it does not reproduce fold-specific candidate/history policies. The
   full-five-label cohort and untested final formula prevent score comparability.
6. **CF scalability and ties.** Dense pivots and an all-user cosine matrix are
   inappropriate for our bounded CPU pipeline. Constant program columns can
   yield division by zero. There is no positive-similarity requirement, and a
   zero diagonal does not strictly exclude self when zero-valued ties enter the
   top ten. Full ranking ties inherit pandas ordering; no explicit ID policy or
   pinned pandas version establishes a portable final tie-break.
7. **Output/ID safeguards are weaker.** `read_csv` has no explicit ID dtypes;
   there is no equivalent final positive-ID/catalog/template-order validation.
   Keep uint64-sized user identifiers as exact strings, never floats; retain
   our five-unique-positive-program and deterministic numeric-ID tie policies.

## Strongest implementable ideas and falsifiable tests

These are **proposed independent experiments**, not measured improvements or
claims of authorship of the published ideas. Preserve all four required models,
our labels, candidates, unioned coverage, and repeat eligibility.

| Priority | Experiment and mechanism relative to ours | Falsifiable test |
| --- | --- | --- |
| 1 | **Recency-weighted repeat expert.** Our `PersonalRepeatModel` uses unweighted qualifying counts (`sweettv/baselines.py:22–33`). Aggregate `sum_e 2**(-age_days_e/h)` over training-only completed airings. Start with completion weight 1; test the published final `q(v)` separately. This distinguishes persistent/current habits from stale repeats; ALS confidence and RRF do not restore discarded event times. | Compare no decay versus a small prespecified half-life grid such as 30/60/120/186 days on identical chronological folds and all users. Ablate completion weighting independently. Require consistent paired MAP@5 gains over plain repeat and our unchanged hybrid; otherwise reject. |
| 2 | **Schedule-opportunity reranking.** First ablate the source's actual historical `log10(H_i)` multiplier, preferably also a `log1p(H_i)` variant to avoid its one-airing zero. Separately test future airing count `F_i` and a shrunk availability ratio, e.g., decayed repeat count × `(F_i+a)/(H_i+a)`. The ratio is **our proposed extension, not notebook behavior**. It aims at expected repeated completions over the 12-week forecast rather than generic affinity. | Count unique nonquarantined airings, not metadata rows; use exactly each fold's known historical/forecast EPG. Compare no multiplier, historical boost, future boost, and ratio separately. Report gains by increased/decreased future availability. Assert that the source-equivalent score is invariant to future EPG multiplicities when its candidate set is unchanged; a true future-count extension should change. Reject a feature that only fits one cutoff. |
| 3 | **Frequency prior as a soft reranker, not a label rewrite.** Movies/very long programs may be less likely to dominate a frequency target. Test a soft category/duration prior before considering the source's hard exclusions. Keep training labels and evaluation relevance unchanged. | Isolate category and duration effects. Report all-user MAP@5 and losses on movie/long-program relevant IDs, plus coverage of previously strong repeat candidates. Do not select a method using only the full-five-label cohort. |
| 4 | **Sparse user-neighbor short-profile fallback.** Item-column-scaled count cosine supplies a different neighborhood from ALS/PPR. Reimplement with sparse/batched top-K neighbors, positive-similarity checks, explicit self exclusion, and evidence-only output. | Compare short-profile MAP@5 against our current popularity/ALS/PPR fallback, with identical main repeats. Ablate unweighted versus similarity-weighted neighbor voting. Reject if gains disappear outside a narrowly selected cohort or bounded memory cannot be maintained. |
| Later | **Time-slot compatibility with future EPG.** The source only constructs unused weekday/hour profiles. A new extension could sum supplied forecast airings weighted by a training-only user/channel/time-slot activity profile. | Compare against opportunity count alone; if time-slot affinity provides no incremental held-out gain, stop. Do not describe this as an implemented or validated published method. |

Implementation boundary: `TrainingData` currently carries count CSR and canonical
item metadata, not event-level times/coverage or airing opportunity histories
(`sweettv/contracts.py:17–23`; `sweettv/data.py:912–972`). Recency must be
aggregated from cutoff-filtered `PreparedData.events` into a separate feature
matrix; schedule counts must come from deduplicated fold-specific airings. Do
not silently replace count interactions for every model or infer recency from
canonical item metadata. Keep the existing repeat baseline as a control.

For every experiment:

- Use global cutoff boundaries and forecast-only schedule candidates; no behavior
  after cutoff may affect any fitted feature, normalization, or popularity prior.
  Assert training airing ends `<= cutoff` and mutate/delete post-cutoff viewing
  to verify fitted features and predictions are unchanged with fixed EPG.
- Use our standard membership-based AP@5 on the same full template-user cohort,
  including short/empty-label diagnostics; never import the custom source metric.
  Preserve identical top-five label construction and tie policies across models.
- Add hand-checked fixtures for fragmented 60%+25% viewing, duplicate overlaps,
  equal-score ties, one historical airing, duration over 24 hours, ID zero, and
  uint64-sized user IDs. Tests are proposed here, **not executed**.
- Evaluate the new repeat expert alone, then replace/add only its hybrid expert
  under controlled weights. RRF discards score magnitudes; a better expected-count
  signal may need a reranker, but adding an uncalibrated score sum is not justified.
  Prevent fallback popularity votes from being duplicated in fusion.
- Report per-fold absolute/paired MAP@5 deltas, runtime, memory, and user-level
  uncertainty; repeated model-selection folds are not an untouched test set.
  Only an authorized, actually scored submission can establish an improvement
  over the user-reported private **0.41001**.

**Checks performed for this review:** source/metadata JSON parsing, inspection of
48 cell sources, SHA-256 hashing of both downloaded files, source-only searches
for license/version/score/link claims, reading our relevant count/fold/ranking
code, and arithmetic conversion of the published decay constants. No model
training, metric evaluation, notebook execution, external intermediate download,
or prediction/label/output-file inspection was performed.
