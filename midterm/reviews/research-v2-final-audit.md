# Final round-two promotion audit

Date: 2026-11-01. Independent, read-only evidence audit; no further tuning.

## Verdict

**PASS for local validation promotion of the preselected `raw_legacy_w05`.**
The frozen standalone `raw_day_h56_p1` also passes, but the recorded development
decision already chooses the blend when it clears confirmation. That rule—not a
new choice using confirmation scores—supports the promoted candidate.

**This is not approval to upload, a verified final-fit artifact, or evidence of
a new private score above 0.41001.** I did not open, list, poll, or verify
`final-001`. The two final-fitting specifications were checked only in
`configs/research-v2-final-grid.json`; they exactly match the frozen candidates.
Only this NEW report was created. No source, tests, configurations, cache,
other reports, package materials, personal-verification state, exports, or remote
resources were changed. No API, credential, submission, push, or subagent was used.

## 1. Selection and freeze: supported by the actual receipts

| Gate | Independent evidence | Assessment |
| --- | --- | --- |
| Development-only selection | Inspected `PROTOCOL.md`, `SELECTION.md`, selector source and receipt inputs. The selector reads only `development` and `late_development`; all recorded selector/input hashes match current files. Reapplied its existing arithmetic and 0.001 simplicity rule to saved development results. | `raw_legacy_w05` is the recorded winner, unchanged. |
| Freeze precedes confirmation | Receipt `created_at`: **2026-10-31T23:11:29.996013+00:00**; its local birth time is 23:11:29.996405. Confirmation grid birth: **23:11:44.747916+00:00**; every completed per-trial confirmation/primary result was created later. | Local chronology supports the freeze; it is not tamper-proof external timestamping or a claim of previously unseen history. |
| Configuration identity | Shortlist and confirmation `grid.json` files are byte-identical. Each of ten executed result parameter dictionaries equals its frozen specification. The final grid is exactly the two-candidate subset, including component weights and omitted/default parameter choices. | No post-confirmation parameter change found. |
| Final grid created after confirmation | Final grid birth is 23:17:52.972263 UTC, after aggregate results at 23:16:45.789336. | This is an export of the identical frozen specifications, not evidence of retuning. |
| Failed/weak trials retained | Verified five grid sizes **14 + 8 + 14 + 5 + 6 = 47** and their **94** development-window CSV hashes against the five independent receipts. No unsuccessful family was removed from that record. | Four new standalone raw models pass; five of six blends beat frozen raw on both development windows/cohorts. The chosen blend follows the recorded rule. |
| Shared evaluation population | Every method has the same context fingerprint within its fold; independent event/label reconstruction and CSV checks confirm **1,199 primary**, **1,046 confirmation**, and **1,260 all-template users**. | No method-specific candidate/cohort filtering found. |
| Original reference parity | Replayed primary legacy CSV is byte-identical to `results/official-run-001/primary/submission_hybrid_balanced.csv`. | Exact prediction parity, not merely equal MAP. |

The development standalone selected by the recorded rule is `raw_day_h56_p1`,
with equal-window all-user mean **0.3342157186948853**. The chosen blend's saved
development mean is **0.34227854938271607**. These checks reproduce the existing
decision; they are not additional parameter comparisons or new trials.

## 2. What I independently recomputed

All audit commands were inline `.venv/bin/python -B` scripts, with no saved
scripts, new experiments, or production computational helpers imported as
oracles. Their full commands/results are retained in this audit's tool record.

1. **All ten completed CSV cases:** parsed 12,600 user rows, preserving string
   user IDs and exact template order. Checked the header, exactly five distinct
   positive program IDs, shared candidate eligibility, and equality to saved
   predictions. Recomputed every AP using standard-library `Fraction` arithmetic,
   then strict-five/all-user MAP, per-user metric alignment, paired deltas, and
   wins/losses/ties. All match the receipts; floating summaries differ at most by
   ordinary last-bit rounding, within the asserted `2e-14` tolerance.
2. **Labels and popularity without the production fold builder:** scanned the
   immutable prepared event Parquet using PyArrow, deduplicating user/airing
   identity and checking signatures. All **611,760** identities were unique.
   Recounted **237,136** primary training events and **461,409** confirmation
   training events, plus held-out frequencies for all template users. The saved
   labels match exactly. This is independent aggregation of prepared events,
   **not a new raw-log-to-EPG reconstruction**.
3. **R5 on actual saved scores:** read only the ten completed trials' explicitly
   referenced score matrices, using read-only NumPy memory maps. Independently
   ranked finite evidence before one-time popularity fallback under ascending
   and reversed final-score ties. Recomputed original/reversed label results and
   exact without-replacement boundary-tie expected AP. All recorded sensitivities
   match; ascending reconstructions reproduce every saved prediction.
4. **Nested fusion arithmetic:** independently rebuilt the fixed RRF60/depth200,
   weights **1:0.5**, from separately fitted raw and legacy saved component scores.
   Checked ordinary and reversed nested-constituent controls in both windows.
   For all **4 × 1,260 user vectors**, finite fused scores agree **exactly**, with
   identical finite/NaN masks and no evidence-top-five difference. This checks
   the actual cached blend, not just its declared configuration. It does not make
   claims about NaN payload bytes or bytewise identity of all learned features.
5. **Eight paired bootstrap intervals:** independently generated the fixed
   5,000 user-resampling replicates with seed `20261101`, starting from the exact
   CSV-derived AP deltas. All eight interval endpoints reproduce within `2e-14`.
6. **Retained-trial evidence:** all **104 CSV hashes** across the five development
   receipts plus the confirmation receipt match their files. I independently
   recomputed AP for the ten confirmation/primary cases, not all 94 development
   cases; the latter's metric recount remains attributed to their existing
   standard-library independent-verification receipts.
7. **Test/source receipt:** parsed the actual JUnit XML and verified **1,352 tests,
   zero failures/errors/skips**, dependency-check success and unchanged-source
   status. All **76** recorded source/test/tool hashes still match. This audits
   the parent's executed full-suite receipt; I did not rerun that suite here.

## 3. Confirmed accuracy on the fixed populations

| Window / method | Strict-five MAP@5 | All-1,260 MAP@5 | Strict delta vs legacy |
| --- | ---: | ---: | ---: |
| Primary12, legacy | 0.400394773 | 0.386755732 | — |
| Primary12, frozen raw | 0.426930776 | 0.411215608 | +0.026536002 |
| Primary12, **chosen blend** | **0.436152349** | **0.420731481** | **+0.035757576** |
| Confirmation, legacy | 0.354821542 | 0.323143959 | — |
| Confirmation, frozen raw | 0.372278521 | 0.338396605 | +0.017456979 |
| Confirmation, **chosen blend** | **0.380940089** | **0.346781966** | **+0.026118547** |

Primary strict-five has 1,199 users; confirmation has 1,046. The all-user metrics
include respectively **20/44 empty-label** users and all short-label users;
61/214 users fall outside the respective strict-five subsets. Every method uses
the same subset within its window and the same 7,045/3,957 candidates.

Exact rational checks for the chosen blend:

- Primary strict MAP = `39221/89925`; paired gain = **`59/1650`**.
  Strict wins/losses/ties: **531 / 263 / 405**.
- Confirmation strict MAP = `119539/313800`; paired gain = **`683/26150`**.
  Strict wins/losses/ties: **411 / 239 / 396**.
- Chosen-blend all-user gains are **+0.033975750 / +0.023638007**, respectively.

The primary strict descriptive interval is
**[0.028987489574645538, 0.04256602724492633]**; confirmation strict is
**[0.019949012109623965, 0.0325540949649458]**. These are **post-selection,
reused-history user-resampling descriptors**, not unseen-test confidence,
selection-adjusted significance, calendar-level uncertainty, or a forecast of
a Kaggle score. Confirmation weeks 27–30 overlap original primary weeks 19–30;
they are not independent test replications. All history has previously been
observed, including during the earlier work.

## 4. R5 is now an executed, independently recounted control

The ascending-ID official/local metric is unchanged. The counterfactual
direction was not selected for promotion. Below are **chosen-blend minus matched
legacy** strict-five deltas:

| Fixed sensitivity | Primary12 delta | Confirmation delta |
| --- | ---: | ---: |
| Original labels and ordinary constituent ranks | +0.035757576 | +0.026118547 |
| Reversed label boundary ties | +0.034259105 | +0.023224984 |
| Boundary-tie marginalized labels | +0.035843437 | +0.025581408 |
| Reversed nested constituent RRF/top-depth ties | +0.037784265 | +0.028078394 |
| Reversed nested constituents + reversed labels | +0.033402836 | +0.023228171 |
| Reversed nested constituents + marginalized labels | +0.035870012 | +0.025922022 |

Reversing final-score/fallback ties independently leaves these aggregate AP
values unchanged in the audited runs; that does not imply that no tied scores
exist. All six independently recomputed tie comparisons are positive for frozen
raw as well. Nested reversals use the corresponding reversed-constituent legacy
control, not an unmatched baseline. Source inspection confirms reversal occurs
before constituent depth truncation at both fusion levels; the saved-score RRF
rebuild corroborates the actual execution. **The earlier R5 promotion gate is
closed for these measured controls**, not universally for every possible tie rule.

## 5. Model semantics and temporal isolation

The promoted raw expert uses the original source-clock channel-viewing intervals,
including partial viewing. It unions per user/channel, clips to
`[history_start, train_end)` **before** channel selection, bin expansion,
weighting, and aggregation, and uses only context users. Supplied forecast EPG
is permitted static metadata; held-out viewing/labels do not fit the occupancy
features. Both raw standalone and blend receipts report maximum behavioral ends
equal to their cutoffs: **1588550400 / 1593388800**. The legacy part uses the
unchanged pre-cutoff qualifying-broadcast counts and original methods.

The exact frozen raw configuration is daily pooling of **30-minute source-clock
bins**, **56-day half-life**, no finite lookback, and **coverage power 1**. The same
calendar-bin recency weights apply to numerator and calendar-seconds denominator.
It integrates occupancy over each broadcast, divides by its full duration, then
sums per program. It is **not a calibrated >=80% completion probability, measured
personal exposure, or causal preference estimate**. Completion labels and their
>=80% union-coverage reconstruction have not been changed to favor this feature.

The blend is evidence-only RRF60/depth200 of raw weight 1 and the original hybrid
weight 0.5. Its constituent score matrices contain unsupported NaNs; no shared
popularity backfill votes are imported into the fusion. Shared fallback runs once.
This is deliberately a **reranker**, not unrestricted auxiliary discovery:
`0.5/61 < 1/65`, so an auxiliary-only item cannot enter the top five when five
raw-supported items exist. Raw schedule transfer itself can support previously
unwatched programs; the blend does not exclude watched programs.

### Reproducibility caveat actually encountered

An audit-only assertion that separate same-config raw fits must have identical
`training_feature_sha256` failed in both windows. The standalone and blend raw
receipts have identical parameters, source/cache bindings, windows, interval and
second counts, occupancy dimensions/nnz, and recorded minimum/maximum occupancy,
but different feature byte hashes. Parallel floating aggregation is a plausible
explanation, **not a diagnosed cause**; the learned matrices were not compared.

This is not evidence of retuning or an established model defect. The subsequent
independent fixed-RRF reconstruction matches **all finite fused scores and NaN
masks exactly**, ordinary and nested-reversed, for every audited user. Thus this
fingerprint difference had no observed effect on the cached fusion evidence.
Do not claim general bytewise-identical refits; preserve each run's actual
fingerprints and the exact ultimately approved prediction file. No source was
changed or new fit performed to suppress the discrepancy.

Other open risks remain: organizer-equivalent reconstruction/ties and source
timezone are not fully known; calendar coverage/activity changes; occupancy is a
proxy; 20-week final training differs from the shorter historical fits; and the
final twelve-week horizon can drift. None is resolved by a positive bootstrap
interval or 1,352 passing correctness tests.

## 6. Before one freshly approved Kaggle test

1. Let the parent's frozen final fits finish. **This audit has not inspected
   their artifacts.** Verify successful completion and exact source/configuration
   bindings, final training cutoff **1595808000** and supplied forecast end
   **1603065600**, with no future viewing fit and no post-confirmation retuning.
2. Independently validate the chosen blend's final CSV: exact 1,260 template
   users/order as strings, exactly five unique positive eligible forecast IDs,
   required header/spacing, no missing users or duplicates, and unchanged repeat
   eligibility. Record its exact bytes/SHA-256 and actual run fingerprints.
3. Preserve the original scored file; use a separate immutable new artifact.
   The unchanged original `submissions/submission.csv` was independently hashed
   to **`6dd285833c5602b699ab2e0d510ac75acb49addc410de59916bb859359107d79`**.
4. Obtain fresh explicit approval for **one identified frozen CSV**. Fitting both
   passing candidates does not authorize two uploads or further adaptive private
   queries. Preserve the eventual genuine score/receipt without treating it as
   permission for another tuning loop.

The six independent receipts and test-verification record state zero new
submissions; all ten completed result records have null Kaggle score and
`submitted=false`. The promotion receipt also explicitly denies new-submission
authorization. I made no API query: this is a local evidence audit, not an
independent inspection of remote account activity. The only established scored
reference here remains **57851385 / private 0.41001**. No new Kaggle score is known
or claimed, and course-package/personal-verification work is outside this audit.

## 7. Critical hashes checked against actual files

All paths below are relative to `midterm/`. `R/` abbreviates
`results/research-v2/`; repeated critical receipts were unchanged on recheck.

```text
research_v2/PROTOCOL.md                 d92047c7ac0bb96cf47c2587a8fd212c9f1406eb1d589ebffbedbe4cc606201d
research_v2/SELECTION.md                a3f09063f4a8d8bedd0777461846807ad4dba31c2728b533c537418e6a289102
R/shortlist-001/selection.json          f5a5ad0fd73bd34a33c12a63c611f6cc16bec7d8eb321ec1c611f5d6e656e63b
R/shortlist-001/grid.json               4e3fca45526b28023c05ad49c87d8385ac32427c1a25781117e740d9c8843795
R/confirmation-001/grid.json            4e3fca45526b28023c05ad49c87d8385ac32427c1a25781117e740d9c8843795
configs/research-v2-final-grid.json     7ad5268aa1a914e63e2d99112740b9a361d673f96593ad7693a3e1369aea2729
R/confirmation-001/results.json         0572ec4d9315052d1dd4356ea4e66e687bb5c44a2ccbfc1ea777fdcd5a3266cc
R/primary12-paired-001.json             0c5d95282dbf7113d8842ea4e3fc7cd155238ff674a694bbb83e9dd43e6e1bce
R/confirmation-paired-001.json          8451fb6fe1057773a9f32b3ff22dea9d7d96cd272fa1cb8b033683b780480b66
R/primary12-tie-sensitivity-001.json    54c59a1dd7c2cac73c93a577c0fd2d9510346cccb18ba2981848cddc693d34cd
R/confirmation-tie-sensitivity-001.json e0ebd4746ce384972b063014883a99fc556c2fa8120dd1acc0e5cfc4e7666380
R/promotion-001.json                    9f9581f3bee9a1a97c599471285ccb68fa23245332d8690e495996c825e000ca
R/verification-001/verification.json    afb667f982638a78b674403c9701598d09a3a2b747051cda5e6f487c3474903d
R/verification-001/pytest.xml           9f7010767a50a9bffc30dd8410d23048aca93c10beb7cee51b059ec36edd58f9

research_v2/raw_habits.py               c10952b6654e90cdebe64f4af5924e709d114adacd8176508b19b96c2b0be5ad
research_v2/fusion.py                   5b3d2d4db214b45c9cd7cbf7e277418dd5173297ea1a6b44709cdaa630ef923d
research_v2/legacy.py                   077d84e2e40c467324c00e06ae0c2ab32a67ce95182f5cceba602752a7af1f33
research_v2/contracts.py                9202448a3b0fb30a8e902112fcbdb825ea78e5901a50744b50e1d5d3aaa9c7e0
research_v2/evaluate.py                 fa5d43eae0269be9f2b635e9ad28c195225395a417cf455bb9292fad130d274c
tools/select_research_shortlist.py      d9a69417b50d2783ec65585bfe5e0e4aacd40a4c7b788f62739f10d3657827a9
tools/research_tie_sensitivity.py       451ef7aa1dd4c0348ad00a86119beea503d1652708fcd28516b3dc1ee3e8fb2d
tools/compare_research.py               414e016042980cf46bb999e17b72d0054a5c34126e0b7ba7f26b6c7e12f0f005
```

The five retained-development receipts and separate completed-confirmation
receipt were all verified, including every recorded CSV hash:

```text
R/independent-priors-001.json                 8e76f08df04d77a3f567413b5864336cdf93808c0b95116e7209e2634ab008b8
R/independent-availability-001.json           1940c7bc3a461bfa1d0f15e0b379906575852b2ce46172e1cd71f8a5626347e1
R/independent-collaborative-schedule-001.json ba5eb60dd70558d46407335c3c17e9ff69489a60d638a212da61711217900b51
R/independent-raw-habits-001.json             573af2754ddca4d5a1620037552ca98235043af1a6f7bbe79498e2c8daa8346a
R/independent-fusion-001.json                 fe052112476a332bf2346b6b1cdd221daccbca353918059fbcdd04e288e3e660
R/independent-confirmation-001.json           01db73d6da3c95b335ae705551f6e3b2e1ec3a2363182701f08d4f2c1236cd17
```

Selected evidence bytes, not final-fitting artifacts:

| Artifact | SHA-256 |
| --- | --- |
| Original/replayed primary legacy CSV | `dd1cfcf06a53ff53216fb437e1aa8997de1222fe6cc527784751939ed186dc13` |
| Primary chosen-blend CSV | `598cac13cb16083eb7dc94fd9ef9b4657887d0a250fbf3b202b3c93238edb1c2` |
| Confirmation chosen-blend CSV | `3eeeae42286884ab420144c59c029d5bbf5e20b4a90f1f82b569babf2e6bd6ab` |
| Primary raw CSV | `534528c9d75a96ffb529b8e2b88abb8de71e3f1544f4953376111d2816cb8cea` |
| Confirmation raw CSV | `52bbc1017e628150dbacc9588c1d31ace55f33575725ce929be84b6c5a604a47` |
| Prepared event Parquet, independently recounted | `9ce9b9c8b38311a1f00a676cc92fd1984e05ae423e404a89a034ac89a98bf918` |
| Primary chosen-blend score matrix | `c4b913e88704d06b47c83e23142daafd71c46039b8878e4c86d706367855beec` |
| Confirmation chosen-blend score matrix | `65856471f31e1fde9f627e2e78af9a7882f2a96e0cd41d68aad587126516b050` |
| Primary nested-reversed blend score matrix | `a2a9d07a1b9044f88ecd2074b5afa2189c0d51e10f23fe7e21fd72e6796f47e0` |
| Confirmation nested-reversed blend score matrix | `e10e14295c1b3b9cc5b13b65b9f33ddcafc0f85b11bcb514dfc0bfba2771f7f9` |

No promotion-evidence integrity failure was found. The justified next decision is
an explicitly approved test of the **same frozen blend after final-file checks**,
not additional tuning or a claim that the private-score objective is already met.
