# Verification status

## Established locally

Four independent Python model implementations are present. Authors challenged
their partners and switched roles: CBF/association rules and MF/PPR. Original
challenge reports and adversarial tests are preserved; additive author responses
record reproductions, fixes, and rechecks. A separate agent reviewed shared
reconstruction, ranking, metrics, output integrity and integration.

The current final implementation record is
`results/verification-2026-11-01-final/verification.json` (earlier checkpoints
are retained separately):

- **667 tests passed**, zero failed/errors/skipped, in one executed full suite.
- Dependency consistency check passed.
- Source hashes were recorded and unchanged during checks.
- Tests include independent PPR linear solves, native ALS normal-equation/dot
  checks, exhaustive rule counts, Unicode CBF fixtures, interval-union oracles,
  temporal isolation, raw-ID preservation, MAP arithmetic and CSV validation.
- Fault-injection defenses are identified as such; they are not invented bugs
  attributed to model authors or evidence of an actual security breach.

## Synthetic integration evidence

`results/synthetic-integration-v1/` and `v2/` retain executed 72-user, all-model,
three-stage runs. Independent stdlib verification of v2 checked all **22**
per-model CSV/metric cases plus exact selected-final-file byte identity. Its
report is `results/synthetic-integration-v2/independent-verification.json`.

These are explicitly synthetic inputs and scores. They establish integration
and arithmetic, **not** SWEET.TV recommendation quality. Historical runs precede
some later coordinator hardening; they are not relabeled as current-source runs.

## Actual-data execution and Kaggle result

Official CLI OAuth completed successfully. The authenticated live file list
differs from the old competition description; `configs/source-files.json` records
the observed mapping and the library validates explicit filename aliases.

Initial403 responses were followed by successful authenticated download. The
CLI's accept-rules hint is generic and the SDK's user_has_entered can default to
false; neither established the cause. The final cause was not isolated.

Actual strict source scans and the production parser agree on source counts and
formats. The source has 5,821,987 logs, 4,914 users and all1,260 target users;
655 target IDs exceed signedint64 and remain strings. Explicit filename, column
and timestamp-format adapters preserve original raw files. There are611,760
qualifying user/airing events. Preparation took119.46s, peak~3.9GiB; validated
Parquet reload took9.87s. Both naive clocks share an explicitly stated numeric
anchor; no observed UTC offset is claimed.

All four required-method models plus both references completed development,
primary validation and final fitting on the official data. The separate
stdlib oracle checked all22 per-model CSV/metric cases and exact selected-file
identity in `results/official-run-001/independent-verification.json`. Another
independent raw-data endpoint-sweep agreed on all203 events, histories and
labels for the same two selected users, without importing production helpers.

Three user-approved preplanned candidates were submitted. Hybrid57851385
returned **private0.41001 / public0.40590**, above repeat57851386
(private0.38950) and PPR57851387 (private0.38258). No training or weight changes
used these returned scores. `submissions/kaggle-manifest.json` records exact
files/hashes; an authenticated download of the actual scored57851385 file is
byte-identical to `submissions/submission.csv`, SHA-256
`6dd285833c5602b699ab2e0d510ac75acb49addc410de59916bb859359107d79`.

## Outstanding course-delivery items

- The original source timezone and organizer fragment/conflict/tie equivalence
  remain uncertain. Historical8-vs-final20-week training mismatch is disclosed.
- User explicitly deferred personal verification; no agent check substitutes.
- Genuine private-score screenshot and final report/export/package inspection.
- GitHub publication is approved and being prepared; LMS submission is separate.

No GPU was needed. Kaggle upload/scoring is confirmed, not merely local presence.
CPU execution is verified; Apple Accelerate thread counts are not claimed known.
Native child-session exports are retained privately under ignored `sessions/`;
the initial checkpoint does not replace later official-data/report evidence.
