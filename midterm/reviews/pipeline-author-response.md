# Coordinator response to independent pipeline challenges

The original challenger report and tests were preserved. This response
distinguishes observed operational defects from fault-injection hardening;
neither implies that current model authors intentionally corrupted inputs.

## Original challenge

The coordinator reproduced **7 failures / 25 passes** in the unchanged
`tests/test_pipeline_adversarial.py` before applying corrections:

- P1: reject unconfirmed or chronologically invalid protocols before raw-data
  loading; validate cross-stage boundaries before creating a full-run directory.
- P2: absence of a synthetic marker no longer implies an official source.
  Unmarked inputs record null official identity and explicit unverified status.
- P3: selected CSV copying now publishes exact bytes using a temporary file and
  exclusive hard link, never replacing an existing final destination.
- P4: per-model read-only user/candidate copies plus post-call equality checks
  prevent query permutations from changing earlier score alignment. Returned
  batch scores are copied immediately to prevent cross-model buffer aliasing.
- P5: the training-state fingerprint now frames dtype/shape/length descriptors
  and includes sparse shape, users, metadata and canonical training provenance.

After these changes, unchanged challenger tests plus parent integration/shared
tests passed **159/159**. The verifier independently confirmed all seven
original failures closed, reran the unchanged **32/32** challenger cases and
the then-current full **561/561** suite. Its additive post-fix report is retained.
Source hashes are now recorded and a source change during scoring aborts the run.

## Supplemental scalar-return ownership finding

The verifier then demonstrated that the scalar fallback collected references to
one reusable score buffer before `vstack` copied them. This was a controlled
double, not evidence that an actual current model used that pattern.

The coordinator added a regression with three literal expected recommendation
lists and reproduced **one failure**: every user incorrectly received the last
user's NaN/fallback row. Each scalar row is now copied immediately before the
next query. The same regression then passed; its expected fallback total is
12 positions, not 15. Original reviewer files remain unchanged.

## Additional observed timing and wrapper corrections

Historical synthetic v1 assigned approximately .502 seconds of all-model cold
imports to popularity's fit. Imports are now lazy and initialization is timed
separately from fit/scoring. This is timing attribution, not a ranking defect.

The independent whole-run wrapper initially passed a JSON list where the
independent verifier requires an explicit positive-integer set. The invocation
failed before producing a success report. It now validates the JSON list and
constructs the required set. The fixed wrapper independently validated all
22 per-model CSVs/metrics and selected-byte identity for synthetic v2. A new
six-user full-run regression covers the wrapper path in isolated stdlib mode.

## Limits

Read-only arrays and fingerprints are consistency defenses, not a sandbox
against arbitrary in-process Python. Passing fixture tests do not establish
organizer label equivalence, real recommendation quality, personal verification,
or any returned Kaggle score. Official file names differ from the old data
description and are explicitly mapped; actual contents/time zones still require
source reconciliation after the user's competition-rule acceptance.
