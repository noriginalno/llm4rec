# Round-two improvement research

The original scored submission (private MAP@5 .41001, ref57851385) and all its
files remain immutable. New local research does not authorize more submissions,
commits, or pushes. Public notebook source may inform ideas, but is not executed;
outputs, answer files, hidden labels and future viewing logs are not imported.

## Targets and invariants

Keep the original reconstructed >=80% completion counts, program identity,
candidate policy, template users, source-clock assumption and AP@5 evaluator.
No matching-policy search to flatter metrics. All methods in a fold share its
catalog/cohorts; sparse/cold cases receive the same explicit one-time fallback.
TemporalContext exposes only pre-cutoff behavioral arrays and permitted static
forecast schedules, never held-out behavior or labels. Schedule frequency is an
opportunity proxy, not measured personal exposure or a causal effect.

## Development and confirmation

Explore on the original development interval (train11–14, forecast15–18) and
a later blocked interval (train11–22, forecast23–26). Freeze a small shortlist
before confirmation on train11–26 -> forecast27–30 and comparison against the
unchanged original 12-week primary (train11–18 ->19–30).

These dates are portions of previously observed source data, including the
old primary validation. They are **not new untouched test data**. Their use is
time-consistent local research and robustness confirmation, not an unbiased
generalization estimate. Do not call repeated primary results an independent
test or treat previous private scores as training labels.

Report fixed strict-five cohorts plus all1260-user diagnostics per window.
Show ascending-ID, reversed-ID and, where useful, tie-marginalized sensitivity;
do not replace official MAP@5 with an optimistic tie metric. Require gains over
the original reference under more than one time window before recommending a
new live submission. Keep every trial and source/configuration fingerprint.

## Initial bounded hypotheses

1. Recency-weighted personal frequency, optional finite history windows.
2. Smoothed forecast/historical schedule adjustment and channel/time backoff.
3. Count-sensitive linear collaborative models (EASE/item-kNN), rather than
   assuming confidence-weighted binary ALS forecasts repeat frequency.
4. Only after standalone results: a small evidence-only fusion of complementary
   predictors, with no duplicated popularity backfill votes.

Separate independent implementation and challenge roles as in round one. All
claims depend on executed local measurements; higher Kaggle score is a goal,
not a guarantee. Existing report/screenshot/personal-check gates remain open.
