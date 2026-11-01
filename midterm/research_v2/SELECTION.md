# Round-two selection decisions

## Frozen before collaborative/schedule scores

The prior-only grid was completed first. None exceeded the unchanged hybrid on
both development intervals; source-inspired priors are hypotheses, not a proven
improvement. Those results remain in `explore-priors-001` and are not replaced.

The next grid is exactly
`configs/research-v2-collaborative-schedule-grid.json`: nine EASE configurations
and five schedule/habit configurations, each on the same two development folds.
No configuration is chosen from the already-known private scores.

The separate schedule-adjusted-repeat ablation is exactly
`configs/research-v2-availability-grid.json` (eight configurations on the same
two development windows). This tests the opportunity ratio independently of
channel/time transfer. It uses the same selection rule below.

The raw-viewing-habit extension is exactly
`configs/research-v2-raw-habits-grid.json` (five configurations): default half-life,
longer/no decay, one per-airing nonlinear-power alternative, and one daily pooling
alternative. No full cross-product of slot widths, powers and history lengths.
All partial/full raw viewing is clipped before estimation; completion labels,
candidate policy and cohorts remain unchanged. These mean-coverage proxies are
not calibrated >=80% completion probabilities. This grid is frozen before any
raw-habit quality scores are inspected.

For standalone promotion, require nonnegative strict-five and all-template delta
against the unchanged hybrid in BOTH development windows; rank passing models by
equal-window mean all-template delta. Prefer the simpler model within 0.001 mean
MAP. If none passes, keep at most one representative of each complementary model
family for a separate small fusion ablation. That ablation and exact weights
will be frozen before its scores, using development only. No full cross-product
of every weak predictor.

Freeze at most two new candidates before confirmation and the original twelve-
week primary replay. Promotion requires positive primary/cohort and all-template
deltas in both confirmation windows, no integrity failure, and tie sensitivity
that does not reverse the overall conclusion. Rejection is an acceptable result;
do not keep changing the same shortlist after checking these windows. These are
reused historical data, not untouched test estimates. No new live submissions
are authorized by local promotion.

## After standalone results, before fusion scores

All41 standalone/reference configurations are retained in
`results/research-v2/scoreboard-stage-003/`. Four linear raw-occupancy variants
pass the two-window/two-cohort gate. The best mean all-template result is the
daily pooled56-day-half-life model (`raw_day_h56_p1`); its mean delta is within
0.001 of the weekly28-day alternative, and daily pooling has fewer time bins.
Choose daily56 as the frozen standalone candidate, not the most favorable model
separately in each fold. No other family passed the standalone gate.

Before confirmation, test exactly six evidence-only blends in
`configs/research-v2-fusion-grid.json`: raw daily56 with either the unchanged
hybrid or unweighted personal repeat at relative weights0.25,0.5,1. This is a
limited complementary-preference ablation around the successful raw model,
not a cross-product of the rejected families. Raw history can identify a habitual
channel/time without identifying the preferred program; repeat/hybrid may help
rerank that ambiguity. Use the same development selection rule, and keep at most
the best passing blend plus frozen raw daily56 for confirmation. If no blend
beats raw in both windows/all-user cohorts, confirm raw alone.

For RRF60, an auxiliary-only rank1 item has at most `w/61`, while five primary
items each receive at least`1/65`. Thus weights0.25/0.5 cannot introduce an
auxiliary-exclusive item when five primary-supported items exist; these are
rerankers, not unrestricted discovery models. Weight1 can overcome this bound.
No popularity backfill votes enter any blend.

## Source calendar audit

`results/research-v2/source-calendar.json` is a raw-input audit, not a model
selection result. The file has 139 source-clock start dates. Weekly log counts
decline from 407,493 in week11 to126,940 in week26 and rebound to271,172 in
week28. Unique logged users also change sharply (2,406 in week26;4,620 in
week28). This is observable source coverage/activity drift; its cause is unknown.
Do not assert attrition, outages, nonwatching, or a timezone correction as fact.
