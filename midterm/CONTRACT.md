# Shared implementation and review contract

## Scope and provenance

The source task requires two approved methods plus popularity. The user's
expanded scope requires all four: CBF, Association Rules, implicit Matrix
Factorization, and Personalized PageRank, implemented by independent agents.
Every author must also challenge another implementation; paired review reverses
author/reviewer roles. No score, timing, human verification, or submission is
claimed until observed.

Public sources: the linked Midterm Quiz presentation (revision 1289 at analysis),
the SWEET.TV Kaggle competition data/evaluation pages retrieved through the
official MCP, and the unchanged upstream homework guidelines at commit
`4b9598b4e670417387aabe3facb7868b6b6143eb`.

## Labels and temporal isolation

- Match log `vsetv_id` to schedule `channel_id` and intersect half-open intervals.
- EPG duration is seconds. Audit actual log timestamp/duration representations.
- Preserve positive `tv_show_id`; zero and missing IDs do not become programs.
- Aggregate unioned viewing fragments per user/airing, not across airings;
  count one completed airing at coverage >=80% before rounding.
- Fragment union, broadcast identity, conflict and boundary policies are explicit
  engineering assumptions where organizer implementation details are unavailable.
- Fit only from behavioral observations available before the split cutoff.
- Rank repeats as eligible; no generic watched-item exclusion.
- Use the same forecast-schedule candidate catalog and evaluation users per fold.
- The five most frequent held-out qualifying program IDs are relevant. Standard
  AP@5 uses min(5, number of relevant IDs), not number of recommendation hits.

## Model interface

`TrainingData` is defined in `sweettv/contracts.py`. Metadata rows align with raw
`item_ids` and may have `tv_show_title`, `tv_show_category`, `tv_show_genre_1/2/3`,
`actors`, `director`, and `channel_title`; missing values are allowed.

Each module exports its named class with `fit(data)` and
`score(user_index, candidate_indices)`; the latter returns a float vector in the
requested candidate order. NaN means no model evidence. Models do not perform
their own backfill, filter watched items, read validation labels, or access
Kaggle credentials. `diagnostics` describes actual dimensions/coverage/backend.
Optional `score_batch` must return users x candidates in exactly the input order.

Modules: `cbf.py:ContentBasedModel`, `association.py:AssociationRulesModel`,
`mf.py:MatrixFactorizationModel`, `ppr.py:PersonalizedPageRankModel`.

Shared ranking: finite scores descending, then numeric raw program ID ascending;
backfill missing positions once using training-only popularity and then raw ID.
No fallback votes are fed back into hybrid fusion. Emit exactly five distinct
positive program IDs per template user, preserving user IDs as strings/order.

## Bounded execution and evidence

Start on CPU with sparse matrices. Apple integrated GPU is optional only after a
measured benefit and numerical checks; the implicit library's CUDA backend does
not run on Apple Metal. No dense global user-item score matrix. Separate runtime
measurements from deterministic prediction artifacts.

Primary proposed split: train weeks 11-18, forecast 19-30; development 11-14 ->
15-18; final 11-30 -> 31-42, subject to verifying actual dates/week semantics.
Training-history/horizon mismatch must be disclosed. All primary metrics are
validation/model-selection results, not untouched test estimates.

No remote predictions, publication, commits, or pushes without explicit approval.
Kaggle rules acceptance and OAuth consent are user actions. Protect original
native session exports, citations, personal verification, and report evidence.
