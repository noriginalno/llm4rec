# What changes when we recommend from a profile?

## Main findings

On a reproducible sample of **200 MovieLens users**, cosine item-to-item and
cosine profile-based recommendation overlap by only **0.84 titles out of five**
on average. The profile therefore offers a substantially different perspective,
but it does **not** automatically improve long-tail discovery: in this run,
item-to-item has a slightly higher tail share (47.8% versus 46.1%) and broader
catalog coverage (14.92% versus 12.78%).

Compared with raw dot product, cosine reduces the average number of genre labels
on recommended movies, increases coverage, and reduces exposure concentration
for both modes. Those are measured exposure properties. Enjoyment, discovery of
genuinely unknown titles, and retention are **not measured** here.

The full generated tables, including every example's two Top-5 lists under both
scoring rules, are in [results/comparison.md](results/comparison.md). The
[JSON artifact](results/comparison.json) contains all sampled user IDs, selected
ratings, histories, ordered recommendations with scores, and summary metrics.

## 1. Data and shared implementation

MovieLens 100K contains **1,682 movie IDs, 100,000 ratings, and 943 users**. The
files are byte-preserved copies of the pinned source documented in the
[README](README.md#sources-and-attribution). Titles are decoded as Latin-1.

The parser verifies 24 pipe-delimited fields per `u.item` row. Five are metadata;
the next is the `unknown` flag; the last 18 map, in order, to Action, Adventure,
Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir,
Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, and Western. Confusing
19 flags with 18 names shifts every genre, invalidating the comparison.

Movie #1, *Toy Story*, correctly maps to Animation/Children's/Comedy; #50,
*Star Wars*, maps to Action/Adventure/Romance/Sci-Fi/War. Two IDs, #267 and
#1373, have zero usable genre vectors. These remain in the catalog denominator
but cannot receive a positive genre score.

Both the browser and experiment import the same functions from
[`script.js`](script.js): `dotProduct`, `cosineSimilarity`, `averageProfile`,
`rankMovies`, and `recommend`. Popularity comes from counting **all** rows per
movie in `u.data`; rating value does not change that count. It never enters the
score or recommendation tie-breaker.

For binary movie vector `x_i` and selected history `H`:

```text
p = (1 / |H|) * sum(x_h for h in H)        # original vectors, not unit vectors
dot(q, x_i) = sum(q[g] * x_i[g])
cos(q, x_i) = dot(q, x_i) / (||q||₂ * ||x_i||₂)

item-to-item: q = x_active
profile-based: q = p
```

For example, histories `[1,0,0]`, `[1,1,1]`, and `[0,1,0]` produce
`p = [2/3, 2/3, 1/3]`. Normalizing each movie before averaging would change
the genre weights and is deliberately not done. Genre-mix percentages in the UI
are fractions of selected movies carrying each genre, so they need not sum to 100%.

Both modes use the same candidate pool: **every catalog ID except all selected
IDs**. Positive scores are ordered descending after rounding to 12 decimal
places, then by ascending numeric ID. Quantization prevents mathematically equal
scores from being ordered by floating-point noise. Empty histories/zero queries
produce empty lists; zero-score candidates are omitted. With one valid selection,
both cosine lists are identical. Every sampled user in this run has five positive
recommendations under all four combinations.

## 2. Reproducible protocol

From `hw2/`, with Node.js 20+:

```sh
npm test
npm run experiment
```

No installation or network access is needed for these commands. The experiment
writes `results/comparison.json` and `results/comparison.md`. It records input
SHA-256 hashes and omits generation timestamps. A second run was checked to
reproduce both files **byte for byte**.

### Sample construction

1. Treat a rating of **4 or 5** as a like. Require at least three such ratings:
   **942 of 943 users** are eligible.
2. Sort eligible IDs numerically. Shuffle using Mulberry32 with seed
   **20260922**, using Fisher–Yates from the last index down to 1. Take the first
   **200**, without replacement. IDs are saved explicitly in the JSON artifact.
3. For each sampled user, sort liked ratings by ascending timestamp, then movie
   ID to resolve ties. Use the first **three distinct rated movie IDs**. Duplicate
   user/movie rows are rejected during parsing. The **third** is active.
4. Evaluate item and profile using cosine, then both using raw dot product, with
   the same histories, candidates, K=5, and tie-breaking rule.
5. Separately change the active movie from third to second while holding the
   three-movie history fixed, then compare the first-two-like history against
   the three-like history. These give two descriptive persistence measures.

The six curated histories below are illustrative, not included in the sampled
aggregate. No parameters are tuned on recommendation outcomes.

### Head/tail definition and metrics

Sort all 1,682 movie IDs by descending rating count, then ascending ID. The first
`ceil(0.20 × 1682) = 337` are **head**; the remaining **1,345** are **tail**. The
rounding makes the head 20.04% of the catalog. The boundary crosses a popularity
tie: the last head item is #682 with 100 ratings, and the first tail item is
#1012, also with 100. Thus a bare `< 100 ratings` threshold is not equivalent.
The exact head IDs are saved.

Each mode/scoring combination produces **1,000 recommendation slots** (200 × 5):

- **Tail share:** tail slots / all recommendation slots; repeats count again.
- **Average recommendation popularity:** mean rating count across slots, not
  an average of star ratings and not an average over unique movies.
- **Catalog coverage:** unique recommended IDs / 1,682.
- **Tail catalog coverage:** unique recommended tail IDs / 1,345.
- **Overlap@5:** shared IDs between two lists / 5, averaged over users.
- **Jaccard overlap:** shared IDs / union size, averaged over users. It is not
  computed from the mean intersection because that would give a different value.
- **Average genre count:** mean number of genre labels across recommended slots.
- **Top-10 exposure share:** fraction of all slots occupied by the ten most
  frequently recommended IDs for that mode; higher means more concentration.

All overlap measures compare **sets**, not ranking positions. In the zero-vector
example empty/empty overlap is defined as zero; it does not affect sampled metrics.

## 3. Example histories: when the lists agree or diverge

Here are the ordered cosine Top-5 IDs. The linked generated report expands them
to titles, genres, scores, and rating counts and includes the raw-dot lists too.

| History (active in bold) | Item-to-item IDs | Profile-based IDs | Shared |
| --- | --- | --- | ---: |
| **Toy Story #1** | 422, 95, 1219, 63, 94 | 422, 95, 1219, 63, 94 | 5/5 |
| Star Wars #50, Empire Strikes Back #172, **Return of the Jedi #181** | 271, 498, 62, 82, 121 | 271, 498, 62, 82, 121 | 5/5 |
| Toy Story #1, Aladdin #95, **Aladdin and the King of Thieves #422** | 1219, 63, 94, 102, 138 | 1219, 993, 63, 94, 102 | 4/5 |
| Toy Story #1, Star Wars #50, **Fargo #100** | 5, 329, 332, 348, 649 | 172, 426, 181, 560, 29 | 0/5 |
| Room with a View #213, Sense and Sensibility #275, **Titanic #313** | 207, 1483, 631, 720, 14 | 14, 20, 36, 125, 131 | 1/5 |
| **unknown #267** | empty | empty | 0 |

- **Coherent histories can agree completely.** The space-adventure trilogy
  yields *Starship Troopers*, *The African Queen*, *Stargate*, *Jurassic Park*,
  and *Independence Day* in both lists, though the profile scores differ slightly.
  Genre similarity is broader than franchise similarity.
- **A repeated genre can steer the profile.** In family animation, the Musical
  flag from *Aladdin* is diluted to 1/3 but still brings *Hercules* into the
  profile Top-5, replacing *D3: The Mighty Ducks*.
- **Mixed interests can separate the lists entirely.** With *Fargo* active,
  item-to-item returns Crime/Drama/Thriller records such as *Copycat* and
  *Kiss the Girls*. The combined profile instead favors cross-genre overlaps:
  *The Empire Strikes Back*, *Transformers: The Movie*, *Return of the Jedi*,
  *A Kid in King Arthur's Court*, and *Batman Forever*. Broad labels alone do
  not establish that these are good recommendations for the viewer.
- **The profile can damp a one-off genre.** *Titanic* has Action/Drama/Romance,
  whereas the two other romance examples have Drama/Romance. The profile's
  Action weight is only 1/3; pure Drama/Romance movies lead its results. Item
  mode prioritizes exact Action/Drama/Romance matches.
- **No genres means no evidence.** Selecting #267 alone gives two helpful empty
  states rather than an arbitrary five-way zero-score tie.

## 4. Why cosine changes genre-count bias

For binary query `q` and a candidate with `g` genre labels, cosine equals
`shared_genres / sqrt(query_genres × g)`. Dot product only counts overlap. Adding
matching labels can increase dot score; adding **nonmatching** labels does not
lower it. A candidate that covers many genres has more opportunities to match.
Cosine introduces a candidate-length penalty for that breadth.

Numerical example with `q = [1, 1, 0, 0]`:

| Candidate | Genre count | Dot product | Cosine |
| --- | ---: | ---: | ---: |
| A = `[1, 0, 0, 0]` | 1 | 1 | 0.707107 |
| B = `[1, 1, 0, 0]` | 2 | 2 | 1.000000 |
| C = `[1, 1, 1, 1]` | 4 | 2 | 0.707107 |

Dot ties B and C and ranks C above A. Cosine distinguishes the exact match B
from the broader C, while A and C tie. This reduces a genre-count advantage; it
does not ban multi-genre movies. In particular, many labels may genuinely align
with a broad profile.

For a fixed query, dividing by the query norm is a constant and **does not
change candidate ordering**. The candidate norm is what changes the ranking.
Query normalization gives comparable, scale-invariant cosine values. Averaging
instead of summing the history also changes only the common query scale, not
the ranking under either score.

### Measured normalization effects

| Scoring | Mode | Avg. genres per recommendation | Tail share | Avg. rating count | Unique IDs / catalog coverage | Top-10 exposure share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Raw dot | Item | 3.250 | 35.10% | 178.75 | 159 / 9.45% | 34.90% |
| Cosine | Item | 2.270 | 47.80% | 138.27 | 251 / 14.92% | 22.20% |
| Raw dot | Profile | 3.913 | 39.00% | 167.90 | 130 / 7.73% | 42.20% |
| Cosine | Profile | 2.961 | 46.10% | 141.14 | 215 / 12.78% | 25.20% |

Cosine reduces mean recommended genre count by **30.15% for item** and **24.33%
for profile** relative to raw dot. Tail share rises **12.7 percentage points**
and **7.1 points**, respectively. These effects occur without any popularity
feature in scoring. They are empirical associations in this catalog, not a
guarantee that cosine always makes a recommender less popular or more diverse.

The cosine and raw-dot lists share only **45.6%** of slots for item mode and
**50.6%** for profile mode, averaged as overlap@5 per user. Normalization materially
changes results rather than merely rescaling displayed numbers. The profile
still recommends more genres than item mode under cosine (2.961 versus 2.270).

## 5. Long-tail discovery and overlap between strategies

Considering the cosine service itself:

| Metric | Item-to-item | Profile-based |
| --- | ---: | ---: |
| Tail slots / 1,000 | 478 | 461 |
| Tail share | 47.80% | 46.10% |
| Mean rating count per slot | 138.274 | 141.143 |
| Unique recommended IDs | 251 | 215 |
| Catalog coverage | 14.92% | 12.78% |
| Unique tail IDs | 136 | 105 |
| Tail catalog coverage | 10.11% | 7.81% |
| Top-10 exposure share | 22.20% | 25.20% |

**Measured:** item-to-item exposes 36 more unique IDs and 31 more tail IDs in
this sample. Profile-based recommendations are slightly more popular on average
and more concentrated. A combined preference vector is not a built-in long-tail
promotion mechanism. No significance test was performed; the 1.7-point tail-share
difference is a descriptive result for this seed/history policy.

The tail occupies about **80% of catalog IDs** but less than half of recommended
slots in either mode. Even without a popularity score, selected preferences,
genre distribution, and the deterministic tie-breaker can concentrate exposure.
Tail status also spans movies with one rating through some with 100: it is a
coarse relative-popularity category, not a measure of user-specific novelty.

| Between item and profile | Cosine | Raw dot |
| --- | ---: | ---: |
| Mean shared titles / 5 | 0.840 | 1.385 |
| Mean overlap@5 | 16.80% | 27.70% |
| Mean Jaccard | 0.1448 | 0.1922 |
| Identical Top-5 sets / 200 | 20 | 5 |
| Disjoint Top-5 sets / 200 | 139 | 59 |

Cosine has both more fully identical and more disjoint sets than raw dot. The
mean alone hides this mixture: coherent histories can agree exactly, while
heterogeneous histories often diverge sharply.

## 6. Repetition, discovery, and retention

### Measured behavior

- Switching only the active movie (third → second, same three selections)
  retains **7.6%** of cosine item results on average. The profile retains
  **100%**, as required by its definition; this is a useful invariance check.
- Adding the third like to the first-two-like history retains **6.8%** of item
  results versus **35.2%** of profile results. This change also excludes the new
  selection and switches the item query to that new active movie. It is not an
  isolated measurement of averaging alone.
- Raw-dot persistence on history growth is **11.5%** for item and **51.9%** for
  profile. Cosine both changes the genre preference and reduces repetition in
  this particular measurement.
- Across users, the ten most-exposed cosine IDs take **22.2%** of item slots
  and **25.2%** of profile slots. For example, *Shanghai Triad* appears 30 times
  in item mode; *Copycat*, *Apollo 13*, and *Wild Things* each appear 27 times
  in profile mode. Raw-dot profile results are more concentrated: *The Empire
  Strikes Back* alone appears in 81 of the 1,000 slots.

### Interpretation and hypotheses

The two-column interface gives users a direct way to explore a current interest
while keeping a broader preference summary visible. The low measured overlap
suggests that the columns can offer complementary choices. A stable profile may
reduce the effort of repeatedly choosing an anchor; an active movie offers
control over a temporary mood. **Those are product hypotheses**, not observed
reductions in effort or improvements in satisfaction.

Genre-only vectors create many ties, and the ascending-ID rule always picks the
same IDs from a tied group. Exact reruns of an unchanged history are deterministic
and will not discover anything new. The observed concentration is consistent
with this mechanism. Furthermore, MovieLens includes distinct IDs for the same
film: #329 and #348 both display *Desperate Measures* in the mixed-history item
list. ID-based coverage can therefore overstate unique-film coverage.

Useful long-tail matches and varied options **could** encourage return visits;
repetitive or semantically odd recommendations could do the opposite. Nothing
in these offline tables estimates retention or causal impact. Testing those
hypotheses would require an online comparison, measuring actual choices,
satisfaction, repeat recommendation exposure, and return visits over a defined
period. Any later diversification or tie-rotation policy should be evaluated
separately from this controlled, deterministic genre-only baseline.

## 7. Scope and limitations

- This is a **descriptive recommendation-exposure study**, not a held-out
  relevance evaluation. No precision, recall, click rate, or retention uplift is
  inferred from popularity or genre similarity.
- Only the three supplied selections are excluded, exactly as in the app.
  Other already-rated movies, including disliked ones, may be candidates;
  therefore recommended does not mean previously unseen.
- The catalog and popularity counts use the entire historical dataset. There
  is no release-date cutoff and no temporal train/test boundary; some candidates
  may be unavailable at the selected ratings' timestamps. Popularity is a
  retrospective descriptor, not a future-safe estimator.
- First-three-like histories are short, unweighted, and often have tied
  timestamps. ID tie-breaking makes them reproducible but cannot recover the
  true order within a batch of ratings. Results can change with longer or
  recent histories, another seed, genre weighting, or a different tie policy.
- Broad selected movies contribute more nonzero genre dimensions because the
  specification averages raw vectors. Cosine normalizes the resulting query,
  but does not remove this aspect of profile construction.
- Coarse genre labels omit plot, tone, cast, language, and individual context.
  This 1997–1998 catalog is not representative of a current streaming service.
- The checks establish parser/scoring correctness, reproducibility, and basic
  browser behavior. Live GitHub Pages deployment, assistive-technology testing,
  and real user outcomes were not verified.
