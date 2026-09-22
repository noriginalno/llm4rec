# Homework 2 — Content-Based Movie Recommender

**Reel Alike** is a static, vanilla HTML/CSS/JavaScript service comparing two
genre-based Top-5 recommenders on MovieLens 100K. No backend, build step, API key,
account, external font, or runtime dependency is needed.

[Service entry point](index.html) · [Analysis](ANALYSIS.md) ·
[Computed comparison](results/comparison.md) · [Verification record](results/verification.md)

GitHub Pages URL after deployment: <https://noriginalno.github.io/llm4rec/hw2/>.

## Run locally

With Python 3 and Git installed, run this from **any directory inside the local
`llm4rec` clone**, including `hw2/`:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory "$(git rev-parse --show-toplevel)"
```

Open <http://127.0.0.1:8000/hw2/>. The explicit `--directory` serves the repository
root (the folder containing `hw1/` and `hw2/`) regardless of your current directory.
If a server is already running on port 8000, stop it with Ctrl+C in its terminal
before running the command above.

**Why `/hw2/` might return 404:** without `--directory`, Python serves your current
working directory. If you ran `python3 -m http.server 8000 --bind 127.0.0.1` from
inside `hw2/`, the app is at <http://127.0.0.1:8000/> instead. In that setup,
`/hw2/` would look for a nonexistent `hw2/hw2/` subfolder. You can keep using `/`
or restart with the explicit command above to use `/hw2/`.

Use HTTP rather than double-clicking `index.html`: ES modules and local data
fetches need an HTTP origin. The server only serves files; the recommender runs
entirely in the browser.

1. Search for a title, choose it from the native select, and **Add to shelf**.
   When a search leaves exactly one match, Enter adds it directly.
2. Add more watched-and-liked movies. The newest addition becomes active.
3. Use a radio button to change the active movie. Item-to-item updates; the
   profile stays tied to the whole shelf.
4. Remove a movie with its × button, clear the shelf, or load the three-title example.

Tab/Shift+Tab move between controls; native selects support keyboard selection;
arrow keys change the active radio. Focus returns to a useful control after
removal. Loading, errors, retry, no-search-results, empty histories, and missing
genres have explicit states. The lists sit side by side on larger screens and
stack on small screens. The shelf lives in this tab's memory and resets on reload.

## Four application files

| File | Responsibility |
| --- | --- |
| [`index.html`](index.html) | Semantic page structure, labeled native controls, live status regions, result containers. |
| [`style.css`](style.css) | Responsive layout, focus states, colors, and typography. |
| [`data.js`](data.js) | Local data loading, Latin-1 title decoding, strict parsing, genre vectors, and rating counts. |
| [`script.js`](script.js) | Exported pure scoring/ranking functions plus UI state and rendering. |

`script.js` imports `data.js` as an ES module. Node tests and experiments import
the same exported functions; a DOM guard prevents UI initialization in Node.
The browser never downloads the experiment code or npm packages.

Supporting files:

- `u.item`, `u.data`: original MovieLens bytes from the pinned source below.
- `tests/core.test.js`: focused unit tests and a full-dataset integration check.
- `tests/browser.mjs`: optional Chromium end-to-end checks; includes a temporary
  static HTTP server under the same `/hw2/` path used by Pages.
- `experiments/compare.js`: seeded sample, metrics, and result generation.
- `results/comparison.json`: all 200 sampled IDs, histories, ordered lists/scores,
  checksums, example histories, and aggregate results.
- `results/comparison.md`: readable generated tables and example Top-5 lists.
- `ANALYSIS.md`: protocol, equations, measured findings, limitations, hypotheses.

## Recommendation rules

- Each movie has an 18-dimensional binary vector. `u.item` contains five
  metadata fields, the `unknown` flag at index 5, and the **18 named genre flags
  at indices 6–23**. All 19 flags are validated; only `unknown` is discarded.
- Item-to-item uses the active movie's vector. Profile-based averages the
  **original** vectors of all selected IDs, before any cosine normalization.
- Both score with `dot(query, candidate) / (norm(query) * norm(candidate))`.
  A zero norm returns zero. Empty/all-zero queries yield empty lists, and only
  positive matches appear; at most five results are shown.
- Both consider every unselected catalog ID. Rank by score rounded to **12
  decimal places**, descending, then numeric movie ID, ascending. The raw score
  is retained for display and analysis. Input order and popularity do not break ties.
- Rating count is the number of rows for that movie in `u.data`, including low
  ratings. It is displayed for context and used for analysis, never for ranking.
- MovieLens IDs are the item identity. Original duplicate-title records are
  retained, so two distinct IDs can display the same title (for example
  *Desperate Measures* #329 and #348). This also affects catalog-level metrics.
- The UI uses cosine only. Raw dot-product scoring is an experimental option on
  the same `rankMovies`/`recommend` functions, not a separate implementation.

Parsing failures identify the file and line instead of silently dropping records.
The integration check expects 1,682 movies, 100,000 ratings, and 943 users.

## Tests and experiments

Requires **Node.js 20+**. From `hw2/`:

```sh
npm test
npm run experiment
```

These two commands use only Node built-ins and need **no `npm install`**. The
experiment overwrites `results/comparison.json` and `results/comparison.md`.
It has no network calls, clock-based fields, or unseeded randomness. Running it
again with the bundled data reproduces the same files. The algorithm and seed
are specified in [ANALYSIS.md](ANALYSIS.md).

Optional browser verification (Playwright is a development-only dependency):

```sh
npm ci
npx playwright install chromium
npm run test:browser
```

The test opens its own ephemeral localhost server and closes it afterward. To
also save desktop/mobile screenshots, set `BROWSER_SCREENSHOTS` to an output
directory before running `npm run test:browser`. No screenshot files are needed
by the service. On a fresh Linux host, Playwright may additionally need its
documented browser system dependencies.

See [the verification record](results/verification.md) for the actual checks run.

## GitHub Pages

1. Publish these files to the repository's `master` branch.
2. In **Settings → Pages → Build and deployment**, select **Deploy from a branch**.
3. Choose **master** and **/ (root)**, then save.
4. Wait for the Pages deployment to finish and visit
   <https://noriginalno.github.io/llm4rec/hw2/>.

All script, style, and data URLs are relative to `hw2/`, including module-relative
data URLs, so they work under the `/llm4rec/` project prefix. The root README links
both homework projects. There is no npm build or custom deployment workflow to
configure. Deployment itself has not been performed as part of this local work.

## Sources and attribution

- Data: **MovieLens 100K**, collected by the **GroupLens Research Project,
  University of Minnesota**. Dataset page:
  <https://grouplens.org/datasets/movielens/100k/>.
- The two data files were obtained unchanged from
  [`dryjins/RecSys-LLMs/week2`](https://github.com/dryjins/RecSys-LLMs/tree/d8a178a50b1997cd0a5e25604be684b0179418db/week2),
  pinned to commit `d8a178a50b1997cd0a5e25604be684b0179418db`.
  That exercise inspired the four-file structure. This implementation replaces
  its recommendation logic with shared cosine/profile scoring and corrects the
  `unknown`-flag alignment.
- The original GroupLens dataset description and usage terms are included in
  [`MOVIELENS_README.txt`](MOVIELENS_README.txt), retrieved from
  <https://files.grouplens.org/datasets/movielens/ml-100k/README>.
  The dataset retains those terms; the repository's [MIT license](../LICENSE)
  does not replace them.
- Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. **The MovieLens
  Datasets: History and Context.** ACM Transactions on Interactive Intelligent
  Systems 5(4), Article 19. <https://doi.org/10.1145/2827872>.

SHA-256 checksums (also saved in the computed results):

```text
u.item  553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701
u.data  06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490
```
