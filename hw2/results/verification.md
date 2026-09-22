# Verification record

Run locally on **2026-09-22**, macOS arm64, Node.js **v24.13.0**, npm **11.6.2**.

## Core tests

Command from `hw2/`: `npm test`.

**13 tests passed; 0 failed.** Coverage includes:

- All 18 genre positions with `unknown` independently enabled; CRLF and BOM.
- Unknown-only/all-zero vectors and Latin-1 accented titles.
- Malformed fields, invalid flags/IDs/ratings, duplicate records, missing references.
- Popularity from all ratings, including rating=1; movies with no ratings.
- Cosine identity, orthogonality, partial overlap, scale invariance, zero norms.
- Original-vector averaging, including mixed-width genres and empty histories.
- Exclusions, five-result cap, deterministic floating-point ties, input-order
  invariance, and independence from popularity.
- Dot/cosine ranking differences, profile invariance when active movie changes.
- Full MovieLens dimensions, known title/genre/count checks, and recommendation integrity.

## Experiment

Command: `npm run experiment`.

**Completed:** 200 seeded users sampled from 942 eligible users; six curated
histories; both scoring functions × both modes; saved JSON and Markdown results.
A second run was programmatically compared with the first: **both files were
byte-identical**.

| Artifact | SHA-256 |
| --- | --- |
| `comparison.json` | `786d45bb3bb7a701499022664e74e5c7fc4ff7c982d0805d1f05836716ab334a` |
| `comparison.md` | `d256ae369bf44f81ae65e536d32c5c48fd9727eb5734fb5e834d711cc8fc3d5b` |

## Browser

Command: `npm run test:browser` (with `BROWSER_SCREENSHOTS` set to a temporary
directory for visual inspection).

**Passed** using Playwright **1.63.0**, Chromium **153.0.8010.12**, a real static
localhost server, and a nested `/hw2/` URL. Checked:

- Visible loading state while ratings request is held; controls disabled until ready.
- Initial empty shelf and successful loading of 1,682 movies/100,000 ratings.
- Keyboard search/Enter addition, multi-selection, duplicate prevention, no matches.
- Native radio arrow navigation, active query changes, profile invariance, focus retention.
- Removing active/inactive/last selected movie, clear, zero-vector movie, example history.
- Exclusion of all selected IDs and presence of titles/genres/scores/rating counts.
- Two-column desktop layout at 1440×1100; single-column layout at 390×844;
  no horizontal overflow at widths 390 or 320.
- HTTP 404 error message, successful retry, and malformed item-data errors.
- **No uncaught page errors.**

Full-page desktop and mobile screenshots were opened and visually reviewed.
They are local inspection artifacts and are not required for deployment.

## Not verified

- A published GitHub Pages deployment (no remote deployment was performed).
- Safari/Firefox behavior, a physical mobile device, or a screen reader session.
- Real-user enjoyment, novelty, satisfaction, or retention.
