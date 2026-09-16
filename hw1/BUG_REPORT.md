# Week 1 HTML audit

Reviewed: 2026-09-15.

Source: [dryjins/RecSys-LLMs/week1](https://github.com/dryjins/RecSys-LLMs/tree/7b4e5a35e15dda8d87e73636e0a6fb178d203534/week1).
Original `index.html` Git blob: `181cac7d0f03cbd5d0f647811b77c4f62edf4992`.
Original `prompt.md` Git blob: `533fce28bdf2a5b031e29524d23df18d3c26262f`.

This review covers both files in `week1`, all 12 menu entries, the inline
styles and script, and the upstream license. The corrected application is
`index.html`; the original is preserved in `upstream/index.html`.

## 1. Missing and misleading food icons

The page loads Font Awesome Free **6.4.0**. Checking the exact
[6.4.0 CSS source](https://github.com/FortAwesome/Font-Awesome/blob/6.4.0/css/all.css)
confirms that three selected class names have no glyph definition in this
stylesheet. Changing `fas` to `fa-solid` would not supply those missing icons.
Older aliases such as `fa-hamburger` and `fa-utensil-spoon` are present.

| Lunch | Original icon | Finding | Replacement drawing |
| --- | --- | --- | --- |
| Pizza | `fa-pizza-slice` | Present, appropriate | Pizza slice |
| Sushi | `fa-fish` | Present, loose proxy: whole fish instead of sushi | Sushi roll with chopsticks |
| Burger | `fa-hamburger` | Present alias, appropriate | Burger |
| Salad | `fa-leaf` | Present, loose proxy: a leaf instead of a dish | Salad bowl |
| Tacos | `fa-utensil-spoon` | Present, misleading: spoon | Filled taco |
| Ramen | `fa-bowl-hot` | Missing from the loaded stylesheet | Noodles, bowl, chopsticks |
| Sandwich | `fa-bread-slice` | Present, loose proxy: bread without filling | Filled sandwich |
| Pasta | `fa-pasta` | Missing from the loaded stylesheet | Pasta on a plate with a fork |
| Curry | `fa-mortar-pestle` | Present, misleading: preparation tool | Rice and curry on a plate |
| Steak | `fa-drumstick-bite` | Present, misleading: poultry drumstick | Steak with grill marks |
| Soup | `fa-bowl` | Missing from the loaded stylesheet | Steaming bowl and spoon |
| BBQ | `fa-fire` | Present, loose proxy: fire without food or grill | Barbecue grill |

The three missing classes are objective rendering defects. The three misleading
choices and four loose proxies are semantic design findings; generic food
symbols are not missing-glyph failures.

**Fix:** use local SVG symbols with a distinct identifier and drawing for each
dish. All 12 menu records now reference the matching symbol. Heading, button,
initial-state, and loading icons are local as well.

## 2. Superseded timers show stale results — medium severity

Original code: `generateRandomLunch`, especially lines 172–191 of the original.

Every click schedules a new `setTimeout(..., 500)` without cancelling the old
one. The page also calls the same function immediately on startup.

Reproduction with a deterministic clock:

1. At 0 ms, page load chooses Pizza and schedules its result for 500 ms.
2. At 400 ms, click again; this request chooses Sushi and should remain loading
   until 900 ms.
3. At 500 ms, the original callback displays Pizza, interrupting the newer
   loading state.
4. At 900 ms, Sushi replaces Pizza.

A second test clicks again one millisecond before an older timer is due and
reproduces the same stale-result flash. A burst of 100 clicks during startup
creates **101 pending callbacks** in the original.

These fixed-delay timers normally complete in scheduling order, so the final
result is usually the last click's choice. The verified bug is intermediate
stale results, interrupted loading, and redundant callbacks. There is no
network request or LLM generation in this application; the delay is simulated.

**Fix:** retain the timer ID and call `clearTimeout` before scheduling its
replacement. Keep the button usable so a new click can replace a pending
choice. Generation completes 500 ms after the latest click; icon and label
are updated together in the same callback.

The original also adds the same `fade-in` class from each pending callback.
An older callback can start the fade and later callbacks add an already-present
class, so their updates do not reliably restart that animation. Cancelling
superseded callbacks fixes the associated animation-state interference.

## 3. All icons depend on a third-party stylesheet and font

Original code: the cdnjs stylesheet link in the document head.

If the stylesheet or its font cannot be loaded, even valid icon names lose
their drawings. The original has no fallback. This is a dependency/failure-mode
finding from the source, not a claim that cdnjs was down during the review.

**Fix:** embedded SVGs remove the external resource dependency. The application
has no network-loaded assets.

## 4. Dynamic results lack an accessible announcement

The original `.lunch-display` is an ordinary `div` with no live-region semantics
or busy state. Changes are visible but are not explicitly announced to a
screen reader while focus remains on the Generate button.

**Fix:** `role="status"`, `aria-live="polite"`, `aria-atomic="true"`, and
`aria-busy`. Decorative icons are hidden from assistive technology; the dish
name supplies the meaningful text. A `main` landmark and explicit keyboard
focus style were also added. Actual screen-reader behavior remains to be
checked in a browser.

## 5. Insufficient text contrast

Calculated sRGB contrast against white:

| Element/state | Original color | Original ratio | New color | New ratio |
| --- | --- | --- | --- | --- |
| Heading; button background under white text | `#ff6b6b` | 2.78:1 | `#b52e40` | 6.12:1 |
| Hovered button background under white text | `#ff5252` | 3.19:1 | `#942437` | 8.18:1 |
| Footer text | `#777777` | 4.48:1 | `#666666` | 5.74:1 |

The normal button and heading fail even the 3:1 large-text threshold. The
hover state falls below the 4.5:1 normal-text threshold, relevant to the smaller
mobile button. The footer is just under 4.5:1. Thresholds follow
[W3C's contrast guidance](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html).

**Fix:** darker colors, retaining the original warm palette. Ratios are
calculated for the solid colors, not intermediate hover-transition frames.

## 6. Motion preferences are ignored

The original fade, movement on hover, and spinner do not disable animation for
`prefers-reduced-motion` users.

**Fix:** disable the fade, spinner animation, transitions, and hover movement
when reduced motion is requested. The text loading state remains available.

## 7. No explanation when JavaScript is disabled

The original keeps an apparently actionable button and question text even
though generation cannot run.

**Fix:** add a `noscript` explanation and initialize the button as disabled.
JavaScript enables it after attaching its event listener. This is progressive
enhancement, not server-side generation.

## 8. Course prompt documentation does not match the supplied project

`prompt.md` is a historical generated README template. It mentions separate
`style.css` and `script.js` files, a `LICENSE.md` link, and placeholder demo and
clone URLs. The actual `week1` folder contains only `index.html` and `prompt.md`;
the repository license is named `LICENSE` at the root.

**Fix:** preserve `prompt.md` verbatim for coursework provenance and provide a
separate accurate README, directory layout, local-run instructions, and links
to the actual license. No fabricated live demo URL is presented.

## What is not established as a bug

- An immediate repeated lunch is valid random sampling: each of 12 options
  has probability 1/12 on each draw. No non-repetition rule was requested.
- The page is a random chooser, not a personalized or LLM-driven recommender.
- `innerHTML` only uses hardcoded data in the original; this review does not
  identify an exploitable injection path. The revised page updates an existing
  SVG reference and uses `textContent` for labels.
- Browser timers may run later than 500 ms when the tab is throttled or the
  main thread is busy. The delay is a minimum, not an exact wall-clock guarantee.
- No mobile overflow defect was confirmed; do not infer one solely from the
  original responsive CSS. The revised button has a defensive maximum width.

## Validation and limits

- Six deterministic tests pass against the corrected application's actual
  inline JavaScript: startup, click during startup, near-boundary rapid clicks,
  a 100-click burst, all 12 icon/name pairs and busy states, and repeated draws.
- The original passes the ordinary-startup test and fails all three selected
  overlap/burst regression checks, demonstrating the tests detect its bug.
- Checked the exact Font Awesome 6.4.0 CSS for all 12 original class names.
- Parsed the revised HTML; verified unique IDs, local icon references, and
  absence of external script/stylesheet dependencies. Parsed all 16 SVG symbols.
- Rasterized the 12 food SVGs and visually reviewed the resulting contact sheet.
- Calculated text contrast from the CSS colors.
- Full browser rendering, mobile layout, keyboard behavior, animation timing,
  real offline loading, and screen-reader announcements are **not verified**.
  The available browser blocked local-file navigation under its security policy.
  These remain manual checks; the Node harness and SVG renderer do not replace
  a browser.

## GitHub status

The corrected project is stored in repository `noriginalno/llm4rec`, directory
`hw1/`, together with the original source, tests, documentation, and MIT
license. GitHub forks operate on whole repositories, so this is a
directory-level copy and adaptation of `week1`; it does not claim GitHub's
repository-fork relationship.

GitHub Pages should be served from the repository's `master` branch and root.
The application path is `https://noriginalno.github.io/llm4rec/hw1/`.
