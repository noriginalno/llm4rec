# Homework 1 — Random Lunch Menu Generator

The page initially gives all 12 lunches equal weight. Pressing **Generate
Lunch!** rejects the displayed suggestion: its weight is halved before the next
weighted draw, while the other probabilities increase proportionally through
normalization. A 500 ms loading state precedes each result. When another click
arrives during that delay, it replaces the pending choice and restarts the
delay. Only the latest choice is displayed, and the previously visible lunch is
not penalized again while the page is already loading.

## Run

Open `index.html` in a current browser. Everything the page needs is embedded,
including the SVG drawings. There are no third-party scripts, icon fonts,
external stylesheets, API requests, or build dependencies.

## Changes

- Replaced missing and misleading Font Awesome icons with 12 matching SVGs.
- Added rejection-aware weighted sampling: clicking again halves the displayed
  lunch's relative weight before regeneration.
- Cancelled superseded timers, including the timer started on page load.
- Added a polite result announcement and a busy state for assistive technology.
- Increased text contrast, added a visible keyboard focus style, and respected
  the operating system's reduced-motion preference.
- Added an explanation when JavaScript is disabled; the button stays disabled
  until initialization completes.

See [BUG_REPORT.md](BUG_REPORT.md) for the full original-icon mapping,
reproduction steps, evidence, and validation limits.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Corrected application, styles, icons, and script |
| `prompt.md` | Original-style project description updated for this repository |
| `upstream/index.html` | Original application, preserved for comparison |
| `tests/generator.test.cjs` | Deterministic tests of the actual inline script |
| `BUG_REPORT.md` | Findings, fixes, evidence, and remaining checks |
| `../LICENSE` | Original MIT license |

## Tests

With Node.js 20 or later, from the repository root:

```sh
node --test hw1/tests/generator.test.cjs
```

To reproduce the three scheduling failures in the original (POSIX shell):

```sh
LUNCH_HTML=hw1/upstream/index.html node --test --test-name-pattern='page load completes|click during|rapid clicks|burst' hw1/tests/generator.test.cjs
```

The tests execute the real inline script with a minimal DOM and a virtual
clock. They do not substitute for browser rendering, keyboard, or screen-reader
testing. The corrected application passes all nine tests; the original fails
three of the four selected scheduling checks.

## GitHub Pages

GitHub Pages is published from `master` and the repository root. The application
is available at <https://noriginalno.github.io/llm4rec/hw1/>.

## Attribution

Based on [dryjins/RecSys-LLMs/week1](https://github.com/dryjins/RecSys-LLMs/tree/7b4e5a35e15dda8d87e73636e0a6fb178d203534/week1),
by Joseph Seungmin Jin. The original [MIT license](../LICENSE) is preserved.
The embedded SVG drawings were created for this revision.
