# Homework 1 — Implementation brief

Audit and improve the Random Lunch Menu Generator from the upstream `week1`
folder, then publish the corrected application in `noriginalno/llm4rec/hw1`.

## Application requirements

- Keep the application in one self-contained `index.html` file.
- Randomly select one of 12 lunch options on initial page load and whenever the
  user presses **Generate Lunch!**.
- Display a visible loading state for 500 ms before showing the result.
- If another generation starts during that delay, cancel the previous pending
  result. Only the latest request may update the page.
- Show a distinct, semantically correct icon for every lunch option: Pizza,
  Sushi, Burger, Salad, Tacos, Ramen, Sandwich, Pasta, Curry, Steak, Soup, and
  BBQ.
- Keep all icons inside the document as SVG symbols. The application must not
  depend on an external icon font, stylesheet, script, API, or build process.
- Update the displayed icon and lunch name together.
- Keep the layout usable on desktop and mobile screens.

## Accessibility requirements

- Announce generated results through a polite live region.
- Expose the loading state with `aria-busy`.
- Treat decorative icons as hidden from assistive technology.
- Provide a visible keyboard-focus style.
- Respect `prefers-reduced-motion`.
- Provide an explanation when JavaScript is disabled.
- Use text and control colors with sufficient contrast.

## Repository requirements

```text
llm4rec/
├── LICENSE
├── README.md
└── hw1/
    ├── index.html
    ├── prompt.md
    ├── README.md
    ├── BUG_REPORT.md
    ├── upstream/
    │   └── index.html
    └── tests/
        └── generator.test.cjs
```

- Preserve the unmodified upstream HTML in `upstream/index.html` for comparison.
- Document every confirmed defect, its evidence, and its correction in
  `BUG_REPORT.md`.
- Include deterministic tests covering initial generation, overlapping timers,
  rapid clicks, a large click burst, all 12 icon/name pairs, accessibility busy
  states, and repeated random results.
- Preserve the upstream MIT license and attribution.

## Validation

Run from the repository root:

```sh
node --test hw1/tests/generator.test.cjs
```

All tests must pass for the corrected page. The overlap and burst regression
tests must demonstrate the scheduling bug when run against the upstream page.

## Deployment

Publish GitHub Pages from the `master` branch and repository root. The completed
application must be available at:

<https://noriginalno.github.io/llm4rec/hw1/>
