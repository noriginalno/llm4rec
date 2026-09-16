# First-repeat distribution experiment

This experiment compares the original uniform lunch generator with the
rejection-weighted generator. It collects 10,000 complete sessions for each
algorithm. Every sequence includes its first repeated suggestion, so
`X = len(sequence)`. With 12 foods, the pigeonhole principle guarantees
`2 <= X <= 13`.

## Algorithms

- **Uniform:** every suggestion has probability `1 / 12` on every draw.
- **Adaptive:** after a unique suggestion is rejected, its weight is halved
  before the next draw. All weights are then normalized by their total, exactly
  as in the web application.

## Reproduce

From the repository root:

```sh
python3 hw1/experiments/first_repeat/collect_sequences.py
python3 -m pip install -r hw1/experiments/first_repeat/requirements.txt
python3 hw1/experiments/first_repeat/plot_distribution.py
```

Both collectors use fixed seeds recorded in `data/summary.json`.

## Saved evidence

| File | Contents |
| --- | --- |
| `collect_sequences.py` | Seeded collection and mathematical validation code |
| `plot_distribution.py` | Plot code reading the aggregated CSV |
| `data/uniform_sessions.csv` | All 10,000 raw uniform sessions |
| `data/adaptive_sessions.csv` | All 10,000 raw rejection-weighted sessions |
| `data/distribution.csv` | Observed counts and exact expected values for each X |
| `data/summary.json` | Seeds, food-ID mapping, means, percentiles, and limits |
| `first_repeat_distribution.svg` | Generated distribution plot |

In the raw files, each character in `sequence_codes` is one suggestion. Codes
`0` through `9` represent the first ten foods, while `A` is Soup and `B` is
BBQ. The complete code-to-food mapping is stored in `data/summary.json`.

## Mathematical check

After `k` different lunches have appeared without a repeat, the conditional
probability that the next suggestion repeats is:

- Uniform: `h(k) = k / 12`
- Adaptive: `h(k) = k / (24 - k)`

For the adaptive case, the `k` seen foods each have weight `0.5`, while the
remaining `12 - k` foods each have weight `1`. Therefore the repeat weight is
`0.5k` and the total weight is `12 - 0.5k`.

The exact probability of the first repeat at position `x` is:

```text
P(X = x) = product(1 - h(k), k=1..x-2) * h(x-1)
```

The generated distribution file includes exact probabilities and expected
counts beside the observed data, making simulation error directly auditable.

## Results

| Metric | Uniform | Rejection-weighted |
| --- | ---: | ---: |
| Collected sessions | 10,000 | 10,000 |
| Observed mean X | 5.0349 | 6.2295 |
| Exact mean X | 5.0361 | 6.2042 |
| Observed median X | 5 | 6 |
| Observed 90th percentile | 8 | 9 |
| First repeat by position 4 | 42.82% | 25.41% |
| Longest observed sequence | 12 | 13 |

The rejection-weighted algorithm delayed the first repeat by `1.1946`
positions in the collected sample, or `23.73%` relative to the uniform mean.
The exact model predicts a `1.1682`-position improvement (`23.20%`), confirming
that the observed shift is an algorithmic effect rather than simulation noise.
