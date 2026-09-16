#!/usr/bin/env python3
"""Collect reproducible first-repeat sessions for both lunch algorithms."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


FOODS = (
    "Pizza",
    "Sushi",
    "Burger",
    "Salad",
    "Tacos",
    "Ramen",
    "Sandwich",
    "Pasta",
    "Curry",
    "Steak",
    "Soup",
    "BBQ",
)
ALGORITHMS = ("uniform", "adaptive")
MAX_SEQUENCE_LENGTH = len(FOODS) + 1
FOOD_CODES = "0123456789AB"


def weighted_choice(rng: random.Random, weights: list[float]) -> int:
    """Return an index using the same cumulative-weight rule as the web app."""
    target = rng.random() * sum(weights)
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if target < cumulative:
            return index
    return len(weights) - 1


def generate_session(algorithm: str, rng: random.Random) -> list[int]:
    """Generate suggestions through and including the first repeated item."""
    weights = [1.0] * len(FOODS)
    seen: set[int] = set()
    sequence: list[int] = []

    for _ in range(MAX_SEQUENCE_LENGTH):
        if algorithm == "adaptive" and sequence:
            # Continuing the session means the user rejected the last suggestion.
            weights[sequence[-1]] *= 0.5

        if algorithm == "uniform":
            suggestion = rng.randrange(len(FOODS))
        elif algorithm == "adaptive":
            suggestion = weighted_choice(rng, weights)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        sequence.append(suggestion)
        if suggestion in seen:
            return sequence
        seen.add(suggestion)

    raise AssertionError("A repeat must occur by the 13th suggestion")


def exact_distribution(algorithm: str) -> dict[int, float]:
    """Calculate the exact first-repeat probabilities for validation."""
    survival_probability = 1.0
    probabilities: dict[int, float] = {}

    for x in range(2, MAX_SEQUENCE_LENGTH + 1):
        distinct_seen = x - 1
        if algorithm == "uniform":
            repeat_probability = distinct_seen / len(FOODS)
        else:
            # Seen foods have weight 0.5; unseen foods retain weight 1.
            repeat_probability = distinct_seen / (2 * len(FOODS) - distinct_seen)

        probabilities[x] = survival_probability * repeat_probability
        survival_probability *= 1.0 - repeat_probability

    return probabilities


def percentile(counts: Counter[int], total: int, probability: float) -> int:
    threshold = total * probability
    cumulative = 0
    for x in range(2, MAX_SEQUENCE_LENGTH + 1):
        cumulative += counts[x]
        if cumulative >= threshold:
            return x
    return MAX_SEQUENCE_LENGTH


def validate_session(sequence: list[int]) -> None:
    assert 2 <= len(sequence) <= MAX_SEQUENCE_LENGTH
    assert len(set(sequence[:-1])) == len(sequence) - 1
    assert sequence[-1] in sequence[:-1]
    assert all(0 <= suggestion < len(FOODS) for suggestion in sequence)


def collect(sessions: int, seeds: dict[str, int]) -> tuple[list[dict], dict[str, Counter[int]]]:
    rows: list[dict] = []
    counts = {algorithm: Counter() for algorithm in ALGORITHMS}

    for algorithm in ALGORITHMS:
        rng = random.Random(seeds[algorithm])
        for session_id in range(1, sessions + 1):
            sequence = generate_session(algorithm, rng)
            validate_session(sequence)
            x = len(sequence)
            counts[algorithm][x] += 1
            rows.append(
                {
                    "algorithm": algorithm,
                    "session_id": session_id,
                    "x": x,
                    "sequence_codes": "".join(FOOD_CODES[item] for item in sequence),
                }
            )

    assert len(rows) == sessions * len(ALGORITHMS)
    assert all(sum(count.values()) == sessions for count in counts.values())
    return rows, counts


def write_outputs(
    output_dir: Path,
    sessions: int,
    seeds: dict[str, int],
    rows: list[dict],
    counts: dict[str, Counter[int]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for algorithm in ALGORITHMS:
        with (output_dir / f"{algorithm}_sessions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("session_id", "x", "sequence_codes"),
            )
            writer.writeheader()
            writer.writerows(
                {
                    "session_id": row["session_id"],
                    "x": row["x"],
                    "sequence_codes": row["sequence_codes"],
                }
                for row in rows
                if row["algorithm"] == algorithm
            )

    exact = {algorithm: exact_distribution(algorithm) for algorithm in ALGORITHMS}
    with (output_dir / "distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["x"]
        for algorithm in ALGORITHMS:
            fieldnames.extend(
                (
                    f"{algorithm}_count",
                    f"{algorithm}_percent",
                    f"{algorithm}_exact_probability",
                    f"{algorithm}_exact_expected_count",
                    f"{algorithm}_observed_minus_expected",
                )
            )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for x in range(2, MAX_SEQUENCE_LENGTH + 1):
            row: dict[str, int | str] = {"x": x}
            for algorithm in ALGORITHMS:
                observed = counts[algorithm][x]
                expected = sessions * exact[algorithm][x]
                row[f"{algorithm}_count"] = observed
                row[f"{algorithm}_percent"] = f"{100 * observed / sessions:.2f}"
                row[f"{algorithm}_exact_probability"] = f"{exact[algorithm][x]:.8f}"
                row[f"{algorithm}_exact_expected_count"] = f"{expected:.3f}"
                row[f"{algorithm}_observed_minus_expected"] = f"{observed - expected:.3f}"
            writer.writerow(row)

    summary: dict[str, object] = {
        "food_count": len(FOODS),
        "foods_by_id": dict(enumerate(FOODS)),
        "food_codes": dict(zip(FOOD_CODES, FOODS)),
        "sessions_per_algorithm": sessions,
        "total_sessions": sessions * len(ALGORITHMS),
        "maximum_allowed_sequence_length": MAX_SEQUENCE_LENGTH,
        "algorithms": {},
    }
    algorithm_summaries: dict[str, object] = {}
    for algorithm in ALGORITHMS:
        observed_mean = sum(x * counts[algorithm][x] for x in counts[algorithm]) / sessions
        exact_mean = sum(x * probability for x, probability in exact[algorithm].items())
        algorithm_summaries[algorithm] = {
            "seed": seeds[algorithm],
            "observed_mean_x": round(observed_mean, 4),
            "exact_mean_x": round(exact_mean, 4),
            "observed_median_x": percentile(counts[algorithm], sessions, 0.5),
            "observed_p90_x": percentile(counts[algorithm], sessions, 0.9),
            "observed_min_x": min(counts[algorithm]),
            "observed_max_x": max(counts[algorithm]),
        }
    summary["algorithms"] = algorithm_summaries

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=10_000)
    parser.add_argument("--uniform-seed", type=int, default=20_260_916)
    parser.add_argument("--adaptive-seed", type=int, default=20_260_917)
    parser.add_argument("--output-dir", type=Path, default=experiment_dir / "data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sessions <= 0:
        raise SystemExit("--sessions must be positive")
    seeds = {"uniform": args.uniform_seed, "adaptive": args.adaptive_seed}
    rows, counts = collect(args.sessions, seeds)
    write_outputs(args.output_dir, args.sessions, seeds, rows, counts)
    print(f"Wrote {len(rows):,} sessions to {args.output_dir}")
    for algorithm in ALGORITHMS:
        mean_x = sum(x * counts[algorithm][x] for x in counts[algorithm]) / args.sessions
        print(f"{algorithm}: count={sum(counts[algorithm].values()):,}, mean X={mean_x:.4f}")


if __name__ == "__main__":
    main()
