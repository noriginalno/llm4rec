#!/usr/bin/env python3
"""Plot the observed first-repeat distribution produced by collect_sequences.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=experiment_dir / "data" / "distribution.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=experiment_dir / "first_repeat_distribution.svg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams["svg.hashsalt"] = "hw1-first-repeat"
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    x = [int(row["x"]) for row in rows]
    uniform = [int(row["uniform_count"]) for row in rows]
    adaptive = [int(row["adaptive_count"]) for row in rows]

    figure, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    width = 0.38
    axis.bar(
        [value - width / 2 for value in x],
        uniform,
        width,
        label="Uniform random",
        color="#5677A6",
    )
    axis.bar(
        [value + width / 2 for value in x],
        adaptive,
        width,
        label="Rejection-weighted",
        color="#D97757",
    )
    axis.set_title("Position of the first repeated lunch")
    axis.set_xlabel("X — position of first repeated suggestion")
    axis.set_ylabel("Y — number of sessions")
    axis.set_xticks(x)
    axis.set_xlim(1.4, 13.6)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    figure.suptitle("10,000 collected sessions per algorithm", fontsize=10, y=0.94)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        metadata={"Creator": "hw1 first-repeat experiment", "Date": None},
    )
    plt.close(figure)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
