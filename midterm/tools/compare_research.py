#!/usr/bin/env python3
"""Paired-user descriptive uncertainty for a frozen research comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def paired(first, second, seed=20261101, replicates=5000):
    users = [r["user_id"] for r in first["per_user"]]
    if users != [r["user_id"] for r in second["per_user"]]:
        raise ValueError("Paired metric users/order differ")
    delta = np.array([b["ap"] - a["ap"] for a, b in zip(first["per_user"], second["per_user"])])
    if not len(delta):
        raise ValueError("Cannot compare an empty cohort")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, replicates, 200):
        draws = rng.integers(0, len(delta), size=(min(200, replicates-start), len(delta)))
        means.extend(delta[draws].mean(axis=1).tolist())
    return {"users": len(delta), "baseline_map": first["map_at_5"],
            "candidate_map": second["map_at_5"], "mean_paired_delta": float(delta.mean()),
            "descriptive_paired_bootstrap_95_interval": np.quantile(means, [.025, .975]).tolist(),
            "wins": int((delta > 1e-12).sum()), "losses": int((delta < -1e-12).sum()),
            "ties": int((np.abs(delta) <= 1e-12).sum()), "seed": seed, "replicates": replicates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = {"baseline": str(args.baseline), "comparisons": {},
              "caveat": "Post-selection reused-history descriptive intervals, not independent test confidence or a Kaggle-score prediction"}
    for candidate in args.candidates:
        for filename in ("candidate_ids.json", "labels.json", "template_users.json"):
            if json.loads((args.baseline.parent / filename).read_text()) != json.loads((candidate.parent / filename).read_text()):
                raise ValueError(f"Comparison alignment differs: {filename}")
        result["comparisons"][str(candidate)] = {
            "strict_five": paired(json.loads((args.baseline / "primary_metrics.json").read_text()),
                                  json.loads((candidate / "primary_metrics.json").read_text())),
            "all_users": paired(json.loads((args.baseline / "diagnostic_metrics.json").read_text()),
                                json.loads((candidate / "diagnostic_metrics.json").read_text())),
        }
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
