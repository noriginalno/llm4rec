#!/usr/bin/env python3
"""Freeze at most two new candidates using development results only."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path


FOLDS = ("development", "late_development")
RAW_NAME = "raw_day_h56_p1"
RAW_ROOT = Path("results/research-v2/explore-raw-habits-001")
RAW_SPEC = {"name": RAW_NAME, "class": "research_v2.raw_habits:RawScheduleModel",
            "parameters": {"half_life_days": 56, "slot_mode": "day", "coverage_power": 1}}


def read_results(root, name):
    return {fold: json.loads((root / fold / name / "result.json").read_text()) for fold in FOLDS}


def mean_all(results):
    return sum(r["all_user_map_at_5"] for r in results.values()) / len(results)


def reverse_spec(spec):
    spec = deepcopy(spec)
    spec["name"] += "_reverse_constituent_ties"
    spec["parameters"]["reverse_score_ties"] = True
    for component in spec["parameters"].get("components", []):
        if component["class"] == "research_v2.legacy:LegacyHybrid":
            component.setdefault("parameters", {})["reverse_score_ties"] = True
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    raw = read_results(RAW_ROOT, RAW_NAME)
    specs = json.loads((args.fusion_root / "grid.json").read_text())
    scored, passing = [], []
    for spec in specs:
        results = read_results(args.fusion_root, spec["name"])
        for fold in FOLDS:
            for filename in ("labels.json", "candidate_ids.json", "template_users.json"):
                if json.loads((args.fusion_root / fold / filename).read_text()) != json.loads((RAW_ROOT / fold / filename).read_text()):
                    raise ValueError("Development alignment differs")
        deltas = [results[f][key] - raw[f][key] for f in FOLDS
                  for key in ("map_at_5", "all_user_map_at_5")]
        entry = {"name": spec["name"], "mean_all_user_map": mean_all(results),
                 "deltas_against_raw": deltas, "beats_raw_both_windows_and_cohorts": min(deltas) > 0,
                 "spec": spec}
        scored.append(entry)
        if entry["beats_raw_both_windows_and_cohorts"]:
            passing.append(entry)
    passing.sort(key=lambda e: -e["mean_all_user_map"])
    best = passing[0] if passing else None
    if best:
        plateau = [e for e in passing if best["mean_all_user_map"] - e["mean_all_user_map"] <= .001]
        # A repeat expert is smaller/faster than fitting all four original models.
        best = min(plateau, key=lambda e: ("legacy" in e["name"], -e["mean_all_user_map"]))
    grid = [
        {"name": "legacy_hybrid", "class": "research_v2.legacy:LegacyHybrid", "parameters": {}},
        {"name": "legacy_reverse_constituent_ties", "class": "research_v2.legacy:LegacyHybrid",
         "parameters": {"reverse_score_ties": True}},
        RAW_SPEC,
    ]
    if best:
        grid.extend([best["spec"], reverse_spec(best["spec"])])
    source_paths = [Path(__file__), args.fusion_root / "grid.json", Path("research_v2/SELECTION.md")]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "development_only": True, "confirmation_seen": False,
        "standalone": RAW_SPEC, "blends": scored,
        "chosen_blend": best["name"] if best else None,
        "confirmation_new_candidates": [RAW_NAME] + ([best["name"]] if best else []),
        "decision_rule": "Use chosen development blend if it passes both positive strict/all-user confirmation deltas and tie sensitivity; otherwise use frozen raw if it passes; otherwise do not promote. Do not retune after confirmation.",
        "confirmation_folds": ["confirmation", "primary12"],
        "new_kaggle_submission_authorized": False,
    }
    for name, value in (("selection.json", result), ("grid.json", grid)):
        with (args.out / name).open("x") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
    print(json.dumps({"new_candidates": result["confirmation_new_candidates"],
                      "grid_entries_including_controls": len(grid), "output": str(args.out)}))


if __name__ == "__main__":
    main()
