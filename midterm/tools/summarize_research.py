#!/usr/bin/env python3
"""Summarize all completed development trials without hiding weak configurations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FOLDS = ("development", "late_development")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    trials = {}
    for root in args.roots:
        for path in root.glob("*/*/result.json"):
            fold, name = path.parent.parent.name, path.parent.name
            if fold not in FOLDS:
                continue
            if (fold, name) in trials:
                raise ValueError("Duplicate trial name/fold in supplied roots")
            value = json.loads(path.read_text())
            value["path"] = str(path.parent)
            value["grid_root"] = str(root)
            trials[(fold, name)] = value
    controls = {f: trials[(f, "legacy_hybrid")] for f in FOLDS}
    summaries = []
    for name in sorted({name for fold, name in trials}):
        if not all((fold, name) in trials for fold in FOLDS):
            raise ValueError(f"Incomplete development comparison: {name}")
        rows = {f: trials[(f, name)] for f in FOLDS}
        for fold, row in rows.items():
            root = Path(row["grid_root"]) / fold
            control_root = Path(controls[fold]["grid_root"]) / fold
            for filename in ("labels.json", "candidate_ids.json", "template_users.json"):
                if json.loads((root / filename).read_text()) != json.loads((control_root / filename).read_text()):
                    raise ValueError(f"Different evaluation alignment: {name}/{fold}/{filename}")
        strict = [rows[f]["map_at_5"] - controls[f]["map_at_5"] for f in FOLDS]
        all_users = [rows[f]["all_user_map_at_5"] - controls[f]["all_user_map_at_5"] for f in FOLDS]
        summaries.append({
            "name": name, "class": rows[FOLDS[0]]["class"],
            "development_map": rows[FOLDS[0]]["map_at_5"],
            "late_development_map": rows[FOLDS[1]]["map_at_5"],
            "development_delta": strict[0], "late_development_delta": strict[1],
            "development_all_delta": all_users[0], "late_development_all_delta": all_users[1],
            "mean_all_delta": sum(all_users) / 2,
            "standalone_nonregression_gate": name != "legacy_hybrid" and min(strict + all_users) >= 0,
            "development_expected_tie_map": rows[FOLDS[0]]["tie_marginalized_map_at_5"],
            "late_development_expected_tie_map": rows[FOLDS[1]]["tie_marginalized_map_at_5"],
            "paths": {f: rows[f]["path"] for f in FOLDS},
        })
    summaries.sort(key=lambda r: -r["mean_all_delta"])
    with (args.out / "scoreboard.json").open("x") as handle:
        json.dump({"trial_count": len(summaries), "fold_count": 2,
                   "controls": {f: controls[f]["path"] for f in FOLDS}, "trials": summaries}, handle, indent=2)
        handle.write("\n")
    with (args.out / "scoreboard.csv").open("x", newline="") as handle:
        fields = [k for k in summaries[0] if k != "paths"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    markdown = ["# Development scoreboard", "",
                "Reused historical validation; not an untouched test or a Kaggle score.", "",
                "| Trial | Early MAP | Later MAP | Mean all-user delta | Both-cohort gate |",
                "| --- | ---: | ---: | ---: | --- |"]
    for row in summaries:
        markdown.append(f'| {row["name"]} | {row["development_map"]:.6f} | '
                        f'{row["late_development_map"]:.6f} | {row["mean_all_delta"]:+.6f} | '
                        f'{"pass" if row["standalone_nonregression_gate"] else "no"} |')
    (args.out / "scoreboard.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps({"configurations": len(summaries), "standalone_passes": [
        r["name"] for r in summaries if r["standalone_nonregression_gate"]], "output": str(args.out)}))


if __name__ == "__main__":
    main()
