#!/usr/bin/env python3
"""Independently validate every CSV and primary metric in a benchmark run.

Uses only the independent stdlib oracle, never sweettv production helpers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    source = Path(__file__).with_name("independent_verify.py")
    spec = importlib.util.spec_from_file_location("independent_oracle", source)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    template = oracle.read_template_users(args.template)
    def allowed_ids(path: Path) -> set[int]:
        values = oracle.load_json(path)
        if not isinstance(values, list) or any(type(i) is not int or i <= 0 for i in values):
            raise ValueError("Candidate IDs must be a JSON list of positive integers")
        return set(values)
    checks = []
    for split in ("development", "primary", "final"):
        directory = args.run / split
        metrics = oracle.load_json(directory / "metrics.json")
        allowed = allowed_ids(directory / "candidate_ids.json")
        for path in sorted(directory.glob("submission_*.csv")):
            name = path.stem.removeprefix("submission_")
            report = oracle.inspect_submission(path, template, allowed)
            result = {"split": split, "method": name, "valid": True,
                      "rows": report["user_count"], "sha256": report["sha256"]}
            if split != "final":
                actual = oracle.recompute_metrics(
                    oracle.load_json(directory / f"predictions_{name}.json"),
                    oracle.load_json(directory / "labels.json"),
                    oracle.load_json(directory / "evaluation_users.json"),
                )
                expected = metrics[name]["primary_five_label_cohort"]
                for key in ("map_at_5", "precision_at_5", "recall_at_5"):
                    if actual[key] is None or expected[key] is None:
                        if actual[key] != expected[key]:
                            raise ValueError(f"{split}/{name}/{key}: empty-cohort mismatch")
                    elif abs(actual[key] - expected[key]) > 1e-12:
                        raise ValueError(f"{split}/{name}/{key}: metric mismatch")
                result["independent_map_at_5"] = actual["map_at_5"]
            checks.append(result)
    if not checks:
        raise ValueError("Run contains no per-model submissions")
    selected = oracle.load_json(args.run / "selection.json")
    final_csv = args.run / "final" / "submission.csv"
    final_report = oracle.inspect_submission(final_csv, template,
                                             allowed_ids(args.run / "final" / "candidate_ids.json"))
    expected = args.run / "final" / f"submission_{selected['selected_method']}.csv"
    if final_csv.read_bytes() != expected.read_bytes():
        raise ValueError("Selected final CSV is not byte-identical to its designated method file")
    result = {"synthetic": selected["synthetic"], "oracle_imports_production": False,
              "per_model_checks": checks, "per_model_pass": len(checks), "fail": 0,
              "selected_csv": final_report, "selected_file_identity": True}
    out = args.out or args.run / "independent-verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"synthetic": result["synthetic"], "per_model_pass": len(checks),
                      "selected_file_identity": True, "fail": 0, "report": str(out)}, indent=2))


if __name__ == "__main__":
    main()
