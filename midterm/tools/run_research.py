#!/usr/bin/env python3
"""Run explicitly enumerated local research trials, never Kaggle submissions."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_v2.contracts import make_context
from research_v2.evaluate import evaluate_model, label_frequencies
from sweettv.cache import load_prepared
from sweettv.pipeline import save_json
from tools.prepare_official import preparation_signature


FOLDS = {
    "development": (1586131200, 1586131200, 1588550400),  # Apr06 -> May04.
    "late_development": (1590969600, 1590969600, 1593388800),  # Jun01 -> Jun29.
    "confirmation": (1593388800, 1593388800, 1595808000),  # Jun29 -> Jul27.
    "primary12": (1588550400, 1588550400, 1595808000),
    "final": (1595808000, 1595808000, 1603065600),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", choices=list(FOLDS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/processed/official-v1"))
    args = parser.parse_args()
    grid = json.loads(args.grid.read_text())
    if not isinstance(grid, list) or not grid or len({r["name"] for r in grid}) != len(grid):
        raise ValueError("Grid must be a nonempty list of uniquely named trials")
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "grid.json", grid)
    prepared = load_prepared(args.cache, data_source_sha256=preparation_signature(Path("data/raw")))
    all_results = {}
    for fold in args.folds:
        context, labels, audit = make_context(prepared, *FOLDS[fold])
        users = list(prepared.template_users)
        frequencies = label_frequencies(prepared, context, users)
        folder = args.out / fold
        folder.mkdir()
        save_json(folder / "labels.json", {u: labels[u] for u in users})
        save_json(folder / "template_users.json", users)
        save_json(folder / "candidate_ids.json", context.data.item_ids[context.candidate_indices])
        save_json(folder / "audit.json", audit)
        all_results[fold] = {}
        for trial in grid:
            module, class_name = trial["class"].split(":", 1)
            if not module.startswith("research_v2."):
                raise ValueError("Only project-owned research classes may be executed")
            cls = getattr(importlib.import_module(module), class_name)
            params = trial.get("parameters", {})
            result = evaluate_model(cls(**params), context, labels, users, frequencies,
                                    folder / trial["name"], parameters=params)
            all_results[fold][trial["name"]] = result
            print(json.dumps({"fold": fold, "trial": trial["name"], "map_at_5": result["map_at_5"],
                              "tie_expected": result["tie_marginalized_map_at_5"],
                              "all_user_map": result["all_user_map_at_5"]}), flush=True)
        del context
    save_json(args.out / "results.json", all_results)


if __name__ == "__main__":
    main()
