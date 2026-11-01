#!/usr/bin/env python3
"""Reverse final score ties without changing labels; label reversal stays separate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_v2.contracts import make_context
from research_v2.evaluate import label_frequencies, tie_marginalized_ap
from sweettv.cache import load_prepared
from sweettv.evaluation import evaluate_predictions
from sweettv.pipeline import save_json
from tools.prepare_official import preparation_signature
from tools.run_research import FOLDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, nargs="+", required=True)
    parser.add_argument("--fold", choices=list(FOLDS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    prepared = load_prepared(Path("data/processed/official-v1"),
                             data_source_sha256=preparation_signature(Path("data/raw")))
    context, labels, _ = make_context(prepared, *FOLDS[args.fold])
    users = list(prepared.template_users)
    primary = [u for u in users if len(labels[u]) == 5]
    frequencies = label_frequencies(prepared, context, users)
    candidates = context.data.item_ids[context.candidate_indices]
    popularity = np.asarray(context.data.interactions.sum(axis=0)).ravel()[context.candidate_indices]
    reversed_labels = {u: sorted(frequencies[u], key=lambda i: (-frequencies[u][i], -i))[:5] for u in users}
    result = {}
    for path in args.trials:
        if json.loads((path.parent / "candidate_ids.json").read_text()) != candidates.tolist() or \
                json.loads((path.parent / "template_users.json").read_text()) != users:
            raise ValueError("Trial cohort/candidate alignment differs")
        meta = json.loads((path / "result.json").read_text())
        scores = np.load(meta["score_file"], mmap_mode="r", allow_pickle=False)
        if scores.shape != (len(users), len(candidates)):
            raise ValueError("Score cache shape differs")
        predictions = {}
        for row, user in enumerate(users):
            values = scores[row]
            positions = np.flatnonzero(np.isfinite(values))
            order = positions[np.lexsort((-candidates[positions], -values[positions]))][:5]
            used = set(order.tolist())
            remaining = np.array([j for j in range(len(candidates)) if j not in used])
            fallback = remaining[np.lexsort((-candidates[remaining], -popularity[remaining]))][:5-len(order)]
            predictions[user] = candidates[np.r_[order, fallback]].tolist()
        result[str(path)] = {
            "ascending_score_ties_map": meta["map_at_5"],
            "reverse_final_score_ties_original_labels": evaluate_predictions(predictions, labels, primary)["map_at_5"],
            "reverse_final_score_ties_reverse_labels": evaluate_predictions(predictions, reversed_labels, primary)["map_at_5"],
            "reverse_final_score_ties_marginalized_labels": float(np.mean([
                tie_marginalized_ap(predictions[u], frequencies[u]) for u in primary])),
            "caveat": "For nested RRF, constituent tie direction is a separately fitted explicit sensitivity trial",
        }
    save_json(args.out, {"fold": args.fold, "strict_five_users": len(primary), "trials": result})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
