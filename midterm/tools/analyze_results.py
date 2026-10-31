#!/usr/bin/env python3
"""Descriptive business/algorithm analysis; does not tune or alter predictions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = args.run / "primary"
    summary = json.loads((root / "summary.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    timings = json.loads((root / "timings.json").read_text())
    users = json.loads((root / "evaluation_users.json").read_text())
    all_users = json.loads((root / "template_users.json").read_text())
    labels = json.loads((root / "labels.json").read_text())
    methods = list(summary["primary_map_at_5"])
    predictions = {name: json.loads((root / f"predictions_{name}.json").read_text()) for name in methods}
    train_end = summary["split"]["provenance"]["train_end"]
    forecast_start = summary["split"]["provenance"]["forecast_start"]
    forecast_end = summary["split"]["provenance"]["forecast_end"]
    history, heldout = defaultdict(Counter), defaultdict(Counter)
    global_popularity = Counter()
    events = pq.read_table(args.cache / "events.parquet", columns=["user_id", "tv_show_id", "start", "end"]).to_pylist()
    for event in events:
        u, i = event["user_id"], event["tv_show_id"]
        if event["end"] <= train_end:
            history[u][i] += 1
            global_popularity[i] += 1
        elif event["start"] >= forecast_start and event["end"] <= forecast_end:
            heldout[u][i] += 1
    del events
    derived_labels = {u: sorted(heldout[u], key=lambda i: (-heldout[u][i], i))[:5] for u in all_users}
    if derived_labels != labels:
        raise ValueError("Cache-derived frequency labels disagree with saved labels")
    reverse_labels = {u: sorted(heldout[u], key=lambda i: (-heldout[u][i], -i))[:5] for u in all_users}
    # Independent formula, not an import of the production evaluator.
    def ap(predicted, relevant):
        relevant = set(relevant)
        if not relevant:
            return 0.0
        found, total = set(), 0.0
        for rank, item in enumerate(predicted[:5], 1):
            if item in relevant and item not in found:
                found.add(item)
                total += len(found) / rank
        return total / min(5, len(relevant))
    table = []
    for method in methods:
        ordinary = [ap(predictions[method][u], labels[u]) for u in users]
        reverse = [ap(predictions[method][u], reverse_labels[u]) for u in users]
        if abs(float(np.mean(ordinary)) - summary["primary_map_at_5"][method]) > 1e-12:
            raise ValueError("Analysis AP disagrees with saved primary metric")
        unseen = sum(i not in history[u] for u in users for i in predictions[method][u])
        unique = {i for u in users for i in predictions[method][u]}
        cost = timings[method]["fit_seconds"] + timings[method]["score_seconds"]
        if method in summary["fusions"]:
            cost += sum(timings[n]["fit_seconds"] + timings[n]["score_seconds"]
                        for n, weight in summary["fusions"][method].items() if weight > 0)
        table.append({
            "method": method, "primary_map_at_5": float(np.mean(ordinary)),
            "all_template_map_at_5": metrics[method]["all_template_users_diagnostic"]["map_at_5"],
            "reverse_id_label_tie_map_at_5": float(np.mean(reverse)),
            "unseen_in_user_training_slots": unseen, "slots": len(users) * 5,
            "unique_recommended_ids": len(unique),
            "mean_training_program_popularity": float(np.mean([global_popularity[i] for u in users for i in predictions[method][u]])),
            "fit_plus_score_seconds_including_hybrid_constituents": cost,
            "fallback": summary["support"][method],
        })
    hybrid = next(iter(summary["fusions"]))
    first = np.array([ap(predictions[hybrid][u], labels[u]) for u in users])
    second = np.array([ap(predictions["ppr"][u], labels[u]) for u in users])
    differences = first - second
    rng = np.random.default_rng(20261101)
    samples = []
    for _ in range(10):
        indices = rng.integers(0, len(users), size=(200, len(users)))
        samples.extend(differences[indices].mean(axis=1).tolist())
    ci = np.quantile(samples, [0.025, 0.975]).tolist()
    tied_users = []
    for u in users:
        ranked = sorted(heldout[u], key=lambda i: (-heldout[u][i], i))
        if len(ranked) > 5 and heldout[u][ranked[4]] == heldout[u][ranked[5]]:
            tied_users.append(u)
    examples = summary["example_users"]
    with (root / "recommendations.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["user_id"] in examples]
    with (args.out / "two_user_recommendations.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "source": str(args.run), "primary_users": len(users), "all_template_users": len(all_users),
        "method_table": table,
        "hybrid_vs_ppr": {"mean_difference": float(differences.mean()), "bootstrap_95_percent_interval": ci,
                           "wins": int((differences > 0).sum()), "losses": int((differences < 0).sum()),
                           "ties": int((differences == 0).sum()), "seed": 20261101, "resamples": 2000,
                           "scope": "Paired users, fitted models and this period fixed; not independent test, temporal uncertainty or causal impact"},
        "primary_users_with_top5_boundary_ties": len(tied_users),
        "label_tie_sensitivity_note": "Reverse numeric-ID tie policy changes relevant membership only; predictions and user cohort remain fixed. Neither tie policy is confirmed as organizer policy.",
        "relevant_training_seen_slots": sum(i in history[u] for u in users for i in labels[u]),
        "relevant_slots": len(users) * 5,
        "example_users": examples,
        "examples": {u: {"history": dict(history[u]), "heldout_frequency": dict(heldout[u]), "labels": labels[u],
                          "predictions": {name: predictions[name][u] for name in methods},
                          "ap": {name: ap(predictions[name][u], labels[u]) for name in methods}} for u in examples},
        "time_scope": "Fit and model scoring kernels plus fusion; initialization, shared ranking/export/ETL excluded. Not online latency or total production cost.",
        "kaggle_score": None, "predictions_modified": False,
    }
    with (args.out / "analysis.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    lines = ["# Actual-data validation analysis", "", "These are temporal model-selection results, not returned Kaggle scores.", "",
             f"Common primary cohort: **{len(users)} users**; all-template diagnostic: **{len(all_users)} users**.", "",
             "| Method | Primary MAP@5 | All-template MAP@5 | Alternate tie MAP@5 | Fit + score seconds* |",
             "|---|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['method']} | {row['primary_map_at_5']:.6f} | {row['all_template_map_at_5']:.6f} | {row['reverse_id_label_tie_map_at_5']:.6f} | {row['fit_plus_score_seconds_including_hybrid_constituents']:.3f} |")
    lines += ["", "*Hybrid includes every positively weighted constituent. Shared ranking/export/ETL and initialization are excluded; these are not online latencies.", "",
              f"Hybrid minus PPR: **{differences.mean():+.6f} MAP**, paired-user descriptive 95% bootstrap interval **[{ci[0]:+.6f}, {ci[1]:+.6f}]**; wins/losses/ties **{(differences>0).sum()}/{(differences<0).sum()}/{(differences==0).sum()}**.",
              "The same validation period selected the final method. This interval does not remove selection bias or forecast-period uncertainty.", "",
              f"**{len(tied_users)}** primary users have a frequency tie across the fifth-label boundary. Alternate-ID sensitivity is diagnostic, not a reason to select the most flattering metric.", "",
              "## Same two training-selected users", "",
              "Labels describe program IDs, not exact episodes; 4,351 source IDs have multiple titles. Full representative source titles/categories are retained in the CSV."]
    for u in examples:
        total = sum(history[u].values())
        lines += ["", f"### User {u}", f"Training: {total} qualifying broadcasts, {len(history[u])} distinct programs; largest share {max(history[u].values())/total:.2%}.",
                  "| Method | AP@5 | Top-5 IDs |", "|---|---:|---|"]
        for method in methods:
            lines.append(f"| {method} | {ap(predictions[method][u], labels[u]):.6f} | {' '.join(map(str,predictions[method][u]))} |")
    with (args.out / "analysis.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"primary_users":len(users), "hybrid_vs_ppr":result["hybrid_vs_ppr"], "boundary_tie_users":len(tied_users), "analysis":str(args.out / 'analysis.md')},indent=2))


if __name__ == "__main__":
    main()
