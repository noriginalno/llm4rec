#!/usr/bin/env python3
"""Read approved submission scores and preserve exact candidate/final CSV bytes.

This script does not upload, select website flags, retrain, or publish anything.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kaggle_environment import configure_isolated_home
configure_isolated_home(ROOT)

from kaggle.api.kaggle_api_extended import KaggleApi
from sweettv.submission import copy_submission_exclusively, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--refs", nargs="+", required=True, help="Approved method:submission_ref pairs")
    args = parser.parse_args()
    refs = {}
    for value in args.refs:
        method, ref = value.split(":", 1)
        if method not in {"hybrid_balanced", "ppr", "repeat"} or method in refs:
            raise ValueError("Only the three approved, unique candidates are accepted")
        refs[method] = int(ref)
    api = KaggleApi()
    api.authenticate()
    if api.get_config_value(api.CONFIG_NAME_USER) != args.account:
        raise ValueError("Active Kaggle account differs from the approved account")
    records = {}
    for method, ref in refs.items():
        submission = api.get_submission(ref)
        status = submission.status.name
        if status != "COMPLETE" or submission.private_score in (None, ""):
            raise ValueError(f"Submission {ref} is not complete with a returned private score")
        score = Decimal(str(submission.private_score))
        if not score.is_finite():
            raise ValueError("Nonfinite returned score")
        source = args.run / "final" / f"submission_{method}.csv"
        records[method] = {
            "submission_ref": ref, "status": status,
            "public_score": str(submission.public_score), "private_score": str(submission.private_score),
            "description": submission.description, "remote_filename": submission.file_name,
            "submission_date_as_returned": str(submission.date),
            "source_file": str(source), "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    selected = max(records, key=lambda name: (Decimal(records[name]["private_score"]), name))
    args.out.mkdir(parents=True, exist_ok=False)
    template = json.loads((args.run / "final" / "template_users.json").read_text())
    allowed = json.loads((args.run / "final" / "candidate_ids.json").read_text())
    for source in sorted((args.run / "final").glob("submission_*.csv")):
        destination = args.out / source.name
        copy_submission_exclusively(source, destination)
        validate_submission(destination, template, allowed)
    final = args.out / "submission.csv"
    copy_submission_exclusively(args.run / "final" / f"submission_{selected}.csv", final)
    validation = validate_submission(final, template, allowed)
    result = {
        "competition": "sweettv-tv-program-recommender", "kaggle_account": args.account,
        "approved_candidate_count": len(records), "records": records,
        "designated_final_method": selected,
        "designated_final_submission_ref": records[selected]["submission_ref"],
        "designated_final_private_score": records[selected]["private_score"],
        "designated_final_file": str(final), "final_validation": validation,
        "selection_policy": "Highest returned private score among exactly three user-approved, preplanned candidates; no retraining or weight tuning from returned scores",
        "screenshot_supplied": False, "personal_verification_completed": False,
        "lms_submission_confirmed": False,
    }
    with (args.out / "kaggle-manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
