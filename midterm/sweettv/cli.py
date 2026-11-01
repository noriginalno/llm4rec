"""Local entrypoint. No upload, rule acceptance, or remote publication commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from .data import DataSchemaError, RawSchema, load_raw, parse_timestamp
from .demo import make_demo
from .pipeline import DEFAULT_FUSIONS, best_method, run_fold, save_json
from .submission import copy_submission_exclusively, validate_submission


def week(year: int, number: int) -> int:
    return int(datetime.fromisocalendar(year, number, 1).replace(tzinfo=timezone.utc).timestamp())


def load(args):
    schema = RawSchema(log_start=args.log_start, log_end=args.log_end,
                       log_duration=args.log_duration, schedule_start=args.schedule_start)
    try:
        return load_raw(args.raw_dir, schema=schema, assumed_timezone=args.assume_timezone,
                        encoding=args.encoding, delimiter=args.delimiter,
                        file_names=json.loads(args.file_map.read_text()),
                        timestamp_formats=json.loads(args.timestamp_formats.read_text()) if args.timestamp_formats else None)
    except DataSchemaError as error:
        if args.audit_failure:
            save_json(args.audit_failure, {"error": str(error), "partial_audit": error.audit})
        raise


def validate_protocol(protocol: dict) -> None:
    stages = ("development", "primary", "final")
    fields = {"train_end", "forecast_start", "forecast_end"}
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a JSON object")
    for stage in stages:
        values = protocol.get(stage)
        if not isinstance(values, dict) or set(values) != fields:
            raise ValueError(f"{stage}: require exactly {sorted(fields)}")
        if any(type(values[field]) is not int for field in fields):
            raise ValueError(f"{stage}: cutoffs must be integer epoch seconds")
        if not values["train_end"] <= values["forecast_start"] < values["forecast_end"]:
            raise ValueError(f"{stage}: invalid temporal ordering")
    for previous, following in zip(stages, stages[1:]):
        if protocol[previous]["forecast_end"] > protocol[following]["train_end"]:
            raise ValueError(f"Protocol leakage: {previous} labels extend beyond {following} training cutoff")


def full_run(prepared, out: Path, protocol: dict, *, synthetic: bool, parameters: dict | None = None) -> dict:
    validate_protocol(protocol)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "source_audit.json", prepared.audit)
    save_json(out / "protocol.json", protocol)
    development = run_fold(prepared, **protocol["development"], output=out / "development",
                           model_parameters=parameters, fusions=DEFAULT_FUSIONS, synthetic=synthetic)
    selected_fusion = best_method(development, list(DEFAULT_FUSIONS))
    frozen_fusion = {selected_fusion: DEFAULT_FUSIONS[selected_fusion]}
    primary = run_fold(prepared, **protocol["primary"], output=out / "primary",
                       model_parameters=parameters, fusions=frozen_fusion, synthetic=synthetic)
    selected = best_method(primary)
    final = run_fold(prepared, **protocol["final"], output=out / "final",
                     model_parameters=parameters, fusions=frozen_fusion, evaluate=False, synthetic=synthetic)
    source = out / "final" / f"submission_{selected}.csv"
    destination = out / "final" / "submission.csv"
    copy_submission_exclusively(source, destination)
    validation = validate_submission(destination, prepared.template_users,
                                     json.loads((out / "final" / "candidate_ids.json").read_text()))
    result = {
        "synthetic": synthetic, "selected_method": selected,
        "selection": "Two predefined fusion weights selected on development; final method selected by primary MAP@5; exact ties prefer repeat then popularity",
        "primary_map_at_5": primary["primary_map_at_5"],
        "final_csv": str(destination), "validation": validation,
        "kaggle_score": None, "submitted": False,
        "limitations": ["No returned Kaggle score", "Primary has shorter training history than final fit", "Unknown organizer fragment/conflict/tie policies", "No author personal verification yet"],
    }
    save_json(out / "selection.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Synthetic end-to-end run; never a competition result")
    demo.add_argument("--out", type=Path, required=True)
    demo.add_argument("--users", type=int, default=72)
    for command in ("audit", "evaluate", "run"):
        p = sub.add_parser(command)
        p.add_argument("--raw-dir", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--assume-timezone", default=None, help="Explicit IANA assumption only if raw timestamps have no offsets")
        p.add_argument("--encoding", default="utf-8-sig", choices=["utf-8-sig", "utf-8", "cp1251"])
        p.add_argument("--delimiter", default=None)
        p.add_argument("--log-start", default="start_time")
        p.add_argument("--log-end", default="end_time")
        p.add_argument("--log-duration", default="duration")
        p.add_argument("--schedule-start", default="start_time")
        p.add_argument("--file-map", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "source-files.json")
        p.add_argument("--timestamp-formats", type=Path, default=None,
                       help="JSON explicit strptime formats keyed by logical source filenames")
        p.add_argument("--audit-failure", type=Path, default=None)
        p.add_argument("--parameters", type=Path, default=None)
        if command == "evaluate":
            p.add_argument("--train-end", required=True)
            p.add_argument("--forecast-start", required=True)
            p.add_argument("--forecast-end", required=True)
        if command == "run":
            p.add_argument("--protocol", type=Path, required=True,
                           help="Explicit audited epoch-second cutoffs; no automatic calendar-week inference")
            p.add_argument("--confirm-protocol", action="store_true",
                           help="Confirm the protocol's dates/week interpretation were checked against the raw-data audit")
    args = parser.parse_args()
    if args.command == "run" and not args.confirm_protocol:
        parser.error("Inspect actual timestamps first, then pass --confirm-protocol with the audited protocol")
    if args.out.exists():
        parser.error("Output already exists; use a new path to preserve previous evidence")
    if args.command == "demo":
        if args.users < 6:
            parser.error("Demo needs at least six users")
        raw = args.out.parent / f"{args.out.name}-raw"
        make_demo(raw, user_count=args.users)
        prepared = load_raw(raw, schema=RawSchema(log_start="start_time", log_end="end_time", schedule_start="start_time"))
        protocol = {
            "synthetic": True,
            "development": {"train_end": week(2020, 15), "forecast_start": week(2020, 15), "forecast_end": week(2020, 19)},
            "primary": {"train_end": week(2020, 19), "forecast_start": week(2020, 19), "forecast_end": week(2020, 31)},
            "final": {"train_end": week(2020, 31), "forecast_start": week(2020, 31), "forecast_end": week(2020, 43)},
        }
        print(json.dumps(full_run(prepared, args.out, protocol, synthetic=True), indent=2))
        return
    protocol = None
    if args.command == "run":
        protocol = json.loads(args.protocol.read_text())
        validate_protocol(protocol)
    prepared = load(args)
    synthetic = (args.raw_dir / "SYNTHETIC.json").exists()
    if args.command == "audit":
        save_json(args.out, prepared.audit)
        print(json.dumps(prepared.audit, indent=2))
        return
    parameters = json.loads(args.parameters.read_text()) if args.parameters else None
    if args.command == "evaluate":
        dates = {key: parse_timestamp(getattr(args, key), assumed_timezone=args.assume_timezone)
                 for key in ("train_end", "forecast_start", "forecast_end")}
        print(json.dumps(run_fold(prepared, **dates, output=args.out,
                                  model_parameters=parameters, synthetic=synthetic), indent=2))
    else:
        print(json.dumps(full_run(prepared, args.out, protocol, synthetic=synthetic, parameters=parameters), indent=2))


if __name__ == "__main__":
    main()
