#!/usr/bin/env python3
"""Prepare the inspected official schema once and cache normalized data safely."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sweettv import data as data_module
from sweettv.cache import load_prepared, save_prepared
from sweettv.data import RawSchema, load_raw
from sweettv.pipeline import save_json


def preparation_signature(raw_dir: Path) -> str:
    """Bind preprocessing code/options and immutable-source stat/manifest state."""
    digest = hashlib.sha256()
    for path in (Path(data_module.__file__), Path(__file__), ROOT / "configs" / "source-files.json",
                 ROOT / "configs" / "timestamp-formats.json", raw_dir / "source-manifest.json"):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    mapping = json.loads((ROOT / "configs" / "source-files.json").read_text())
    for name in sorted(mapping.values()):
        stat = (raw_dir / name).stat()
        digest.update(json.dumps([name, stat.st_size, stat.st_mtime_ns]).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--run-out", type=Path, default=None)
    parser.add_argument("--parameters", type=Path, default=None)
    args = parser.parse_args()
    signature = preparation_signature(args.raw_dir)
    started = perf_counter()
    if args.cache.exists():
        prepared = load_prepared(args.cache, data_source_sha256=signature)
        mode = "validated_parquet_cache"
    else:
        prepared = load_raw(
            args.raw_dir,
            schema=RawSchema(log_end="stop_time", log_duration="duraton"),
            assumed_timezone="UTC",
            file_names=json.loads((ROOT / "configs" / "source-files.json").read_text()),
            timestamp_formats=json.loads((ROOT / "configs" / "timestamp-formats.json").read_text()),
        )
        source_manifest = json.loads((args.raw_dir / "source-manifest.json").read_text())
        for file_audit in prepared.audit["files"].values():
            original = source_manifest["files"][file_audit["source_filename"]]
            if original["sha256"] != file_audit["sha256"] or original["bytes"] != file_audit["size_bytes"]:
                raise ValueError("Raw file no longer matches the authenticated-download manifest")
        prepared.audit["schema_verified_on_actual_source"] = True
        prepared.audit["source_acquisition"] = source_manifest["acquisition"]
        prepared.audit["source_manifest_matches_raw_bytes"] = True
        prepared.audit["clock_policy"] = (
            "Both sources supply naive wall-clock datetimes. UTC is an explicit common-clock "
            "numeric anchor, not evidence that original timestamps were recorded in UTC. "
            "No inferred Kyiv offset or DST conversion is applied."
        )
        prepared.audit["unknowns"] = [
            "Original time zone is unspecified; common naive-clock alignment is an engineering assumption.",
            "Organizer fragment, duplicate, conflict, and boundary-tie policies remain incompletely documented.",
            "Upstream-rounded identifiers cannot be repaired; source text identifiers are preserved exactly.",
        ]
        save_prepared(prepared, args.cache, data_source_sha256=signature)
        mode = "raw_prepared_and_cached"
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "source_audit.json", prepared.audit)
    timing = {"mode": mode, "seconds": perf_counter() - started,
              "process_peak_rss_bytes_darwin": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "preparation_signature": signature, "events": len(prepared.events),
              "schedules": len(prepared.schedules), "template_users": len(prepared.template_users)}
    save_json(args.out / "preparation.json", timing)
    print(json.dumps(timing, indent=2), flush=True)
    if args.protocol or args.run_out:
        if not args.protocol or not args.run_out:
            parser.error("--protocol and --run-out must be supplied together")
        from sweettv.cli import full_run
        protocol = json.loads(args.protocol.read_text())
        parameters = json.loads(args.parameters.read_text()) if args.parameters else None
        result = full_run(prepared, args.run_out, protocol, synthetic=False, parameters=parameters)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
