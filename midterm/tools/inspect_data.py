#!/usr/bin/env python3
"""Inspect bounded CSV samples before choosing exact schema/timezone adapters."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encoding", choices=["utf-8-sig", "utf-8", "cp1251"], default="utf-8-sig")
    parser.add_argument("--delimiter", default=None)
    parser.add_argument("--file-map", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "source-files.json")
    args = parser.parse_args()
    mapping = json.loads(args.file_map.read_text())
    names = list(mapping.values())
    if len(names) != 5 or len(set(names)) != 5 or any(Path(name).name != name for name in names):
        raise ValueError("File map must name five distinct relative basenames")
    result = {"bounded_sample_only": True, "reconstruction_run": False,
              "logical_to_source_filenames": mapping, "files": {}}
    for name in names:
        path = args.raw_dir / name
        with path.open(encoding=args.encoding, errors="strict", newline="") as handle:
            sample = handle.read(65_536)
            delimiter = args.delimiter or csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            handle.seek(0)
            reader = csv.DictReader(handle, delimiter=delimiter, strict=True)
            rows = []
            for _, row in zip(range(3), reader):
                rows.append(row)
            result["files"][name] = {"bytes": path.stat().st_size, "encoding": args.encoding,
                                      "delimiter": delimiter, "columns": reader.fieldnames, "sample_rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
