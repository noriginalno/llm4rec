#!/usr/bin/env python3
"""Download official competition files through the already-authorized CLI.

No submissions, rule acceptance, or authentication are performed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--file-map", type=Path, default=root / "configs" / "source-files.json")
    parser.add_argument("--archive", type=Path, default=None,
                        help="Use an official ZIP downloaded in your browser; do not call the API")
    args = parser.parse_args()
    mapping = json.loads(args.file_map.read_text())
    files = list(mapping.values())
    if len(files) != 5 or len(set(files)) != 5 or any(Path(name).name != name for name in files):
        raise ValueError("File map must name five distinct relative basenames")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    acquisition = "existing_files"
    if not all((args.raw_dir / name).is_file() for name in files):
        if args.archive is None:
            command = [sys.executable, str(root / "tools" / "kaggle_local.py"),
                       "competitions", "download", "sweettv-tv-program-recommender",
                       "-p", str(args.raw_dir)]
            subprocess.run(command, check=True)
            archive = args.raw_dir / "sweettv-tv-program-recommender.zip"
            acquisition = "official_kaggle_cli"
        else:
            archive = args.archive
            acquisition = "user_provided_browser_archive"
        if archive.is_file():
            with zipfile.ZipFile(archive) as zipped:
                missing_members = sorted(set(files) - set(zipped.namelist()))
                if missing_members:
                    raise ValueError(f"Archive is missing official filenames: {missing_members}; no files extracted")
                for name in files:
                    target = args.raw_dir / name
                    if target.exists():
                        with zipped.open(name) as source, target.open("rb") as existing:
                            while True:
                                block = source.read(1024 * 1024)
                                if block != existing.read(1024 * 1024):
                                    raise ValueError(f"Existing {name} differs from archive; preserved original, use a new raw directory")
                                if not block:
                                    break
                        continue
                    with zipped.open(name) as source, target.open("xb") as destination:
                        while block := source.read(1024 * 1024):
                            destination.write(block)
        else:
            raise FileNotFoundError(f"No downloaded/provided archive found: {archive}")
    manifest = {"competition": "sweettv-tv-program-recommender", "acquisition": acquisition,
                "logical_to_source_filenames": mapping, "files": {}}
    for name in files:
        path = args.raw_dir / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    destination = args.raw_dir / "source-manifest.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        if any(existing.get(key) != manifest[key] for key in ("competition", "logical_to_source_filenames", "files")):
            raise ValueError("Existing source manifest differs; refusing to replace it")
        manifest = existing
    else:
        with destination.open("x") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
