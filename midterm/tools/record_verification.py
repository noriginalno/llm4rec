#!/usr/bin/env python3
"""Record actual pytest/dependency results and source hashes, without inventing checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    files = sorted([*root.glob("sweettv/**/*.py"), *root.glob("tests/*.py"),
                    *root.glob("tools/*.py"), *root.glob("reviews/*.md"),
                    root / "CONTRACT.md", root / "requirements.lock"])
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    command = [sys.executable, "-m", "pytest", "-q", "--junitxml", str(args.out / "pytest.xml")]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    (args.out / "pytest.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    dependency = subprocess.run([sys.executable, "-m", "pip", "check"], cwd=root, capture_output=True, text=True)
    (args.out / "dependency-check.log").write_text(dependency.stdout + dependency.stderr, encoding="utf-8")
    suites = ET.parse(args.out / "pytest.xml").getroot().findall("testsuite")
    counts = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    changed = [str(p.relative_to(root)) for p in files if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(root))]]
    result = {
        "recorded_at": datetime.now(timezone.utc).isoformat(), "command": command,
        "exit_code": completed.returncode, "counts": counts,
        "dependency_check_exit_code": dependency.returncode,
        "source_hashes": hashes, "sources_changed_during_checks": changed,
        "scope": "Fixture/synthetic implementation checks; not personal verification or a returned Kaggle score",
    }
    with (args.out / "verification.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(completed.stdout + completed.stderr)
    print(json.dumps({"counts": counts, "exit_code": completed.returncode,
                      "dependency_check_exit_code": dependency.returncode, "sources_changed": changed}, indent=2))
    raise SystemExit(completed.returncode or dependency.returncode or bool(changed))


if __name__ == "__main__":
    main()
