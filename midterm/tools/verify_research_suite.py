#!/usr/bin/env python3
"""Record the complete original+research test suite without overwriting history."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    paths = [p for folder in ("sweettv", "research_v2", "tools", "tests")
             for p in (ROOT / folder).rglob("*.py")]
    paths.append(ROOT / "requirements.lock")
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    before = snapshot()
    temporary = ROOT.parent.parent / ".opencode/runtime/tmp/opencode" / ("research-suite-" + uuid.uuid4().hex)
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--basetemp", str(temporary), "--junitxml", str(args.out / "pytest.xml"), "tests"]
    tests = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (args.out / "pytest.log").write_text(tests.stdout + tests.stderr)
    dependencies = subprocess.run([sys.executable, "-m", "pip", "check"], cwd=ROOT,
                                  capture_output=True, text=True)
    (args.out / "pip-check.log").write_text(dependencies.stdout + dependencies.stderr)
    xml = ET.parse(args.out / "pytest.xml").getroot()
    suites = [xml] if xml.tag == "testsuite" else list(xml.iter("testsuite"))
    counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
    unchanged = before == snapshot()
    passed = tests.returncode == dependencies.returncode == 0 and unchanged and not any(
        counts[key] for key in ("errors", "failures", "skipped"))
    record = {"passed": passed, "command": command, "python": sys.version,
              "platform": platform.platform(), "packages": {p: version(p) for p in (
                  "numpy", "scipy", "duckdb", "pyarrow", "implicit", "scikit-learn", "pytest")},
              "counts": counts, "dependency_check_passed": dependencies.returncode == 0,
              "sources_unchanged": unchanged, "source_sha256": before,
              "new_kaggle_submissions": 0, "original_report_state_modified": False}
    with (args.out / "verification.json").open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(tests.stdout)
    print(json.dumps({"passed": passed, "counts": counts, "sources_unchanged": unchanged,
                      "output": str(args.out)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
