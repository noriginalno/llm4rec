"""Offline acquisition-tool regressions: no actual account or network requests."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
MAPPING = json.loads((ROOT / "configs" / "source-files.json").read_text())


def archive(path, *, omit=None):
    payloads = {name: f"fixture only: {name}\n".encode() for name in MAPPING.values() if name != omit}
    with zipfile.ZipFile(path, "w") as out:
        for name, data in payloads.items():
            out.writestr(name, data)
        out.writestr("../../must-not-extract.txt", "unrelated")
    return payloads


def run_import(path, raw):
    return subprocess.run([sys.executable, "-I", "-S", str(ROOT / "tools" / "download_data.py"),
                           "--raw-dir", str(raw), "--archive", str(path)], capture_output=True, text=True)


def test_browser_archive_import_is_whitelisted_hashed_and_network_free(tmp_path):
    path = tmp_path / "official-name-fixture.zip"
    payloads = archive(path)
    raw = tmp_path / "raw"
    p = run_import(path, raw)
    assert p.returncode == 0, p.stderr
    result = json.loads(p.stdout)
    assert result["acquisition"] == "user_provided_browser_archive"
    assert result["logical_to_source_filenames"] == MAPPING
    assert not (tmp_path / "must-not-extract.txt").exists()
    for name, data in payloads.items():
        assert (raw / name).read_bytes() == data
        assert result["files"][name]["sha256"] == hashlib.sha256(data).hexdigest()
    original_manifest = (raw / "source-manifest.json").read_bytes()
    p = run_import(path, raw)
    assert p.returncode == 0, p.stderr
    assert (raw / "source-manifest.json").read_bytes() == original_manifest


def test_missing_archive_member_fails_before_extraction(tmp_path):
    path = tmp_path / "missing.zip"
    archive(path, omit="submission.csv")
    raw = tmp_path / "raw"
    p = run_import(path, raw)
    assert p.returncode != 0 and "Archive is missing" in p.stderr
    assert list(raw.iterdir()) == []


def test_conflicting_existing_file_is_not_overwritten_or_silently_mixed(tmp_path):
    path = tmp_path / "archive.zip"
    archive(path)
    raw = tmp_path / "raw"
    raw.mkdir()
    existing = raw / "dataset11-30.csv"
    existing.write_bytes(b"existing user file\n")
    p = run_import(path, raw)
    assert p.returncode != 0 and "differs from archive" in p.stderr
    assert existing.read_bytes() == b"existing user file\n"
    assert not (raw / "source-manifest.json").exists()
