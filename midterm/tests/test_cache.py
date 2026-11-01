"""Prepared-cache correctness/integrity checks using only synthetic local data."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_array_equal

from sweettv import cache, data as data_module
from sweettv.cache import CacheValidationError, load_prepared, save_prepared
from sweettv.data import PreparedData, build_fold
from sweettv.pipeline import training_fingerprint


SOURCE_SHA = hashlib.sha256(b"synthetic preprocessor code fixture, not raw source identity").hexdigest()
PAYLOADS = {"events.parquet", "schedules.parquet", "audit.json", "template_users.json"}
BIG_USER = "18446744073709551615"


def schedule(airing="a", start=0, end=10, item=11, **extra):
    return {
        "airing_id": airing, "channel_id": "0007", "start": start,
        "end": end, "tv_show_id": item, "metadata": {}, **extra,
    }


def event(airing, user=BIG_USER, **extra):
    return {
        "user_id": user,
        **{key: airing[key] for key in ("tv_show_id", "start", "end", "airing_id")},
        **extra,
    }


@pytest.fixture
def prepared():
    first = schedule(metadata={
        "tv_show_title": "Кіно / 東京 / café 🍿", "null": None, "empty": {},
        "array": [None, {}, [], True, False, 1, 1.25, "null"],
        "nested": {"異なる型": ["A", 3, None], "": ""}, "huge_integer": 2**80,
    }, extra={"optional": None})
    second = schedule("b", 20, 30, metadata=None, quarantined=True)
    third = schedule("c", 40, 50, metadata={"same_key": [1, "two", None]}, quarantined=False)
    missing_metadata = schedule("missing", 60, 70)
    del missing_metadata["metadata"]
    return PreparedData(
        template_users=[BIG_USER, "0000042", "José/用户"],
        schedules=[second, first, third, deepcopy(first), missing_metadata],
        events=[event(first, coverage=0.8), event(second, "0000042", coverage=None),
                event(first), event(first, coverage=0.8)],
        audit={
            "files": {"synthetic.csv": {"sha256": "a" * 64, "size_bytes": 17}},
            "native": [None, True, 2, 0.5, "Україна", {}, []],
            "unknown": None, "empty": {},
        },
    )


def save_fixture(prepared, tmp_path):
    directory = tmp_path / "new-cache"
    manifest = save_prepared(prepared, directory, data_source_sha256=SOURCE_SHA)
    return directory, manifest


def read_manifest(directory):
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def write_manifest(directory, manifest):
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def refresh_record(directory, filename):
    """Test-only forged manifest: integrity is not source authentication."""
    manifest = read_manifest(directory)
    contents = (directory / filename).read_bytes()
    manifest["files"][filename] = {
        "sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents),
    }
    write_manifest(directory, manifest)


def forbid_arrow(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Opened Arrow before completing all cache validation")

    monkeypatch.setattr(cache.pq, "ParquetFile", forbidden)
    monkeypatch.setattr(cache.pq, "read_table", forbidden)


def test_strict_roundtrip_unicode_nulls_optional_keys_duplicate_order_and_raw_audit(
    prepared, tmp_path,
):
    original = deepcopy(prepared)
    directory, manifest = save_fixture(prepared, tmp_path)
    loaded = load_prepared(directory, data_source_sha256=SOURCE_SHA)
    assert loaded == original
    assert prepared == original
    assert loaded.events[0] == loaded.events[3]
    assert loaded.schedules[1] == loaded.schedules[3]
    assert "coverage" not in loaded.events[2]
    assert loaded.events[1]["coverage"] is None
    assert loaded.schedules[0]["metadata"] is None
    assert "metadata" not in loaded.schedules[4]
    assert "quarantined" not in loaded.schedules[1]
    assert type(loaded.events[0]["user_id"]) is str
    assert loaded.events[0]["user_id"] == BIG_USER
    assert loaded.template_users == [BIG_USER, "0000042", "José/用户"]
    assert loaded.audit["files"] == original.audit["files"]
    assert manifest == read_manifest(directory)
    # Returned objects do not alias the caller's input, other rows, or disk.
    loaded.schedules[1]["metadata"]["nested"]["異なる型"].append("changed")
    assert loaded.schedules[3]["metadata"] == original.schedules[3]["metadata"]
    assert load_prepared(directory, data_source_sha256=SOURCE_SHA) == original


def test_manifest_versions_exact_source_string_and_all_payload_hashes(prepared, tmp_path):
    directory = tmp_path / "cache"
    supplied = SOURCE_SHA.upper()
    manifest = save_prepared(prepared, directory, data_source_sha256=supplied)
    assert manifest["version"] == 1
    assert manifest["data_source_sha256"] == supplied
    assert manifest["preprocessor_code_sha256"] == supplied
    assert set(manifest["files"]) == PAYLOADS
    assert {path.name for path in directory.iterdir()} == PAYLOADS | {"manifest.json"}
    for filename, record in manifest["files"].items():
        contents = (directory / filename).read_bytes()
        assert record == {"bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}
    assert load_prepared(directory, data_source_sha256=SOURCE_SHA) == prepared


def test_explicit_parquet_types_metadata_strings_and_default_quarantine(prepared, tmp_path):
    directory, _ = save_fixture(prepared, tmp_path)
    events = pq.ParquetFile(directory / "events.parquet").read()
    schedules = pq.ParquetFile(directory / "schedules.parquet").read()
    assert events.schema == pa.schema([
        pa.field("user_id", pa.string(), False), pa.field("tv_show_id", pa.int64(), False),
        pa.field("start", pa.int64(), False), pa.field("end", pa.int64(), False),
        pa.field("airing_id", pa.string(), False), pa.field("coverage", pa.float64(), True),
        pa.field("row_json", pa.string(), False),
    ])
    assert schedules.schema == pa.schema([
        pa.field("airing_id", pa.string(), False), pa.field("channel_id", pa.string(), False),
        pa.field("start", pa.int64(), False), pa.field("end", pa.int64(), False),
        pa.field("tv_show_id", pa.int64(), False), pa.field("metadata_json", pa.string(), False),
        pa.field("quarantined", pa.bool_(), False), pa.field("row_json", pa.string(), False),
    ])
    assert events["coverage"].to_pylist() == [0.8, None, None, 0.8]
    assert schedules["quarantined"].to_pylist() == [True, False, False, False, False]
    metadata = schedules["metadata_json"].to_pylist()
    assert metadata[0] == "null"  # Not str(None), an inferred empty struct, or {}.
    assert json.loads(metadata[1]) == prepared.schedules[1]["metadata"]
    assert metadata[4] == "null"


@pytest.mark.parametrize("kind", ["all", "events", "schedules"])
def test_empty_lists_roundtrip_with_explicit_schemas(prepared, tmp_path, kind):
    if kind in ("all", "events"):
        prepared.events = []
    if kind in ("all", "schedules"):
        prepared.schedules = []
    if kind == "all":
        prepared.template_users, prepared.audit = [], {}
    directory, _ = save_fixture(prepared, tmp_path)
    assert load_prepared(directory, data_source_sha256=SOURCE_SHA) == prepared
    for filename in ["events.parquet", "schedules.parquet"]:
        schema = pq.ParquetFile(directory / filename).schema_arrow
        assert schema.field("start").type == pa.int64()
        assert schema.field("row_json").type == pa.string()


def test_template_user_duplicates_are_preserved_not_filtered(prepared, tmp_path):
    prepared.template_users = ["0007", BIG_USER, "0007"]
    directory, _ = save_fixture(prepared, tmp_path)
    assert load_prepared(directory, data_source_sha256=SOURCE_SHA).template_users == prepared.template_users


@pytest.mark.parametrize("filename", sorted(PAYLOADS))
def test_same_size_checksum_corruption_rejected_before_any_arrow_open(
    prepared, tmp_path, monkeypatch, filename,
):
    directory, _ = save_fixture(prepared, tmp_path)
    path = directory / filename
    contents = bytearray(path.read_bytes())
    contents[len(contents) // 2] ^= 1
    path.write_bytes(contents)
    forbid_arrow(monkeypatch)
    with pytest.raises(CacheValidationError, match="checksum/bytes"):
        load_prepared(directory, data_source_sha256=SOURCE_SHA)


def test_byte_count_mismatch_rejected_before_arrow(prepared, tmp_path, monkeypatch):
    directory, manifest = save_fixture(prepared, tmp_path)
    manifest["files"]["audit.json"]["bytes"] += 1
    write_manifest(directory, manifest)
    forbid_arrow(monkeypatch)
    with pytest.raises(CacheValidationError, match="checksum/bytes"):
        load_prepared(directory, data_source_sha256=SOURCE_SHA)


@pytest.mark.parametrize("version", [0, 2, True, "1", None])
def test_manifest_version_rejected_before_arrow(prepared, tmp_path, monkeypatch, version):
    directory, manifest = save_fixture(prepared, tmp_path)
    manifest["version"] = version
    write_manifest(directory, manifest)
    forbid_arrow(monkeypatch)
    with pytest.raises(CacheValidationError, match="version"):
        load_prepared(directory, data_source_sha256=SOURCE_SHA)


def test_wrong_expected_preprocessor_code_hash_is_rejected(prepared, tmp_path, monkeypatch):
    directory, _ = save_fixture(prepared, tmp_path)
    forbid_arrow(monkeypatch)
    with pytest.raises(CacheValidationError, match="Preprocessor code SHA-256"):
        load_prepared(directory, data_source_sha256="0" * 64)


@pytest.mark.parametrize("key", ["data_source_sha256", "preprocessor_code_sha256"])
def test_missing_or_inconsistent_manifest_code_hash_is_not_bypassed(
    prepared, tmp_path, monkeypatch, key,
):
    directory, manifest = save_fixture(prepared, tmp_path)
    manifest[key] = "0" * 64
    write_manifest(directory, manifest)
    forbid_arrow(monkeypatch)
    with pytest.raises(CacheValidationError, match="Preprocessor code SHA-256"):
        load_prepared(directory, data_source_sha256=SOURCE_SHA)
    del manifest[key]
    write_manifest(directory, manifest)
    with pytest.raises(CacheValidationError, match="preprocessor code SHA-256"):
        load_prepared(directory, data_source_sha256=SOURCE_SHA)


@pytest.mark.parametrize("bad_sha", [None, "", "not-a-sha", "a" * 63, "x" * 64])
def test_required_code_sha_has_no_permissive_default(prepared, tmp_path, bad_sha):
    with pytest.raises(ValueError, match="SHA-256"):
        save_prepared(prepared, tmp_path / "bad", data_source_sha256=bad_sha)
    with pytest.raises(ValueError, match="SHA-256"):
        load_prepared(tmp_path / "missing", data_source_sha256=bad_sha)
    assert not (tmp_path / "bad").exists()


def test_code_sha_keyword_cannot_be_omitted(prepared, tmp_path):
    with pytest.raises(TypeError, match="data_source_sha256"):
        save_prepared(prepared, tmp_path / "cache")
    with pytest.raises(TypeError, match="data_source_sha256"):
        load_prepared(tmp_path / "cache")


def test_nonoverwrite_existing_cache_empty_directory_or_file(prepared, tmp_path):
    directory, _ = save_fixture(prepared, tmp_path)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(FileExistsError):
        save_prepared(prepared, directory, data_source_sha256=SOURCE_SHA)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original
    existing = tmp_path / "already-empty"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        save_prepared(prepared, existing, data_source_sha256=SOURCE_SHA)
    assert not list(existing.iterdir())
    existing_file = tmp_path / "already-file"
    existing_file.write_text("untouched")
    with pytest.raises(FileExistsError):
        save_prepared(prepared, existing_file, data_source_sha256=SOURCE_SHA)
    assert existing_file.read_text() == "untouched"


@pytest.mark.parametrize("unsafe", ["../outside.json", "/outside.json", "nested/row.parquet", "evil.pkl"])
def test_manifest_cannot_supply_arbitrary_paths(prepared, tmp_path, monkeypatch, unsafe):
    directory, manifest = save_fixture(prepared, tmp_path)
    manifest["files"][unsafe] = manifest["files"].pop("audit.json")
    write_manifest(directory, manifest)
    forbid_arrow(monkeypatch)
    with pytest.raises(CacheValidationError, match="whitelist"):
        load_prepared(directory, data_source_sha256=SOURCE_SHA)


def test_extra_unmanifested_files_and_missing_payloads_are_rejected(prepared, tmp_path):
    directory, _ = save_fixture(prepared, tmp_path)
    extra = directory / "unexpected.txt"
    extra.write_text("untracked payload")
    with pytest.raises(CacheValidationError, match="nonwhitelisted"):
        load_prepared(directory, data_source_sha256=SOURCE_SHA)
    extra.unlink()
    (directory / "audit.json").unlink()
    with pytest.raises(CacheValidationError, match="missing"):
        load_prepared(directory, data_source_sha256=SOURCE_SHA)


@pytest.mark.parametrize("filename", ["events.parquet", "manifest.json"])
def test_symlink_payload_or_manifest_cannot_escape_directory(prepared, tmp_path, filename):
    directory, _ = save_fixture(prepared, tmp_path)
    external = tmp_path / "outside"
    external.write_bytes((directory / filename).read_bytes())
    (directory / filename).unlink()
    (directory / filename).symlink_to(external)
    with pytest.raises(CacheValidationError, match="nonsymlink"):
        load_prepared(directory, data_source_sha256=SOURCE_SHA)


def test_root_symlink_is_rejected(prepared, tmp_path):
    directory, _ = save_fixture(prepared, tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(CacheValidationError, match="nonsymlink directory"):
        load_prepared(alias, data_source_sha256=SOURCE_SHA)


def test_schema_and_typed_payload_disagreement_rejected_even_with_updated_hashes(prepared, tmp_path):
    directory, _ = save_fixture(prepared, tmp_path)
    path = directory / "events.parquet"
    original = pq.ParquetFile(path).read()
    stored = original.to_pylist()
    stored[0]["user_id"] = "different-but-valid-string"
    pq.write_table(pa.Table.from_pylist(stored, schema=original.schema), path)
    refresh_record(directory, "events.parquet")
    with pytest.raises(CacheValidationError, match="disagree"):
        load_prepared(directory, data_source_sha256=SOURCE_SHA)
    # Checksums alone are not schema validation or authentication.
    pq.write_table(pa.table({"user_id": [123]}), path)
    refresh_record(directory, "events.parquet")
    with pytest.raises(CacheValidationError, match="schema"):
        load_prepared(directory, data_source_sha256=SOURCE_SHA)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), ("tuple",), {1: "int key"}])
def test_non_json_metadata_fails_closed_without_a_completed_manifest(prepared, tmp_path, bad):
    prepared.schedules[0]["metadata"] = {"bad": bad}
    directory = tmp_path / "incomplete"
    with pytest.raises((ValueError, TypeError)):
        save_prepared(prepared, directory, data_source_sha256=SOURCE_SHA)
    assert directory.is_dir()
    assert not (directory / "manifest.json").exists()
    with pytest.raises(CacheValidationError):
        load_prepared(directory, data_source_sha256=SOURCE_SHA)


def test_user_ids_not_coerced_from_integers_or_none(prepared, tmp_path):
    for index, bad in enumerate([2**64 - 1, None]):
        prepared.events[0]["user_id"] = bad
        with pytest.raises(ValueError, match="user_id must already be a string"):
            save_prepared(prepared, tmp_path / f"invalid-{index}", data_source_sha256=SOURCE_SHA)


def test_parent_partition_like_name_is_not_inferred_as_an_extra_column(prepared, tmp_path):
    directory = tmp_path / "cohort=all" / "cache"
    save_prepared(prepared, directory, data_source_sha256=SOURCE_SHA)
    assert load_prepared(directory, data_source_sha256=SOURCE_SHA) == prepared


@pytest.mark.parametrize("split", [(100, 100, 200), (15, 15, 200)])
def test_build_fold_training_fingerprint_candidates_labels_audit_identical_after_cache(tmp_path, split):
    past = schedule("past", 0, 10, 11, metadata={"title": "Кіно", "nullable": None})
    repeat = schedule("repeat", 20, 30, 11, metadata={"title": "same ID, later airing"})
    future_repeat = schedule("forecast-repeat", 100, 110, 11)
    future_new = schedule("forecast-new", 110, 120, 22)
    future_cold = schedule("forecast-cold", 120, 130, 33)
    prepared = PreparedData(
        [BIG_USER, "0000042"],
        [future_cold, past, repeat, future_repeat, future_new, deepcopy(past)],
        [event(repeat), event(past, coverage=1), event(past, coverage=1),
         event(future_new), event(future_repeat, "0000042", coverage=None)],
        {"files": {"synthetic.csv": {"sha256": "b" * 64}}, "raw_identity_verified": False},
    )
    # This fixture uses the CURRENT source fingerprint, not a bypass/default.
    source_sha = hashlib.sha256(Path(data_module.__file__).read_bytes()).hexdigest()
    before = build_fold(prepared, *split)
    directory = tmp_path / "fold-cache"
    save_prepared(prepared, directory, data_source_sha256=source_sha)
    restored = load_prepared(directory, data_source_sha256=source_sha)
    after = build_fold(restored, *split)
    assert training_fingerprint(before[0]) == training_fingerprint(after[0])
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert restored == prepared
