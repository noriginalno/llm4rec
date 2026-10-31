import importlib.util
import os
from pathlib import Path


def module():
    source = Path(__file__).resolve().parents[1] / "tools" / "kaggle_environment.py"
    spec = importlib.util.spec_from_file_location("kaggle_environment_under_test", source)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_standalone_clone_keeps_credentials_inside_ignored_project_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "original-home"))
    root = tmp_path / "checkout" / "midterm"
    root.mkdir(parents=True)
    previous = os.umask(0o077)
    try:
        home = module().configure_isolated_home(root)
    finally:
        os.umask(previous)
    assert home == root / ".cache" / "kaggle-home"
    assert os.environ["HOME"] == str(home)
    assert home.is_dir()
    assert not (tmp_path / ".opencode").exists()


def test_recognized_course_workspace_reuses_its_existing_isolated_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "original-home"))
    root = tmp_path / "llm4rec" / "midterm"
    root.mkdir(parents=True)
    config = tmp_path / ".opencode" / "profile" / "opencode" / "opencode.jsonc"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    previous = os.umask(0o077)
    try:
        home = module().configure_isolated_home(root)
    finally:
        os.umask(previous)
    assert home == tmp_path / ".opencode" / "runtime" / "home"
    assert home.is_dir()
    assert not (root / ".cache").exists()
