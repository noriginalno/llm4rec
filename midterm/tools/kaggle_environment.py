"""Credential-location policy shared by local Kaggle entrypoints."""

import os
from pathlib import Path


def configure_isolated_home(project_root: Path) -> Path:
    root = Path(project_root).resolve()
    workspace = root.parents[1]
    if (workspace / ".opencode" / "profile" / "opencode" / "opencode.jsonc").is_file() and root.parent == workspace / "llm4rec":
        home = workspace / ".opencode" / "runtime" / "home"
    else:
        home = root / ".cache" / "kaggle-home"
    os.umask(0o077)
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["HOME"] = str(home)
    return home
