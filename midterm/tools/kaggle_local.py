#!/usr/bin/env python3
"""Run the official CLI with this workspace's protected, isolated HOME."""

import os
from pathlib import Path
import sys

from kaggle_environment import configure_isolated_home


root = Path(__file__).resolve().parents[1]
configure_isolated_home(root)
binary = root / ".venv" / "bin" / "kaggle"
if not binary.is_file():
    sys.exit("Install midterm/requirements.in into midterm/.venv first")
if sys.argv[1:3] == ["auth", "login"]:
    # The SDK's default HTTP callback logger prints its complete GET URL,
    # including the one-time authorization code. Suppress callback/debug logs
    # before invoking the official CLI so the unmodified AI export stays clean.
    code = (
        "import http.server,logging,runpy,sys;"
        "http.server.BaseHTTPRequestHandler.log_message=lambda *args:None;"
        "logging.getLogger('kagglesdk.kaggle_oauth').setLevel(logging.CRITICAL);"
        "sys.argv=['kaggle',*sys.argv[1:]];"
        "runpy.run_module('kaggle',run_name='__main__')"
    )
    python = root / ".venv" / "bin" / "python"
    os.execv(str(python), [str(python), "-u", "-c", code, *sys.argv[1:]])
os.execv(str(binary), [str(binary), *sys.argv[1:]])
