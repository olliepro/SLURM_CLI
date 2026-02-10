from __future__ import annotations

import os
from pathlib import Path

# Shared configuration paths/defaults
CONFIG_DIR = Path.home() / ".slurmcli"
CONFIG_PATH = CONFIG_DIR / "config.json"

# Default resource selections
DEFAULT_TIME_MINUTES = 30
DEFAULT_MEM_GB = 50
DEFAULT_GPUS = 1
DEFAULT_CPUS = 4
MAX_CPUS = 128
DEFAULT_TIMEOUT_LIMIT_SECONDS = 30 * 60  # 30 minutes

# Shell/UI defaults
DEFAULT_SHELL = os.environ.get("SHELL", "zsh")
UI_TERMINAL = "terminal"
UI_VSCODE = "vscode"

# Timeout modes
TIMEOUT_IMPATIENT = "impatient"
TIMEOUT_NOTIFY = "notify"

# Search defaults
SEARCH_DEFAULT_MIN_TIME_MINUTES = 30
SEARCH_DEFAULT_MIN_GPUS = 1
SEARCH_SWITCH_MINUTES = 60
SEARCH_SUBMIT_GAP_SECONDS = 5
SEARCH_DASHBOARD_CLOSE_SECONDS = 5
SEARCH_JOB_PREFIX = "slurmcli-search"

MAX_RECENT_ACCOUNTS = 5
