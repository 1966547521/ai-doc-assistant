"""Central paths for runtime artifacts kept outside source code."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("APP_DATA_DIR", PROJECT_ROOT / "data")).resolve()
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = DATA_DIR / "chroma"
LOG_DIR = DATA_DIR / "logs"
SESSIONS_FILE = DATA_DIR / "sessions.json"
HISTORY_FILE = DATA_DIR / "history.json"


def ensure_runtime_directories() -> None:
    """Create the persistent data directory before a service writes to it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
