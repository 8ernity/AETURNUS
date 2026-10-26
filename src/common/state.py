"""Single source of truth for runtime SQLite state.

All mutable databases (triage work orders, self-healing ledger, drift history,
and MLflow tracking) live under one state directory.  The directory is selected
by ``AETURNUS_STATE_DIR`` when set, otherwise by ``state.dir`` in
``configs/pipeline.yaml``.  State is never shipped inside a package because it
is not tracked by git.
"""

import os
from pathlib import Path
from typing import Optional

from src.common.utils import get_root_dir, load_yaml


def get_state_dir(root_dir: Optional[Path] = None) -> Path:
    root = Path(root_dir) if root_dir is not None else get_root_dir()
    env_value = os.environ.get("AETURNUS_STATE_DIR")
    if env_value:
        state_dir = Path(env_value).expanduser()
    else:
        state_dir = Path(load_yaml("configs/pipeline.yaml")["state"]["dir"]).expanduser()
    if not state_dir.is_absolute():
        state_dir = root / state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def state_db_path(filename: str, root_dir: Optional[Path] = None) -> Path:
    if not filename or Path(filename).name != filename:
        raise ValueError("state_db_path expects a plain filename, not a path")
    return get_state_dir(root_dir) / filename
