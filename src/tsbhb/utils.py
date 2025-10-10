from __future__ import annotations

from pathlib import Path
import os
import random
from typing import Optional


def set_seed(seed: int = 42) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Find project root by walking up until `pyproject.toml` is found.

    Fallback: use 3 parents up from this file (src/tsbhb/...).
    """
    if start is None:
        start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists():
            return p
    # Fallback for safety
    return Path(__file__).resolve().parents[3]


def default_data_file() -> Path:
    return find_repo_root() / "data" / "online_retail.csv"


def default_out_dir() -> Path:
    out = find_repo_root() / "experiments" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out

