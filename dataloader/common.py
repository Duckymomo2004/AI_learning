from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


def get_data_root(root: str | Path | None = None) -> Path:
    data_root = Path(root) if root is not None else DATA_ROOT
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root
