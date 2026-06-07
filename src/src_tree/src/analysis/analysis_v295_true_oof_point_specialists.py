from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_root_module() -> ModuleType:
    root_path = Path(__file__).resolve().parents[2] / "analysis_v295_true_oof_point_specialists.py"
    spec = importlib.util.spec_from_file_location("_v295_true_oof_point_specialists_root", root_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {root_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_root = _load_root_module()

for _name in dir(_root):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_root, _name)


if __name__ == "__main__":
    _root.main()
