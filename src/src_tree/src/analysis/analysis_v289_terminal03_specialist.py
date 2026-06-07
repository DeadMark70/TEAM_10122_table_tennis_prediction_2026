"""Source mirror for the V289 terminal action 0/3 specialist."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMPL = ROOT / "analysis_v289_terminal03_specialist.py"
SPEC = importlib.util.spec_from_file_location("_analysis_v289_terminal03_specialist_impl", IMPL)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load V289 implementation from {IMPL}")
_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_module)

for _name in dir(_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_module, _name)


if __name__ == "__main__":
    main()
