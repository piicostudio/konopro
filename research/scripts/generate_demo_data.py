from __future__ import annotations

import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RESEARCH_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from konopro_research.demo_data import ensure_demo_data  # noqa: E402


if __name__ == "__main__":
    outputs = ensure_demo_data(RESEARCH_ROOT / "data" / "demo")
    for name, path in outputs.items():
        print(f"{name}: {path}")
