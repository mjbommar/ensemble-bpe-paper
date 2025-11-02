import os
import sys
from pathlib import Path


def pytest_configure():
    # Ensure `src` is on sys.path for imports
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

