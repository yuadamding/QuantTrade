"""Shared fixtures for the test suite (split out of the former monolithic test_correctness.py).

ROOT / SRC locate the repo and its source tree (rl_quant is editable-installed, so importing it does
not depend on the sys.path insert below -- it is kept only as a fallback for a non-installed checkout).
load_script imports a scripts/*.py module by path for the tests that exercise the utility scripts.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
