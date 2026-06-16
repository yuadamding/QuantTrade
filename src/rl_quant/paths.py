"""Shared filesystem paths for QuantTrade, centralized so workflows and presets agree on roots.

Previously each script re-derived the data root and universe/covariate paths independently, which
let defaults drift between entry points. These helpers are the single source of truth.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """QuantTrade repo root (the directory containing ``src/``, ``scripts/``, ``README.md``)."""
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    """Workspace data directory.

    Mirrors the logic the scripts duplicated: prefer ``<workspace>/data`` (a sibling of the repo
    when laid out as ``<workspace>/QuantTrade``) when it exists, else ``<repo>/data``.
    """
    root = project_root()
    shared = root.parent / "data"
    if root.name in {"QuantTrade", "rl_quant"} and shared.exists():
        return shared
    return root / "data"


def scripts_dir() -> Path:
    """Directory holding the workflow entry-point scripts the CLI dispatches to."""
    return project_root() / "scripts"
