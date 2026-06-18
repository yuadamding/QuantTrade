from __future__ import annotations

import unittest
from _support import ROOT


class RepositoryHygieneTests(unittest.TestCase):
    def test_no_local_absolute_path_literals(self) -> None:
        forbidden = [
            "/" + part
            for part in [
                "home/yding1995",
                "tmp/",
                "mnt/",
                "root/",
                "Users/",
                "path/to",
            ]
        ]
        excluded_parts = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
        excluded_names = {"test_repository_hygiene.py"}  # this file defines the forbidden needles
        for path in ROOT.rglob("*"):
            if not path.is_file() or excluded_parts.intersection(path.parts) or path.name in excluded_names:
                continue
            text = path.read_text(errors="ignore")
            for needle in forbidden:
                self.assertNotIn(needle, text, f"Local absolute path literal {needle!r} found in {path}")

    def test_no_current_tree_secret_literals(self) -> None:
        forbidden = [
            "X-RapidAPI-Key",
            "X-RapidAPI-Proxy-Secret",
            "TRADING_PWD =",
        ]
        excluded_parts = {".git", "__pycache__"}
        excluded_names = {".env.example", "test_repository_hygiene.py"}  # this file defines the forbidden needles
        for path in ROOT.rglob("*"):
            if not path.is_file() or excluded_parts.intersection(path.parts) or path.name in excluded_names:
                continue
            try:
                text = path.read_text(errors="ignore")
            except UnicodeDecodeError:
                continue
            for needle in forbidden:
                self.assertNotIn(needle, text, f"Forbidden secret-like literal {needle!r} found in {path}")
