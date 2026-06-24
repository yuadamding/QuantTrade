"""Architecture enforcement: the protocol-first layering must hold as a RUNTIME import DAG.

Each ``rl_quant`` subpackage sits at a layer; a package may import only packages at a strictly lower (more
foundational) layer -- never a higher one. This locks the "ownership consolidation" the architecture migration
plan calls for (protocol/data_sources/... own foundations; training/workflows sit on top) and makes the
review's #1 systemic risk -- semantic duplication / silent drift across layers -- a CI failure rather than a
slow rot. The check is over RUNTIME imports only: ``if TYPE_CHECKING:`` imports are annotation-only and do not
create runtime coupling, so they are excluded (e.g. envs/intraday.py imports training.intraday.TrainingConfig
under TYPE_CHECKING purely for a type hint).

To add a NEW subpackage you MUST place it in LAYER_ORDER (an unplaced package fails the test, forcing a
deliberate layering decision). A genuinely necessary upward edge must be added to ALLOWED_UPWARD_EXCEPTIONS with
a written justification -- there are none today (the runtime graph is a clean DAG).
"""

from __future__ import annotations

import ast
import pathlib
import unittest

# Layer order, lowest (most foundational) first. Verified to make the current runtime import graph a clean DAG.
LAYER_ORDER = [
    "protocol",
    "data_sources",
    "models",
    "datasets",
    "execution",
    "features",
    "reportability",
    "evaluation",
    "training",
    "workflows",
]

# (importer_pkg, imported_pkg) edges that are KNOWINGLY allowed to violate the layer order, each with a reason.
# Empty today -- the runtime graph is acyclic under LAYER_ORDER. Add here only with an explicit justification.
ALLOWED_UPWARD_EXCEPTIONS: set[tuple[str, str]] = set()

_SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "rl_quant"


def _is_type_checking_if(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _runtime_imported_modules(tree: ast.AST) -> list[str]:
    """Module strings imported at RUNTIME (excluding anything inside an ``if TYPE_CHECKING:`` block)."""
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_if(node):
            for child in ast.walk(node):
                skip.add(id(child))
    modules: list[str] = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


class ImportBoundaryTests(unittest.TestCase):
    def _packages(self) -> list[str]:
        return sorted(p.name for p in _SRC_ROOT.iterdir() if p.is_dir() and p.name != "__pycache__")

    def test_layer_order_covers_every_package(self) -> None:
        # A new subpackage must be placed in LAYER_ORDER (forces a deliberate layering decision).
        self.assertEqual(
            sorted(LAYER_ORDER), self._packages(),
            "LAYER_ORDER must list exactly the rl_quant subpackages; place any new package at its layer.",
        )

    def test_no_lower_layer_imports_a_higher_layer(self) -> None:
        index = {pkg: i for i, pkg in enumerate(LAYER_ORDER)}
        packages = set(self._packages())
        violations: list[str] = []
        for pkg in packages:
            for path in (_SRC_ROOT / pkg).rglob("*.py"):
                tree = ast.parse(path.read_text(), filename=str(path))
                for module in _runtime_imported_modules(tree):
                    if not module.startswith("rl_quant."):
                        continue
                    parts = module.split(".")
                    if len(parts) < 2:
                        continue
                    imported = parts[1]
                    if imported not in packages or imported == pkg:
                        continue
                    if index[pkg] <= index[imported] and (pkg, imported) not in ALLOWED_UPWARD_EXCEPTIONS:
                        rel = path.relative_to(_SRC_ROOT.parent.parent)
                        violations.append(f"{rel}: {pkg} (layer {index[pkg]}) imports {imported} (layer {index[imported]})")
        self.assertEqual(
            violations, [],
            "Layer-boundary violations (a lower/equal layer importing a higher one):\n"
            + "\n".join(violations)
            + "\nFix the import, move the shared code to a lower layer, or (rarely, with justification) add the "
            "edge to ALLOWED_UPWARD_EXCEPTIONS.",
        )


if __name__ == "__main__":
    unittest.main()
