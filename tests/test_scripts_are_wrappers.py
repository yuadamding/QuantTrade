"""Architecture enforcement: scripts/ must become THIN WRAPPERS that delegate to the package.

The target ("ownership consolidation") is: rl_quant/ owns implementation; scripts/ are thin entrypoints that
parse/delegate to a package command and call it -- no business logic, no dataset semantics, no large parsers
living in scripts. A "wrapper" here is operationalised as a script module with NO top-level function or class
definitions (business logic = functions/classes belongs in the package). The five current wrappers already
delegate to rl_quant.cli.main(...).

This test locks that contract WITHOUT a big-bang migration:
  * every script NOT in LEGACY_NON_WRAPPER_SCRIPTS must already be a wrapper -- so the existing wrappers cannot
    regress, and a NEW script must be a wrapper (or be a deliberate, listed exception);
  * every script IN the allowlist must STILL own logic -- so once a script is migrated (its logic moves into a
    package and it becomes a wrapper) it MUST be removed from the allowlist, making the backlog shrink visibly;
  * the allowlist may not contain stale / non-existent names.

LEGACY_NON_WRAPPER_SCRIPTS is the migration backlog. It should only ever SHRINK.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"

# Scripts that still own implementation logic (top-level functions/classes). Each is a migration target:
# move its logic into rl_quant/ (workflows.commands / datasets.*/builder / reportability.gates / ...) and turn
# the script into a thin wrapper, then DELETE it from this set. Do not ADD to this set without a strong reason.
LEGACY_NON_WRAPPER_SCRIPTS = {
    "build_hourly_from_minute_context_dataset.py",
    "build_hourly_transformer_dataset.py",
    "build_news_article_table.py",
    "build_news_llm_aggregates.py",
    "build_news_llm_features.py",
    "build_second_context_decision_dataset.py",
    # build_stock_covariate_silver_features.py MIGRATED -> rl_quant.workflows.commands.* (now a thin wrapper).
    # build_stock_second_silver_features.py MIGRATED -> rl_quant.workflows.commands.* (now a thin wrapper).
    "convert_polygon_second_to_protocol.py",
    "download_daily_ohlcv.py",
    "download_hourly_ohlcv.py",
    "download_intraday_ohlcv.py",
    "evaluate_second_context_dataset.py",
    "export_live_bundle.py",
    "extract_nbbo_features.py",
    "fetch_top_us_market_cap_universe.py",
    "fetch_top_volume_universes.py",
    "generate_qwen_news_precomputed.py",
    "integrate_stock_covariates_with_hour_partitions.py",
    "train_dqn_agent.py",
    "train_hourly_causal_transformer_rl.py",
    "train_hourly_from_minute_context_rl.py",
    "train_hourly_from_second_calendar_holdout.py",
    "train_hourly_from_second_protocol_partitions.py",
    "train_second_context_action_scorer.py",
    "train_strategy_allocator.py",
    # validate_research_protocol.py MIGRATED -> rl_quant.workflows.commands.validate (now a thin wrapper).
}


def _owns_logic(path: pathlib.Path) -> bool:
    """True if the script defines top-level functions/classes (business logic belongs in the package)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in tree.body
    )


class ScriptsAreWrappersTests(unittest.TestCase):
    def _scripts(self) -> list[pathlib.Path]:
        return sorted(p for p in _SCRIPTS_DIR.glob("*.py") if p.name != "__init__.py")

    def test_non_legacy_scripts_are_thin_wrappers(self) -> None:
        offenders = [
            p.name for p in self._scripts()
            if p.name not in LEGACY_NON_WRAPPER_SCRIPTS and _owns_logic(p)
        ]
        self.assertEqual(
            offenders, [],
            "These scripts own business logic (top-level functions/classes) but are not in the migration "
            f"backlog: {offenders}. Make them thin wrappers that delegate to rl_quant.* (the package owns "
            "logic), or -- with justification -- add them to LEGACY_NON_WRAPPER_SCRIPTS.",
        )

    def test_legacy_allowlist_entries_still_own_logic(self) -> None:
        # Once a backlog script is migrated it becomes a wrapper; it must then be REMOVED from the allowlist so
        # the backlog visibly shrinks and the wrapper cannot silently regress.
        migrated = [
            name for name in sorted(LEGACY_NON_WRAPPER_SCRIPTS)
            if (_SCRIPTS_DIR / name).exists() and not _owns_logic(_SCRIPTS_DIR / name)
        ]
        self.assertEqual(
            migrated, [],
            f"These scripts are now thin wrappers -- remove them from LEGACY_NON_WRAPPER_SCRIPTS: {migrated}.",
        )

    def test_legacy_allowlist_has_no_stale_entries(self) -> None:
        existing = {p.name for p in self._scripts()}
        stale = sorted(name for name in LEGACY_NON_WRAPPER_SCRIPTS if name not in existing)
        self.assertEqual(stale, [], f"LEGACY_NON_WRAPPER_SCRIPTS names non-existent scripts: {stale}.")


if __name__ == "__main__":
    unittest.main()
