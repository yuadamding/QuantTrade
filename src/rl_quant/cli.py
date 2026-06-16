"""``qt`` / ``quanttrade`` unified command-line entry point.

This is a thin dispatcher: it maps ``qt <group> <workflow>`` to the existing workflow entry-point
script, expands ``--preset NAME`` (and the ``--interval`` / ``--source`` default-preset sugar) into
that script's CLI arguments, forwards any remaining arguments verbatim, and runs the script. It does
not reimplement the workflows -- it removes the need for per-workflow wrapper scripts and gives one
discoverable surface. Workflow implementations still live in ``scripts/`` for now; moving them into
``rl_quant.workflows`` is a later migration step.
"""

from __future__ import annotations

import argparse
import runpy
import sys

from rl_quant.paths import scripts_dir
from rl_quant.presets import PRESETS, resolve_preset

# (group, workflow) -> the single underlying entry-point script for that workflow family.
_DISPATCH: dict[tuple[str, str], str] = {
    ("train", "strategy"): "train_strategy_allocator.py",
    ("train", "direct-bar"): "train_hourly_causal_transformer_rl.py",
    ("train", "subhour"): "train_hourly_from_minute_context_rl.py",
    ("train", "partitions"): "train_hourly_from_second_protocol_partitions.py",
    ("train", "second-context"): "train_second_context_action_scorer.py",
    ("train", "intraday-nbbo"): "train_dqn_agent.py",
    ("build", "direct-bar"): "build_hourly_transformer_dataset.py",
    ("build", "subhour"): "build_hourly_from_minute_context_dataset.py",
    ("build", "second-context"): "build_second_context_decision_dataset.py",
    ("evaluate", "second-context"): "evaluate_second_context_dataset.py",
    ("validate", "protocol"): "validate_research_protocol.py",
}

# When no explicit --preset is given, an --interval/--source selector can pick a default preset.
_DEFAULT_PRESETS: dict[tuple[str, str, str], str] = {
    ("train", "direct-bar", "1m"): "train.direct-bar.minute",
    ("build", "direct-bar", "1m"): "build.direct-bar.minute",
    ("train", "subhour", "1s"): "train.subhour.second-context",
    ("build", "subhour", "1s"): "build.subhour.second-context",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qt", description="QuantTrade unified CLI.")
    groups = parser.add_subparsers(dest="group", required=True)
    for group in ("train", "build", "evaluate", "validate"):
        group_parser = groups.add_parser(group, help=f"{group} workflows")
        workflows = group_parser.add_subparsers(dest="workflow", required=True)
        for (dispatch_group, workflow) in _DISPATCH:
            if dispatch_group != group:
                continue
            # allow_abbrev=False so script-specific args are never mis-consumed as qt options.
            wp = workflows.add_parser(workflow, allow_abbrev=False, help=f"{group} {workflow}")
            wp.add_argument("--preset", help="Named preset (see `qt preset list`).")
            if workflow == "direct-bar":
                wp.add_argument("--interval", choices=["1h", "1m"], default="1h", help="Bar interval (selects default preset).")
            if workflow == "subhour":
                wp.add_argument("--source", choices=["1m", "1s"], default="1m", help="Context source (selects default preset).")
    preset_parser = groups.add_parser("preset", help="Inspect named presets")
    preset_sub = preset_parser.add_subparsers(dest="preset_cmd", required=True)
    preset_sub.add_parser("list", help="List all presets")
    show = preset_sub.add_parser("show", help="Print the expanded args for a preset")
    show.add_argument("name")
    return parser


def resolve_workflow(args: argparse.Namespace, passthrough: list[str]) -> tuple[str, list[str]]:
    """Map a parsed (group, workflow) + selector/preset to (script_name, full script argv).

    Preset args come first; passthrough (user-supplied) args come after so they override defaults.
    """
    script = _DISPATCH[(args.group, args.workflow)]
    selector = getattr(args, "interval", None) or getattr(args, "source", None)
    preset_name = args.preset or _DEFAULT_PRESETS.get((args.group, args.workflow, selector or ""))
    preset_args = resolve_preset(preset_name) if preset_name else []
    return script, [*preset_args, *passthrough]


def _run_script(script_name: str, script_argv: list[str]) -> int:
    path = scripts_dir() / script_name
    if not path.exists():
        raise SystemExit(f"qt: workflow script not found: {path}")
    saved_argv = sys.argv
    sys.argv = [str(path), *script_argv]
    try:
        runpy.run_path(str(path), run_name="__main__")
        return 0
    except SystemExit as exc:  # scripts end with `raise SystemExit(main())`
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    finally:
        sys.argv = saved_argv


def _preset_command(args: argparse.Namespace) -> int:
    if args.preset_cmd == "list":
        for name, preset in sorted(PRESETS.items()):
            print(f"{name}\t[{preset.workflow}]\t{preset.description}")
        return 0
    if args.name not in PRESETS:
        raise SystemExit(f"qt: unknown preset {args.name!r}; run `qt preset list`.")
    print(" ".join(resolve_preset(args.name)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    if args.group == "preset":
        return _preset_command(args)
    script, script_argv = resolve_workflow(args, passthrough)
    return _run_script(script, script_argv)


if __name__ == "__main__":
    raise SystemExit(main())
