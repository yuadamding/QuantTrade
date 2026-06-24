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
import shlex
import sys

from rl_quant.paths import scripts_dir
from rl_quant.presets import PRESETS, resolve_preset

# (group, workflow) -> the single underlying entry-point script. The per-second train/build entry points
# were removed with the precomputed-feature stack (2026-06-23, "keep the LLM-generated part only").
_DISPATCH: dict[tuple[str, str], str] = {
    ("validate", "protocol"): "validate_research_protocol.py",
}

# When no explicit --preset is given, the source selector picks the default preset. (Per-second presets removed.)
_DEFAULT_PRESETS: dict[tuple[str, str, str], str] = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qt", description="QuantTrade unified CLI.")
    groups = parser.add_subparsers(dest="group", required=True)
    for group in ("train", "build", "validate"):
        group_parser = groups.add_parser(group, help=f"{group} workflows")
        workflows = group_parser.add_subparsers(dest="workflow", required=True)
        for (dispatch_group, workflow) in _DISPATCH:
            if dispatch_group != group:
                continue
            # allow_abbrev=False so script-specific args are never mis-consumed as qt options.
            wp = workflows.add_parser(workflow, allow_abbrev=False, help=f"{group} {workflow}")
            wp.add_argument("--preset", help="Named preset (see `qt preset list`).")
            if workflow == "second":
                wp.add_argument("--source", choices=["1s"], default="1s", help="Context source (1-second bars).")
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
    preset_args: list[str] = []
    if preset_name:
        if preset_name not in PRESETS:
            raise SystemExit(f"qt: unknown preset {preset_name!r}; run `qt preset list`.")
        expected = f"{args.group}.{args.workflow}"
        if PRESETS[preset_name].workflow != expected:
            # Refuse to feed a preset's args to a different workflow (e.g. a direct-bar preset to a
            # second-context command), which would forward the wrong CLI flags to the script.
            raise SystemExit(
                f"qt: preset {preset_name!r} targets workflow {PRESETS[preset_name].workflow!r}, "
                f"not {expected!r}; run `qt preset list` to see each preset's workflow."
            )
        preset_args = resolve_preset(preset_name)
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
    # shlex.join so the printed args stay safe to copy/paste even if a path contains spaces.
    print(shlex.join(resolve_preset(args.name)))
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
