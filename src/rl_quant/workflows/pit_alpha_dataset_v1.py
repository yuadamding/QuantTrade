"""CLI for auditing and staging the organized Polygon PIT-alpha inputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rl_quant.data_sources.polygon_pit_alpha import (
    audit_organized_polygon_for_pit_alpha,
    convert_symbol_day_to_five_minute_staging,
    default_organized_polygon_shards,
    load_exchange_session_authority,
    resolve_symbol_day_source,
    write_conversion_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit organized Polygon data or stage one ordered five-minute symbol-day."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser(
        "audit", help="write a nonreportable conversion-readiness audit"
    )
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--verify-canonical-files", action="store_true")

    convert = commands.add_parser(
        "convert-symbol-day", help="stage one five-minute symbol-day"
    )
    convert.add_argument("--symbol", required=True)
    convert.add_argument("--date", required=True)
    convert.add_argument("--session-authority", type=Path, required=True)
    convert.add_argument("--session-authority-file-sha256", required=True)
    convert.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shards = default_organized_polygon_shards(args.data_root)
    if args.command == "audit":
        audit = audit_organized_polygon_for_pit_alpha(
            shards,
            verify_canonical_files=bool(args.verify_canonical_files),
        )
        file_sha = write_conversion_audit(args.output, audit)
        print(
            json.dumps(
                {
                    "audit_file": str(args.output),
                    "audit_file_sha256": file_sha,
                    "staging_conversion_possible": audit.staging_conversion_possible,
                    "bar_source_inventory_verified": (
                        audit.bar_source_inventory_verified
                    ),
                    "pit_alpha_training_ready": audit.pit_alpha_training_ready,
                    "reportable_pit_authority_ready": audit.reportable_pit_authority_ready,
                    "blockers": audit.blockers,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    source = resolve_symbol_day_source(shards, args.symbol, args.date)
    session = load_exchange_session_authority(
        args.session_authority,
        expected_file_sha256=args.session_authority_file_sha256,
    )
    publication = convert_symbol_day_to_five_minute_staging(
        source, session, args.output
    )
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
