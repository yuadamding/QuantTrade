"""Golden-artifact guards against SILENT semantic drift.

The repo is artifact-heavy and provenance-sensitive: a return basis, its content hash, and the hashing
algorithm itself are load-bearing for reportability and resume. Broad unit tests do not reliably catch a change
that quietly alters those stamps. These golden assertions PIN the exact hashes; an intentional change must
update the golden value here (a visible, reviewed diff), and an UNintentional change fails CI.

Pinned today:
  * the canonical content_hash() algorithm over a fixed v2 ReturnBasis -- if this changes, EVERY persisted
    return_basis_content_hash silently becomes stale, so a change must be deliberate and migration-aware;
  * the direct-hourly builder's declared basis hash -- catches a silent change to that dataset's economics.
"""

from __future__ import annotations

import unittest

from _support import load_script
from rl_quant.protocol.action_return_basis import ReturnBasis

# A FIXED canonical v2 basis. Its content hash pins the content_hash() ALGORITHM (field set, ordering, encoding).
_FIXED_V2_BASIS = ReturnBasis(
    weight_semantics="full_capital_single_slot_returns",
    formula="clipped_simple_return(decision_bar_close, next_bar_close)",
    clip_min=-1.0,
    clip_max=1.0,
    semantics_version="v1",
    fill_convention="decision_bar_close_to_next_bar_close",
    basis_version="v2",
    entry_fill_rule="decision_bar_close",
    exit_fill_rule="next_bar_close",
    execution_latency_ms=0,
    source_bar_interval="1h",
    price_source="bar_close",
)
_FIXED_V2_BASIS_HASH = "d37f183d3f56d74175d4557a66f0871bd343cf00969ce0e4770384965b261e88"

# The direct-hourly builder's declared basis (1h). Pins that dataset's declared economics.
_DIRECT_HOURLY_1H_BASIS_HASH = "f1f0e0abe3664180001be75991d3352e4faf97f795da877765050057150091b6"


class GoldenArtifactTests(unittest.TestCase):
    def test_return_basis_content_hash_algorithm_is_stable(self) -> None:
        # If this fails, content_hash() changed -- EVERY persisted return_basis_content_hash is now stale.
        # Update the golden ONLY as part of a deliberate, migration-aware change to the hashing.
        self.assertEqual(_FIXED_V2_BASIS.content_hash(), _FIXED_V2_BASIS_HASH)
        # Recomputation is deterministic within a process.
        self.assertEqual(_FIXED_V2_BASIS.content_hash(), _FIXED_V2_BASIS.content_hash())

    def test_direct_hourly_builder_basis_hash_is_pinned(self) -> None:
        module = load_script("build_hourly_transformer_dataset")
        basis = ReturnBasis.from_mapping(module._direct_hourly_action_return_basis("1h"))
        self.assertTrue(basis.is_complete())
        self.assertEqual(basis.validation_errors(), [])
        # A change to the direct-hourly economics (fill rule, latency, clip, price source, ...) changes this
        # hash -- intentional changes update the golden; accidental drift fails here.
        self.assertEqual(basis.content_hash(), _DIRECT_HOURLY_1H_BASIS_HASH)


if __name__ == "__main__":
    unittest.main()
