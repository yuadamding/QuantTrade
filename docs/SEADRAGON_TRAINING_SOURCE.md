# Seadragon training source

This checkout is the local, reviewable companion to the QuantTrade source used
for the TOP2000 Seadragon training run. The authoritative bundle was read from

```text
/rsrch8/home/bcb/yding4/quant/training/bundles/top2000-fsdp2-a1999-17745275-9f1bcd35/QuantTrade
```

Its source manifest is pinned by SHA-256:

```text
17745275eb47e3458d9452759a05a1c9d98fff0061a42e352ef0d8dd1a2e556c
```

A controller-side copy of the manifest must be retained outside the repository.
Set `QT_TRAINING_MANIFEST` to that verified file, confirm its own digest equals
the value above, and then run the content check from the repository root:

```bash
sha256sum "$QT_TRAINING_MANIFEST"
sha256sum -c "$QT_TRAINING_MANIFEST"
```

The command is intentionally expected to report any local divergence. At the
time this document was added, the library and workflow additions in this
working tree were not all byte-identical to the sealed parent bundle. The
working-tree changes are preserved for review; do not reset or overwrite them
without first recording a new source manifest and its reason for divergence.

## What this source means

- The bundle is the parent training source, not a claim that the resulting
  models are production-ready.
- The S0–S7 benchmark used frozen checkpoints and a separate 2026 evaluation;
  it did not select or promote a model.
- Later A100x4 pilot bundles derive from the same parent manifest. Their source
  delta is recorded in the pilot `BUNDLE-RECEIPT.json`; do not mix pilot-only
  orchestration with the parent source without a new immutable manifest.
- Remote access is restricted to the approved chain
  `station001 -> work-mac -> seadragon` and the non-PHI project root
  `/rsrch8/home/bcb/yding4/quant/training`.

## Safe refresh procedure

1. Run `codex-chain-linux verify` from the enrolled controller.
2. Inspect the exact bundle path and manifest on Seadragon through the chain.
3. Compare the manifest against this checkout before changing files.
4. Preserve unrelated local edits; stage any source refresh as a reviewable
   patch and re-run the focused test suite.
5. Record the new manifest hash and source delta here before using the code for
   another GPU run.
