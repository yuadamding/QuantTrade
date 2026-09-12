# QT200 historical-message census

This is a lossless, nonauthorizing diagnostic for the historical semantic-adapter
boundary. It is not a corrected trade dataset, security master, bar/tape/fill
source, feature builder, or training-readiness authority.

`scan_and_publish_qt200_trade_census_v1` reads an exact committed gzip transaction
using `scan_massive_selected_trade_file_v1`. The latter preserves every selected
original row, including canonicalization failures. An optional original-line
callback supplies the exact bytes, including the original line terminator, for
bounded samples; callbacks remain provisional until CRC, EOF, complete payload
hash, metadata, and worker-local identity checks pass.

The census separates:

- internally derived source-record identity (object transaction, row ordinal,
  original-line hash);
- the original provider trade ID, including blank values;
- correction-target references, which remain null/not assessed.

Blank ID, size, price, conditions, raw correction, tape and venue/TRF fields are
counted independently. Counts include the full joint grouping and marginal tables.
The exact selected QT200 symbols and diagnostic aliases are separate inventories;
neither symbol matching nor census success proves historical issue identity.

The bound native correction inventory supplies candidate semantic labels only.
It does not establish its applicability to a particular historical export.
The census makes no event-stream-versus-final-snapshot assumption. Resolved and
unresolved correction counts remain **null**, not fabricated zeroes.

Sequence observations are scoped to source date and ticker. A bounded in-memory
index distinguishes first occurrences from repeated equal/different original-line
hashes, comparing against the first occurrence. It neither deduplicates messages
nor uses a sequence number as a provider trade ID or original-message reference.

Only `census.json` and its `COMPLETE.json` inventory are published. Allocation or
source failures leave an incomplete attempt, never a completion marker. Existing
attempts cannot be overwritten. Original market data remain a required dependency;
the diagnostic stores bounded examples, not another full tick copy.

## Qualification and resources

The caller must bind an immutable source request, ordered symbol inventories,
the actual correction authority, explicit allocations, and remote runtime.
The operational wrapper owns wall-clock, independent RSS, filesystem and fileset
quota guards. Group cardinality/key bytes, selected rows, sequence keys, sample
count/bytes and total output are bounded independently. No unbounded spill occurs.

The regression suite covers overlapping defects, exact original bytes, record-ID
scope, all quantity classes, unsupported corrections, sequence collisions,
source damage/mutation, table reconciliation, no-clobber publication, and the
full proposed sequence-index allocation. Run it only in the user-approved remote
test environment. Host-memory capacity on an H100 node is **not** evidence of
GPU PPO optimization.

Successful diagnostic output always retains:

```text
diagnostic_only = true
identity_qualified = false
message_semantics_qualified = false
correction_replay_qualified = false
bars_tape_fills_qualified = false
training_ready = false
```

Subsequent economic qualification still requires supported historical correction
linkage, versioned price-message rules, availability timing, permanent security
identity, and all dependent bars/tape/fill/feature/forecast gates. A diagnostic
must never be used to bypass those requirements.

## Historical correction orientation and native preflight

The provider's NYSE [correction glossary](https://massive.com/glossary/conditions-indicators)
describes retrospective encoding: code 01 carries corrected values at original
time, while code 12 carries original incorrect values at correction time.
This does not prove applicability to every historical export or venue.
Do not treat this pair as a forward new-trade/replacement stream. The QT200
native prerequisite now blocks 01/12 until their historical representation,
predecessor linkage and revision availability are qualified. Other native
canonicalization and replay contracts are not relaxed.

`qt200_historical_message_adapter_v1` classifies losslessly retained rows and
keeps source-record identity separate from nullable provider identity. The
narrow zero-volume corrected-close candidate has no executable volume, fill
authorization or backdated availability. Reference condition rules are not
automatically valid historical rules.

`qt200_correction_pair_probe_v1` reads a hash-bound original-string Parquet in
two bounded passes. It retains all rows sharing a candidate raw ticker/sequence
key, including ordinary/unknown-code collisions, without choosing a correction
target or discarding ambiguity. Parquet values do not reproduce original CSV
quoting or raw-line hashes; those remain tied to the preserved gzip.

Candidate selection parses a bounded unsigned integer independently of the
preserved raw code. The declared diagnostic families are 1/12, 8/10 and 7/11;
zero padding therefore cannot hide a candidate. Raw ticker/sequence keys are
not normalized. Pair classification remains observational, including when IDs
match: no economic roles, correction targets or revision times are inferred.

The selected market-day spool also blocks 01/12, including rows rejected by
canonicalization for blank IDs. It retains the original rows without applying
the unsupported update, invalidates all dependent ticker-day masks, and labels
every output `terminal_corrected_diagnostic`, never decision-time-qualified.
Synthetic forward-replacement regression cases use an explicitly synthetic
contract; their success is not historical correction qualification. Full-file
native preflight still checks off-panel records. No panel-scoped authority or
compact-Parquet-to-native-input promotion is introduced by these changes.
