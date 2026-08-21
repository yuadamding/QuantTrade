# PITAlphaDatasetV1 data authority

`PITAlphaDatasetV1` is the reportable data boundary for the alpha-learning
roadmap. It replaces static ticker identity and one convenience availability
mask with permanent security identity, event-sourced membership, separate
decision/fill/mark states, and explicit terminal economic disposition.

The first source-owned implementation lives in:

```text
src/rl_quant/alpha/contracts.py
src/rl_quant/alpha/accounting.py
```

It is deliberately standard-library only. Dataset bytes can therefore be
validated before importing PyTorch, pandas, PyArrow, a model, or a vendor
adapter.

## Frozen invariants

- `CASH` is explicit at action index zero. Every risky action maps one-to-one
  to a permanent `security_id`.
- Tickers are time-varying aliases. Ticker changes never create or liquidate
  an economic position.
- Membership is an event stream. Its ranking observation must precede its
  availability time, which must not follow its effective time.
- Future survival is forbidden from the universe rule.
- Availability has five distinct dimensions: observable,
  decision-eligible, fill-eligible, markable, and terminal-event.
- Missing market data never implies a zero return or liquidation. A held
  security requires a market mark, a validated fallback mark, or a terminal
  disposition.
- Every delisted security has exactly one terminal economic disposition at
  its delisting time.
- Corporate actions are identified globally and booked exactly once.
- Cash earns a causal, source-receipted one-step return.
- Post-fill targets begin at the declared fill session. They require a
  complete economic path through the horizon or exact terminal-value carry.
- A total loss remains simple return `-1.0`; the code does not fabricate a
  finite log return for zero terminal value.
- The manifest is canonical JSON with a semantic receipt and exact sorted
  byte inventory. Unlisted files, links, special files, byte drift, and
  noncanonical field types fail closed.

## Corporate-action coverage

The contract represents ordinary and special dividends, splits, reverse
splits, spin-offs, rights distributions, tender offers, cash and stock
mergers, return of capital, ticker/exchange changes, delisting proceeds,
bankruptcy recovery, and worthless disposition.

The accounting primitive currently books these events against a permanent-ID
position and preserves cash plus successor securities. A vendor adapter must
normalize its source events into this contract; it may not introduce a new
economic interpretation inside a model loader.

## Golden tests

`tests/test_pit_alpha_dataset_v1.py` covers:

- causal ticker and membership history;
- future-availability rejection;
- split value preservation;
- exactly-once dividends;
- stock-merger conversion;
- delisting and worthless outcomes;
- temporary missing marks;
- post-fill target alignment and terminal carry;
- causal cash accrual;
- permanent action-axis closure;
- canonical manifest round-trip, mutation, extra-file, and symlink rejection.

## What this milestone does not claim

No vendor dataset has been ingested, reconciled, or declared reportable by
this source change. The next data milestone must select approved sources and
materialize:

```text
security_master.parquet
ticker_history.parquet
membership_events.parquet
corporate_actions.parquet
session_calendar.parquet
cash_returns.parquet
authorities/decision_availability.parquet
authorities/fill_availability.parquet
authorities/total_return_ledger.parquet
authorities/terminal_event_ledger.parquet
```

That materialization must produce `DatasetFileRecord` entries from the actual
bytes, pass the authority validator, and reconcile total returns and terminal
events against an independent source before five-minute model training is
authorized.

## Organized Polygon conversion boundary

The prior organized TOP2000 cache can be audited and sampled through:

```bash
python -m rl_quant.workflows.pit_alpha_dataset_v1 \
  --data-root /approved/quant/data \
  audit --output /approved/staging/conversion-audit.json \
  --verify-canonical-files

python -m rl_quant.workflows.pit_alpha_dataset_v1 \
  --data-root /approved/quant/data \
  convert-symbol-day --symbol AAPL --date 2022-01-03 \
  --session-authority /approved/calendars/XNYS/2022-01-03.json \
  --session-authority-file-sha256 <exact-file-sha256> \
  --output /approved/staging/AAPL/2022-01-03.parquet
```

The audit reports four deliberately different readiness states:

```text
staging_conversion_possible
bar_source_inventory_verified
pit_alpha_training_ready
reportable_pit_authority_ready
```

The old cache may satisfy the first state and, where its manifest contains
correct hashes, the second. It cannot satisfy either training or reportability
state by itself.

The converter no longer accepts a bare file path. It resolves exactly one
accepted legacy manifest row, opens the canonical regular file without
following a final symlink, hashes its current bytes, validates Parquet schema
and row count, and reconciles any declared size, row count, and SHA-256. The
source authority also binds the exact legacy manifest-file bytes and row
number. A legacy row with no SHA can receive a new observed source hash for
nonreportable staging, but it is explicitly not historically hash-verified.

Each conversion also requires an exact exchange-session authority supplied by
an external calendar source. The authority freezes exchange, open, close,
scheduled interval count, special-session reason, and a model-availability
lag. Early closes therefore have a shorter structural grid; intervals after a
scheduled close are not mislabeled as missing observations.

The converter records FIGI/CIK identity transitions, extracts dividend and
split candidates with source-file, line, and record provenance, and aggregates
authorized session rows into sparse ordered five-minute intervals. Missing
scheduled intervals are not zero-filled. Economic interval end and assumed
strategy availability are separate timestamps.

Publication is transactional and create-only. Each staged symbol-day consists
of:

```text
<symbol-day>.parquet
<symbol-day>.parquet.receipt.json
<symbol-day>.parquet.commit.json
```

The receipt binds source and output semantic table hashes, the physical Parquet
hash, frozen schema and writer settings, source authority, and session
authority. The commit marker is written last and binds the exact Parquet and
receipt files. A bundle without a valid commit marker is incomplete. Every
bundle remains explicitly nonreportable and unauthorized for alpha training.

The audit always refuses to mint `PITAlphaDatasetV1` from this cache alone. Its
2026-ranked universe is future-selected for the 2022--2026 history; it lacks
historical membership events, a terminal-event/successor ledger, causal cash
returns, and independent total-return reconciliation. Polygon overview
snapshots are identity observations rather than permanent-ID authority, and
adjusted prices must not be combined with split share transformations until one
coherent accounting convention has been independently reconciled.

Adjusted-price semantics remain a hard barrier. Staging currently accepts the
organized adjusted observations only to preserve the source as found. Those
bars cannot become model inputs until the protocol freezes whether observations
are unadjusted or constructed from adjustment factors causally effective at
each decision time, and economic labels are reconciled from one independent
total-return and terminal-event authority.

The source-to-authority mapping is therefore deliberately staged:

| Organized source | Safe conversion now | Final V1 destination | Remaining authority |
|---|---|---|---|
| One-second adjusted aggregates | Sparse ordered five-minute development bars | `partitions/<date>/bars_5m.parquet` | Session calendar, permanent IDs, and adjusted/unadjusted accounting policy |
| Monthly overview snapshots | FIGI/CIK identity observations | `security_master.parquet`, `ticker_history.parquet` | Effective ticker/security event reconciliation |
| One 2026 dollar-volume rank | Source evidence only | `membership_events.parquet` | Historical point-in-time eligibility and rank events |
| Dividend and split JSONL | Deduplicated action candidates | `corporate_actions.parquet` | Complete announcement timing and independent reconciliation |
| Missing from organized cache | None | cash, availability, total-return, and terminal ledgers | Approved causal sources must be acquired |

Only after the rightmost column is closed may the candidate tables be renamed
to the authoritative V1 filenames and included in a canonical dataset manifest.
