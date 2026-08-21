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
  --output /approved/staging/AAPL/2022-01-03.parquet
```

The converter rebases stale manifest paths onto the canonical organized roots,
rejects malformed symbols, records FIGI/CIK identity transitions, extracts
dividend and split candidates, and aggregates regular-session second bars into
sparse ordered five-minute intervals. Missing five-minute intervals are not
zero-filled. Every staged symbol-day has an exact source/output receipt and is
explicitly nonreportable.

The audit always refuses to mint `PITAlphaDatasetV1` from this cache alone. Its
2026-ranked universe is future-selected for the 2022--2026 history; it lacks
historical membership events, a terminal-event/successor ledger, causal cash
returns, and independent total-return reconciliation. Polygon overview
snapshots are identity observations rather than permanent-ID authority, and
adjusted prices must not be combined with split share transformations until one
coherent accounting convention has been independently reconciled.

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
