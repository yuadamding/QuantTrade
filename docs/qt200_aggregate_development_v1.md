# QT200 aggregate-based development track

## Scope and status

This is a separately identified research track, `QT200-AGG-DEV-01`, requested
on 2026-09-10. It is not a new interpretation of an existing native V5 run.
The initial deliverable is a bounded provider-source pilot, not training,
scientific registration of a complete executable experiment, or profitability
evidence. `launch_ready=false` until the bars-only dataset, execution proxy,
study selection barrier, and requested GPU route have their own acceptance.

The user changed the requested training topology on 2026-09-11 to **one H100
with one CUDA worker**, replacing the earlier joint-two-H100 requirement.
Train one model at a time; retain the global data/optimizer budget without
GPU-count learning-rate scaling. LSF GPU regression remains separate from
Kubernetes H100 training. The distributed adapter is optional historical
engineering work, not a prerequisite for this route. This change does not
qualify data, freeze the final learning/study contract or authorize outer
access. Existing immutable artifacts retain their original runtime fields;
do not rewrite their `joint_h100_runtime_qualified` flags or receipts.

Single-device acceptance uses `test_massive_adaptive_single_gpu_v1.py` and its
fresh-process probe. It runs the existing trainer on `cuda:0` for two actual
63-session numerical-market rollouts with four PPO epochs each, proves actor
and critic updates plus CUDA optimizer state, and exercises the existing
checkpoint serializer/parser. Same-profile resume must reproduce the next
update, RNGs and economic trace exactly. CPU inference uses a declared numerical
tolerance and may not modify checkpoint evidence. These synthetic regressions
run only on LSF GPUs; they do not qualify the historical QT200 inputs, the
bars-only forecast integration, or the Kubernetes H100 runtime.

The permitted claim is **fixed-panel development research using
provider-finalized aggregates with explicit availability and execution
assumptions**. It is not source-exact SIP replay, historical as-received data,
trade-derived capacity, a point-in-time selected universe, or live execution.

The native V5 archive and its correction/identity gates remain unchanged.
Unsupported native corrections do not prevent this separate bars-only route;
neither does aggregate acquisition discharge those native gates.

## Three independent work tracks

1. **Native investigation:** retain the original CSV rows and source receipts;
   obtain the historical export contract and compare paginated REST trades
   for ABBV, WMT, and AAPL on 2017-01-03. Matching products is not independent
   truth or an authenticated correction link. Ask about a documented final-state
   filter as well as full lifecycle reconstruction. Preserve an unsent request
   as unsent until the external service returns a submission receipt.
2. **Aggregate research:** acquire unadjusted provider daily bars, with optional
   minute bars for an explicitly modeled execution window. Use separate source,
   feature, experiment, and report identities. Never mint native trade/tape/fill
   authorities from those bars.
3. **Prospective observations:** use new immutable captures with actual request
   and receipt times. Later observations never overwrite earlier versions.
   Scheduling a decision-cutoff collector and proving its timely execution are
   separate from a one-off historical download. A historical capture made today
   does not retroactively become a historical vintage.

## Bounded acquisition pilot

Freeze the plan before the first HTTP request. The pilot has nine queries:

| Product | Symbols | Inclusive dates | Purpose |
| --- | --- | --- | --- |
| REST trades | ABBV, WMT, AAPL | 2017-01-03 | Compare fields/populations with preserved original rows |
| REST daily bars, unadjusted | ABBV, WMT, AAPL | 2017-01-03 through 2017-12-29 | Check compact bars ingestion and missingness |
| REST minute bars, unadjusted | ABBV, WMT, AAPL | 2017-01-03 | Inspect execution-proxy inputs, not produce fills |

The acquisition plan binds the exact ordered QT200 universe, implementation
hashes, endpoints, dates, sort/adjustment settings, page and byte bounds, and
no-retry policy. The 50,000-row request limit is not a completeness claim.
Follow every validated provider `next_url`, reject loops/redirects and secret-
bearing URLs, and retain each original response body with its digest, request
ID, query URL, retrieval interval, and page-chain receipt. A failed or bounded-
out query remains incomplete; never publish it as a completed day.

Store raw responses compressed losslessly. Do not deduplicate apparently equal
trades, replace blank IDs, reinterpret correction flags, or reconstruct ticks
from aggregates. A successful empty response means the provider returned no
bars; it does not prove listing status or justify a zero-return observation.

Controller acquisition is CPU/I/O work, not a local test or training run.
Numerical and regression tests run only on the approved remote environment.
Credentials stay in the protected controller-only acquisition child and never
enter URLs, logs, source packages, remote mounts, or receipts.

## Full research data contract to freeze before training

### Full-panel daily observation expansion

`qt200_daily_capture_v1` requests the exact ordered QT200 symbols over
2017-01-03 through 2026-08-31, inclusive, using only unadjusted daily bars.
It shares the pilot's header-only, no-redirect, receipt-linked HTTP transport;
the original pilot population and limits remain unchanged. The new fixed
profile allows 200 queries, at most eight pages each, 32 MiB per response,
512,000,000 total raw response bytes and 1,800 seconds. There are no automatic
retries. Failed responses and incomplete attempts remain preserved.

Normalization retains decimal OHLCV, optional field masks, page/row provenance
and actual retrieval times in compressed JSONL. Raw responses remain immutable.
It rejects duplicate/unordered intervals and reports malformed rows without
silently dropping them from the readiness decision. Relative missingness uses
the union of *observed* panel dates, explicitly not an authenticated exchange
calendar. An empty response, pre-listing interval, ticker change or reused
symbol is not imputed or joined to an intended issue automatically.

This expansion completes a requested observation surface only. Historical
identity, corporate actions, calendar, features, execution proxy and runtime
acceptance remain separate. `training_ready` stays false.

### Remaining scientific inputs

### Bars-only numerical preparation and execution kernels

The next engineering surface is `qt200_aggregate_features_v1` plus
`qt200_aggregate_execution_v1`. These are implementations to qualify, not
permission to start the settings panel or a claim that issue history is solved.
They do not change the frozen native V5 observation/reward specifications.

The numeric feature schema has **12 per-security fields**: log close/open,
log high/low, close location in the range, log-one-plus volume and transaction
count, optional log VWAP/close, close log returns over 1/5/20/63 sessions,
one-session log-one-plus volume change, and 20-session population volatility.
All features for decision t stop at t-1. Missing values have explicit masks;
a zero-range close location is missing rather than an invented midpoint.
No return window bridges a missing session, known issue conflict or candidate
corporate action in the unadjusted series. No global normalization is fitted
during full-history preparation; the fit normalizer rejects a boundary that
overlaps held-out sessions. Numeric features are persisted separately from
minute execution data and future targets. These 12 fields are not a fabricated
substitute for the native 90-dimensional PPO observation.

For qualification, the proposed execution proxy sweeps the ten minute **open
prices** in [09:35,09:45) ET, taking at most floor(2% × observed minute volume)
newly traded shares in each minute. Orders are committed before the window.
This is a model, not exact VWAP, observed fills or capacity evidence. Missing
slots invalidate that instrument's execution window; observed zero volume
remains distinguishable from missing data. Sells settle before buys; all buys
share cash proportionally, with whole-share flooring and retained residual
cash. The same fixed orders can fill differently under 10/20/40 bp, so a
nonmonotone terminal cost ladder remains a reportable economic outcome.

The accounting kernel changes shares at a supported split, creates a dividend
receivable for pre-trade ex-date holdings, and transfers that receivable to
cash only when payment is due. Selling before pay date does not erase the
entitlement. Duplicate events and ambiguous same-day split/dividend share
bases are rejected. Unresolved mergers, successors, event applicability and
terminal outcomes are not silently handled as ordinary dividends or splits.
Terminal mark-based liquidation charges the registered cost but is explicitly
an adjustment, not proof of liquidity; unpaid dividends remain receivables.

Acceptance tests include independently calculated cash/shares/fees, split
invariance, dividend timing, no-trade costs, incomplete-window rejection,
proportional cash allocation, nonmonotone cost results, future-price feature
invariance, missing-day propagation, fit-only scaling, persisted feature
replay and parent tamper rejection. Run them only on LSF GPUs. Any actual
feature materialization consumes the exact completed observed panel with
before/after source hashes; it must not race or replace its producer.

### Source-bound price labels and forecast-input reader

`qt200_aggregate_targets_v1` prepares separate **gross price-prediction** labels
from the accepted daily/window panel and the unchanged feature artifact. The
seven nonoverlapping boundary intervals are 0–1, 1–5, 5–10, 10–21, 21–42,
42–63 and 63–126 exchange sessions after the next-session entry. Each boundary
uses the observed 09:35 minute open, conditional on a complete ten-minute
window. This price proxy is neither window VWAP nor an observed/modeled fill.
No fee, dividend, split, terminal event, factor residual or profitable result is
imputed. These labels are not the native V5 economic target authority.

A bucket is masked when its daily path or either boundary window is missing,
a candidate action or known issue conflict occurs anywhere in its inclusive
interval, or the required calendar/maturity is unavailable. This conservative
mask includes candidate events with unresolved applicability; absence of a
known event is not proof of event or identity completeness. The full 200-name
calendar rectangle remains intact, including unlisted/missing periods and the
unmatured tail. Nullable values, Boolean masks, reason bitsets, exact feature
row links and individual bucket maturity sessions are persisted in Zstandard
Parquet. Future target availability never changes a feature mask.

`qt200_aggregate_forecast_data_v1.prepare_fit_inputs` reopens the exact feature
and label completions and validates row identities, chronology, masks, source
hashes and calendar geometry. Its explicit contiguous fit interval must end
before the held-out start. It reveals a label only after close of the exchange
session following the bucket end, under the same **modeled** finalized-bar
availability convention. This does not prove actual historical revision times.
Unmatured labels become null with false loss masks in the fit view; their
future completeness/value does not filter the feature population. The scaler
uses only the requested fit features, never future labels or held-out rows.

Full-history materialization does not fit a scaler or forecast, select a
checkpoint/configuration, or authorize training. A later bars-only learning
specification must explicitly consume these price labels and masks; they must
not be passed as native factor-adjusted total returns. Source identity/event
qualification and the requested single-H100 pathway remain independent gates.

Acceptance runs on LSF GPUs and includes actual persisted full-panel fixtures,
next-session entry and seven-bucket compounding, missing-day and split-jump
masking, delayed maturity, fit-only scaling, future-label invariance, corrupted
source/row-link rejection, and read-only reconstruction. Input mounts are
read-only. An interrupted output has no completion and is never overwritten.

### Persisted cutoff-masked forecast-input preparation

`qt200_aggregate_fit_package_v1` persists explicit expanding-history input
views, fit-only normalizers and source-row mappings in compact Parquet. The
engineering plan chooses cutoffs by **calendar length**, not observed returns
or target completeness. Each cutoff is the last fit session; the immediately
following session is excluded. All fit rows remain, including those with no
mature label. Unobserved features remain null with false masks after scaling.
No held-out values or target-completeness filters enter the normalizer.

The initial operational preflight uses 756, 882, 1,008 and 1,134 session
prefixes. These are **support-check views, not the final experiment's four
folds**, validation/outer assignments or a configuration-selection rule. No
study selection is frozen and no model is fitted by this stage. Its cutoff
and label-availability convention remains modeled finalized-bar availability,
not proof of historical data vintages. Native V5 gates remain unchanged.

The reader reconstructs every persisted row and normalizer from the original
hash-bound feature/target parents, rejects rehashed but inconsistent content,
and creates nothing during verification. Tests cover wrong cutoffs, missing
or changed parents, mask/link/scaler tampering, publication interruption,
no-clobber recovery and fresh-process replay. Acceptance runs only on LSF GPUs.
The real preparation is followed by a separate-process read-only reconstruction
before its worker can report success. Security identity, corporate-action
applicability and single-H100 runtime qualification remain independent blockers.

### Compact session-aligned observation integration

`qt200_aggregate_panel_v1` consumes hash-bound accepted daily, reference,
supplement, calendar and minute-normalization products. It writes lossless
Zstandard Parquet daily/minute tables and a 200 × 2,428 session index. Row
indices link actual observations; missing rows stay null and missing minute
slots have an explicit ten-bit mask. Observed zero volume is distinct from
an absent observation. Decimal strings and page/row provenance survive.

Features refer only to the preceding session, while execution-input references
point to the following session. No feature returns, targets, prices, fills or
liquidity are invented. `observation_support_complete` is only a presence check,
not training eligibility. Exact-date issue conflicts override matches; sparse
reference observations do not propagate into continuous identity intervals.
FB/ANTM/HONI remain separate candidate alias observations. Corporate-action
indices identify candidate observations, not accounting-approved cash/shares.

Every consumed normalized artifact is rehashed before and after integration.
The operator binds its successful LSF producer receipts. This is explicitly
not another raw HTTP replay or historical-availability certification. The
original sources remain read-only and native/PIT/training flags remain false.
Regressions and actual full-panel integration run only on LSF GPU allocations.

### Reference supplements

`qt200_reference_supplement_v1` fills the source-observed FB/ANTM/HONI daily
query gaps, captures the final three August dates of split/dividend records,
and asks for dated reference observations. Its 340-query inventory consists
of three alias series, two economic-tail queries, 200 end-date ticker queries,
six alias-boundary queries and three issuer-reference dates for each of the 43
review cases. Issuer CIKs come from the hash-bound retained identity evidence;
they are discovery filters, never security identifiers or successor links.
All returned classes, conflicting identifiers and empty responses are retained.
The reference endpoints use explicit historical dates and active-status filters;
an empty response is not proof of no listing, trading or corporate action.

This separately bounded profile shares the reviewed immutable transport but
does not broaden the nine-query pilot or 200-query daily capture. It allows
128 MB of raw responses, eight pages per query and 1,800 seconds, with no
retry, redirect, URL credential or automatic ticker stitching. The documented
`/stocks/v1/splits` and `/stocks/v1/dividends` response contracts are retained
as new sources, not relabeled as the older economic capture generation.
See the provider's [dated ticker filters](https://massive.com/docs/rest/stocks/tickers/all-tickers),
[split contract](https://massive.com/docs/rest/stocks/corporate-actions/splits)
and [dividend contract](https://massive.com/docs/rest/stocks/corporate-actions/dividends).
Successful acquisition does not assert historical availability, continuous
identity, correct issue-level event accounting or training readiness.

### Rolling minute-input preparation

`qt200_minute_capture_v1` defines one immutable full-history capture per QT200
symbol and each of the three source-observed alias intervals (203 captures).
It preserves **all** returned minute observations and pagination, not only the
future execution window. Each capture is bounded at 512 MB raw responses,
128 pages and 1,800 seconds; the operator must additionally freeze study-wide
time, total-byte and storage limits before acquisition. Exact tests must pass
on LSF GPUs before the controller opens its acquisition credential.

The separate execution-input view retains actual minute OHLCV rows in
[09:35,09:45) New York time, with original row/page hashes, optional VWAP/count
masks and explicit missing slots. Other rows remain in the original capture.
It does not construct a VWAP from closes, infer available liquidity, create
fills, establish issue continuity or mark the new experiment training-ready.
The chronological calendar/clock plan uses the pinned calendar library and
reproduces all predecessor sessions before extending coverage; it never takes
the union of observed bar dates as an exchange calendar.

These inputs remain separate from the not-yet-frozen bars-only feature schema,
execution proxy, corporate-action accounting and requested single-H100 runtime.
No full-history capture or passing data-adapter test qualifies those interfaces.

- Keep the exact user-defined ordered 200 equities, including `BRK.B`; cash
  is separate. Establish issue identities and listing intervals before joining
  by ticker. Present-day panel selection remains survivor-conditioned.
- Preserve the full requested calendar and explicit missingness. Reconcile
  aggregate coverage independently of the 2,425 acquired trade-file dates;
  that inventory is not proof of aggregate coverage or all August 2026 dates.
- Use `adjusted=false` with explicit split/share and dividend/cash accounting.
  Qualify successors, mergers and terminal outcomes for positions actually held.
  Do not apply a split to an already split-adjusted price series.
- Declare a genuinely bars-only feature schema: price/return/range/volatility
  and volume observations with their masks. Remove trade-size quantiles,
  signed trade flow, venue-level tape, and inferred correction populations.
  Do not insert observed-looking zeros for unavailable tape features.
- Proposed information clock: decision after session t's close, feature bars
  no later than session t-1, execution in the next exchange session. Bind the
  actual exchange calendar, early closes, timezone and daylight-saving rules.
  Treat the provider's currently documented next-day 11 a.m. ET flat-file
  publication as an assumption for historical research, not as an observed
  timestamp or proof that later revisions cannot exist.
- Freeze the minute-based price proxy and participation model before training.
  The proposed next-session window is [09:35, 09:45) America/New_York. Commit
  targets before the window; its eventual prices/volume belong only to the
  simulator. An average of minute closes is not VWAP. Optional provider VWAP
  must retain its presence mask and exact window/population interpretation.
  No aggregate price or volume proves that an order would actually fill.
- Preserve the applicable long-only, cash-constrained accounting, independent
  benchmark and fit-selected fixed control, $10M primary capital, 20 bp primary
  cost, 10/20/40 bp stress, and 2% modeled participation ceiling. These are
  modeling inputs, not measured market impact or capacity.
- Reuse downstream code only after its input semantics match the new contract.
  The native 90-dimensional PPO observation/forecast graph cannot be filled
  with fabricated bars-derived substitutes just to satisfy its shape.
- Freeze chronological splits and study selection before outcome access. Start
  with one configuration. The proposed 12-setting, three-common-seed panel
  remains separately gated: all candidates stop before O0; a committed common
  validation rule selects a configuration, not an outer-return winner.
- One H100 must perform the model's CUDA optimization. CPU computation on an
  H100 node is not GPU-training qualification. No real training is authorized
  by the acquisition pilot; the exact single-device runtime must be accepted.

## Acceptance and reporting

The separate [joint-PPO engineering adapter](massive_adaptive_joint_ppo_v1.md)
addresses the previous two-rank optimizer proposal with synthetic remote
regressions. It is not required for the current single-H100 route. It neither
consumes this bars-only dataset nor fills missing native observation fields.
A passing LSF GPU test is not H100 qualification;
the real study remains closed until its independent source, forecast,
observation and study-selection gates pass.

### Receipt-bound pilot ingestion

`qt200_aggregate_pilot_v1.analyze_pilot()` consumes the completed capture by
its externally bound plan and completion hashes, plus hash-bound original CSV
examples and correction-pair diagnostics. It replays the entire capture before
and after analysis. It has no network, credential, training, or source-promotion
interface and writes only into a previously absent, separate output directory.

The compact `observed-bars.jsonl.gz` preserves interval timestamps, decimal
OHLCV values, optional VWAP/transaction-count masks, retrieval times, original
row hashes and page/row provenance. Validation checks finite values, OHLC range,
unadjusted-query identity, timezone-aware interval alignment, query-date bounds,
and unique ascending intervals. Invalid rows remain in the immutable raw
capture and are identified in the report; they prevent observed-schema
acceptance. No bars or exchange-calendar gaps are synthesized. An empty capture
cannot establish a usable observed bars schema. Calendar completeness, stable
issue identity and corporate-action treatment remain separate prerequisites.

`source-comparisons.json.gz` retains every matching ticker/query-date/sequence
candidate for the preserved historical examples. Field equality is diagnostic:
it neither assigns a provider ID nor chooses a correction target. Missing REST
correction/TRF fields are distinguished from explicit zero values. Matching
REST and flat-file rows does not prove independent accuracy, complete revision
history, or what was observable at a historical decision.

The proposed minute execution window is inspected only for observed slots,
volume, and optional VWAP presence. No fill or execution price is derived at
this pilot boundary. `observed_bar_schema_accepted` is distinct from
`full_history_dataset_complete`, historical point-in-time qualification, native
V5 qualification, and `training_ready`; the latter claims remain false.

Run the corresponding regressions on the user-approved LSF GPU runtime, then
run this analysis against the real immutable pilot. Tests use synthetic HTTP
responses only for capture persistence and failure cases; the actual pilot
analysis consumes the retained provider responses without stubs.

Keep separate facts: capture complete, identity/accounting complete, bars
schema accepted, availability modeled, dataset complete, runtime qualified,
study selection frozen, economic execution complete, and profitability passed.
Historical point-in-time certification remains false for retrieved-now
finalized bars without historical vintage evidence, even after adding a lag.

An unsuccessful investment result must still produce a complete report.
Report absolute after-cost return separately from benchmark, neutral, and
fit-selected-control differences, plus cost stress, drawdown and fold results.
Disclose turnover, costs, fill shortfall, cash/exposure, concentration and
holding durations as diagnostics. Do not change reward or enforce a 30-session
holding lock as part of this data-adapter pilot.

## Provider references

Checked on 2026-09-10; current documentation is not a historical applicability
attestation for the acquired 2017 export.

- [REST trade identity, clocks and pagination](https://massive.com/docs/rest/stocks/trades-quotes/trades)
- [Correction and condition glossary](https://massive.com/glossary/conditions-indicators)
- [Custom bars, adjustment flag, optional VWAP and pagination](https://massive.com/docs/rest/stocks/aggregates/custom-bars)
- [Daily aggregate product and publication schedule](https://massive.com/docs/flat-files/stocks/day-aggregates)

Machine-specific roots, source hashes, support submission state, acquisition
receipts, runtime packages and scheduler identities live in the private
QuantTrade operations index, not in this scientific contract note.
# Daily/reference preparation sidecar

`qt200_daily_reference_v1.reconcile_daily_references()` now connects the
full-panel raw daily capture to the retained support bundle, dated issue
responses and bound exchange-calendar source. It replays both source chains
before and after processing, retaining one source-keyed sidecar row per bar.

Exact-date identifier conflicts override matches; sparse identity responses
are never expanded into continuous validity. Ticker-event brackets remain
candidates. A stale last event, duplicate event dates or an unlinked event
response cannot create a historical alias interval. This matters for current
symbols whose historical observations can describe another security, such as
META and SNOW. No automatic price-history stitching is performed.

The output records per-symbol calendar gaps, dates beyond the original
calendar/action coverage, corporate-action source problems and narrowly scoped
candidate alias requests. It does not infer delisting from HTTP 404 or missing
bars, insert prices/liquidity, or qualify observed dates as exchange sessions.
Original identity and split/dividend raw captures must also be independently
reconciled; the operational integration runs those existing readers first.

Regression and actual-data integration execute on LSF GPUs, not local CPU.
These are data-preparation checks, not H100 optimizer qualification.
Training, native V5 and point-in-time authorization remain false until the
separate aggregate research dataset's remaining contracts are satisfied.
# Bounded acquisition acceleration

Independent ticker histories may be captured with the package-owned bounded
pipeline in `qt200_capture_pipeline_v1.py`: at most four tasks, one global
0.3-second request-grant spacing, serial pagination per history, deterministic
result order, and cancellation after a failure. This does not increase the
global request-rate allowance or change raw response/compression semantics.
The operator must serialize capacity checking and source publication, reserve
in-flight bytes, preserve previous attempts and consume exact remote regression
results before launch. Generic thread concurrency is not source qualification.

Tests in `test_qt200_capture_pipeline_v1.py` exercise real shared-thread pacing,
disjoint work, failure shutdown and byte-preserving transport/replay against
synthetic HTTP responses. They make no historical availability, investment or
GPU-optimization claims. Machine-specific attempts and measured throughput live
in the private operations handoff; no tests are run on the local controller.
