# QT200 inputs for adaptive V5 PPO

QT200 is a user-selected, fixed 200-equity research panel. This note describes
the real-data integration boundary, not a new manifest generation, qualified
dataset, launch permission, or profitability result. Keep the native V5
runner, timing, targets, costs, selection and verification rules unchanged.

## Correct consumer and scope

The legacy `datasets/raw_window.py` consumer constructs intraday OHLCV blocks
and next-block labels with CASH plus one action per equity. Adaptive V5 instead
consumes a reconstructed source graph, causal forecasts and a ten-control PPO
action. A passing raw-window format audit cannot promote data into V5.

The native PIT-1500 and PIT-500 rules are rank cutoffs, not mandatory population
sizes. `alpha/pit_universe.py::build_historical_membership` requires complete
rank inputs relative to the supplied security master, then selects ranks below
the rule limit. An eligible population can contain fewer than 200 names. Do not
lower eligibility thresholds, invent rows or use future target validity to fill
the population.

Using a fixed QT200 candidate master conditions the historical experiment on a
later selection. Bind the ordered panel, selection date, stable security IDs and
dated aliases explicitly; describe results as **fixed-panel development
evidence**, not whole-market historical PIT selection. This candidate-scope
binding and its root enforcement remain required implementation work. Merely
supplying a smaller master does not create that binding.

## Required data, in dependency order

| Layer | Required production evidence |
|---|---|
| Original trades | Complete committed gzip transaction, original row number and raw-line hash; all selected corrections retained |
| Identities | Unique permanent-security and listing intervals; evidence-backed ticker continuity, including ticker reuse |
| Canonical market data | Full correction replay, separate price/high-low/volume condition eligibility, daily bars and tape, source receipts |
| Economics | Source-qualified splits, distributions, successors and terminal outcomes; causal availability and complete accounting |
| Features and fills | Exact 64-session histories; 19 bars and 15 tape features with independent masks; next-session `[09:35,09:45)` ET fills |
| Targets and forecasts | Seven native economic buckets through 126 sessions, 127 subsequent marks, separate maturity/terminal masks, prior fitted checkpoint and calibration |
| V5 handoff | Source bundle and dependency graph reconstructed in the actual evaluation runtime, then the ordinary V5 training/release workflow |

The source-selection scanner is an engineering extraction step only. Its
callback rows are provisional until the complete scan succeeds. Preserve the
original raw-line digest; hashing reconstructed CSV or a canonical dictionary
does not reproduce original-row provenance. A selected-row completion must
never masquerade as the whole-market canonical scan or an identity authority.

The existing daily-input and Feature V3 builders are P0-origin/accounting-bound.
Do not manufacture their qualification flags from arrays or copy the synthetic
fixture's resealing helpers into a production importer. The adaptive context
feature bridge must authenticate its genuine upstream sources explicitly.
Finalized historical files alone also do not establish what was available at
the close-plus-60-minute decision; retain the native availability/parity gates.

## History and access

`training/massive_adaptive_split_plan_v1.py` requires at least **2,016 consecutive
candidate exchange sessions**, including its purges, four outer folds, embargo
and lockbox. Extra feature history and target maturity are separate checks. The
organized 2022–2026 QT200 windows are insufficient for this unchanged geometry;
the full-history archive must supply the qualified candidate inventory.

Never shorten the geometry merely to fit the prepared files. Freeze dates from
source-qualified coverage, not investment outcomes. Validation, outer and
lockbox inputs remain subject to their existing release rules. Raw extraction
does not authorize policy evaluation on a protected date.

## Storage and execution

The approved execution strategy reuses an existing research PVC with narrow
`subPath` mounts: QT200 data read-only at `/mnt/qt200`, one immutable
attempt-specific package read-only at `/mnt/package`, and that attempt's
dedicated results directory writable at `/mnt/output`. Do not create a new
PVC, mount the home root, broaden the data scope or substitute `hostPath`.
The exact claim and source subpaths are bound in private operator evidence.

A bounded one-H100 mount-validation Job can establish access, permissions and
hardware for its exact attempt; it is not a regression run, model training or
a V5 data qualification. Existing-claim reuse resolves the storage strategy,
not the historical identity, economic, feature, forecast or source-graph gates
above. The organized QT200 mount does not expose the separate raw archive or
make the shorter organized history sufficient for the native V5 geometry.

Retain the original compressed archive as an external immutable dependency.
Prefer bounded rolling correction spools and packed daily/feature outputs over
retaining another complete tick copy. Check actual bytes and inodes before
publication, including existing datasets and temporary space. Delete no source
or failed evidence implicitly.

Loaded-source device/inode/ctime identities are runtime-specific. Byte-identical
copies require fresh loading and source reconstruction in the execution mount;
do not disable identity checks to make a remount pass.

Source review is not a test pass. Run regression and model tests on the approved
compute route and record exact source, runtime and test results before using
these changes. Single-run V5 completion is also distinct from a qualified joint
two-rank PPO adapter or a study-level pre-outer selection barrier.

The requested training route is H100, but the existing registered supervised
entrypoint `train_and_publish_massive_adaptive_alpha_v1` retains its frozen
authorizing CPU numerical contract. A passing H100 regression does not resolve
that mismatch or establish a fitted checkpoint. Treat it as a blocking runtime
contract decision: neither run CPU training nor change the scientific protocol
or registered trainer implicitly to satisfy the H100 request.

## Source-processing implementation boundary

The current source adapters deliberately expose distinct intermediate products:

- `qt200_identity_evidence_v1` checks immutable support and dated-reference
  capture chains. It reports issuer/issue/listing conflicts and candidate
  corporate-action joins; it never manufactures a security master or terminal
  event coverage. Provider `active`, a matching ticker, or a sparse historical
  snapshot is insufficient to establish tradability or continuous identity.
- `qt200_market_day_v1` consumes every provisional selected original trade
  into a bounded disk spool, reconciles it to full-gzip completion, replays
  corrections, then emits bars, legacy tape, separately labeled volume and
  price-plus-volume flows, and the V5 morning-fill population. An invalid
  ticker-day keeps its evidence and explicit false masks. The compact report
  remains dependent on the original gzip; it is not a native partition archive
  or a replacement for qualified daily-input authorities.
- `qt200_native_scan_v1` advances the existing native whole-file scan without
  retaining another tick copy. A first complete pass verifies the original gzip
  transaction, compressed hash/CRC and selected original-row provenance; a
  second uses `scan_massive_daily_trade_file_v0(retain_rows=False)` on **every**
  source row. It persists three sealed metadata files and supports read-only
  full-source reconstruction with fresh runtime loading. Original verification
  times are not backdated to force receipt equality. It does not resolve issue
  identities, replay correction chains into partitions, qualify condition
  eligibility, construct daily inputs or authorize training. Failed publication
  remains unaccepted; replay never regenerates missing evidence.
- `massive_adaptive_real_source_bridge_v1` composes existing typed accounting,
  Feature V3, adaptive context/action and decision-root builders. It checks
  sourced identity/rank equality and separate feature/action economic clocks.
  It cannot enlarge the existing feature population to a broader adaptive
  context universe. Population disagreement fails closed, but disagreement has
  not been established for real QT200 origins: a fixed panel may satisfy both
  existing eligibility rules. Check actual memberships before proposing another
  authority or adapter. Forecast fitting, calibration, target maturity and
  source-graph reconstruction remain the
  downstream native workflow; this adapter does not grant training readiness.

The dated-reference evidence also makes ticker reuse an operational concern,
not merely a hypothetical one. Never join an old same-ticker observation to a
modern QT200 issuer without source-backed issue continuity. Preserve conflicting
listing dates and provider corrections with their separate query/capture times.
Do not use retrospective reference financial fields as point-in-time features.

`build_qt200_issue_resolution_evidence_v1` now combines both identity diagnostics
by independently reopening their hash-bound support and dated-capture roots;
it accepts no caller-produced reports. It preserves the ordered 200 targets and
classifies dated composite/share-class identifier assertions as supporting,
conflicting or unknown. A conflicting identifier overrides a partial match;
CIK and name cannot substitute for issue identity. Query dates remain separate
from capture times. Economic rows retain candidate-only review flags, never
event-validity masks or qualified joins. The support extraction still cannot
reauthenticate omitted original provider bodies, whereas the dated loader
verifies its retained raw-body hashes and decoded-response links. Neither path
establishes continuous identity, historical knowledge times, PIT qualification
or training readiness; the existing standalone loaders remain unchanged.

`reconcile_qt200_split_source_v1` separately reopens the retained original
split payload, receipt and commit through the native source loader and V8
capture parser. It authenticates the exact hash-bound support selection by
page/result position, provider row, response hash and capture time. Missing,
additional, duplicated or modified selected positions fail closed; identical
provider rows at different positions are preserved. Candidate ticker aliases
retain their existing candidate-only meaning. This read-only operation can
establish original split-source provenance, not issue continuity, historical
availability, price adjustments, complete corporate-action coverage or training
readiness. It does not change the existing all-support diagnostic: other
provider surfaces must be authenticated separately. Its focused regressions
and actual retained-source reconciliation require remote acceptance evidence;
file presence or static review is not a passing result.

`reconcile_qt200_dividend_source_v1` applies the same source-provenance check
to retained dividend pages. Its original-payload reader streams a hash with
a separate 256-MiB bound; support files and transaction sidecars remain capped
at 16 MiB. Native replay is limited to 128 pages and 500,000 original results,
including unselected rows. Declaration, ex-dividend, record and payment dates
are preserved as provider fields, not converted into historical knowledge times.
Provenance authentication cannot authorize a dividend cash flow, establish the
security entitled to it or repair missing corporate-action coverage. Selected
rows remain candidate-only, and the aggregate training flag remains false.

`reconcile_qt200_identity_source_v1` handles the three retained reference and
ticker-event captures in their original self-receipted canonical-JSON format.
They are not native payload/receipt/commit transactions; no substitute sidecars
are created. The reader checks exact capture and declared-producer hashes,
newline-bearing self-receipts, original raw response bodies, pagination,
complete queried identifier inventories and the exact selected support rows.
Both unselected and 404 responses are checked. Original positions remain
distinct even when their visible provider values coincide.

The original concurrent event producer does not establish a serial ordering
between responses. Likewise, the support extractor used unordered target sets;
only contiguous target-block order within one response is normalized, while
response order and event order within each target remain exact. Neither rule
permits deduplicating observations or guessing chronology. Producer source
hashes bind a declared implementation, not an independent execution attestation
or provider signature. A successful read therefore qualifies retained capture
provenance only: continuous historical issue identity, listing/tradability,
historical availability, successor joins, economic accounting and V5 training
remain unqualified. The reader performs no provider requests or source writes.
Acceptance requires the focused remote tests and real-capture reconciliation;
the private operations index records the exact frozen attempt and result.

The September 9, 2026 H100 diagnostic passed all 127 focused cases and
reconciled the retained original identity captures: 36,575 reference rows on
38 pages and 11,347 event responses. It reproduced 231 selected reference
records, 200 selected captures, 214 ticker-change observations and 209
candidate alias intervals, while leaving the earlier split/dividend reports
byte-identical. The tested module SHA-256 was
`3fa7bb478c19c22426cf116de106d079779b80b6bda7f0e520a2daa2ca5b945f`;
identity diagnostic semantic receipt:
`cc4e92cb5c68b51a760962046242d5f0ef8e0a193834366af126564b49225e93`.
This is source-provenance regression evidence, not the V5 economic vertical,
historical issue qualification or joint-two-GPU optimizer qualification.

For the first original-day processing attempt, retain the bounded correction
spool and measure its peak bytes, throughput and memory before choosing the
full-history worker count. Both filesystem availability and fileset quota matter;
neither overrides the total user-approved project storage budget. A failed
pilot is retained evidence, not an invitation to overwrite its output root.

## Existing native prerequisites

The immediate integration path uses existing authorities, not another data
schema. A real bridge origin still requires the following:

1. `PITSecurityUniverseAuthority` instances under the existing P0, context and
   action rules, sharing sourced issue masters, ticker intervals, listing and
   delisting records, and rank inputs. Permanent issue IDs need not be FIGIs;
   ticker or issuer CIK equality is insufficient. Dated provider observations
   cannot supply unobserved historical `available_at_ms`, prove continuous
   listing, or establish successor/terminal coverage. Keep query, effective and
   capture times distinct rather than backdating knowledge.
2. `MassiveProfitabilityProductionAcquisitionV2`, qualified monthly rank inputs
   derived from exact 63-session bars, and a production
   `MassiveProfitabilityDecisionOriginPlanV2` containing the selected P0 origin.
   Committed raw files and calendar/condition/correction snapshots contribute
   upstream evidence; their presence alone does not establish these authorities.
3. `MassiveProfitabilityArchiveFreezeV1`, `MassiveProfitabilitySecuritySupportV2`
   and the native daily-input builder's complete inventory of whole-file scans,
   semantic/persisted partitions, bars and tape. The production
   [`build_massive_profitability_daily_input_authority_v1`](../src/rl_quant/features/massive_profitability_daily_input_authority_v1.py)
   is **archive-wide**, not a one-day importer. A successful selected market-day
   engineering report cannot replace its typed artifacts or qualification.
4. Native economic REST capture objects, `MassiveTerminalCoverageSourceV8`,
   `MassiveProfitabilityTerminalCoverageAuthorityV1` and the accounting freeze.
   Candidate corporate-action observations must first acquire valid issue joins,
   required-surface coverage and native source receipts. The bridge requires two
   separately persisted `MassiveEconomicOriginCoverageV8` objects: feature
   economics bound to the P0 origin's 12:30 ET clock, and action economics bound
   to the adaptive session close plus 60 minutes. Never substitute one for the
   other or reseal source flags to bypass missing evidence.
5. Once those roots genuinely qualify, the existing bridge and
   `materialize_massive_adaptive_decision_tensor_v1` can derive/replay model
   inputs. Forecasts additionally need a previously fitted, authorized checkpoint,
   training window and target archive, the native split/inference plans, and
   checkpoint-specific calibration. `materialize_massive_adaptive_forecast_archive_v2`
   produces a complete target-free inner-validation schedule, not a caller-chosen
   one-row forecast. A single valid bridge origin cannot replace that chronology,
   target maturity or the separate causal RL-training forecast authority.

None of these prerequisites is discharged by the adapter regression result;
the dataset and forecast route remain unqualified until their own gates pass.

## Verification scope

The next original-source failure exposed a separate cross-clock assumption:
the reader required every participant timestamp to be no later than its SIP
timestamp. Four hash-matched source rows had participant observations about
22–23 ms later than their SIP observations. The provider defines these as
different [timestamp sources](https://massive.com/docs/rest/stocks/trades-quotes/trades);
the observed difference alone does not establish calibrated event latency or
justify swapping, clamping or rounding either field. Preserve participant,
SIP and TRF clocks exactly. Canonicalization now binds this rule in its
specification digest, so old canonical receipts cannot be silently promoted
under the changed implementation.

Causal normalization retains the registered SIP entitlement delay and the
independent delayed-receive upper bound. Strategy availability must not precede
either participant or SIP time; an offset extending beyond qualified
availability still fails. Session/date, source-byte, correction, identity and
decision-cutoff checks are unchanged. This source-reader correction is not a
claim that every vendor timestamp is accurate, nor permission to use raw
historical rows as executable training input. The bounded error report now
also distinguishes up to four rejection reasons with an overflow count;
unseen errors remain failures, not silently discarded rows.

The expanded 240-case H100 check exposed four **unresolved legacy runtime**
acceptance failures in `test_massive_finalized_whole_file_v0.py`: two frozen
origin-policy/live implementation bindings differ, and the pinned `ml2` image
has an empty machine-id and no `/usr/bin/git` for that host-capture path.
Those failures are retained, not skipped or converted into authorization.
Historical policy receipts must not be resealed to hide implementation drift,
and host-source mocks cannot qualify the actual runtime. A separate 213-case
source-adapter test inventory excludes that entire legacy runtime module
(which remains available as a fixture dependency). Any passing result for
that narrower inventory applies only to the nonauthorizing original-day
reader/scan; it is not a claim that the broad suite or legacy runtime passed.

That focused inventory passed **213 tests on Kubernetes H100** on 2026-09-07,
with zero failures, errors or skips, including 24 clock-order regressions and
the actual four hash-matched rows. All three real identity diagnostics remained
byte-identical. The failed broad result remains a separate, unresolved finding.
Original-source retry results are separate from regression acceptance.

The first real native scan exposed a provider-format gap that array-only
synthetic fixtures had missed: a bare condition field such as `12` decoded to
a JSON integer and was rejected as a malformed list. The provider documents
both bare singleton and comma-delimited condition fields in its
[trade-level example](https://www.massive.com/blog/insights-from-trade-level-data).
The per-security extractor and whole-file reader now share the same strict
parser for empty, singleton, comma-delimited and existing JSON-array inputs.
Floats, booleans, nulls, negative codes, empty elements and damaged brackets
remain errors; original line hashes, row ordinals and economic fields are
unchanged. Numeric decoding does not confer condition eligibility.

That fix passed **137 tests on Kubernetes H100** on 2026-09-07, including the
actual retained failing AAPL row, the real per-security extraction route,
whole-file publication/replay and bounded failure diagnostics. There were no
failures, errors or skips, and all three real identity diagnostic receipts
remained identical. A native rejection now retains up to four original-row
numbers, raw-line hashes and error reasons rather than only an aggregate
failure message. Failed source rows are still rejected, not silently skipped.
The original failed scan and source files are preserved. Its new receipt-bound
data retry is separate from this passing regression result and cannot be called
complete or training-ready until its own terminal and source gates pass.

An earlier native full-file scan wrapper passed a fresh **96-test** H100
regression, including 21 new cases for native-reference equality, original-line
hashes, fresh verification time, full-market rejection, damaged gzip, source
mutation, incomplete publication, changed support and nonmaterializing replay.
There were no failures, errors or skips; all three real identity diagnostics
reproduced exactly. This qualifies those source-adapter test cases, not a real
archive-wide scan or native V5 dataset. Real original-file materialization and
its terminal receipts remain a separate data-preparation step.

The cross-capture API and its 14 additional cases passed a fresh **75-test**
focused regression on Kubernetes H100 on 2026-09-07, with zero failures, errors
or skips. That immutable attempt reproduced both existing real support/dated
diagnostic files exactly, then produced the new reconciliation report from the
same captured sources. Its source inventory, actual JUnit result, digest-pinned
runtime and ownership-checked Job/Pod cleanup are recorded in private operator
evidence. No local CPU tests or model training were performed. This tests the
reconciliation implementation; it does not qualify historical identity or V5
training inputs.

The focused scanner, source-target/fill, identity-evidence, market-day and
native feature-bridge suites passed **61 tests** on the approved Kubernetes
H100 runtime on 2026-09-06. No local CPU regression tests were run. The first
attempt exposed two fixture setup defects (an absent nested source directory
and too few securities for native exposure residual degrees of freedom); the
fixtures were corrected without relaxing production validators, then tested
in a new immutable H100 attempt. Private operator receipts bind the exact
source inventory, image, device, JUnit results and cleanup.

The subsequent partial-index change passed healthy-day and invalid-day
comparison against the previous full-index layout. All source, correction,
economic and validity-mask fields matched; physical spool metadata and the
enclosing physical-dependent receipt were intentionally compared separately.
Both layouts reject corrupted stored rows. The change retains all provisional
rows and validation boundaries. Passing parity proves behavior for these
fixtures, not a real-data throughput improvement; measure a complete original
day before claiming acceleration or scaling full-history concurrency.

These are adapter regression tests, not complete V5 implementation qualification
or proof that the historical QT200 dataset is training-ready. The real identity
diagnostics deliberately remain nonauthorizing. Full-history processing and
native daily-source, forecast, calibration and graph qualification must be
demonstrated independently before training is launched.

See [native adaptive semantics](massive_pit_adaptive_alpha_v1.md) and the
[separate P0 boundary](massive_profitability_p0_execution.md). Machine-specific
mounts, quotas, job IDs and audit hashes belong in private operator receipts.
