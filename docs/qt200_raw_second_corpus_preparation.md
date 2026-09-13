# QT200 full-history raw-second preparation

This is a source-preparation extension of the existing raw-second research
workflow, not a new model or an authorization to train. The requested input
scope is the frozen ordered 200-stock QT200 panel, 2017-01-03 through
2026-08-31. A planning registry is not proof that all requested bars exist.
Input expansion does not change experiment split definitions or permit using
held-out performance for selection.

## Original bytes and bounded source access

`SecondPackWriter` stores complete original response pages in individually
framed gzip members plus a hash-bound index. It retains supplied original
compressed bytes and page receipts when repacking a captured source. Each
committed store has at most 24 one-hour queries, eight pages per query,
128,000,000 decoded response bytes, a 160,000,000-byte pack and a 4-MiB index.
These are storage/parser bounds, not provider subscription entitlements.

`PackedSecondCaptureRef` loads the same `SecondQuery` and `CapturedSecondPage`
objects as the existing expanded source. Its native capture identity is
unchanged; its physical index has a separate hash. Reference restoration
accepts only the explicit original or packed schemas. Partition and catalog
identities retain their raw-source meaning, not an engineered-cache identity.

Local interval access verifies query metadata before reading relevant page
bodies. Requested frames are bounded-decompressed and checked against both
compressed and original-byte hashes, then the unchanged raw-page validator
checks timestamps, pagination, OHLCV and missingness. Full packed-store replay
additionally hashes every byte. A local interval lookup does not claim that
unrelated partitions were freshly replayed.

There is no numeric or learned-embedding cache in this version. No model
normalization, resampling, return calculation, forward fill or corporate-action
price rewrite is introduced. Known-empty, unknown and delayed observations
retain their separate masks. Cross-partition duplicate conflicts still fail.

`pack_verified_second_capture` replays a retained acquisition before and after
packing. It does not remove the originals. This remains the default legacy
capture route: both copies and partial failures count against storage. A
separately identified direct-packed route is described below. It does not
rewrite, remove or retroactively qualify legacy captures.

## Direct packed acquisition generation

`raw_second_direct_capture_v1` exposes a separate plan, acquisition and replay
interface. It uses the same header-authenticated HTTP loop, with a private
package-owned second-response storage sink; the legacy transport has no new
default behavior and does not accept caller-provided storage implementations.
Plans still precede credential loading. No new credential loader or remote
credential transport is introduced.

For a successful direct batch, the exact retained layout is:

```text
plan.json
STARTED.json
packed/index.json
packed/pages.gzpack
COMPLETE.json
```

The original response is compressed once into its framed pack with acquisition
receipt bytes. There are no per-query directories or expanded native JSON
copies. HTTP/body/semantic failures preserve already written frames and an
explicit failure receipt, never a successful acquisition terminal. An index
alone is not acquisition completion. Interrupted or consumed roots cannot be
retried in place.

Before issuing another GET, the direct sink reserves room for a complete
maximum-sized next response and bounded frame/index metadata. This deliberately
stops conservatively near the batch ceiling rather than receiving a page whose
original bytes cannot be retained. Such a stop is an incomplete acquisition,
not a successfully truncated query. Provider status, pagination, original-byte
hashes, receipt chronology, census and native second semantics are replayed
before successful publication and again by the read-only verifier.

The corpus controller opts in through `storage_mode="direct-packed-v1"`, under
`rl-quant.raw-second-direct-packed-corpus-v1`. Omitting the option retains the
legacy `loose-and-packed-v1` generation. A successor must retain its
predecessor's storage generation; it cannot silently migrate formats or reset
request/byte counters. Reused captures declare their original mode and are
fully replayed using that mode's verifier.

Direct batches reserve 160,000,000 bytes plus 8 MiB for the pack/index and
bounded bookkeeping, and 32 inodes, independently of fresh project/fileset
admission. Those are conservative per-batch reservations, not measured corpus
compression or permission to consume the full project ceiling. Legacy batches
continue to reserve both copies. Completed actual byte counts are charged
through the immutable corpus lineage.

This new direct route requires its own LSF regression evidence; the earlier
packed-reader acceptance does not qualify code added afterward. Source
implementation, mocked HTTP tests and actual bulk acquisition are separate
milestones.

## Historical routing is not ticker substitution

The fixed policy slot and literal provider query ticker are different fields.
The narrow alias adapter supports the reviewed FB/META and ANTM/ELV changes
only with original ticker-event evidence and an original exact-date reference
identifying the intended common-stock issue. It preserves the provider ticker
and response bytes; it never rewrites historical OHLCV payloads.

An endpoint reference observation does not qualify every intervening date.
Ordinary fixed-symbol requests also require exact-date evidence against the
frozen intended-issue population. A historical ticker collision, absent
reference, unsupported predecessor/successor or prelisting interval cannot
silently become a valid QT200 history. Acquisition routing alone still does
not qualify continuous tradability, corporate-event accounting or historical
information availability.

The driver blocks a batch when any required slot lacks a supported exact-date
route or an explicitly reviewed unavailability disposition. An absent reference
alone cannot become a known-unavailable mask.

### Explicit pre-public-trading slot accounting

`SecondPrelistingDisposition` binds affirmative original SEC/Nasdaq documents,
their acquisition receipts, retained intended-issue provider observations, and
the frozen ordered 200-issue population. Eight Class-A intervals are reviewed:
PLTR, CRWD, DDOG, SNOW, ROKU, PINS, ABNB and COIN. ROKU retains its conflicting
September 28/29 date statements and uses the earlier exclusive boundary; it
does not qualify first-day intraday trading. DELL Class C is supported only
for the exact reviewed January 3, 2017 session, not an interpolated interval.
Its Class-V/VMware-tracking shares cannot substitute for the intended issue.

The new corpus generation explicitly opts in with
`slot_accounting="explicit-prelisting-v1"`. Each batch retains its exact
original ordered slots, accounting for each as acquired, reused, or
known-unavailable. All-unavailable batches issue no HTTP request and create
no provider capture or market records. Mixed batches request only supported
acquisition slots. Replay revalidates the original evidence, exact disjoint
slot cover and counters; rehashed caller-authored mappings are insufficient.
No slot is removed from the fixed universe.

This is acquisition accounting, not a native tensor or economic qualification.
The native catalog still needs explicit unavailable masks, dated tradability,
corporate actions and availability rules. No blank/404 response, current ticker
match, or missing file can authorize this disposition. Existing corpus plans
retain their original schemas and limits; a new generation cannot silently
extend a consumed pilot's budget or relabel its old completion receipts.

The added prelisting and mixed-slot suites require new LSF acceptance on the
exact source bytes, including original evidence fixtures. Earlier regression
receipts do not qualify this addition.

### Explicit predecessor/successor acquisition route

`SecondSuccessorRoute` adds one reviewed AVGO predecessor route. It requires
the original exact-date Broadcom Limited ordinary-share reference, an original
dated Broadcom Inc. common-share reference, the frozen intended-issue table,
and the exact reviewed SEC and Nasdaq document bytes. The predecessor's
missing FIGIs remain missing; it is not assigned the successor's identifiers.
Caller-authored mapping reports or matching text snippets cannot substitute
for those sources.

The relationship is the specific mandatory one-for-one exchange completed
after the April 4, 2018 close, effective for Nasdaq trading April 5. Queries
are restricted to the reviewed predecessor dates and regular-session clock
hours. Each requested date still needs its own original provider observation;
the event does not interpolate identity or tradability over the interval.
The corpus binds this route separately from a same-issue ticker alias and
revalidates its evidence when replaying completed batches.

This permits source acquisition only. It does not implement a portfolio
conversion, authorize historical observation vintages, adjust model prices,
or resolve other QT200 successor and listing cases. The corporate ledger
still needs its own supported event coverage before training.

### Explicit read-only evidence relocation

`EvidenceRelocation` allows `replay_direct_second_capture` to read a published
byte-identical copy of identity evidence while preserving the original plan,
absolute references, source hashes and raw page contents. The caller must
explicitly bind the package inventory, path map and original capture hashes.
There is no prefix substitution, original-path fallback, automatic repair,
or acquisition-time relocation option.

The resolver checks the exact package population and original-to-published
file mappings. Missing entries, ambiguous mappings, noncanonical paths,
links, unexpected files/directories, changed bytes and mismatched capture
identities reject. The mapping changes only where bytes are read, not which
evidence they represent. Original source implementation hashes remain
historical provenance; current replay implementation qualification is separate.

Both additions require a new LSF acceptance package. Include
`test_raw_second_successor_v1.py` with its mandatory hash-verified original
evidence fixture, `test_raw_second_evidence_v1.py`, affected corpus/capture
suites, and a separate fresh-process replay of a retained actual capture from
the worker's real published mount. Missing evidence or a skipped test must
not count as acceptance. Neither a successful byte transfer nor older tests
qualify newly edited source.

### Exact-day documented PCLN/BKNG route

`SecondDocumentedAliasRoute` is a separate, explicitly typed acquisition
route for BKNG's fixed slot 56 and literal provider ticker PCLN on
January 3, 2017 only. It binds the retained complete issuer-query response,
the original SEC name/ticker-change filing and capture receipt, the modern
BKNG reference, and the intended ordered QT200 issue population. Missing
historical FIGIs remain absent. The filing does not supply an earlier ticker
event or historical observation vintage; neither is invented.

The existing META/FB and ELV/ANTM alias contract is unchanged. The new route
has a strict serialization discriminator, with a conditional source hash in
loose/direct capture plans. Existing plan key sets and legacy payloads remain
unchanged when this route is absent. Corpus use additionally requires the
predeclared `alias_routing="documented-rename-v1"` option; predecessor tranches
must have the same option. Old tranches are not silently promoted or edited.
Unknown routes, unsupported dates and differing intended-issue evidence
reject before batch publication or HTTP acquisition.

This is not continuous identity, event accounting, economic coverage or
training qualification. It does not resolve HLT, RCL or CCL. Include the
documented-alias evidence and integration suites in a new exact-source LSF
acceptance package, with original fixtures and provider-disabled replay.
Static checks or a prior package's pass cannot qualify this new route.

## Bounded preparation, restart and validation

### Exact-day reviewed HLT/RCL/CCL classes

`SecondReviewedClassRoute` provides three acquisition-only routes for the
January 3, 2017 regular session. Each binds three exact original dated
provider wrappers, the unchanged intended QT200 population, and its complete
reviewed set of original SEC/NYSE bodies and acquisition receipts. The
ordinary fixed-slot identifier validator is unchanged. No generic matching-CIK
or missing-FIGI exception is introduced.

- HLT preserves regular-way common shares before the distributions and
  after-close reverse split; HLT WI, PK and HGV cannot substitute for HLT.
- RCL binds the specifically documented NYSE common-stock class and legal
  registrant, including the operating-name distinction. Provider FIGIs remain
  absent at all three observations; no synthetic issue identifier is emitted.
- CCL distinguishes old paired stock from redomiciled Bermuda common shares
  and records the specific share-count continuity. CUK is not a historical
  alias. Current missing FIGIs remain absent, not copied from historical CCL.

Corpus use requires `class_routing="reviewed-common-class-v1"` and an explicit
typed `CorpusIdentity.reviewed_class`. Plans bind the new implementation;
successor tranches must preserve the policy. Existing serialized identities
have no new field when the route is absent. Replay reopens all original
evidence before accepting the completed batch. The direct provider capture
still contains literal ticker/OHLCV bytes; corpus identity proof does not
relabel that capture as native economic qualification.

Neither later endpoint observations nor these source relationships establish
all intervening dates. Unsupported dates, classes, evidence changes, missing
sources, fabricated identifiers and conflicting routes reject before batch
publication or new requests. Distribution entitlements, fractional proceeds,
redomiciliation accounting, historical availability and native model catalogs
remain independent requirements; all training flags remain false.

Include `test_raw_second_reviewed_class_v1.py` and
`test_raw_second_reviewed_class_integration_v1.py` in a new exact-source LSF
acceptance package. They require the original evidence fixture and cover
persisted corpus capture/replay, strict routing opt-in, prior-contract
compatibility, corruption rejection and read-only evidence relocation.
Earlier source passes and static checks do not qualify the changed code.

### Tranche limits

The corpus registry yields deterministic session/hour batches lazily. A
preparation tranche contains at most 64 batches; it does not materialize
millions of query files just to plan the job. Request, response-byte and
retained-byte budgets are explicit. Fresh project/fileset bytes and inode
checks remain an operator responsibility before admitting acquisition.

Completed captures are replayed before reuse. A consumed but incomplete batch
is ambiguous, not an instruction to redownload. Failed and partial attempts
remain evidence. This driver cannot open credentials, change institutional
quotas, delete old data, repair missing market observations, or start training.
Historical identity and data completeness failures remain separate from an
unprofitable strategy result.

Resume checks the plan against the owner's original publication reservation
and replays the bound predecessor's completed range and counters. Supplying a
new hash for edited plan fields cannot reset consumed budgets or skip batches.
Registry admission uses the complete raw-input contract, the declared
900,000-ms availability assumption, and integral, second-aligned session clocks.

Required regression evidence is remote execution of the packed, alias and
corpus tests plus affected capture/partition/raw-input/experiment checks. The
direct generation additionally requires `test_raw_second_direct_capture_v1.py`
and the legacy shared-transport consumers, including failure preservation,
pre-request budget, explicit storage-generation and read-only replay cases. An
actual retained-source parity probe should compare complete page bytes and
receipt times, native and packed raw tensors, and all availability/missingness
masks without network acquisition or model updates. Lint or static parsing
alone is not runtime qualification. Use the assigned LSF GPU and retain exact
source inventory, logs and JUnit outcomes; fresh-process tests remain separate.

These preparation changes do not qualify full-200-stock H100 memory,
throughput, economic-event coverage or profitability. Those remain downstream
checks of the unchanged train–select–frozen-test workflow.
