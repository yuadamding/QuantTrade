# Massive P0 profitability execution boundary

This document is the durable project contract for the research-only Massive
profitability vertical slice. It describes implemented semantics and the
evidence required to execute them. It is not evidence that a historical run
occurred or that the strategy is profitable.

## Scientific claim

The primary contrast is incremental delayed-tape profitability at 20 basis
points of one-way cost:

\[
\Delta_{\mathrm{tape},20}
=
\overline r_{\mathrm{net},20}(\mathrm{MV04})
-
\overline r_{\mathrm{net},20}(\mathrm{MV02}).
\]

The capacity-matched falsification contrast is:

\[
\overline r_{\mathrm{net},20}(\mathrm{MV04})
-
\overline r_{\mathrm{net},20}(\mathrm{MV04\mbox{-}SHUFFLE}).
\]

| Setting | Role |
|---|---|
| `MV00` | fixed bars sanity baseline |
| `MV02` | bars-only two-layer MLP |
| `MV04` | bars plus finalized Massive tape |
| `MV04-SHUFFLE` | MV04 with tape shuffled within decision date |

The information source is exactly two XNYS sessions before the decision. A
positive result can support persistent cross-sectional tape alpha. It cannot
support same-day, quote, spread-capture, auction, latency, or production-
equivalence claims.

## Implemented authority chain

The minimum framework at commit
`bd54cda43102b26c500efef16ed9777f6a169ecc` implements:

```text
authenticated Massive acquisition
→ archive and accounting freezes
→ PhasePlan V2 with a 63-session outer-to-lockbox embargo
→ DataGate V2
→ source-derived Features V3 and Targets V2
→ replay-promoted TournamentDataset V3
→ TournamentPlan V2
→ deterministic Training V4 and Checkpoint V3 replay
→ TrainingReplayAuthority V1
→ Prediction V3 replay
→ PredictionReplayAuthority V1
→ EvaluationPlan V6
→ SourceBundle V7
→ stitched, costed OuterEvidence V6
```

Generic dataset, checkpoint, prediction, training-replay, and prediction-
replay files are nonauthorizing. Package-owned root promotion must reproduce
their committed content before downstream evaluation.

The preferred one-fold V6 engineering vertical-slice command is:

```bash
PYTHONPATH=src conda run -n quanttrade python -m pytest -q \
  tests/test_massive_profitability_v6_vertical_slice.py
```

It passed under the project Python 3.11.15 `quanttrade` environment in 169.78
seconds. The same test also passed under the local Python 3.13.12 `.venv` in
125.21 seconds. This proves wiring, deterministic replay, corruption
rejection, and costed P&L on typed synthetic inputs. It does not replay raw
Massive files, execute the frozen 100-epoch historical workload, or establish
historical profitability.

## Frozen chronology and data semantics

The archive derives dates from committed source availability; callers do not
choose free start and end dates.

```text
minimum candidate dates:       1,827
initial fit:                    at least 756
inner purge:                    63
inner validation:               126
outer purge:                    63
outer tests:                    4 x 126 entry dates
outer-to-lockbox embargo:       63 maturation-only sessions
sealed lockbox:                 252 decision dates
feature rectangle:              64 sessions
targets:                        H1, H5, H21, H63
```

The embargo may mature an outer H63 position but cannot create an outer or
lockbox entry. Outer folds form one chronological portfolio: a prior-fold
tranche remains active after the next fold begins, and each calendar date
appears once.

```text
action universe:            decision-time monthly PIT-500
monthly rank input:         exact 63 sessions ending at decision t-1
feature source:             decision t-2 XNYS sessions
minimum vendor lead:        18 hours
decision time:              12:30 ET
entry/exit benchmark:       [15:50,16:00) ET qualifying-trade VWAP
qualifying fill trade:      terminal-active, price-forming, volume-forming
feature support:            exactly 64 exchange sessions
missing feature encoding:   value 0, mask false
```

The predictive tensor has 19 bars and 15 tape features. Corporate-action
fields are accounting inputs, never direct predictive features. Tape-flow
numerators and denominators use the same terminal-active regular-session
price-and-volume-forming population.

Targets are economic fill-to-fill returns at H1, H5, H21, and H63. Accounting
owns split-adjusted shares, cash, dividends, successors, mergers, terminal
outcomes, and the complete mark path. A selected position cannot disappear
because a future exit is missing. An unresolved total-loss fallback remains a
full selected-long loss and cannot become a selected-short windfall.

Before selection, every setting uses the same deterministic source-time
residualization: intercept, log price, log trailing ADV, five-session reversal,
21-minus-5-session momentum, and 63-session volatility. Target validity is not
read when causal eligibility or ranks are formed.

## Frozen training and portfolio contract

For trainable settings, confirmation requires:

\[
4\ \text{folds} \times 3\ \text{settings} \times 5\ \text{seeds}
=60\ \text{runs}.
\]

Those 60 fits require 60 independent deterministic CPU training replays.
Sixteen fold-setting prediction artifacts require root-bound replay. Five seed
outputs are averaged before residualization and portfolio construction; seeds
are not independent market histories.

The trainable architecture has separate 64-dimensional bars and tape
projections, an 8-dimensional staleness projection, two 128-unit GELU fusion
layers, LayerNorm, and 0.05 dropout. Every horizon emits mean, q10, median,
q90, and positive scale. Training uses AdamW at `3e-4`, weight decay `1e-4`,
at most 100 epochs, early-stopping patience 10, and complete decision-date
cross-sections. Objective weights are Huber 1.00, ranking 0.20, quantile 0.25,
calibration 0.10, and SSL 0.00.

```text
score:                    predicted mean
selection:                top/bottom 20%, deterministic security-ID ties
horizons:                 H1, H5, H21, H63
new-tranche scaling:      1/H
horizon allocation:       fixed 25% each
one-way costs:            10, 20, 40 bp
primary cost:             20 bp
short-borrow stress:      100 bp annualized
capacity:                 $1M, $10M, $50M total composite capital
entry cap:                2% of trailing 63-session ADV
inference:                at least 2,000 nonwrapping 63-session blocks
```

There is no portfolio optimizer, position-age input, duration loss, 30-day
holding objective, or duration-based checkpoint selection.

## Data-first execution gate

Do not allocate training compute from provider listings or directory names.
The exact attempt must prove, in order:

1. complete authenticated object GETs and whole-file scans;
2. independently replayed corrections and qualified conditions;
3. permanent identity, PIT membership/rank input, and exact rank bars;
4. semantic and persisted partitions plus independently reconstructed bars
   and same-population tape;
5. archive/accounting freezes, fills, terminal coverage, and source-derived
   feature/target accounting;
6. a passing DataGate V2 and replay-promoted Dataset V3; and
7. the V6 vertical slice in the execution runtime.

Listing receipts prove expected provider inventory, not local payloads. A
bootstrap rehash or ticker inventory does not replace canonical scans,
partitions, bar/tape reconstruction, or DataGate qualification.

Acquisition, scanning, partitioning, accounting, and dataset construction are
CPU/I/O work. At this generation the authorizing trainer is deterministic
float32 CPU with one intra-op thread, and its 60 fits are rerun for root-bound
replay. CUDA/H100 checkpoints cannot satisfy this contract. H100s may be used
after Dataset V3 promotion only for clearly labeled nonauthorizing candidate
diagnostics unless a new protocol generation changes the training authority.

Machine-specific roots, package and scheduler identities, credential metadata,
transfer rules, mount caveats, and exact recovery steps belong to the
Seadragon skill's `quanttrade-massive-p0.md` reference. Do not copy live job
status, cluster credentials, or secret values into repository documentation.

## Outer pass and authorization boundary

At 20 basis points, an outer development pass requires all three lower bounds
to be positive:

```text
LCB95[MV04 net return]
LCB95[MV04 - MV02 net return]
LCB95[MV04 - MV04-SHUFFLE net return]
```

It also requires positive MV04 contribution in at least three of four entry
folds, nonnegative mean MV04 return at 40 bp, positive $10M clipped MV04 return
at 20 bp, and break-even one-way cost of at least 20 bp.

An outer pass is a development conclusion only. It does not authorize final
profitability reporting, lockbox opening, retuning on the lockbox, or
reinforcement learning.

Before lockbox opening can be considered, the evaluator must also provide
positive top-quintile long-only active return at 20 bp and prove that no single
calendar year contributes more than 50% and no single sector more than 35% of
gross signal profit. These are requirements, not permission to inspect sealed
outcomes early.
