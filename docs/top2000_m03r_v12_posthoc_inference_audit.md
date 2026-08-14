# TOP2000 M03R-v12 post-hoc inference audit

## Purpose and authority

This is a frozen, inference-only diagnostic for the completed negative
M03R-v12 three-session panel. It consumes the exact update-64 checkpoint bytes
and already-consumed pre-2026 qualification chronology. It performs no
training or checkpoint selection, cannot mint an economic generation, and
does not authorize 2026 access. All evidence remains future-selected,
development-only, nonreportable, and nonpromotable.

The audit answers two narrow questions before any new predictive generation:

1. Did the dedicated rank head learn ordering that the economic mean head did
   not, despite never being used by the v12 sleeve?
2. How much score and economic evidence changes after correcting the
   action-mask and action/return chronology defects found in the completed
   run?

## Frozen comparisons

Both `economic-mean` and `rank-score` are evaluated independently. The
original signal receives a predeclared active-mass ladder of 25, 50, 100, and
200 bp. Zero, sign-flipped, and deterministically shuffled controls are each
evaluated once at 200 bp. Repeating every control at every cap is deliberately
excluded because it adds storage and compute without identifying a new
scientific boundary.

The cost ladder is 0, 10, 20, and 40 bp per unit one-way turnover. Six folds
are concatenated in chronological order, while bootstrap blocks remain
fold-bounded. Primary inference uses 21-session circular moving blocks with
10- and 30-session sensitivities. Aggregate break-even cost is calculated from
summed gross active return divided by summed incremental turnover; fold ratios
are never averaged.

## Corrected causal boundary

Target validity and action eligibility are distinct:

```text
diagnostic label mask = future label path valid AND origin action eligible
action signal mask     = decision-origin availability
                         AND decision-origin positive regression weight
fill execution mask    = availability actually observed at the next fill
```

Future label-path availability may remove a diagnostic row, but it cannot add
an asset to the action universe or shape a score. Both raw score heads are
projected with a fresh decision-origin residual operator before construction;
the audit records raw-to-residual norm retention separately from final
portfolio projection retention.

The corrected chronology is:

```text
decision at state t
-> target book formed without outcomes
-> fill-time availability repair
-> that action earns return t+1
```

Every one of the 63 qualification decisions must therefore have one aligned
post-fill return. The previous sleeve evaluated 63 predictions against 63
transitions but earned the first return before the first action and charged
the final action without letting it earn a return.

## Required lineage

Each fold binds:

```text
parent predictive terminal file and semantic receipt
parent fold-terminal file
exact checkpoint file SHA-256
semantic model-state SHA-256
package plan and paired episode schedule
cache and asset axis
point-in-time risk source
origin-only action-mask source
post-fill return source
raw and residualized mean/rank arrays
selected scale array
all target-blind control identities
```

Checkpoint bytes must load through the package-owned strict V12 evaluator
loader. A raw file-hash match is necessary but insufficient. Result-bearing
workers use exactly one visible NVIDIA H100 80GB under the pinned image;
CPU replays are structural diagnostics only and cannot publish a terminal.

### Device-boundary and failure-capture lessons

Risk evidence is intentionally persisted on CPU while a result-bearing audit
constructs decision masks on its single H100. Any causal mask that combines
those objects must explicitly move the persisted regression-weight tensor to
the decision tensor's device before the Boolean intersection. The one-H100
capacity gate exercises this exact mixed-device operation; merely loading a
checkpoint on CUDA is not sufficient capacity evidence for this audit.

Terminal failure evidence must be captured before applying success-only Pod
cardinality checks. A failed Indexed Job may retain fewer Pods than its
declared completion count after cluster garbage collection. On failure, the
supervisor therefore records the exact retained Job, owned Pods, and bounded
logs first, then exact-cleans by run ID, Job UID, and fresh resourceVersion.
Only a completed Job is required to retain the full three-Pod scientific
inventory used to validate all worker outputs.

## Detached execution lifecycle

The three-worker audit is operated only by its package-owned attach-only
supervisor. The supervisor cannot create, replace, or discover a Job. It
accepts one predeclared suspended Job binding, validates the exact Job UID,
run ID, source archive, plan, image, and one-H100 capacity receipt, then
activates that Job with the frozen resourceVersion precondition.

The detached handoff is successful only after an immutable launch-success
receipt binds the supervisor process and live Job identity. Healthy execution
is then left alone; status inspection resumes only for an explicit request, a
bound terminal notification, or an anomaly already returned by a current
call. At terminal, the supervisor validates exactly three worker receipts,
six fold artifacts per worker, and the predeclared panel reports before it
performs UID/resourceVersion-guarded cleanup and verifies absence of the Job
and its UID-owned Pods.

Adding or changing this lifecycle source changes the source archive. Static,
capacity, and full-run evidence from an earlier package identity cannot be
reused across that boundary; a fresh package and run identity are required.

## Completed a08 result and decision

The immutable a08 audit completed all three settings and all eighteen fold
artifacts, validated the declared outputs, and exact-cleaned its Job and
UID-owned Pods. The phase-receipt file SHA-256 is
`e7ded042c6e36e4213535770e6d16f3d3a94793cc42f73bf4dc6fa4df96aa892`;
its semantic receipt is
`684e1f8b3eeceab8eebb0059bc7aa5376826410e0550dd1dc810358604d22342`.
The cleanup-receipt file SHA-256 is
`1dc1e91d2e902bdd4a38edf704b171d2e33c88af50c215de9f8cbf4373362ebd`.
These identifiers prove the audit closure, not a positive alpha result.

At the smallest predeclared 25-basis-point active-mass cap, the original
economic means produced the following annualized evidence:

| Setting | Mean IC | Positive IC / spread folds | Gross active | Net active at 10 bp | 21-session gross / net LCB | Spread LCB | Break-even cost | Raw-to-residual retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 listwise | 0.00483 | 5 / 5 of 6 | 0.01675% | -0.05246% | -0.00381% / -0.07290% | -0.01691% | 2.42 bp | 14.80% |
| P1 rank-Gaussian | 0.00392 | 3 / 5 of 6 | 0.01616% | -0.05315% | -0.00477% / -0.07398% | -0.01486% | 2.33 bp | 15.16% |
| P2 no-rank | 0.00426 | 4 / 6 of 6 | 0.01434% | -0.05493% | -0.00514% / -0.07413% | 0.00842% | 2.07 bp | 15.05% |

P1's dedicated rank score was the only rank-head path with a modest coherent
tail result: mean IC `0.00585`, positive mean-IC folds `4/6`, positive-spread
folds `5/6`, and a positive 21-session spread LCB of `0.05556%`. It was still
economically unqualified: gross active return was only `0.00724%`, 10-bp net
active return was `-0.06979%`, the gross and net lower bounds were negative,
break-even cost was `0.94` bp, and only `2.21%` of its raw score norm survived
the exposure-null operator. P0's dedicated rank score had the wrong sign; P2's
untrained rank control also had the wrong sign. Zero, shuffled, and sign-flip
controls did not establish a clean alternative qualification.

Every original sleeve traded on every scored date. Increasing its active-mass
cap scaled turnover and did not create cost-surviving evidence. The result
therefore blocks economic training. It supports only a fresh predictive study
that matches the 252-session context in training and qualification, supervises
only the selected h3 target, and applies rank loss to the exact mean consumed
by execution.

## Advancement rule

This audit is explanatory, not a delayed v12 selection surface. No outcome can
make v12 pass or authorize economic training. Its result chooses only between
fresh predictive research hypotheses:

- positive rank-head evidence with failed mean-head evidence supports a new
  direct rank-to-trade interface;
- severe raw-to-residual loss supports aligning the learned score with the
  executable exposure-null space;
- failure of both heads supports correcting context/sampling first and then
  comparing daily aggregates with a true five-minute sequence encoder;
- control parity or sign inconsistency blocks reuse of the apparent sleeve
  effect.

The next training generation must still use a fresh protocol, source package,
run, Job, and output identity. It must match training and qualification at a
full 252-session local context, make the selected three-session horizon
primary rather than a 10% auxiliary target, sample eligible dates uniformly,
separate causal action masks from future label masks, and keep economic and
2026 stages fail-closed.
