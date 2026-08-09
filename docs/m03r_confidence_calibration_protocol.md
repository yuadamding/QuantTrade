# M03R confidence-calibration fit and replay protocol

**Protocol ID:**
`freeze-checkpoint-fit-inner-validation-calibrator-freeze-no-policy-updates-v1`  
**Evidence schema:**
`rl-quant.m03r-confidence-calibration-fit-evidence-v2`  
**Status:** package qualification only; no governed H100 launch authority

## Generation use

M03R v7 reuses the frozen two-stage ordering and standardized-unit-risk target
semantics below. That semantic reuse does not let a v6 receipt label a v7
artifact: a governed v7 path must bind its own protocol, design, setting,
checkpoint, model, folds, and proposal-path identities through a
generation-qualified public surface. The TOP2000 compatibility runtime's raw
sigmoid sizing is explicitly uncalibrated development behavior and is not
evidence that this deployment calibration was applied.

## Frozen two-stage order

Confidence calibration is downstream of policy training. The only permitted
order is:

1. train the policy and raw confidence-logit head without applying deployment
   calibration or confidence-dependent active-risk sizing;
2. freeze the selected checkpoint and model-state identities;
3. run that frozen checkpoint over the declared inner-validation rows;
4. build typed v6 outcome evidence from the exact detached
   `[observation, 30]` standardized-unit-risk policy and C1 net simple-return
   paths, their exact ordered `(trading_session, fold_id)` row identities, and
   the authoritative proposal-path manifest SHA-256;
5. pass the actual detached logits, typed v6 outcome evidence (or immutable-v5
   binary targets), fold IDs, trading dates, seed, checkpoint SHA-256, and
   model-state SHA-256 to
   `fit_and_bind_m03r_confidence_calibration`;
6. freeze the returned calibrator and content-addressed fit evidence;
7. validate or deploy that exact checkpoint with that exact calibrator.

No policy parameter, checkpoint, checkpoint-selection decision, or confidence
head may be updated using the fitted calibrator. A later policy update creates
a new model-state/checkpoint identity and requires a new calibration fit. This
prevents calibration diagnostics from feeding back into the policy that
generated the fit logits.

The low-level manifest constructor remains available for historical fixtures.
It is not governed fit evidence. Governed qualification requires the
package-owned fit receipt plus retained source arrays that replay its digest.

During the first stage, the raw confidence head is supervised by the
package-owned binary-log-loss primitive over internally derived signs of the
continuous standardized-unit-risk 30-session active-log-return outcome. The
confidence head consumes a detached market representation and uses a separate
optimizer route, so this auxiliary loss cannot perturb the alpha core. Until
the governed driver implements and receipts that isolated step, v6 launch
remains blocked.

## Target construction and circularity boundary

The v6 binary outcome is produced from the frozen checkpoint's standardized
unit-risk 30-session proposal before confidence-dependent risk sizing. It is
not produced from the final confidence-sized policy path. V6 callers cannot
provide either pre-aggregated outcomes or binary targets. They must use
`build_m03r_v6_confidence_outcome_evidence` with exact detached policy and C1
net simple-return arrays of shape `[observation, 30]`, aligned fold IDs and
trading sessions, and the authoritative proposal-path manifest SHA-256. The
package computes

\[
o_j=\sum_{h=1}^{30}\log(1+r^P_{j,h})
-\sum_{h=1}^{30}\log(1+r^{C1}_{j,h})
\]

and then derives `(o_j > 0)` internally. Immutable v5 retains its historical
binary-target API.

Every fit receipt includes a content-addressed target-construction contract
binding the protocol generation, design, benchmark, 30 post-fill returns,
unit-risk normalization, and the exclusion of confidence sizing from the
outcome path. A nested economic-path receipt binds the proposal-path manifest,
all 30 daily policy returns, all 30 daily C1 returns, the package-computed
continuous outcomes, the exact ordered row identity, and the aggregation rule.
The fitter requires this ordered row identity to match the logit/fold/date rows
before canonical sorting. The derived target-array
SHA-256 then includes both the computed outcome-array digest and the target
contract digest. Consequently, changing either daily path—even without
changing an outcome sign—or asserting the same bits under different semantics
changes the evidence identity and cannot satisfy replay.

Immutable v5 retains its original, less-specific target text. Its fit receipt
labels the unit-risk and confidence-sizing relationship as unspecified by v5;
it does not retroactively claim v6 semantics. V6 receipts require the exact
`frozen-standardized-unit-risk-proposal-before-confidence-sizing-v1` path and
fail closed for a final confidence-sized target.

## Deterministic fit

The fitter canonicalizes rows by `(trading_session, fold_id)` and rejects
duplicate row identities, noncanonical dates, attached logits, nonbinary or
single-class targets, and constant logits. It then fits

\[
c_t=\sigma(a z_t+b),\qquad T=1/a,
\]

in CPU float64 using the content-bound bounded-Newton contract. Optimizer ID,
iteration limit, line-search rule, convergence tolerance, L2 regularization,
slope bounds, intercept bounds, completed iterations, and final loss are part
of the evidence digest.

The fitter—not its caller—computes:

- canonical row, fold-array, date-array, logit-array, v6 continuous-outcome,
  derived-target-array, and calibrated-probability SHA-256 digests;
- temperature and intercept;
- Brier score and observed target rate;
- expected calibration error and its complete bin evidence;
- the ordinary calibration manifest and the enclosing replay receipt.

Callers cannot provide fitted parameters or calibration metrics to the
governed function.

## Frozen ECE rule

ECE uses ten equal-width probability bins. Bins are left-closed and
right-open, except the final `[0.9, 1.0]` bin, whose upper edge is inclusive.
The receipt binds every edge, count, mean confidence, observed target rate,
and absolute gap. Empty bins have zero means and zero gap. ECE is

\[
\sum_{k=1}^{10}\frac{n_k}{N}
\left|\bar c_k-\bar y_k\right|.
\]

The binning rule is not caller-selectable.

## Replay and retention

Retain the exact detached logits, typed v6 daily-path outcome evidence (or
historical v5 binary targets), fold IDs, and trading dates with the checkpoint
receipt. `replay_m03r_confidence_calibration_fit` recomputes the v6 economic
path receipt, verifies the exact pre-canonicalization row-order binding,
re-canonicalizes rows, re-derives target signs, refits the
calibrator, recomputes all metrics and hashes, and requires byte-identical
canonical evidence. Any altered daily return, path manifest, row identity,
checkpoint/model identity, fitted value, metric, bin, or protocol field fails
closed.

This evidence demonstrates deterministic calibration mechanics. It does not
by itself authorize outer-data access, checkpoint promotion, or deployment.
