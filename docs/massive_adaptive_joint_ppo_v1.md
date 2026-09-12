# Joint two-GPU adaptive PPO engineering adapter

The user changed the current training request on 2026-09-11 to one H100 and
one CUDA worker. This adapter documents the earlier two-GPU proposal and is
retained as optional engineering work; it is not a launch prerequisite for
the current single-GPU route. Preserve existing source bundles and regression
attempts rather than resizing or relabeling their two-rank evidence.

`massive_adaptive_joint_ppo_v1` implements the proposed panel's optimization
topology, not a new authorizing V5 runner or permission to fit QT200 data.
`launch_ready=false` for the real study. Existing V5 gates and the bars-only
track's separate feature/forecast requirements remain unchanged.

One rank-zero CPU collector executes the existing chronological three-book
environment and actor. It computes GAE over all 63 transitions. Both CUDA
ranks optimize replicas of the same 90-input, two-layer 128-unit actor/critic,
with one common seed and fold. Rank one does not create an environment, reset
an economic episode, choose a separate seed or fit a different fold.

The global permutation is generated once per epoch. Rank zero receives 32
samples and rank one 31, with no padding, duplicate counting or dropped row.
Advantages are normalized globally before the split. Each local loss is
`2/63 * sum(per_sample_loss)`; DDP's rank average is therefore the 63-sample
global mean. Gradient clipping follows reduction, with finite-gradient checks
before either optimizer step. Learning rates are not multiplied by two.
Four epochs produce four actor and four critic steps, not eight logical steps.
This follows the [DDP gradient-averaging contract](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html).

The launcher must use one node, two ranks, NCCL, one assigned CUDA device per
rank, FP32, deterministic algorithms and disabled TF32/autocast. It must set
`--max-restarts=0`: [torchrun stops surviving workers on peer failure](https://docs.pytorch.org/docs/main/elastic/run.html).
The originally proposed training execution profile was two H100s; passing A40/L40
regressions does not qualify that hardware/software profile or prove speedup.

## Persistence and evaluation

Only rank zero publishes a create-only, hash-bound, safe-torch checkpoint.
It contains the actor/critic, both optimizers, both ranks' CPU/CUDA/Python/NumPy
RNG states, global shuffle state, trial/seed/fold/configuration, runtime profile,
and the existing collector's economic book, chronology cursor and transition
history. Existing files are not overwritten; partial files remain failed
evidence. Resume requires identical source/configuration and runtime profiles.

The existing safe collector serializer is reused, without issuing a native
checkpoint authority. A separate CPU reader restores inference and economic
state without writes. Neither it nor a passing optimizer test authorizes
native V5 data, policy selection, outer access or positive profitability.
The adapter rejects collectors attached to authorizing native fit/forecast
roots until that production integration is explicitly qualified.

## Acceptance scope

Regression tests use the existing synthetic numerical market fixture. Its
upstream source/calibration shells are **not** source qualification. The real
actor, distribution, PPO loss, economic collector, compiler, fills and
checkpoint code execute; those outputs are not monkeypatched. This suite is
not a replacement for the persisted V5 vertical qualification.

The dedicated two-GPU checks exercise:

- rank-one data changing synchronized actor/critic gradients;
- both replicas matching an independent global-batch gradient/update reference;
- exact 32/31 coverage and four optimizer steps per global rollout;
- 126 distinct economic transitions across consecutive 63-session updates;
- fresh-process resume reproducing weights, both optimizers, both rank RNGs,
  next sampled actions, transitions and economic state;
- CPU checkpoint inference without modifying evidence;
- committed-checkpoint corruption and overwrite rejection;
- rank-one failure stopping the whole trial with zero retries.

For cross-batch FP32 arithmetic, the gradient reference tolerance is `atol=2e-6,
rtol=3e-5`. The update reference uses `atol=2e-5, rtol=3e-4` and separately
limits maximum parameter error to `2e-5`. Same-profile resumed execution must
match exactly; CPU/GPU training equality is not claimed. Freeze these tolerances
before the remote tests and preserve failures rather than adapting thresholds
to make a failed run pass.

Run regression only on allocated LSF GPUs. Test skip markers allow ordinary
CPU CI to report missing hardware honestly; a dedicated acceptance worker
must prove two GPUs and reject all skipped cases. Run the same frozen adapter
on the approved Kubernetes H100 profile before treating it as H100-qualified.
Study-wide pre-O0 selection and the bars-only forecast/observation integration
are still required before a real candidate panel.
