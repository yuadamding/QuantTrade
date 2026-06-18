"""Run registry: an auditable record of EVERY trial in an experiment family.

The Deflated Sharpe Ratio, PBO, White's Reality Check, and Hansen's SPA are only honest if the trial count
they deflate for is the number of configs/seeds/universes ACTUALLY tried -- not just the winner a researcher
chooses to submit. Counting only the submitted candidate is the classic data-snooping leak (the review's #19).

This registry records each trial (including FAILED ones -- a failed trial is still a "try") and exposes the
count to feed into ``statistical_credibility_report(..., n_trials=registry.n_for_multiple_testing())``. It is a
pure, frozen data structure plus a manifest snapshot; a sweep runner that auto-populates it is a separate
(larger) concern. Pure, stdlib only; changes no backtest number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TRIAL_STATUSES = ("complete", "failed", "running")


@dataclass(frozen=True)
class TrialRecord:
    """One trial in an experiment family. ``included_in_multiple_testing`` marks whether this trial counts
    toward the data-snooping correction (default True -- every genuine attempt should count; set False only
    for non-comparable trials, e.g. a smoke test, and document why)."""

    run_id: str
    status: str  # one of TRIAL_STATUSES
    included_in_multiple_testing: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError(f"run_id must be a non-empty string; got {self.run_id!r}.")
        if self.status not in TRIAL_STATUSES:
            raise ValueError(f"status must be one of {TRIAL_STATUSES}; got {self.status!r}.")
        if not isinstance(self.included_in_multiple_testing, bool):
            raise ValueError(f"included_in_multiple_testing must be a bool; got {self.included_in_multiple_testing!r}.")


@dataclass(frozen=True)
class RunRegistry:
    """An auditable, duplicate-free record of every trial in one experiment family. Use
    ``n_for_multiple_testing()`` as the ``n_trials`` for DSR / PBO / RC / SPA so the multiple-testing
    correction reflects the WHOLE search, not the cherry-picked winner."""

    experiment_family_id: str
    trials: tuple[TrialRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_family_id, str) or not self.experiment_family_id:
            raise ValueError(f"experiment_family_id must be a non-empty string; got {self.experiment_family_id!r}.")
        run_ids = [t.run_id for t in self.trials]
        if len(run_ids) != len(set(run_ids)):
            duplicates = sorted({r for r in run_ids if run_ids.count(r) > 1})
            raise ValueError(f"duplicate run_ids in the registry: {duplicates}")

    def n_declared(self) -> int:
        """Total trials recorded (the full search size)."""
        return len(self.trials)

    def n_completed(self) -> int:
        return sum(1 for t in self.trials if t.status == "complete")

    def n_failed(self) -> int:
        return sum(1 for t in self.trials if t.status == "failed")

    def n_for_multiple_testing(self) -> int:
        """The honest trial count for DSR / PBO / RC / SPA: every trial flagged ``included_in_multiple_testing``
        (completed OR failed -- a failed attempt is still a try that could have been the winner). At least 1
        so the controls are always defined for a non-empty included set."""
        return sum(1 for t in self.trials if t.included_in_multiple_testing)

    def to_manifest(self) -> dict[str, object]:
        """A JSON-serializable snapshot for the run/credibility artifact -- the auditable counts plus the
        per-trial records, so a reader can see the full search behind a reported result."""
        return {
            "experiment_family_id": self.experiment_family_id,
            "n_declared_trials": self.n_declared(),
            "n_completed_trials": self.n_completed(),
            "n_failed_trials": self.n_failed(),
            "n_trials_for_multiple_testing": self.n_for_multiple_testing(),
            "trials": [
                {"run_id": t.run_id, "status": t.status,
                 "included_in_multiple_testing": t.included_in_multiple_testing}
                for t in self.trials
            ],
        }
