"""Run registry: an auditable record of EVERY trial in an experiment family.

The Deflated Sharpe Ratio, PBO, White's Reality Check, and Hansen's SPA are only honest if the trial count
they deflate for is the number of configs/seeds/universes ACTUALLY tried -- not just the winner a researcher
chooses to submit. Counting only the submitted candidate is the classic data-snooping leak (the review's #19).

This registry records each trial (including FAILED ones -- a failed trial is still a "try") and exposes the
count to feed into ``statistical_credibility_report(..., n_trials=registry.n_for_multiple_testing())``. It is a
pure, frozen data structure plus a manifest snapshot; a sweep runner that auto-populates it is a separate
(larger) concern. Pure, stdlib only; changes no backtest number.

``validate_final_reportability_inputs`` is the fail-closed gate a FINAL credibility report should route
through: it refuses a still-running (non-final) family, an empty honest-trial count, or a missing/incoherent
submitted winner, and returns the honest ``n_trials`` to hand straight to the credibility report.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

TRIAL_STATUSES = ("complete", "failed", "running")


@dataclass(frozen=True)
class TrialRecord:
    """One trial in an experiment family.

    ``included_in_multiple_testing`` marks whether this trial counts toward the data-snooping correction
    (default True -- every genuine attempt should count; set False only for non-comparable trials, e.g. a
    smoke test, and document why in ``notes``).

    ``selected`` marks THE submitted candidate -- the single winner a final report is written about. A registry
    holds at most one selected trial (enforced by ``RunRegistry``); a final report requires exactly one, and it
    must be ``complete`` and included (enforced by ``validate_final_reportability_inputs``) -- you cannot submit
    a winner you did not count in the search, nor one that failed or never finished.

    ``config_hash`` is an optional fingerprint of the trial's configuration (hyperparameters / seed / universe),
    recorded for audit so a reader can see which points in the search space were tried. ``notes`` is free-text
    rationale (e.g. why a trial is excluded). Both are recorded, never enforced."""

    run_id: str
    status: str  # one of TRIAL_STATUSES
    included_in_multiple_testing: bool = True
    selected: bool = False
    config_hash: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError(f"run_id must be a non-empty string; got {self.run_id!r}.")
        if self.status not in TRIAL_STATUSES:
            raise ValueError(f"status must be one of {TRIAL_STATUSES}; got {self.status!r}.")
        if not isinstance(self.included_in_multiple_testing, bool):
            raise ValueError(f"included_in_multiple_testing must be a bool; got {self.included_in_multiple_testing!r}.")
        if not isinstance(self.selected, bool):
            raise ValueError(f"selected must be a bool; got {self.selected!r}.")
        if self.config_hash is not None and (not isinstance(self.config_hash, str) or not self.config_hash):
            raise ValueError(f"config_hash must be None or a non-empty string; got {self.config_hash!r}.")
        if not isinstance(self.notes, str):
            raise ValueError(f"notes must be a string; got {self.notes!r}.")


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
        # Coerce the container to a tuple so the frozen guarantee is REAL: a list (the natural shape a sweep
        # runner accumulates trials into) would otherwise stay live and mutable, letting .append() smuggle in a
        # duplicate run_id or a second selected winner AFTER the invariants below were checked at construction.
        # Also type-check elements -- a non-TrialRecord would silently defeat every per-trial check.
        try:
            trials = tuple(self.trials)
        except TypeError as exc:
            raise ValueError(f"trials must be an iterable of TrialRecord; got {self.trials!r}.") from exc
        object.__setattr__(self, "trials", trials)
        for trial in self.trials:
            if not isinstance(trial, TrialRecord):
                raise ValueError(f"every trial must be a TrialRecord; got {type(trial).__name__}: {trial!r}.")
        run_ids = [t.run_id for t in self.trials]
        if len(run_ids) != len(set(run_ids)):
            duplicates = sorted({r for r in run_ids if run_ids.count(r) > 1})
            raise ValueError(f"duplicate run_ids in the registry: {duplicates}")
        # At most one submitted winner per family -- two selected trials is incoherent data, not a valid search.
        selected_ids = [t.run_id for t in self.trials if t.selected]
        if len(selected_ids) > 1:
            raise ValueError(f"at most one trial may be selected (the submitted winner); got {selected_ids}")

    def n_declared(self) -> int:
        """Total trials recorded (the full search size)."""
        return len(self.trials)

    def n_completed(self) -> int:
        return sum(1 for t in self.trials if t.status == "complete")

    def n_failed(self) -> int:
        return sum(1 for t in self.trials if t.status == "failed")

    def n_for_multiple_testing(self) -> int:
        """The honest trial count for DSR / PBO / RC / SPA: the number of FINISHED included trials --
        ``complete`` or ``failed`` (a failed attempt is still a try that could have been the winner) flagged
        ``included_in_multiple_testing``. Running (unfinished) trials are EXCLUDED -- they have no result to
        compare; use ``is_final()`` to confirm the family is complete before treating this as the search size.
        May be 0 for an empty / all-running / all-excluded registry, in which case the downstream DSR/PBO/RC/SPA
        controls (which require n_trials >= 1) fail closed. Counts trials, not distinct ``config_hash`` values:
        re-running an identical config under a fresh run_id counts again (the SAFE/conservative direction -- it
        over-deflates, never under). A caller wanting per-config counting must dedup on ``config_hash`` itself."""
        return sum(
            1 for t in self.trials if t.included_in_multiple_testing and t.status in ("complete", "failed")
        )

    def is_final(self) -> bool:
        """True iff NO included trial is still ``running`` -- the experiment family is complete and its
        multiple-testing count is stable. A final credibility report should require this."""
        return not any(t.included_in_multiple_testing and t.status == "running" for t in self.trials)

    def running_included_run_ids(self) -> tuple[str, ...]:
        """The included trials still ``running`` -- i.e. exactly why ``is_final()`` is False (empty when final).
        Surfaced so the finality gate can name the unfinished trials blocking a final report."""
        return tuple(t.run_id for t in self.trials if t.included_in_multiple_testing and t.status == "running")

    def selected_trial(self) -> TrialRecord | None:
        """The unique submitted winner, or None if the family has not designated one yet. (``RunRegistry``
        guarantees at most one selected trial, so this never has to choose between candidates.)"""
        for trial in self.trials:
            if trial.selected:
                return trial
        return None

    def to_manifest(self) -> dict[str, object]:
        """A JSON-serializable snapshot for the run/credibility artifact -- the auditable counts plus the
        per-trial records, so a reader can see the full search behind a reported result."""
        winner = self.selected_trial()
        return {
            "experiment_family_id": self.experiment_family_id,
            "n_declared_trials": self.n_declared(),
            "n_completed_trials": self.n_completed(),
            "n_failed_trials": self.n_failed(),
            "n_trials_for_multiple_testing": self.n_for_multiple_testing(),
            "is_final": self.is_final(),
            "selected_run_id": winner.run_id if winner is not None else None,
            "trials": [
                {"run_id": t.run_id, "status": t.status,
                 "included_in_multiple_testing": t.included_in_multiple_testing,
                 "selected": t.selected, "config_hash": t.config_hash, "notes": t.notes}
                for t in self.trials
            ],
        }


def validate_final_reportability_inputs(
    registry: RunRegistry,
    *,
    candidate_returns: Sequence[float],
    min_returns: int = 2,
) -> int:
    """Fail-closed gate guarding the inputs to a FINAL credibility report. Returns the honest ``n_trials``
    (``registry.n_for_multiple_testing()``) to hand straight to ``statistical_credibility_report``.

    A "final" report is the single artifact written about ONE submitted candidate selected from the whole
    search; deflating it for anything other than the stable, complete search size is the data-snooping leak
    this module exists to prevent. The gate raises ``ValueError`` (never silently downgrades) when:

    * the family is not final -- an included trial is still ``running``, so the multiple-testing count is still
      growing (the blocking run_ids are named);
    * the honest trial count is 0 -- an empty / all-running / all-excluded registry has nothing to deflate for,
      and the DSR/PBO/RC/SPA primitives require ``n_trials >= 1``;
    * there is not exactly one selected trial -- a final report needs an unambiguous submitted winner. This
      models a SINGLE submitted candidate; an ensemble report validates each member's series separately;
    * the selected trial is not ``complete`` or not ``included_in_multiple_testing`` -- you cannot report a
      winner that failed, never finished, or was excluded from the search it is deflated against;
    * the candidate has fewer than ``min_returns`` return observations, or any non-finite (NaN/inf) /
      non-numeric value -- a final report cannot be computed on a series too short for a finite Sharpe or one
      carrying corrupt values.

    Pass the candidate's ACTUAL return series as ``candidate_returns`` (not a length): the gate calls ``len``
    and scans for non-finite values itself, so the floor is bound to the real data rather than a caller-supplied
    integer that could drift away from it. ``min_returns`` defaults to 2 (the minimum for a finite per-period
    Sharpe) and may be raised but not lowered below 2."""
    if not isinstance(min_returns, int) or isinstance(min_returns, bool) or min_returns < 2:
        raise ValueError(
            f"min_returns must be an int >= 2 (a finite per-period Sharpe needs >= 2 observations); "
            f"got {min_returns!r}."
        )
    try:
        n_candidate_returns = len(candidate_returns)
    except TypeError as exc:
        raise ValueError(
            f"candidate_returns must be a sized sequence of returns (pass the actual series, not a length); "
            f"got {candidate_returns!r}."
        ) from exc
    if not registry.is_final():
        raise ValueError(
            "final reportability gate: the experiment family is not final -- these included trials are still "
            f"running, so the multiple-testing count is unstable: {list(registry.running_included_run_ids())}. "
            "Wait for them to finish (or mark them failed/excluded with a documented reason) before a final report."
        )
    n_trials = registry.n_for_multiple_testing()
    if n_trials < 1:
        raise ValueError(
            "final reportability gate: the honest trial count is 0 (empty / all-running / all-excluded "
            "registry); there is nothing to deflate for and the DSR/PBO/RC/SPA controls require n_trials >= 1."
        )
    winner = registry.selected_trial()
    if winner is None:
        raise ValueError(
            "final reportability gate: no trial is flagged selected -- a final report needs an unambiguous "
            "submitted winner. Mark the reported candidate with TrialRecord(..., selected=True)."
        )
    if winner.status != "complete":
        raise ValueError(
            f"final reportability gate: the selected trial {winner.run_id!r} has status {winner.status!r}, "
            "not 'complete'; you cannot report a winner that failed or never finished."
        )
    if not winner.included_in_multiple_testing:
        raise ValueError(
            f"final reportability gate: the selected trial {winner.run_id!r} is excluded from multiple testing; "
            "the submitted winner must be part of the search it is deflated against."
        )
    if n_candidate_returns < min_returns:
        raise ValueError(
            f"final reportability gate: the candidate has {n_candidate_returns} return observations, fewer than "
            f"the minimum {min_returns} required for a finite Sharpe / credibility report."
        )
    if not all(isinstance(r, (int, float)) and not isinstance(r, bool) and math.isfinite(r)
               for r in candidate_returns):
        raise ValueError(
            "final reportability gate: the candidate return series carries non-finite or non-numeric values "
            "(NaN / inf / None / bool); a final report cannot be computed on corrupt returns."
        )
    return n_trials
