"""Deterministic invariants for the structural Hold-30 sleeve comparator."""

from __future__ import annotations

import unittest

import torch

from rl_quant.execution.hold30_sleeves import (
    HOLD30_SLEEVE_COUNT,
    Hold30SleeveState,
)


ASSETS = 101  # CASH plus 100 names; the common 1% cap can hold full exposure.


def _all_cash(batch: int = 1, assets: int = ASSETS) -> torch.Tensor:
    weights = torch.zeros(batch, assets, dtype=torch.float64)
    weights[:, 0] = 1.0
    return weights


def _review_inputs(batch: int = 1, assets: int = ASSETS):
    scores = torch.zeros(batch, assets, dtype=torch.float64)
    benchmark = torch.zeros_like(scores)
    benchmark[:, 1:] = 1.0 / (assets - 1)
    trade_mask = torch.ones(batch, assets, dtype=torch.bool)
    caps = torch.ones_like(scores)
    gross = torch.ones(batch, dtype=torch.float64)
    return scores, benchmark, trade_mask, caps, gross


class Hold30SleevePhases(unittest.TestCase):
    def test_exact_thirty_session_spacing_and_locked_sleeves(self) -> None:
        state = Hold30SleeveState.from_portfolio(_all_cash())
        inputs = _review_inputs()
        observed = []
        for session in range(60):
            before = state.books.clone()
            review = state.review_maturing_(*inputs)
            observed.append((review.session_index, review.maturing_sleeve, review.review_age))
            locked = torch.arange(HOLD30_SLEEVE_COUNT) != review.maturing_sleeve
            torch.testing.assert_close(state.books[:, locked], before[:, locked], rtol=0.0, atol=0.0)

        self.assertEqual([sleeve for _, sleeve, _ in observed[:30]], list(range(30)))
        self.assertEqual([sleeve for _, sleeve, _ in observed[30:]], list(range(30)))
        self.assertTrue(all(age == 30 for _, _, age in observed))
        self.assertTrue(torch.equal(state.review_count, torch.full((30,), 2, dtype=torch.int64)))

    def test_cap_censor_residual_stays_in_sleeve_and_does_not_retry(self) -> None:
        books = torch.zeros(1, HOLD30_SLEEVE_COUNT, ASSETS, dtype=torch.float64)
        books[:, 0, 0] = 0.50
        books[:, 1:, 0] = 0.50 / 29.0
        state = Hold30SleeveState(books)
        inputs = _review_inputs()

        first = state.review_maturing_(*inputs)
        self.assertTrue(bool(first.maturity_cap_censored.item()))
        self.assertAlmostEqual(float(first.requested_turnover), 0.50, places=12)
        self.assertAlmostEqual(float(first.constructed_turnover), 0.10, places=12)
        self.assertAlmostEqual(float(state.books[0, 0, 0]), 0.40, places=12)
        sleeve_zero_after_censor = state.books[:, 0].clone()

        second = state.review_maturing_(*inputs)
        self.assertEqual(second.maturing_sleeve, 1)
        torch.testing.assert_close(state.books[:, 0], sleeve_zero_after_censor, rtol=0.0, atol=0.0)
        self.assertEqual(int(state.last_review_session[0]), 0)
        self.assertEqual(int(state.review_count[0]), 1)


class Hold30SleeveEconomics(unittest.TestCase):
    def test_drift_does_not_recapitalize_sleeves(self) -> None:
        books = torch.zeros(1, HOLD30_SLEEVE_COUNT, 3, dtype=torch.float64)
        books[:, 0, 1] = 1.0 / 30.0
        books[:, 1:, 0] = 1.0 / 30.0
        state = Hold30SleeveState(books)
        growth = state.drift_(torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64))
        self.assertAlmostEqual(float(growth), 31.0 / 30.0, places=12)
        self.assertAlmostEqual(float(state.sleeve_navs()[0, 0]), 2.0 / 31.0, places=12)
        torch.testing.assert_close(
            state.sleeve_navs()[0, 1:],
            torch.full((29,), 1.0 / 31.0, dtype=torch.float64),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertNotAlmostEqual(float(state.sleeve_navs()[0, 0]), 1.0 / 30.0, places=6)

    def test_forced_exits_are_cause_separated_pro_rata_and_keep_sleeve_cash(self) -> None:
        books = torch.zeros(1, HOLD30_SLEEVE_COUNT, 4, dtype=torch.float64)
        books[:, :, 0] = 0.25 / 30.0
        books[:, :, 1] = 0.25 / 30.0
        books[:, :, 2] = 0.25 / 30.0
        books[:, :, 3] = 0.25 / 30.0
        state = Hold30SleeveState(books)
        membership = torch.tensor([[True, True, True, False]])
        available = torch.tensor([[True, False, True, True]])
        caps = torch.ones(1, 4, dtype=torch.float64)
        gross = torch.ones(1, dtype=torch.float64)
        before_navs = state.sleeve_navs().clone()
        session_before = state.session_index

        repair = state.apply_forced_repairs_(membership, available, caps, gross)
        self.assertEqual(state.session_index, session_before)
        torch.testing.assert_close(state.sleeve_navs(), before_navs, rtol=0.0, atol=1e-12)
        # Membership deletion of asset 3 and availability exit of asset 1 are
        # distinct; the remaining asset 2 is reduced pro rata to the 1% cap.
        self.assertAlmostEqual(float(repair.membership_forced_delta[0, :, 3].sum()), -0.25, places=12)
        self.assertAlmostEqual(float(repair.availability_forced_delta[0, :, 1].sum()), -0.25, places=12)
        self.assertAlmostEqual(float(repair.risk_forced_delta[0, :, 2].sum()), -0.24, places=12)
        torch.testing.assert_close(
            state.books[0, :, 2],
            torch.full((30,), 0.01 / 30.0, dtype=torch.float64),
            rtol=0.0,
            atol=1e-12,
        )
        torch.testing.assert_close(
            repair.total_sleeve_delta.sum(dim=-1), torch.zeros(1, 30, dtype=torch.float64), rtol=0.0, atol=1e-12
        )
        torch.testing.assert_close(repair.aggregate_delta, state.aggregate_weights() - books.sum(1))

    def test_maturity_cross_net_and_reconciliation(self) -> None:
        books = torch.zeros(1, HOLD30_SLEEVE_COUNT, ASSETS, dtype=torch.float64)
        books[:, 0, 1:] = (1.0 / 30.0) / (ASSETS - 1)
        books[:, 1:, 0] = 1.0 / 30.0
        state = Hold30SleeveState(books)
        before = state.books.clone()
        review = state.review_maturing_(*_review_inputs())

        # Same-name renewal is crossed to an exact zero order, not represented
        # as a sell and repurchase that would reset economic age.
        self.assertAlmostEqual(float(review.same_name_cross_net_notional), 1.0 / 30.0, places=12)
        torch.testing.assert_close(review.requested_delta, torch.zeros_like(review.requested_delta), atol=1e-12, rtol=0.0)
        torch.testing.assert_close(review.constructed_delta, torch.zeros_like(review.constructed_delta), atol=1e-12, rtol=0.0)
        torch.testing.assert_close(
            state.aggregate_weights() - before.sum(dim=1), review.constructed_delta, atol=1e-12, rtol=0.0
        )
        torch.testing.assert_close(review.requested_sleeve_delta.sum(-1), torch.zeros(1, dtype=torch.float64))

    def test_residual_capacity_accounts_for_locked_sleeves(self) -> None:
        books = torch.zeros(1, HOLD30_SLEEVE_COUNT, ASSETS, dtype=torch.float64)
        books[:, 0, 0] = 1.0 / 30.0
        # Locked sleeve 1 consumes the full common cap of asset 1.
        books[:, 1, 1] = 0.01
        books[:, 1, 0] = 1.0 / 30.0 - 0.01
        books[:, 2:, 0] = 1.0 / 30.0
        state = Hold30SleeveState(books)
        scores, benchmark, mask, caps, gross = _review_inputs()
        scores[:, 1] = 20.0
        review = state.review_maturing_(scores, benchmark, mask, caps, gross)
        self.assertEqual(float(review.residual_asset_capacity[0, 1]), 0.0)
        self.assertEqual(float(state.books[0, 0, 1]), 0.0)
        self.assertAlmostEqual(float(state.books[0, 1, 1]), 0.01, places=12)

    def test_maturity_builder_retains_entry_score_gradient(self) -> None:
        state = Hold30SleeveState.from_portfolio(_all_cash())
        scores, benchmark, mask, caps, gross = _review_inputs()
        scores.requires_grad_()
        state.review_maturing_(scores, benchmark, mask, caps, gross)
        state.books[0, 0, 1].backward()
        self.assertIsNotNone(scores.grad)
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        self.assertGreater(float(scores.grad[:, 1:].abs().max()), 0.0)


class Hold30SleeveRestart(unittest.TestCase):
    def test_capture_detach_restore_is_exact(self) -> None:
        initial = _all_cash().requires_grad_()
        state = Hold30SleeveState.from_portfolio(initial)
        state.review_maturing_(*_review_inputs())
        snapshot = state.capture(detach=True)
        self.assertFalse(snapshot.books.requires_grad)

        restored = Hold30SleeveState.from_snapshot(snapshot)
        torch.testing.assert_close(restored.books, state.books, rtol=0.0, atol=0.0)
        self.assertEqual(restored.session_index, state.session_index)
        self.assertTrue(torch.equal(restored.last_review_session, state.last_review_session))
        self.assertTrue(torch.equal(restored.review_count, state.review_count))

        restored.review_maturing_(*_review_inputs())
        restored.restore_(snapshot)
        torch.testing.assert_close(restored.books, snapshot.books, rtol=0.0, atol=0.0)
        self.assertEqual(restored.session_index, snapshot.session_index)
        restored.detach_()
        self.assertFalse(restored.books.requires_grad)


if __name__ == "__main__":
    unittest.main()
