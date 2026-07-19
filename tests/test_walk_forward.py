from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta

import pytest

from rl_quant.datasets import WalkForwardConfig, generate_walk_forward_folds


def _dates(count: int) -> tuple[date, ...]:
    start = date(2024, 1, 1)
    return tuple(start + timedelta(days=index) for index in range(count))


def _config(**overrides: int | str | None) -> WalkForwardConfig:
    values: dict[str, int | str | None] = {
        "decision_axis_id": "snapshot-sha256:fixture-a",
        "initial_train_size": 6,
        "validation_size": 2,
        "test_size": 2,
        "label_horizon": 2,
        "purge_size": 2,
        "embargo_size": 1,
        "max_train_size": None,
        "fold_count": None,
    }
    values.update(overrides)
    return WalkForwardConfig(**values)  # type: ignore[arg-type]


def test_expanding_folds_have_explicit_blocks_dates_and_stable_identities() -> None:
    config = _config()
    folds = generate_walk_forward_folds(_dates(40), config)
    assert len(folds) == 3

    first, second, third = folds
    assert first.train.positions == tuple(range(0, 6))
    assert first.train_validation_purge.positions == (6, 7)
    assert first.validation.positions == (8, 9)
    assert first.validation_test_purge.positions == (10, 11)
    assert first.test.positions == (12, 13)
    assert first.test_label_tail.positions == (14, 15)
    assert first.embargo.positions == (16,)
    assert first.embargo_complete
    assert first.reuse_not_before_position == 17
    assert first.train.dates == _dates(6)

    assert second.train.positions == tuple(range(0, 17))
    assert third.train.positions == tuple(range(0, 28))
    assert [fold.train.size for fold in folds] == [6, 17, 28]
    assert all(fold.validation.size == config.validation_size for fold in folds)
    assert all(fold.test.size == config.test_size for fold in folds)
    assert first.identity.value == first.fold_id
    assert first.fold_id.startswith("wf-0000-tr-00000000-00000006")
    assert hash(first.identity) == hash(generate_walk_forward_folds(_dates(40), config)[0].identity)
    assert folds == generate_walk_forward_folds(_dates(40), config)

    changed_interior = list(_dates(40))
    changed_interior[1] = changed_interior[0]
    changed = generate_walk_forward_folds(tuple(changed_interior), config)[0]
    assert changed.identity.decision_digest != first.identity.decision_digest
    assert changed.fold_id != first.fold_id

    revised_snapshot = generate_walk_forward_folds(
        _dates(40),
        _config(decision_axis_id="snapshot-sha256:fixture-b"),
    )[0]
    assert revised_snapshot.identity.decision_axis_id != first.identity.decision_axis_id
    assert revised_snapshot.identity.decision_digest != first.identity.decision_digest
    assert revised_snapshot.fold_id != first.fold_id


def test_rolling_folds_cap_training_and_advance_the_left_edge() -> None:
    folds = generate_walk_forward_folds(_dates(40), _config(max_train_size=10))
    assert [fold.train.size for fold in folds] == [6, 10, 10]
    assert [fold.train.start_position for fold in folds] == [0, 7, 18]
    assert [fold.train.stop_position for fold in folds] == [6, 17, 28]
    assert folds[1].train.positions == tuple(range(7, 17))
    assert folds[2].train.positions == tuple(range(18, 28))


def test_purge_label_tail_and_embargo_prevent_overlap_and_early_reuse() -> None:
    config = _config()
    folds = generate_walk_forward_folds(_dates(40), config)

    out_of_sample_positions: set[int] = set()
    for fold in folds:
        blocks = (
            fold.train,
            fold.train_validation_purge,
            fold.validation,
            fold.validation_test_purge,
            fold.test,
            fold.test_label_tail,
            fold.embargo,
        )
        sets = [set(block.positions) for block in blocks]
        for index, left in enumerate(sets):
            assert all(not left.intersection(right) for right in sets[index + 1 :])

        assert fold.train.positions[-1] + config.label_horizon < fold.validation.positions[0]
        assert fold.validation.positions[-1] + config.label_horizon < fold.test.positions[0]
        assert fold.test.positions[-1] + config.label_horizon == fold.test_label_tail.positions[-1]
        current_oos = set(fold.validation.positions + fold.test.positions)
        assert not out_of_sample_positions.intersection(current_oos)
        out_of_sample_positions.update(current_oos)

    for earlier_index, earlier in enumerate(folds[:-1]):
        immediate_next = folds[earlier_index + 1]
        assert immediate_next.train.stop_position == earlier.reuse_not_before_position
        assert set(earlier.test.positions).issubset(immediate_next.train.positions)
        for later in folds[earlier_index + 1 :]:
            reused = set(earlier.test.positions).intersection(later.train.positions)
            if reused:
                assert later.train.stop_position >= earlier.reuse_not_before_position
                assert later.train.stop_position >= earlier.embargo.stop_position


def test_duplicate_dates_are_disambiguated_by_immutable_integer_positions() -> None:
    repeated = tuple(date(2024, 1, 1) + timedelta(days=index // 3) for index in range(20))
    fold = generate_walk_forward_folds(repeated, _config(fold_count=1))[0]
    assert fold.validation.decisions[0].position == 8
    assert fold.validation.decisions[0].decision_date == repeated[8]
    assert len(set(fold.train.dates)) < fold.train.size

    with pytest.raises(FrozenInstanceError):
        fold.identity.ordinal = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        fold.train.decisions[0].position = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"initial_train_size": 0}, "initial_train_size"),
        ({"validation_size": True}, "validation_size"),
        ({"label_horizon": -1}, "label_horizon"),
        ({"purge_size": 1}, "purge_size must be >= label_horizon"),
        ({"max_train_size": 5}, "cannot be smaller"),
        ({"fold_count": 0}, "fold_count"),
        ({"decision_axis_id": ""}, "decision_axis_id"),
        ({"decision_axis_id": " snapshot "}, "decision_axis_id"),
    ],
)
def test_walk_forward_config_fails_closed(
    overrides: dict[str, int | str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_generation_rejects_bad_dates_insufficient_data_and_partial_exact_plan() -> None:
    with pytest.raises(ValueError, match="chronological"):
        generate_walk_forward_folds(
            (date(2024, 1, 2), date(2024, 1, 1), *_dates(18)),
            _config(fold_count=1),
        )
    with pytest.raises(ValueError, match="canonical YYYY-MM-DD"):
        generate_walk_forward_folds(tuple(["20240101"] * 20), _config(fold_count=1))
    with pytest.raises(TypeError, match="datetime"):
        generate_walk_forward_folds(
            tuple([datetime(2024, 1, 1), *_dates(19)]),  # type: ignore[arg-type]
            _config(fold_count=1),
        )
    with pytest.raises(ValueError, match="at least 16"):
        generate_walk_forward_folds(_dates(15), _config())
    with pytest.raises(ValueError, match="requested 4 complete folds but only 3"):
        generate_walk_forward_folds(_dates(40), _config(fold_count=4))


def test_exact_fold_count_is_deterministic_and_ignores_only_unneeded_suffix() -> None:
    folds = generate_walk_forward_folds(_dates(100), _config(fold_count=2))
    assert len(folds) == 2
    assert folds[-1].embargo.stop_position == 28


def test_final_fold_needs_label_support_but_not_a_trailing_embargo() -> None:
    config = _config()
    final_at_axis_end = generate_walk_forward_folds(_dates(16), config)
    with_full_embargo = generate_walk_forward_folds(_dates(17), config)

    assert len(final_at_axis_end) == 1
    assert final_at_axis_end[0].test_label_tail.positions == (14, 15)
    assert final_at_axis_end[0].embargo.positions == ()
    assert not final_at_axis_end[0].embargo_complete
    assert final_at_axis_end[0].reuse_not_before_position == 17

    # An unused suffix may fill the waiting period, but it is not part of the
    # evaluated decision axis and therefore cannot rename the fold.
    assert with_full_embargo[0].embargo.positions == (16,)
    assert with_full_embargo[0].embargo_complete
    assert final_at_axis_end[0].identity == with_full_embargo[0].identity
    assert final_at_axis_end[0].fold_id == with_full_embargo[0].fold_id


def test_zero_horizon_purge_and_embargo_have_explicit_empty_blocks() -> None:
    config = WalkForwardConfig(
        decision_axis_id="snapshot-sha256:zero-geometry",
        initial_train_size=3,
        validation_size=1,
        test_size=1,
        label_horizon=0,
        purge_size=0,
        embargo_size=0,
        fold_count=2,
    )
    first, second = generate_walk_forward_folds(_dates(7), config)
    assert first.train_validation_purge.positions == ()
    assert first.validation_test_purge.positions == ()
    assert first.test_label_tail.positions == ()
    assert first.embargo.positions == ()
    assert first.test.positions == (4,)
    assert second.train.positions == tuple(range(5))
