from uuid import UUID

import pytest

from models.player import GameContext, PlayerSnapshot, TurnContext
from players.player_8.player import SOCK_COST, Player8


def player(*, days: int = 1000) -> Player8:
	return Player8(PlayerSnapshot(UUID(int=0), 0), GameContext(24, 3, 4, days))


def turn(day: int, *, spent: float = 0.0, remaining: float = 250.0) -> TurnContext:
	return TurnContext(
		day=day,
		total_spent=spent,
		total_embarrassment=0.0,
		budget_remaining=remaining,
	)


def observe_age(p: Player8, *, day: int, black: int, white_age: int) -> None:
	p.history.record(day=day, offered=(black, black, 255 - 2 * white_age, 255 - 2 * white_age))


def test_young_drawer_preserves_current_budget() -> None:
	p = player()
	observe_age(p, day=100, black=4, white_age=4)
	target = p._adaptive_budget_target(turn(100), 250.0)

	assert target == 250.0
	assert not p.replacement_active


def test_mature_drawer_starts_replacement_cycle() -> None:
	p = player()
	observe_age(p, day=500, black=64, white_age=64)
	target = p._adaptive_budget_target(turn(500), 250.0)

	assert target == 10.0
	assert p.replacement_active


def test_young_cohort_stops_active_replacement_cycle() -> None:
	p = player()
	p.replacement_active = True
	observe_age(p, day=500, black=4, white_age=4)
	target = p._adaptive_budget_target(turn(500), 250.0)

	assert target == 250.0
	assert not p.replacement_active


def test_projected_roommate_spending_raises_budget_floor() -> None:
	p = player()
	observe_age(p, day=500, black=64, white_age=64)
	target = p._adaptive_budget_target(turn(500, spent=100.0, remaining=150.0), 250.0)

	assert target == 110.0
	assert p.replacement_active


def test_own_discard_estimate_is_not_charged_to_roommates() -> None:
	p = player()
	p.estimated_own_spend = 100.0
	observe_age(p, day=500, black=64, white_age=64)
	target = p._adaptive_budget_target(turn(500, spent=100.0, remaining=150.0), 250.0)

	assert target == 10.0


def test_selection_charges_credit_and_tracks_own_discard_spend() -> None:
	p = player()
	for day in range(1, 21):
		observe_age(p, day=day, black=64, white_age=64)
	selection = p.select_socks((0, 0, 64, 127), turn(500))

	assert selection.wear == (0, 1)
	assert selection.discard == (2, 3)
	assert p.estimated_own_spend == pytest.approx(2 * SOCK_COST)


def test_endgame_does_not_start_new_replacement_cycle() -> None:
	p = player()
	observe_age(p, day=990, black=64, white_age=64)
	target = p._adaptive_budget_target(turn(990), 250.0)

	assert target == 250.0
	assert not p.replacement_active
