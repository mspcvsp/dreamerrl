"""Episode movement diagnostics using labels that never enter the policy."""

from collections import Counter
from typing import TypedDict


class _OptionalMovementMetrics(TypedDict, total=False):
    failed_movement_rate: float


class MovementMetrics(_OptionalMovementMetrics):
    steps: int
    stationary_steps: int
    movement_steps: int
    no_movement_rate: float
    longest_stationary_streak: int
    unique_tiles: int
    two_step_returns: int
    two_step_return_rate: float
    movement_attempts: int
    failed_movement_attempts: int
    frozen_game_turns: int
    observed_game_turn_pairs: int
    action_counts: dict[str, int]
    action_frequencies: dict[str, float]
    action_names: dict[str, str]


class MovementTracker:
    def __init__(self, position, action_names):
        self.positions = [tuple(position)]
        self.action_names = action_names
        self.counts: Counter[int] = Counter()
        self.stationary = 0
        self.streak = 0
        self.longest = 0
        self.reversals = 0
        self.moved = 0
        self.frozen_turns = 0
        self.turn_pairs = 0
        self.movement_attempts = 0
        self.failed_moves = 0

    def step(self, action, position, *, movement_action=False, turn_advanced=None):
        position = tuple(position)
        same = position == self.positions[-1]
        self.counts[int(action)] += 1
        self.stationary += int(same)
        self.moved += int(not same)
        self.streak = self.streak + 1 if same else 0
        self.longest = max(self.longest, self.streak)
        self.reversals += int(not same and len(self.positions) >= 2 and position == self.positions[-2])
        self.movement_attempts += int(movement_action)
        self.failed_moves += int(movement_action and same)
        if turn_advanced is not None:
            self.turn_pairs += 1
            self.frozen_turns += int(not turn_advanced)
        self.positions.append(position)

    def report(self) -> MovementMetrics:
        steps = sum(self.counts.values())
        result = MovementMetrics(
            steps=steps,
            stationary_steps=self.stationary,
            movement_steps=self.moved,
            no_movement_rate=self.stationary / steps if steps else 0,
            longest_stationary_streak=self.longest,
            unique_tiles=len(set(self.positions)),
            two_step_returns=self.reversals,
            two_step_return_rate=self.reversals / (steps - 1) if steps > 1 else 0,
            movement_attempts=self.movement_attempts,
            failed_movement_attempts=self.failed_moves,
            frozen_game_turns=self.frozen_turns,
            observed_game_turn_pairs=self.turn_pairs,
            action_counts={str(i): self.counts[i] for i in range(len(self.action_names))},
            action_frequencies={str(i): self.counts[i] / steps if steps else 0 for i in range(len(self.action_names))},
            action_names={str(i): name for i, name in enumerate(self.action_names)},
        )
        if self.movement_attempts:
            result["failed_movement_rate"] = self.failed_moves / self.movement_attempts
        return result
