"""First stair-entry opportunity on the standard MiniHack R2 layout."""

from enum import IntEnum
from typing import TypedDict, cast

import torch
from nle import nethack

CORRIDOR_R2 = "MiniHack-Corridor-R2-v0"
COMPASS_OFFSETS = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
    "NE": (1, -1),
    "SE": (1, 1),
    "SW": (-1, 1),
    "NW": (-1, -1),
}


class StairOpportunity(TypedDict):
    step: int
    winning_action: int
    winning_action_probability: float
    winning_action_rank: int
    chosen_action: int


class CorridorEpisode(TypedDict):
    seed: int
    actor_mode: str
    opportunity: StairOpportunity | None
    immediate_entry: bool
    end_status: str
    terminated: bool
    truncated: bool
    episode_return: float


def compass_actions(commands):
    """Map actual environment indices, checking the NetHack key codes."""
    compass = cast(type[IntEnum], nethack.CompassDirection)
    offsets = {int(getattr(compass, name)): delta for name, delta in COMPASS_OFFSETS.items()}
    moves = {i: offsets[int(command)] for i, command in enumerate(commands) if int(command) in offsets}
    if set(moves.values()) != set(COMPASS_OFFSETS.values()) or len(moves) != len(offsets):
        raise ValueError("Corridor R2 diagnostics require the eight NetHack compass moves")
    return moves


def stair_action(obs, moves) -> int | None:
    """Find an adjacent visible down stair on the stock two-room layout."""
    x, y = map(int, obs["blstats"][:2])
    chars = obs["chars"]
    height, width = chars.shape
    for action, (dx, dy) in moves.items():
        nx, ny = x + dx, y + dy
        if not (0 <= nx < width and 0 <= ny < height and chars[ny, nx] == ord(">")):
            continue
        # A diagonal within the room has open floor on both flanks. This
        # excludes doorway and tight-corner moves that can't be confirmed.
        if dx and dy and (chr(chars[y, nx]) not in ".<>" or chr(chars[ny, x]) not in ".<>"):
            continue
        return action
    return None


def winning_action_stats(logits: torch.Tensor, action: int) -> tuple[torch.Tensor, torch.Tensor]:
    probability = logits.softmax(dim=-1)[..., action]
    # Break ties by action ID, matching torch.argmax for the greedy action.
    greater = logits > logits[..., action, None]
    tied_earlier = logits[..., :action] == logits[..., action, None]
    rank = 1 + greater.sum(dim=-1) + tied_earlier.sum(dim=-1)
    return probability, rank


def stair_opportunity(obs, moves, *, step, logits, chosen_action) -> StairOpportunity | None:
    winning = stair_action(obs, moves)
    if winning is None:
        return None
    probability, rank = winning_action_stats(logits, winning)
    return {
        "step": step,
        "winning_action": winning,
        "winning_action_probability": probability.item(),
        "winning_action_rank": int(rank.item()),
        "chosen_action": chosen_action,
    }


def summarize_stair_opportunities(episodes: list[CorridorEpisode]):
    with_opportunity = [episode for episode in episodes if episode["opportunity"] is not None]
    count = len(with_opportunity)
    entries = sum(episode["immediate_entry"] for episode in with_opportunity)
    wins = sum(episode["end_status"] == "TASK_SUCCESSFUL" for episode in with_opportunity)
    return {
        "episodes": len(episodes),
        "wins": sum(episode["end_status"] == "TASK_SUCCESSFUL" for episode in episodes),
        "episodes_with_opportunity": count,
        "immediate_entries": entries,
        "eventual_wins_with_opportunity": wins,
        "opportunity_rate": count / len(episodes) if episodes else None,
        "immediate_entry_rate_given_opportunity": entries / count if count else None,
        "win_rate_given_opportunity": wins / count if count else None,
    }
