from functools import lru_cache
from typing import Any

import numpy as np


def glyph_vocab_size() -> int:
    """Read the integer constant exported dynamically by NLE's extension."""
    from nle import nethack

    return int(getattr(nethack, "MAX_GLYPH"))


def extract_glyphs(obs: Any) -> np.ndarray:
    """
    Unified MiniHack glyph extractor.

    Always returns glyphs of shape (B, H, W).
    """

    # Case 1: Single dict
    if isinstance(obs, dict):
        glyph = np.asarray(obs["glyphs"])
        if glyph.ndim == 2:
            return glyph[None]
        if glyph.ndim == 3:
            return glyph
        raise RuntimeError(f"Unexpected glyph shape: {glyph.shape}")

    # Case 2: list/tuple of dicts
    if isinstance(obs, (list, tuple)) and len(obs) > 0 and isinstance(obs[0], dict):
        return np.stack([np.asarray(o["glyphs"]) for o in obs], axis=0)

    # Case 3: ndarray of dicts (SyncVectorEnv or AsyncVectorEnv)
    if isinstance(obs, np.ndarray) and obs.dtype == object:
        flat = obs.flatten()
        if isinstance(flat[0], dict):
            glyphs = np.stack([np.asarray(o["glyphs"]) for o in flat], axis=0)
        else:
            glyphs = np.asarray(obs)
    else:
        glyphs = np.asarray(obs)

    # Normalize shapes:
    # (B, H, W) → OK
    # (B, B, H, W) → collapse duplicated env axis
    if glyphs.ndim == 4 and glyphs.shape[0] == glyphs.shape[1]:
        glyphs = glyphs[:, 0]

    # (H, W) → add batch dimension
    if glyphs.ndim == 2:
        return glyphs.reshape(1, *glyphs.shape)

    # (B, H, W) → correct
    if glyphs.ndim == 3:
        return glyphs

    raise RuntimeError(f"Unexpected MiniHack observation format: shape={glyphs.shape}")


def glyph_entity_ids() -> tuple[int, ...]:
    """Observable NLE monster/object glyphs, including the visible player glyph."""
    from nle import nethack

    is_monster = getattr(nethack, "glyph_is_monster")
    is_object = getattr(nethack, "glyph_is_object")
    return tuple(glyph for glyph in range(glyph_vocab_size()) if is_monster(glyph) or is_object(glyph))


@lru_cache(maxsize=1)
def glyph_scoring_schema():
    """Conservative observable terrain proxy plus standard player-role glyphs."""
    from nle import nethack

    lookup = [-1] * glyph_vocab_size()
    passable = {
        "doorway",
        "open door",
        "floor of a room",
        "corridor",
        "lit corridor",
        "staircase up",
        "staircase down",
        "ladder up",
        "ladder down",
        "altar",
        "grave",
        "opulent throne",
        "sink",
        "fountain",
        "ice",
        "lowered drawbridge",
    }
    blocked = {"wall", "closed door", "iron bars", "tree", "raised drawbridge"}
    offset = int(getattr(nethack, "GLYPH_CMAP_OFF"))
    symbol = getattr(nethack, "symdef")
    for index in range(int(getattr(nethack, "MAXPCHARS"))):
        name = symbol.from_idx(index).explanation
        if name in passable:
            lookup[offset + index] = 1
        elif name in blocked:
            lookup[offset + index] = 0
    monster = getattr(nethack, "permonst")
    roles = {
        "human",
        "archeologist",
        "barbarian",
        "caveman",
        "cavewoman",
        "healer",
        "knight",
        "monk",
        "priest",
        "priestess",
        "ranger",
        "rogue",
        "samurai",
        "tourist",
        "valkyrie",
        "wizard",
    }
    player_ids = tuple(
        int(getattr(nethack, "GLYPH_MON_OFF")) + index
        for index in range(int(getattr(nethack, "NUMMONS")))
        if monster(index).mname in roles
    )
    return tuple(lookup), player_ids
