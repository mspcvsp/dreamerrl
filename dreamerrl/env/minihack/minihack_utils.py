from typing import Any

import numpy as np


def extract_glyphs(obs: Any):
    """
    Unified MiniHack glyph extractor.

    Handles all possible MiniHack observation formats:
      • dict                      → obs["glyphs"]
      • list[dict]                → stack glyphs
      • np.ndarray(dtype=object)  → stack glyphs
      • np.ndarray(H, W)          → already glyph grid

    Always returns:
        glyphs: np.ndarray of shape (B, H, W)
    """

    # Case 1: Single dict (non-vector env)
    if isinstance(obs, dict):
        glyphs = obs["glyphs"]
        glyphs = np.asarray(glyphs)
        return glyphs.reshape(1, *glyphs.shape)

    # Case 2: Vector env: list/tuple of dicts
    if isinstance(obs, (list, tuple)) and len(obs) > 0 and isinstance(obs[0], dict):
        glyphs = np.stack([o["glyphs"] for o in obs], axis=0)
        return np.asarray(glyphs)

    # Case 3: Vector env: ndarray of dicts
    if isinstance(obs, np.ndarray) and obs.dtype == object and isinstance(obs[0], dict):
        glyphs = np.stack([o["glyphs"] for o in obs], axis=0)
        return np.asarray(glyphs)

    # Case 4: Already a glyph grid (H, W) or (B, H, W)
    arr = np.asarray(obs)

    if arr.ndim == 2:
        # Single glyph grid
        return arr.reshape(1, *arr.shape)

    if arr.ndim == 3:
        # Already (B, H, W)
        return arr

    raise RuntimeError(f"Unexpected MiniHack observation format: {type(obs)}, ndim={arr.ndim}")
