from typing import Any

import numpy as np


def extract_glyphs(obs: Any) -> np.ndarray:
    """
    Unified MiniHack glyph extractor.

    Always returns glyphs of shape (B, H, W).
    """

    # Case 1: Single dict
    if isinstance(obs, dict):
        glyph = np.asarray(obs["glyphs"])
        return glyph.reshape(1, *glyph.shape)

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
