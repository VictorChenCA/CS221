"""Linear-Q feature extractor (proposal §6, MDP-Q variant).

The state s = bot's observation dict. The feature vector φ(s) ∈ ℝ^17:

  φ[0..7]  : closeness of nearest unvisited-biome cell per compass sector
             = 1 / (1 + d_min / R)  ∈ (0, 1]
             where d_min = min cell-distance to a novel cell in that sector
                   R     = grid radius (e.g. 128)
  φ[8..15] : normalized count of novel cells per sector
             = (# novel cells in sector) / (size² / 8)  ∈ [0, 1]
  φ[16]    : bias term, always 1.0

A "novel" cell has a known biome (grid value ≠ -1) whose id is not in
obs["visitedBiomes"]. We deliberately drop biome identity (we treat all
unvisited biomes as equivalent) so Q can learn direction + density
preferences with a tiny 17-parameter state representation. See proposal
§6 (linear FA) and the milestone discussion.

φ-dim = 17. Per-action weights = 8 × 17 = 136 params total.
"""

import math
import numpy as np

from mdp.env import NUM_ACTIONS

PHI_DIM = 2 * NUM_ACTIONS + 1  # 8 closeness + 8 count + 1 bias = 17


def featurize(obs: dict) -> np.ndarray:
    """Return the 17-dim feature vector for `obs` (see module docstring)."""
    grid = obs["grid"]
    r = obs["gridRadius"]
    size = 2 * r + 1
    visited = set(obs.get("visitedBiomes", []))
    R = float(r)

    sector_min_d = np.full(NUM_ACTIONS, np.inf, dtype=np.float32)
    sector_count = np.zeros(NUM_ACTIONS, dtype=np.float32)
    step = 2 * math.pi / NUM_ACTIONS

    for row in range(size):
        for col in range(size):
            dx = col - r
            dz = row - r
            if dx == 0 and dz == 0:
                continue
            b = grid[row * size + col]
            if b < 0 or b in visited:
                continue
            angle = math.atan2(dx, -dz) % (2 * math.pi)
            sector = int(round(angle / step)) % NUM_ACTIONS
            d = math.hypot(dx, dz)
            if d < sector_min_d[sector]:
                sector_min_d[sector] = d
            sector_count[sector] += 1

    # closeness = 1/(1 + d/R) if a novel cell exists in the sector, else 0.
    # The inf sentinel turns into 0 naturally: 1/(1+inf) → 0.
    closeness = 1.0 / (1.0 + sector_min_d / R)                # (0, 1] or 0
    sector_area = (size * size) / NUM_ACTIONS
    count_norm = np.clip(sector_count / sector_area, 0.0, 1.0)  # [0, 1]

    phi = np.empty(PHI_DIM, dtype=np.float32)
    phi[:NUM_ACTIONS] = closeness
    phi[NUM_ACTIONS:2 * NUM_ACTIONS] = count_norm
    phi[-1] = 1.0  # bias
    return phi
