"""Linear-Q feature extractor (proposal §6, MDP-Q variant).

The state s = bot's observation dict. The feature vector φ(s) ∈ ℝ^18:

  φ[0..7]  : closeness of nearest unvisited-biome cell per compass sector
             = 1 / (1 + d_min / R)  ∈ (0, 1]
             where d_min = min cell-distance to a novel cell in that sector
                   R     = grid radius (e.g. 128)
  φ[8..15] : normalized count of novel cells per sector
             = (# novel cells in sector) / (size² / 8)  ∈ [0, 1]
  φ[16]    : visited_progress = len(visited) / NUM_BIOMES  ∈ [0, 1]
             tells Q how far along exploration is — useful for choosing
             between aggressive long-range hops (early) vs cautious
             near-novel hops (late, when most signals are dim)
  φ[17]    : bias term, always 1.0

A "novel" cell has a known biome (grid value ≠ -1) whose id is not in
obs["visitedBiomes"]. We deliberately drop per-biome identity (treating
all unvisited biomes as equivalent) so Q can learn direction + density
+ progress preferences with a small parameter count.

φ-dim = 18. Per-action weights = 8 × 18 = 144 params total.
"""

import math
import numpy as np

from mdp.env import NUM_ACTIONS

NUM_BIOMES = 64                          # mineflayer-data 1.20.1 has 64 biomes
PHI_DIM = 2 * NUM_ACTIONS + 2            # 8 closeness + 8 count + progress + bias = 18


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

    visited_progress = min(1.0, len(visited) / NUM_BIOMES)

    phi = np.empty(PHI_DIM, dtype=np.float32)
    phi[:NUM_ACTIONS] = closeness
    phi[NUM_ACTIONS:2 * NUM_ACTIONS] = count_norm
    phi[-2] = visited_progress
    phi[-1] = 1.0  # bias
    return phi
