"""Linear-Q feature extractor (proposal §6, MDP-Q variant).

The state s = bot's observation dict. The feature vector φ(s) ∈ ℝ^19:

  φ[0..7]  : closeness of nearest unvisited-biome cell per compass sector
             = 1 / (1 + d_min / R)  ∈ (0, 1]
             where d_min = min cell-distance to a novel cell in that sector
                   R     = grid radius (e.g. 128)
  φ[8..15] : normalized count of novel cells per sector
             = (# novel cells in sector) / (size² / 8)  ∈ [0, 1]
  φ[16]    : visited_progress = len(visited) / NUM_BIOMES  ∈ [0, 1]
  φ[17]    : was_stuck = 1.0 if the previous action returned stuck=true
             else 0.0. A global "I'm currently wedged" signal.
  φ[18..25]: stuck_dirs — a per-direction decaying memory of which compass
             directions recently returned stuck. stuck_dirs[k] is set to
             1.0 the step direction k fails and decays by STUCK_TRACE_DECAY
             each subsequent step. This is the key to *learned* escape:
             because Q(s,a)=w_a·φ(s) gives each action its own weight
             vector, action k can learn w_k[stuck_dir_k] < 0 — i.e. lower
             its own Q when it just failed — so the greedy policy routes
             around blocked directions instead of looping. Without this,
             φ is identical step-to-step while wedged (the bot doesn't
             move, so the biome-grid features don't change) and the argmax
             never changes. The caller maintains the trace across steps
             and passes it in as obs["stuck_dirs"] (train.py / eval.py).
  φ[26]    : bias term, always 1.0

A "novel" cell has a known biome (grid value ≠ -1) whose id is not in
obs["visitedBiomes"]. We deliberately drop per-biome identity (treating
all unvisited biomes as equivalent) so Q can learn direction + density
+ progress preferences with a small parameter count.

φ-dim = 27. Per-action weights = 8 × 27 = 216 params total.
"""

import math
import os
import numpy as np

from mdp.env import NUM_ACTIONS

NUM_BIOMES = 64                          # mineflayer-data 1.20.1 has 64 biomes
# 8 closeness + 8 count + progress + was_stuck + 8 stuck_dirs + bias = 27
PHI_DIM = 3 * NUM_ACTIONS + 3

# Per-step multiplicative decay of the stuck-direction memory. 0.5 → a
# blocked direction is ~forgotten after 3 steps, so it can be retried once
# terrain elsewhere has been explored. Tunable via env for the train loop.
STUCK_TRACE_DECAY = float(os.environ.get("STUCK_TRACE_DECAY", "0.5"))


def init_stuck_trace() -> list[float]:
    """Fresh per-episode stuck-direction memory (all directions open)."""
    return [0.0] * NUM_ACTIONS


def update_stuck_trace(trace: list[float], action: int | None,
                       was_stuck: bool,
                       decay: float = STUCK_TRACE_DECAY) -> list[float]:
    """Decay every direction, then mark `action` freshly-stuck if it just
    failed. Returns a new list (callers attach it to the next obs)."""
    new = [v * decay for v in trace]
    if was_stuck and action is not None and 0 <= action < NUM_ACTIONS:
        new[action] = 1.0
    return new


def novelty_potential(obs: dict) -> float:
    """Φ(s) ∈ [0,1] for potential-based reward shaping: closeness to the
    nearest novel-biome cell anywhere in the grid, = 1/(1 + d_min/R).
    1.0 = standing on novelty, → 0 = no novelty in view. Moving toward
    novelty raises Φ, so γΦ(s') − Φ(s) rewards directed exploration.

    Returns 0.0 if the grid is absent (e.g. a dead/NaN obs)."""
    grid = obs.get("grid")
    if grid is None:
        return 0.0
    r = obs["gridRadius"]
    size = 2 * r + 1
    visited = set(obs.get("visitedBiomes", []))
    R = float(r)
    d_min = math.inf
    for row in range(size):
        base = row * size
        dz = row - r
        for col in range(size):
            b = grid[base + col]
            if b < 0 or b in visited:
                continue
            dx = col - r
            if dx == 0 and dz == 0:
                continue
            d = math.hypot(dx, dz)
            if d < d_min:
                d_min = d
    return 1.0 / (1.0 + d_min / R) if d_min < math.inf else 0.0


def featurize(obs: dict) -> np.ndarray:
    """Return the 27-dim feature vector for `obs` (see module docstring)."""
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
    was_stuck = 1.0 if obs.get("was_stuck") else 0.0
    stuck_dirs = obs.get("stuck_dirs")
    if stuck_dirs is None:
        stuck_dirs = (0.0,) * NUM_ACTIONS

    phi = np.empty(PHI_DIM, dtype=np.float32)
    phi[:NUM_ACTIONS] = closeness                          # [0:8]
    phi[NUM_ACTIONS:2 * NUM_ACTIONS] = count_norm          # [8:16]
    phi[2 * NUM_ACTIONS] = visited_progress                # [16]
    phi[2 * NUM_ACTIONS + 1] = was_stuck                   # [17]
    phi[2 * NUM_ACTIONS + 2:3 * NUM_ACTIONS + 2] = stuck_dirs  # [18:26]
    phi[-1] = 1.0                                          # [26] bias
    return phi
