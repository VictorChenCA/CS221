"""Non-learning baselines.

Both policies share a minimal interface: given the latest observation
dict from `Env`, return an integer action in [0, NUM_ACTIONS). State
that needs to persist across steps (RNG, frontier map) lives on the
policy instance, so `eval.py` can just call `policy.act(obs)` in a
loop.
"""

import math
import random
from typing import Protocol

from mdp.env import NUM_ACTIONS


class Policy(Protocol):
    def act(self, obs: dict) -> int: ...
    def reset(self) -> None: ...


class RandomPolicy:
    """Uniform random walk — proposal §6 lower-bound baseline."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> int:
        return self.rng.randrange(NUM_ACTIONS)


class FrontierPolicy:
    """Closest-unfamiliar variant of Yamauchi (1997).

    Scans the local biome grid for the cell whose biome has not yet
    been visited and is geometrically closest to the bot, then snaps
    the bot-to-cell direction onto one of the 8 compass actions.

    Notes on Yamauchi correspondence:
      - 'Frontier' here = a cell whose biome ∉ visited. Yamauchi's
        occupancy-based notion doesn't directly apply (in complete-
        knowledge mode every cell is known), so we substitute biome
        novelty for known-but-unentered cells.
      - We use Euclidean distance, not A* over walkable cells. For
        vanilla flat-ish overworld with 100-block hops this is fine.
      - We don't cluster cells into regions or compute centroids; the
        single closest unfamiliar cell becomes the target.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> int:
        grid = obs["grid"]
        r = obs["gridRadius"]
        size = 2 * r + 1
        visited = set(obs["visitedBiomes"])

        best_d2 = float("inf")
        best_dx = best_dz = 0
        for row in range(size):
            for col in range(size):
                dx = col - r
                dz = row - r
                if dx == 0 and dz == 0:
                    continue
                biome_id = grid[row * size + col]
                if biome_id == -1 or biome_id in visited:
                    continue
                d2 = dx * dx + dz * dz
                if d2 < best_d2:
                    best_d2, best_dx, best_dz = d2, dx, dz

        # No unfamiliar cell visible -> random fallback.
        if best_d2 == float("inf"):
            return self.rng.randrange(NUM_ACTIONS)

        angle = math.atan2(best_dx, -best_dz)
        if angle < 0:
            angle += 2 * math.pi
        return int(round(angle / (2 * math.pi / NUM_ACTIONS))) % NUM_ACTIONS