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
    """Greedy frontier exploration baseline.

    Scores each compass direction by counting nearby cells whose biome
    has not yet been visited. Chooses the highest-scoring direction.
    Falls back to random if all scores are zero.
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

        # One score per compass direction.
        scores = [0 for _ in range(NUM_ACTIONS)]

        for row in range(size):
            for col in range(size):
                dx = col - r
                dz = row - r

                # Skip center cell.
                if dx == 0 and dz == 0:
                    continue

                biome_id = grid[row * size + col]

                # Ignore invisible/invalid cells.
                if biome_id == -1:
                    continue

                # Only reward unseen biomes.
                if biome_id in visited:
                    continue

                # Convert cell direction into compass sector.
                angle = math.atan2(dx, -dz)
                if angle < 0:
                    angle += 2 * math.pi

                sector = int(round(angle / (2 * math.pi / NUM_ACTIONS))) % NUM_ACTIONS
                scores[sector] += 1

        best_score = max(scores)

        # No frontier found -> random fallback.
        if best_score == 0:
            return self.rng.randrange(NUM_ACTIONS)

        best_actions = [
            action
            for action, score in enumerate(scores)
            if score == best_score
        ]

        return self.rng.choice(best_actions)