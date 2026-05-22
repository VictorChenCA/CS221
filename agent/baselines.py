"""Non-learning baselines.

Both policies share a minimal interface: given the latest observation
dict from `Env`, return an integer action in [0, NUM_ACTIONS). State
that needs to persist across steps (RNG, frontier map) lives on the
policy instance, so `eval.py` can just call `policy.act(obs)` in a
loop.
"""

import random
from typing import Protocol

from agent.env import NUM_ACTIONS


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
    """Greedy frontier exploration — proposal §6 non-trivial baseline.

    Yamauchi (1997). At each step, pick the compass direction whose
    sector contains the largest unexplored area (cells with unknown
    biome id within some radius of the agent). Falls back to random
    when no frontier is visible.

    NOTE: stub — implementation owned by Victor. Needs:
      - a persistent known-world map `(cellX, cellZ) -> biomeId`,
        updated from each obs (proposal §2: 4x4 biome cells)
      - sector binning of unknown cells into the 8 compass actions
      - tie-breaking + fallback when all sectors are zero
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.known: dict[tuple[int, int], int] = {}

    def reset(self) -> None:
        self.known.clear()

    def act(self, obs: dict) -> int:
        raise NotImplementedError("FrontierPolicy.act — TODO (Victor)")
