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
    """Yamauchi (1997) frontier exploration, adapted.

    1. Build a "novel" mask: cells whose biome is known (≠ -1) and not
       in the visited set.
    2. 4-connected flood-fill to group novel cells into clusters.
    3. Drop clusters smaller than MIN_CLUSTER (Yamauchi-style noise
       rejection).
    4. Pick the cluster whose centroid is closest to the bot.
    5. Compass-snap the bot→centroid direction onto one of the 8
       compass actions.

    Differences from the paper:
      - 'Frontier' = biome novelty (not occupancy boundary). Under
        complete knowledge there's no unknown space; biome ∉ visited
        is the closest analog.
      - Euclidean centroid distance instead of A* path distance —
        the overworld is mostly traversable with 100-block hops.
      - No reachability prune; relies on pathfinder to fail-stuck.
    """

    MIN_CLUSTER = 3  # cells; drop tiny novel-cell specks (~12-block patches)

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> int:
        grid = obs["grid"]
        r = obs["gridRadius"]
        size = 2 * r + 1
        visited = set(obs["visitedBiomes"])
        n = size * size

        # Step 1: novel mask.
        novel = [False] * n
        for k in range(n):
            b = grid[k]
            if b != -1 and b not in visited:
                novel[k] = True

        # Step 2 + 3: flood-fill clusters; record (size, Σdx, Σdz) per cluster.
        label = [-1] * n
        clusters = []
        for start in range(n):
            if not novel[start] or label[start] != -1:
                continue
            cid = len(clusters)
            label[start] = cid
            stack = [start]
            cnt = sdx = sdz = 0
            while stack:
                k = stack.pop()
                row, col = divmod(k, size)
                cnt += 1
                sdx += col - r
                sdz += row - r
                if col > 0:
                    nk = k - 1
                    if novel[nk] and label[nk] == -1:
                        label[nk] = cid; stack.append(nk)
                if col < size - 1:
                    nk = k + 1
                    if novel[nk] and label[nk] == -1:
                        label[nk] = cid; stack.append(nk)
                if row > 0:
                    nk = k - size
                    if novel[nk] and label[nk] == -1:
                        label[nk] = cid; stack.append(nk)
                if row < size - 1:
                    nk = k + size
                    if novel[nk] and label[nk] == -1:
                        label[nk] = cid; stack.append(nk)
            clusters.append((cnt, sdx, sdz))

        # Filter and choose nearest centroid.
        candidates = [(c, sx, sz) for (c, sx, sz) in clusters if c >= self.MIN_CLUSTER]
        if not candidates:
            return self.rng.randrange(NUM_ACTIONS)
        cnt, sx, sz = min(candidates,
                          key=lambda c: (c[1] / c[0]) ** 2 + (c[2] / c[0]) ** 2)
        cdx, cdz = sx / cnt, sz / cnt

        # Compass-snap.
        angle = math.atan2(cdx, -cdz)
        if angle < 0:
            angle += 2 * math.pi
        return int(round(angle / (2 * math.pi / NUM_ACTIONS))) % NUM_ACTIONS