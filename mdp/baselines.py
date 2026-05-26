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


class FrontierSectorVote:
    """Original 'largest-sector' frontier — the version used before the
    May 23 Yamauchi switch.

    For each of 8 compass directions, count cells inside that wedge
    whose biome ∉ visited. Pick the direction with the highest count.
    Random tiebreak. Falls back to random if no novel cells visible.

    Why this can beat the closest-centroid version: voting across many
    cells per direction is robust to a single unreachable novel cell
    fixating the agent. There's no 'target a single point' fixation
    mode — the agent always moves toward where the most undiscovered
    biome is, in aggregate.
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
        scores = [0] * NUM_ACTIONS

        for row in range(size):
            for col in range(size):
                dx = col - r
                dz = row - r
                if dx == 0 and dz == 0:
                    continue
                b = grid[row * size + col]
                if b == -1 or b in visited:
                    continue
                angle = math.atan2(dx, -dz)
                if angle < 0:
                    angle += 2 * math.pi
                sector = int(round(angle / (2 * math.pi / NUM_ACTIONS))) % NUM_ACTIONS
                scores[sector] += 1

        best = max(scores)
        if best == 0:
            return self.rng.randrange(NUM_ACTIONS)
        return self.rng.choice([a for a, s in enumerate(scores) if s == best])


class FrontierClosestCell:
    """First Yamauchi-style rewrite (commit 28e1614, before clustering).

    Find the single closest cell whose biome ∉ visited, snap its
    direction onto a compass action. No noise rejection, no clustering.

    Failure mode: a tiny isolated patch of novel biome two cells away
    will pull the agent toward it forever even if a richer region is
    farther in a different direction."""

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
                b = grid[row * size + col]
                if b == -1 or b in visited:
                    continue
                d2 = dx * dx + dz * dz
                if d2 < best_d2:
                    best_d2, best_dx, best_dz = d2, dx, dz

        if best_d2 == float("inf"):
            return self.rng.randrange(NUM_ACTIONS)
        angle = math.atan2(best_dx, -best_dz)
        if angle < 0:
            angle += 2 * math.pi
        return int(round(angle / (2 * math.pi / NUM_ACTIONS))) % NUM_ACTIONS


class FrontierClusterCentroid:
    """Current 'cluster + nearest-centroid' Yamauchi variant.

    1. Build a "novel" mask: cells whose biome is known (≠ -1) and not
       in the visited set.
    2. 4-connected flood-fill to group novel cells into clusters.
    3. Drop clusters smaller than MIN_CLUSTER (Yamauchi-style noise
       rejection).
    4. Pick the cluster whose centroid is closest to the bot.
    5. Compass-snap the bot→centroid direction onto one of the 8
       compass actions.
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

        novel = [False] * n
        for k in range(n):
            b = grid[k]
            if b != -1 and b not in visited:
                novel[k] = True

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

        candidates = [(c, sx, sz) for (c, sx, sz) in clusters if c >= self.MIN_CLUSTER]
        if not candidates:
            return self.rng.randrange(NUM_ACTIONS)
        cnt, sx, sz = min(candidates,
                          key=lambda c: (c[1] / c[0]) ** 2 + (c[2] / c[0]) ** 2)
        cdx, cdz = sx / cnt, sz / cnt

        angle = math.atan2(cdx, -cdz)
        if angle < 0:
            angle += 2 * math.pi
        return int(round(angle / (2 * math.pi / NUM_ACTIONS))) % NUM_ACTIONS


# Backwards compat: 'frontier' continues to mean the current cluster-centroid
# version so existing CLI / test scripts keep working.
FrontierPolicy = FrontierClusterCentroid