"""Non-learning baselines.

Both policies share a minimal interface: given the latest observation
dict from `Env`, return an integer action in [0, NUM_ACTIONS). State
that needs to persist across steps (RNG, frontier map) lives on the
policy instance, so `eval.py` can just call `policy.act(obs)` in a
loop.

All policies expose `stats: dict[str, int|float]` accumulated across
the episode for post-hoc analysis. Counters are written to the result
JSON so we can answer questions like 'how often does frontier fall
back to random' or 'how big is the average target cluster.'
"""

import math
import random
from collections import Counter
from typing import Protocol

from mdp.env import NUM_ACTIONS


class Policy(Protocol):
    stats: Counter
    def act(self, obs: dict) -> int: ...
    def reset(self) -> None: ...


class RandomPolicy:
    """Uniform random walk — proposal §6 lower-bound baseline."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.stats: Counter = Counter()

    def reset(self) -> None:
        self.stats = Counter()

    def act(self, obs: dict) -> int:
        self.stats["actions"] += 1
        return self.rng.randrange(NUM_ACTIONS)


class FrontierSectorVote:
    """Original 'largest-sector' frontier — the version used before the
    May 23 Yamauchi switch.

    For each of 8 compass directions, count cells inside that wedge
    whose biome ∉ visited. Pick the direction with the highest count.
    Random tiebreak. Falls back to random if no novel cells visible.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.stats: Counter = Counter()

    def reset(self) -> None:
        self.stats = Counter()

    def act(self, obs: dict) -> int:
        self.stats["actions"] += 1
        grid = obs["grid"]
        r = obs["gridRadius"]
        size = 2 * r + 1
        visited = set(obs["visitedBiomes"])
        scores = [0] * NUM_ACTIONS
        n_novel = 0

        for row in range(size):
            for col in range(size):
                dx = col - r
                dz = row - r
                if dx == 0 and dz == 0:
                    continue
                b = grid[row * size + col]
                if b == -1 or b in visited:
                    continue
                n_novel += 1
                angle = math.atan2(dx, -dz)
                if angle < 0:
                    angle += 2 * math.pi
                sector = int(round(angle / (2 * math.pi / NUM_ACTIONS))) % NUM_ACTIONS
                scores[sector] += 1

        self.stats["novel_cells_seen_sum"] += n_novel
        best = max(scores)
        if best == 0:
            self.stats["random_fallback"] += 1
            return self.rng.randrange(NUM_ACTIONS)
        self.stats["directed"] += 1
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
        self.stats: Counter = Counter()

    def reset(self) -> None:
        self.stats = Counter()

    def act(self, obs: dict) -> int:
        self.stats["actions"] += 1
        grid = obs["grid"]
        r = obs["gridRadius"]
        size = 2 * r + 1
        visited = set(obs["visitedBiomes"])

        best_d2 = float("inf")
        best_dx = best_dz = 0
        n_novel = 0
        for row in range(size):
            for col in range(size):
                dx = col - r
                dz = row - r
                if dx == 0 and dz == 0:
                    continue
                b = grid[row * size + col]
                if b == -1 or b in visited:
                    continue
                n_novel += 1
                d2 = dx * dx + dz * dz
                if d2 < best_d2:
                    best_d2, best_dx, best_dz = d2, dx, dz

        self.stats["novel_cells_seen_sum"] += n_novel
        if best_d2 == float("inf"):
            self.stats["random_fallback"] += 1
            return self.rng.randrange(NUM_ACTIONS)
        self.stats["directed"] += 1
        self.stats["target_dist_cells_sum"] += math.sqrt(best_d2)
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
        self.stats: Counter = Counter()

    def reset(self) -> None:
        self.stats = Counter()

    def act(self, obs: dict) -> int:
        self.stats["actions"] += 1
        grid = obs["grid"]
        r = obs["gridRadius"]
        size = 2 * r + 1
        visited = set(obs["visitedBiomes"])
        n = size * size

        novel = [False] * n
        n_novel = 0
        for k in range(n):
            b = grid[k]
            if b != -1 and b not in visited:
                novel[k] = True
                n_novel += 1
        self.stats["novel_cells_seen_sum"] += n_novel

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
        self.stats["clusters_seen_sum"] += len(clusters)

        candidates = [(c, sx, sz) for (c, sx, sz) in clusters if c >= self.MIN_CLUSTER]
        self.stats["candidate_clusters_sum"] += len(candidates)
        if not candidates:
            self.stats["random_fallback"] += 1
            return self.rng.randrange(NUM_ACTIONS)
        cnt, sx, sz = min(candidates,
                          key=lambda c: (c[1] / c[0]) ** 2 + (c[2] / c[0]) ** 2)
        cdx, cdz = sx / cnt, sz / cnt
        self.stats["directed"] += 1
        self.stats["target_dist_cells_sum"] += math.hypot(cdx, cdz)
        self.stats["target_cluster_size_sum"] += cnt

        angle = math.atan2(cdx, -cdz)
        if angle < 0:
            angle += 2 * math.pi
        return int(round(angle / (2 * math.pi / NUM_ACTIONS))) % NUM_ACTIONS


class FrontierUnvisitedCells:
    """'True Yamauchi' variant: explore physically-unvisited cells, not
    unfamiliar biome ids.

    In complete-knowledge mode the biome of every cell is known offline,
    so the biome-novelty signal saturates fast — after the bot has
    physically entered one cell of each common biome, all other cells
    of those biomes are 'familiar' and the policy collapses to random.
    There may still be vast unexplored space, but biome-novelty can't
    see it.

    This variant tracks the (cellX, cellZ) coordinates the bot has
    actually visited and treats any other cell as a frontier. It uses
    the same cluster + nearest-centroid scoring as FrontierClusterCentroid
    but on the physical-visit mask instead of the biome-novelty mask.
    Now 'go where I haven't been' beats 'go where there's a new biome'
    once biomes saturate."""

    MIN_CLUSTER = 3

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.visited_cells: set[tuple[int, int]] = set()
        self.stats: Counter = Counter()

    def reset(self) -> None:
        self.visited_cells.clear()
        self.stats = Counter()

    def act(self, obs: dict) -> int:
        self.stats["actions"] += 1
        cellX = obs.get("cellX")
        cellZ = obs.get("cellZ")
        if cellX is not None and cellZ is not None:
            self.visited_cells.add((cellX, cellZ))

        grid = obs["grid"]
        r = obs["gridRadius"]
        size = 2 * r + 1
        n = size * size

        # Novel = known biome cell (b≠-1) AND not physically visited.
        novel = [False] * n
        n_novel = 0
        for k in range(n):
            b = grid[k]
            if b == -1:
                continue
            row, col = divmod(k, size)
            world_cx = cellX + (col - r) if cellX is not None else None
            world_cz = cellZ + (row - r) if cellZ is not None else None
            if world_cx is None or (world_cx, world_cz) not in self.visited_cells:
                novel[k] = True
                n_novel += 1
        self.stats["novel_cells_seen_sum"] += n_novel

        # Cluster the novel cells.
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
                for nk in (k - 1 if col > 0 else -1,
                           k + 1 if col < size - 1 else -1,
                           k - size if row > 0 else -1,
                           k + size if row < size - 1 else -1):
                    if nk >= 0 and novel[nk] and label[nk] == -1:
                        label[nk] = cid
                        stack.append(nk)
            clusters.append((cnt, sdx, sdz))
        self.stats["clusters_seen_sum"] += len(clusters)

        candidates = [(c, sx, sz) for (c, sx, sz) in clusters if c >= self.MIN_CLUSTER]
        self.stats["candidate_clusters_sum"] += len(candidates)
        if not candidates:
            self.stats["random_fallback"] += 1
            return self.rng.randrange(NUM_ACTIONS)
        cnt, sx, sz = min(candidates,
                          key=lambda c: (c[1] / c[0]) ** 2 + (c[2] / c[0]) ** 2)
        cdx, cdz = sx / cnt, sz / cnt
        self.stats["directed"] += 1
        self.stats["target_dist_cells_sum"] += math.hypot(cdx, cdz)
        self.stats["target_cluster_size_sum"] += cnt

        angle = math.atan2(cdx, -cdz)
        if angle < 0:
            angle += 2 * math.pi
        return int(round(angle / (2 * math.pi / NUM_ACTIONS))) % NUM_ACTIONS


# Backwards compat: 'frontier' continues to mean the current cluster-centroid
# version so existing CLI / test scripts keep working.
FrontierPolicy = FrontierClusterCentroid