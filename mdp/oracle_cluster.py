"""Cluster-then-target oracle variant.

Instead of greedily picking the single nearest unvisited biome, this
oracle:
  1. Spatially clusters biome-region centroids using single-linkage
     agglomeration (no external deps — pure stdlib).
  2. Scores each cluster by (# biomes in cluster) / (distance to
     cluster centroid from current position) — richness per travel cost.
  3. Greedily picks the highest-scoring cluster, then visits its member
     biomes in nearest-neighbor order before re-scoring remaining clusters.

Same public interface as oracle.py: plan(seed, start_cell, radius_cells,
time_budget_s, biome_at, visited) -> Plan.
"""

import math
import os
from typing import Callable

from mdp.world import CELL_BLOCKS, NpzWorldView
from mdp.oracle import Hop, Plan, _block_dist, _snap_to_compass, DEFAULT_INTERIOR_RADIUS

# Maximum block distance between two biome centroids to merge into one cluster.
CLUSTER_RADIUS_BLOCKS = int(os.environ.get("CLUSTER_RADIUS", "200"))
# Minimum block distance for a hop — skip targets closer than this.
MIN_HOP_BLOCKS = int(os.environ.get("MIN_HOP_BLOCKS", "32"))

INTERIOR_RADIUS = DEFAULT_INTERIOR_RADIUS
WALK_SPEED_BPS = 4.3
PATH_EFFICIENCY_PENALTY = 1.5

BiomeFn = Callable[[int, int], int]


def plan(
    seed: int,
    start_cell: tuple[int, int],
    radius_cells: int,
    time_budget_s: float,
    biome_at: BiomeFn | None = None,
    visited: set[int] | None = None,
) -> Plan:
    if biome_at is None:
        biome_at = NpzWorldView(seed).biome_at

    sx, sz = start_cell

    # ------------------------------------------------------------------
    # 1. Scan biome window
    # ------------------------------------------------------------------
    biome_cells: dict[int, list[tuple[int, int]]] = {}
    cell_biome: dict[tuple[int, int], int] = {}

    for dz in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            cx, cz = sx + dx, sz + dz
            b = biome_at(cx, cz)
            if b < 0:
                continue
            biome_cells.setdefault(b, []).append((cx, cz))
            cell_biome[(cx, cz)] = b

    start_biome = biome_at(sx, sz)
    if start_biome in biome_cells:
        del biome_cells[start_biome]
    if visited:
        for v in list(visited):
            biome_cells.pop(v, None)

    if not biome_cells:
        return Plan(hops=[], expected_biomes=[start_biome] if start_biome >= 0 else [])

    # ------------------------------------------------------------------
    # 2. Interior-cell filtering
    # ------------------------------------------------------------------
    def is_interior(cx: int, cz: int, biome_id: int) -> bool:
        for dz2 in range(-INTERIOR_RADIUS, INTERIOR_RADIUS + 1):
            for dx2 in range(-INTERIOR_RADIUS, INTERIOR_RADIUS + 1):
                if cell_biome.get((cx + dx2, cz + dz2)) != biome_id:
                    return False
        return True

    biome_interior: dict[int, list[tuple[int, int]]] = {}
    for bid, cells in biome_cells.items():
        interior = [c for c in cells if is_interior(c[0], c[1], bid)]
        biome_interior[bid] = interior if interior else cells

    # ------------------------------------------------------------------
    # 3. Per-biome centroid
    # ------------------------------------------------------------------
    def biome_centroid(bid: int) -> tuple[float, float]:
        cells = biome_interior[bid]
        return (
            sum(c[0] for c in cells) / len(cells),
            sum(c[1] for c in cells) / len(cells),
        )

    centroids: dict[int, tuple[float, float]] = {
        bid: biome_centroid(bid) for bid in biome_cells
    }

    # ------------------------------------------------------------------
    # 4. Single-linkage agglomeration
    # ------------------------------------------------------------------
    biome_ids = list(biome_cells.keys())
    cluster_label = {bid: i for i, bid in enumerate(biome_ids)}

    def centroid_block_dist(a: int, b: int) -> float:
        ca, cb = centroids[a], centroids[b]
        return math.hypot(
            (ca[0] - cb[0]) * CELL_BLOCKS,
            (ca[1] - cb[1]) * CELL_BLOCKS,
        )

    changed = True
    while changed:
        changed = False
        members: dict[int, list[int]] = {}
        for bid, lbl in cluster_label.items():
            members.setdefault(lbl, []).append(bid)
        ids = list(members.keys())
        for i, lbl_a in enumerate(ids):
            if changed:
                break
            for lbl_b in ids[i + 1:]:
                if lbl_a == lbl_b:
                    continue
                for ba in members[lbl_a]:
                    for bb in members[lbl_b]:
                        if centroid_block_dist(ba, bb) <= CLUSTER_RADIUS_BLOCKS:
                            for bid in members[lbl_b]:
                                cluster_label[bid] = lbl_a
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break

    cluster_members: dict[int, list[int]] = {}
    for bid, lbl in cluster_label.items():
        cluster_members.setdefault(lbl, []).append(bid)

    clusters: list[list[int]] = list(cluster_members.values())

    # ------------------------------------------------------------------
    # 5. Greedy cluster selection + intra-cluster nearest-neighbor
    # ------------------------------------------------------------------
    hops: list[Hop] = []
    visited_biomes: list[int] = []
    if start_biome >= 0:
        visited_biomes.append(start_biome)

    remaining_clusters = [list(c) for c in clusters]
    cur = (sx, sz)
    time_used = 0.0

    def cluster_centroid_pos(members: list[int]) -> tuple[float, float]:
        xs = [centroids[b][0] for b in members]
        zs = [centroids[b][1] for b in members]
        return sum(xs) / len(xs), sum(zs) / len(zs)

    def dist_to_cluster(pos: tuple[int, int], members: list[int]) -> float:
        ccx, ccz = cluster_centroid_pos(members)
        return math.hypot(
            (pos[0] - ccx) * CELL_BLOCKS,
            (pos[1] - ccz) * CELL_BLOCKS,
        )

    while remaining_clusters:
        def score(members: list[int]) -> float:
            d = dist_to_cluster(cur, members)
            return len(members) / (d / CELL_BLOCKS + 1.0)

        best_cluster = max(remaining_clusters, key=score)
        remaining_clusters.remove(best_cluster)

        unvisited = list(best_cluster)
        while unvisited:
            best_biome = None
            best_cell = None
            best_dist = float("inf")
            for bid in unvisited:
                candidates = biome_interior.get(bid, biome_cells.get(bid, []))
                if not candidates:
                    continue
                nearest = min(candidates, key=lambda c: _block_dist(cur, c))
                d = _block_dist(cur, nearest)
                if d < best_dist:
                    best_dist, best_biome, best_cell = d, bid, nearest

            if best_cell is None:
                break

            # Skip targets that are too close — pathfinder can't make
            # progress on very short hops and will loop indefinitely.
            if best_dist < MIN_HOP_BLOCKS:
                unvisited.remove(best_biome)
                # Still count it as visited since we're essentially on it.
                visited_biomes.append(best_biome)
                biome_cells.pop(best_biome, None)
                continue

            travel_s = PATH_EFFICIENCY_PENALTY * best_dist / WALK_SPEED_BPS
            if time_used + travel_s > time_budget_s:
                remaining_clusters.clear()
                unvisited.clear()
                break

            dx_total = (best_cell[0] - cur[0]) * CELL_BLOCKS
            dz_total = (best_cell[1] - cur[1]) * CELL_BLOCKS
            if math.hypot(dx_total, dz_total) > 1:
                hops.append(_snap_to_compass(dx_total, dz_total))

            visited_biomes.append(best_biome)
            cur = best_cell
            time_used += travel_s
            unvisited.remove(best_biome)
            biome_cells.pop(best_biome, None)

    return Plan(hops=hops, expected_biomes=visited_biomes)