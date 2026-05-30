"""K-step lookahead oracle variant.

Instead of purely greedy nearest-biome, evaluates all permutations of
the next LOOKAHEAD_DEPTH biome choices and picks the ordering with the
best total score:

    score = biomes_collected - DISTANCE_WEIGHT * total_travel_blocks

Configurable via env vars:
    LOOKAHEAD_DEPTH   (default 2)
    DISTANCE_WEIGHT   (default 0.001)
    MIN_HOP_BLOCKS    (default 32, skip targets closer than this)
"""

import itertools
import math
import os
from typing import Callable

from mdp.world import CELL_BLOCKS, NpzWorldView
from mdp.oracle import Hop, Plan, _block_dist, _snap_to_compass, DEFAULT_INTERIOR_RADIUS

LOOKAHEAD_DEPTH = int(os.environ.get("LOOKAHEAD_DEPTH", "2"))
DISTANCE_WEIGHT = float(os.environ.get("DISTANCE_WEIGHT", "0.001"))
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
    # 1. Scan + interior filtering
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

    def best_cell_for(bid: int, from_pos: tuple) -> tuple[tuple[int, int], float]:
        candidates = biome_interior.get(bid, biome_cells.get(bid, []))
        if not candidates:
            return from_pos, float("inf")
        c = min(candidates, key=lambda x: _block_dist(from_pos, x))
        return c, _block_dist(from_pos, c)

    # ------------------------------------------------------------------
    # 2. Lookahead planning loop
    # ------------------------------------------------------------------
    hops: list[Hop] = []
    visited_biomes: list[int] = []
    if start_biome >= 0:
        visited_biomes.append(start_biome)

    remaining = set(biome_cells.keys())
    cur = (sx, sz)
    time_used = 0.0

    while remaining:
        # Filter out any candidates that are too close to move to.
        reachable = [
            bid for bid in remaining
            if best_cell_for(bid, cur)[1] >= MIN_HOP_BLOCKS
        ]

        # If nothing is reachable by hop, count nearby biomes as visited
        # and remove them so we don't loop forever.
        if not reachable:
            for bid in list(remaining):
                cell, dist = best_cell_for(bid, cur)
                if dist < MIN_HOP_BLOCKS:
                    visited_biomes.append(bid)
                    remaining.discard(bid)
            continue

        candidates = reachable
        depth = min(LOOKAHEAD_DEPTH, len(candidates))

        best_sequence: list[int] | None = None
        best_score = float("-inf")

        for perm in itertools.permutations(candidates, depth):
            pos = cur
            t = time_used
            total_dist = 0.0
            feasible_count = 0
            valid = True

            for bid in perm:
                cell, dist = best_cell_for(bid, pos)
                if dist < MIN_HOP_BLOCKS:
                    # Too close — skip in simulation but count it.
                    feasible_count += 1
                    continue
                travel_s = PATH_EFFICIENCY_PENALTY * dist / WALK_SPEED_BPS
                if t + travel_s > time_budget_s:
                    valid = False
                    break
                total_dist += dist
                t += travel_s
                pos = cell
                feasible_count += 1

            if feasible_count == 0:
                continue

            score = feasible_count - DISTANCE_WEIGHT * total_dist
            if score > best_score:
                best_score = score
                best_sequence = list(perm[:feasible_count]) if not valid else list(perm)

        if not best_sequence:
            break

        next_biome = best_sequence[0]
        best_c, dist = best_cell_for(next_biome, cur)

        if dist < MIN_HOP_BLOCKS:
            # Close enough — count as visited without a hop.
            visited_biomes.append(next_biome)
            remaining.discard(next_biome)
            continue

        travel_s = PATH_EFFICIENCY_PENALTY * dist / WALK_SPEED_BPS
        if time_used + travel_s > time_budget_s:
            break

        dx_total = (best_c[0] - cur[0]) * CELL_BLOCKS
        dz_total = (best_c[1] - cur[1]) * CELL_BLOCKS
        if math.hypot(dx_total, dz_total) > 1:
            hops.append(_snap_to_compass(dx_total, dz_total))

        visited_biomes.append(next_biome)
        cur = best_c
        time_used += travel_s
        remaining.discard(next_biome)

    return Plan(hops=hops, expected_biomes=visited_biomes)