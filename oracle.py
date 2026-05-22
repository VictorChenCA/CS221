"""Offline max-coverage oracle (proposal §6 upper bound).

The oracle is *not* a Policy — it doesn't see observations one at a
time. Given a seed and a time budget, it loads the full biome map
inside a fixed search radius and plans a route that visits as many
distinct biomes as possible under the budget. `eval.py` then replays
the planned action sequence through the same bridge the learned
policies use.

Algorithm (greedy orienteering / set-cover):
  1. Scan biome_at over the (2R+1)x(2R+1) cell window around start.
  2. Group cells by biome id; for each biome (other than the starting
     one) keep the cell nearest to start — that's the candidate target.
  3. Greedy loop: from the current position, pick the unvisited biome
     whose target is cheapest to reach. Append the hop, update position,
     mark visited. Stop when no remaining hop fits in the budget.
  4. Snap each (dx, dz) vector to the bridge's 8-way compass action.

This is a 1-(1/e) approximation to weighted orienteering when the
"reward" is uniform set coverage, which is the case here. Tractable for
10-minute budgets at the radii we use.

Biome data comes from `NpzBiomeSource(seed)`, which reads a pre-extracted
biome dump from `data/biomes_<seed>.npz`. Generate that file once per
seed using a cubiomes binding (preferred — no server needed) or by
harvesting `mc-server/world/region/*.mca` with anvil-parser. See
`tools/extract_biomes.py` (TODO).
"""

import math
from dataclasses import dataclass
from typing import Callable

from agent.env import NUM_ACTIONS
from agent.world import CELL_BLOCKS, NpzWorldView

# Empirical pathfinder speed under sprint+jump. Re-measure once the
# bridge is profiled; conservative estimate for budgeting.
WALK_SPEED_BPS = 4.3

BiomeFn = Callable[[int, int], int]  # (cellX, cellZ) -> biome id (or -1 = unknown)


@dataclass
class Hop:
    theta_deg: float
    distance_blocks: int


@dataclass
class Plan:
    hops: list[Hop]
    expected_biomes: list[int]  # in visit order, including the starting biome


def plan(
    seed: int,
    start_cell: tuple[int, int],
    radius_cells: int,
    time_budget_s: float,
    biome_at: BiomeFn | None = None,
) -> Plan:
    """Greedy orienteering plan. See module docstring for the algorithm."""
    if biome_at is None:
        biome_at = NpzWorldView(seed).biome_at

    sx, sz = start_cell

    # Step 1: scan the window, group cells by biome.
    biome_cells: dict[int, list[tuple[int, int]]] = {}
    for dz in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            cx, cz = sx + dx, sz + dz
            b = biome_at(cx, cz)
            if b < 0:
                continue
            biome_cells.setdefault(b, []).append((cx, cz))

    start_biome = biome_at(sx, sz)

    # Step 2: collapse each biome to its closest representative cell.
    targets: dict[int, tuple[int, int]] = {}
    for b, cells in biome_cells.items():
        if b == start_biome:
            continue
        targets[b] = min(cells, key=lambda c: _block_dist(start_cell, c))

    # Step 3: greedy hop loop under the time budget.
    hops: list[Hop] = []
    visited: list[int] = [start_biome] if start_biome >= 0 else []
    cur = (sx, sz)
    time_used = 0.0

    while targets:
        b, tgt = min(targets.items(), key=lambda kv: _block_dist(cur, kv[1]))
        dist_blocks = _block_dist(cur, tgt)
        travel_s = dist_blocks / WALK_SPEED_BPS
        if time_used + travel_s > time_budget_s:
            break
        dx_b = (tgt[0] - cur[0]) * CELL_BLOCKS
        dz_b = (tgt[1] - cur[1]) * CELL_BLOCKS
        hops.append(_snap_to_compass(dx_b, dz_b))
        visited.append(b)
        cur = tgt
        time_used += travel_s
        del targets[b]

    return Plan(hops=hops, expected_biomes=visited)


def _block_dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot((a[0] - b[0]) * CELL_BLOCKS, (a[1] - b[1]) * CELL_BLOCKS)


def _snap_to_compass(dx: float, dz: float) -> Hop:
    """Convert a free (dx, dz) vector in blocks to the nearest 8-way hop.

    The bridge accepts (theta_deg, distance_blocks); theta is measured
    clockwise from +z (north), matching `agent/env.py::action_to_theta`.
    """
    distance = int(round(math.hypot(dx, dz)))
    theta = (math.degrees(math.atan2(dx, dz)) + 360.0) % 360.0
    step = 360.0 / NUM_ACTIONS
    theta_snapped = round(theta / step) * step % 360.0
    return Hop(theta_deg=theta_snapped, distance_blocks=distance)


