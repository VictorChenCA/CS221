"""Offline max-coverage oracle (proposal §6 upper bound).

The oracle is *not* a Policy — it doesn't see observations one at a
time. Given a seed and a time budget, it loads the full biome map
inside a fixed search radius and plans a route that visits as many
distinct biomes as possible under the budget. `eval.py` then replays
the planned action sequence through the same bridge the learned
policies use.

Algorithm (greedy orienteering / set-cover):
  1. Scan biome_at over the (2R+1)x(2R+1) cell window around start.
  2. Group cells by biome id.
  3. Greedily choose the nearest unvisited biome from the CURRENT
     position.
  4. Break long paths into smaller macro-hops that match the actual
     action space used by RL/baselines.
"""

import math
from dataclasses import dataclass
from typing import Callable

from mdp.env import NUM_ACTIONS
from mdp.world import CELL_BLOCKS, NpzWorldView

# Conservative real-world movement estimate for Mineflayer movement.
WALK_SPEED_BPS = 4.3

# Removed in v20: oracle no longer chunks paths into 50-block hops.
# Each biome target is now reached via a single pathfinder hop of the
# full distance. Pathfinder has 30s to compute and execute the path;
# may fail on long routes through water/mountains, but avoids the
# compounding failure rate of 4-5 sequential 50-block chunks.

# Pathfinding is never perfectly straight in Minecraft.
PATH_EFFICIENCY_PENALTY = 1.5

BiomeFn = Callable[[int, int], int]


@dataclass
class Hop:
    theta_deg: float
    distance_blocks: int


@dataclass
class Plan:
    hops: list[Hop]
    expected_biomes: list[int]


def plan(
    seed: int,
    start_cell: tuple[int, int],
    radius_cells: int,
    time_budget_s: float,
    biome_at: BiomeFn | None = None,
) -> Plan:
    """Greedy biome-coverage oracle."""

    if biome_at is None:
        biome_at = NpzWorldView(seed).biome_at

    sx, sz = start_cell

    # ------------------------------------------------------------
    # Scan biome window around spawn/current location.
    # ------------------------------------------------------------

    biome_cells: dict[int, list[tuple[int, int]]] = {}

    for dz in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            cx = sx + dx
            cz = sz + dz

            b = biome_at(cx, cz)

            if b < 0:
                continue

            biome_cells.setdefault(b, []).append((cx, cz))

    start_biome = biome_at(sx, sz)

    # Remove starting biome from targets.
    if start_biome in biome_cells:
        del biome_cells[start_biome]

    # ------------------------------------------------------------
    # Greedy coverage loop.
    # ------------------------------------------------------------

    hops: list[Hop] = []

    visited_biomes: list[int] = []

    if start_biome >= 0:
        visited_biomes.append(start_biome)

    cur = (sx, sz)

    time_used = 0.0

    while biome_cells:

        best_biome = None
        best_cell = None
        best_dist = float("inf")

        # --------------------------------------------------------
        # Choose nearest remaining biome from CURRENT position.
        # --------------------------------------------------------

        for biome_id, cells in biome_cells.items():

            nearest_cell = min(
                cells,
                key=lambda c: _block_dist(cur, c)
            )

            dist = _block_dist(cur, nearest_cell)

            if dist < best_dist:
                best_dist = dist
                best_biome = biome_id
                best_cell = nearest_cell

        if best_cell is None:
            break

        # --------------------------------------------------------
        # Estimate realistic Minecraft travel time.
        # --------------------------------------------------------

        travel_s = (
            PATH_EFFICIENCY_PENALTY
            * best_dist
            / WALK_SPEED_BPS
        )

        if time_used + travel_s > time_budget_s:
            break

        # --------------------------------------------------------
        # Single direct hop to the target biome cell — no 50-block
        # chunking. Each chunked hop was a separate pathfinder call
        # that could fail; with 20-30 chunked hops per plan, failure
        # rate compounded and only ~35% of planned biomes were hit.
        # Going direct: pathfinder gets one shot at the full route.
        # --------------------------------------------------------

        dx_total = (best_cell[0] - cur[0]) * CELL_BLOCKS
        dz_total = (best_cell[1] - cur[1]) * CELL_BLOCKS

        if math.hypot(dx_total, dz_total) > 1:
            hops.append(_snap_to_compass(dx_total, dz_total))

        visited_biomes.append(best_biome)

        cur = best_cell

        time_used += travel_s

        del biome_cells[best_biome]

    return Plan(
        hops=hops,
        expected_biomes=visited_biomes,
    )


def _block_dist(
    a: tuple[int, int],
    b: tuple[int, int],
) -> float:

    return math.hypot(
        (a[0] - b[0]) * CELL_BLOCKS,
        (a[1] - b[1]) * CELL_BLOCKS,
    )


def _snap_to_compass(dx: float, dz: float) -> Hop:
    """Snap arbitrary vector to nearest 8-way action."""

    distance = int(round(math.hypot(dx, dz)))

    theta = (
        math.degrees(math.atan2(dx, dz))
        + 360.0
    ) % 360.0

    step = 360.0 / NUM_ACTIONS

    theta_snapped = (
        round(theta / step) * step
    ) % 360.0

    return Hop(
        theta_deg=theta_snapped,
        distance_blocks=distance,
    )