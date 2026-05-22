"""Offline max-coverage oracle (proposal §6 upper bound).

The oracle is *not* a Policy — it doesn't see observations one at a
time. Given a seed and a time budget, it computes the full biome map
inside a fixed search radius and plans a route that visits as many
distinct biomes as possible under the budget. `eval.py` then replays
the planned action sequence through the same bridge the learned
policies use.

Approach (greedy orienteering / set-cover):
  1. Look up `biome_at(x, z)` for every cell in a (2R+1)x(2R+1) window
     around `start`. R is in 4-block cells, matching the agent's MDP.
  2. For each distinct biome, take the *nearest* representative cell
     to the current position as its target.
  3. Greedy loop until budget exhausted:
       - of the unvisited biomes, pick the one whose target is
         cheapest to reach from the current position;
       - append that hop; advance position; mark biome visited.
  4. Convert the resulting waypoint sequence to (theta, distance) hops
     matching the bridge's action contract.

This is the standard greedy set-cover approximation for orienteering;
it doesn't hit the true optimum but gets us a strong upper bound
that's tractable for ~10-min budgets and the radii we care about.

NOTE: scaffold — `biome_at` is injected so we can swap in cubiomes
later without changing this file. See `_BiomeSource` below.
"""

from dataclasses import dataclass
from typing import Callable, Iterable

# Match the bridge's locomotion contract (8-way compass, hop = blocks).
NUM_ACTIONS = 8
CELL_BLOCKS = 4  # Minecraft 1.18+ biome cell size


BiomeFn = Callable[[int, int], int]  # (cellX, cellZ) -> biome id


@dataclass
class Hop:
    theta_deg: float
    distance_blocks: int


@dataclass
class Plan:
    hops: list[Hop]
    expected_biomes: list[int]  # biomes visited in order, including start


class _BiomeSource:
    """Stub biome source. Replace with cubiomes binding.

    Real implementation will be something like:
        cubiomes.set_seed(seed)
        return cubiomes.get_biome(cell_x * 4, y, cell_z * 4)
    For now this raises so callers don't silently get garbage.
    """

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, cell_x: int, cell_z: int) -> int:
        raise NotImplementedError(
            "biome_at — wire up cubiomes (see oracle.py docstring)"
        )


def plan(
    seed: int,
    start_cell: tuple[int, int],
    radius_cells: int,
    time_budget_s: float,
    biome_at: BiomeFn | None = None,
) -> Plan:
    """Compute an oracle plan. See module docstring for the algorithm.

    `biome_at` defaults to the cubiomes-backed `_BiomeSource(seed)`;
    tests can pass a dict-backed fake.
    """
    if biome_at is None:
        biome_at = _BiomeSource(seed)

    # TODO(Victor):
    #   1. scan the (2R+1)^2 window, build biome -> [cells] index
    #   2. for each biome, keep only the nearest cell to start
    #   3. greedy loop under `time_budget_s` (use a walk-speed constant
    #      to convert distance -> seconds; ~4.3 blocks/s for sprinting)
    #   4. emit Hops in compass-snapped form (match agent/env.py)
    raise NotImplementedError("oracle.plan — TODO")


def _snap_to_compass(dx: float, dz: float) -> Hop:
    """Convert a free vector (in blocks) to the nearest 8-way compass hop.

    Kept here because the oracle's plan must replay through the same
    discrete action space as the learned policies.
    """
    import math
    distance = int(round(math.hypot(dx, dz)))
    theta = (math.degrees(math.atan2(dx, dz)) + 360.0) % 360.0
    step = 360.0 / NUM_ACTIONS
    theta_snapped = round(theta / step) * step % 360.0
    return Hop(theta_deg=theta_snapped, distance_blocks=distance)


def replay(plan: Plan) -> Iterable[dict]:
    """Yield bridge action dicts for `eval.py` to send over the socket."""
    for hop in plan.hops:
        yield {"theta": hop.theta_deg, "distance": hop.distance_blocks}
