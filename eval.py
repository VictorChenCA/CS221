"""Episode orchestrator (proposal §3, §6).

Connects to a running bot bridge on port 9000+id, drives a chosen policy
(or replays the oracle) for a time budget, and writes per-episode metrics
to `results/<policy>_<seed>_<ep>.json`. Aggregation across seeds/episodes
is handled by a separate analysis script (TODO) to keep this file simple.

Assumes the Paper server is already running with the requested world
seed; switching seeds between runs needs a server restart (PaperMC reads
level-seed only at world creation). The wrapping shell script that loops
over seeds owns that — eval.py just runs one episode against whatever
world is currently up.

Usage:
    python eval.py --policy random   --seed 123 --episode 0
    python eval.py --policy frontier --seed 123 --episode 1 --budget-s 600
    python eval.py --policy oracle   --seed 123 --episode 2 --radius 64
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from mdp.env import Env
from mdp.baselines import RandomPolicy, FrontierPolicy
from mdp.world import NpzWorldView

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_BUDGET_S = 600.0     # 10 minutes, proposal §2
POS_GRID_BLOCKS = 50         # state-visitation entropy cell size
COVERAGE_RADIUS = 1000       # blocks; for position-coverage metric


def make_policy(name: str, seed: int):
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "frontier":
        return FrontierPolicy(seed=seed)
    raise ValueError(f"unknown policy '{name}' (use random/frontier/oracle)")


def run_policy_episode(env: Env, policy, budget_s: float) -> list[dict]:
    """Step `policy` against `env` until budget elapses. Return obs trail."""
    policy.reset()
    trail = [env.step(0)]  # warm-up obs to learn current biome
    t0 = time.monotonic()
    while time.monotonic() - t0 < budget_s:
        action = policy.act(trail[-1])
        trail.append(env.step(action))
    return trail


def run_oracle_episode(env: Env, seed: int, radius_cells: int,
                       budget_s: float) -> list[dict]:
    """Plan offline from start_cell, replay hops through the bridge."""
    from mdp import oracle  # local: keep numpy out of the policy path
    warmup = env.step(0)
    start_cell = (warmup["cellX"], warmup["cellZ"])
    plan = oracle.plan(seed=seed, start_cell=start_cell,
                       radius_cells=radius_cells, time_budget_s=budget_s)
    trail = [warmup]
    for hop in plan.hops:
        trail.append(env.step_raw(hop.theta_deg, hop.distance_blocks))
    return trail


def compute_metrics(trail: list[dict]) -> dict:
    """Primary + secondary metrics from proposal §3."""
    visited_biomes: set[int] = set()
    biome_step_counts: Counter[int] = Counter()
    pos_cells: Counter[tuple[int, int]] = Counter()
    for obs in trail:
        b = obs.get("biomeId")
        if b is not None and b >= 0:
            visited_biomes.add(b)
            biome_step_counts[b] += 1
        x, z = obs["x"], obs["z"]
        pos_cells[(x // POS_GRID_BLOCKS, z // POS_GRID_BLOCKS)] += 1

    n_actions = max(len(trail) - 1, 1)
    # Coverage: fraction of POS_GRID cells inside the COVERAGE_RADIUS disk
    # that were visited. Cell is "in disk" iff its center is within radius.
    in_disk = lambda cx, cz: math.hypot(
        cx * POS_GRID_BLOCKS + POS_GRID_BLOCKS / 2,
        cz * POS_GRID_BLOCKS + POS_GRID_BLOCKS / 2,
    ) <= COVERAGE_RADIUS
    visited_in_disk = sum(1 for c in pos_cells if in_disk(*c))
    total_in_disk = _cells_in_disk(COVERAGE_RADIUS, POS_GRID_BLOCKS)
    return {
        "unique_biomes": len(visited_biomes),                           # primary
        "biomes_per_action": len(visited_biomes) / n_actions,
        "position_entropy": _entropy(pos_cells.values()),
        "position_coverage": visited_in_disk / total_in_disk,
        "biome_entropy": _entropy(biome_step_counts.values()),
        "n_actions": n_actions,
    }


def _cells_in_disk(radius_blocks: int, cell_blocks: int) -> int:
    n = 0
    r_cells = radius_blocks // cell_blocks + 1
    for cx in range(-r_cells, r_cells + 1):
        for cz in range(-r_cells, r_cells + 1):
            x = cx * cell_blocks + cell_blocks / 2
            z = cz * cell_blocks + cell_blocks / 2
            if math.hypot(x, z) <= radius_blocks:
                n += 1
    return max(n, 1)


def _entropy(counts) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts if c > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    choices=["random", "frontier", "qlearn", "oracle"])
    ap.add_argument("--seed", type=int, required=True,
                    help="world seed (must match the running server)")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--bot-id", type=int, default=0,
                    help="bridge bot id (port = 9000 + id)")
    ap.add_argument("--budget-s", type=float, default=DEFAULT_BUDGET_S)
    ap.add_argument("--radius", type=int, default=64,
                    help="oracle search radius in cells")
    ap.add_argument("--mode", choices=["complete", "los"], default="complete",
                    help="proposal §2 world setting; 'complete' loads "
                         "data/biomes_<seed>.npz, 'los' uses the bridge grid")
    args = ap.parse_args()

    view = NpzWorldView(args.seed) if args.mode == "complete" else None
    env = Env(port=9000 + args.bot_id, timeout=args.budget_s + 60,
              world_view=view)
    try:
        if args.policy == "oracle":
            trail = run_oracle_episode(env, args.seed, args.radius, args.budget_s)
        elif args.policy == "qlearn":
            raise NotImplementedError("qlearn integration — TODO (§C7)")
        else:
            policy = make_policy(args.policy, seed=args.seed)
            trail = run_policy_episode(env, policy, args.budget_s)
    finally:
        env.close()

    metrics = compute_metrics(trail)
    n_stuck = sum(1 for o in trail if o.get("stuck"))
    print(f"[eval] {n_stuck} stuck of {metrics['n_actions']} actions")
    out = RESULTS_DIR / f"{args.policy}_{args.seed}_{args.episode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "policy": args.policy,
        "seed": args.seed,
        "episode": args.episode,
        "budget_s": args.budget_s,
        **metrics,
    }, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
