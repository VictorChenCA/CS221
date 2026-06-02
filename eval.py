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
    python3 eval.py --policy random   --seed 123 --episode 0
    python3 eval.py --policy frontier --seed 123 --episode 1 --budget-s 600
    python3 eval.py --policy oracle   --seed 123 --episode 2 --radius 64
"""

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from mdp.env import Env
from mdp.baselines import (
    RandomPolicy,
    FrontierPolicy,
    FrontierSectorVote,
    FrontierClosestCell,
    FrontierClusterCentroid,
    FrontierUnvisitedCells,
)
from mdp.world import NpzWorldView
from mdp.features import init_stuck_trace, update_stuck_trace
import mdp.oracle_cluster as oracle_cluster_mod
import mdp.oracle_lookahead as oracle_lookahead_mod

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_BUDGET_S = 600.0     # 10 minutes, proposal §2
POS_GRID_BLOCKS = 50         # state-visitation entropy cell size
COVERAGE_RADIUS = 1000       # blocks; for position-coverage metric


def make_policy(name: str, seed: int):
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "frontier":
        return FrontierPolicy(seed=seed)  # = FrontierClusterCentroid (current default)
    if name == "frontier_sector":
        return FrontierSectorVote(seed=seed)
    if name == "frontier_sector_grid32":
        return FrontierSectorVote(seed=seed, max_radius=32)   # 128-block window
    if name == "frontier_sector_grid64":
        return FrontierSectorVote(seed=seed, max_radius=64)   # 256-block window
    if name == "frontier_sector_penalty":
        return FrontierSectorVote(seed=seed, penalize_visited=True)
    if name == "frontier_sector_short":
        p = FrontierSectorVote(seed=seed)
        p.distance = 25
        return p
    if name == "frontier_sector_long":
        p = FrontierSectorVote(seed=seed)
        p.distance = 75
        return p
    if name == "frontier_closest":
        return FrontierClosestCell(seed=seed)
    if name == "frontier_cluster":
        return FrontierClusterCentroid(seed=seed)
    if name == "frontier_cells":
        return FrontierUnvisitedCells(seed=seed)
    raise ValueError(f"unknown policy '{name}'")


def run_policy_episode(env: Env, policy, budget_s: float) -> tuple[list[dict], dict]:
    """Step `policy` against `env` until budget elapses. Return (trail, termination_info).

    Stuck-escape: deterministic policies (qlearn, frontier_sector_penalty)
    enter same-state-same-action loops when stuck (94-99% of qlearn's actions
    in v25 picked the same theta as the previous, with 78% stuck rate). After
    any STUCK_ESCAPE_STREAK consecutive stuck actions, force a random
    compass action for one step to break the loop. Random/oracle ignore this
    (random never gets stuck-looped; oracle has its own plan)."""
    import random as _random
    policy.reset()
    t0 = time.monotonic()
    trail = [env.observe()]  # warmup: no-op observe (no pathfinder run)
    if trail[-1].get("dead"):
        reason = trail[-1].get("reason")
        elapsed = time.monotonic() - t0
        print(f"[eval] bot dead at warmup: {reason}")
        print(f"[eval-done] termination=dead_at_warmup elapsed={elapsed:.1f}s "
              f"actions=0 dead_reason={reason}")
        return trail, {"termination": "dead_at_warmup", "elapsed_s": elapsed,
                       "dead_reason": reason, "dead_at_action": 0}

    STUCK_ESCAPE_STREAK = int(os.environ.get("STUCK_ESCAPE_STREAK", "1"))
    # STUCK_ESCAPE_STREAK = 999 effectively disables hardcoded escape — use
    # this to evaluate whether qlearn has *learned* to escape on its own.
    escape_rng = _random.Random()
    stuck_streak = 0
    n_escapes = 0
    trace = init_stuck_trace()
    while time.monotonic() - t0 < budget_s:
        # Annotate the obs the policy will see with was_stuck + the
        # per-direction stuck memory so qlearn's featurizer can route
        # around recently-failed directions on its own.
        trail[-1]["was_stuck"] = bool(stuck_streak > 0)
        trail[-1]["stuck_dirs"] = trace
        if stuck_streak >= STUCK_ESCAPE_STREAK:
            action = escape_rng.randrange(8)
            n_escapes += 1
            stuck_streak = 0
        else:
            action = policy.act(trail[-1])
        obs = env.step(action)
        stuck_now = bool(obs.get("stuck"))
        trace = update_stuck_trace(trace, action, stuck_now)
        obs["stuck_dirs"] = trace
        trail.append(obs)
        if stuck_now:
            stuck_streak += 1
        else:
            stuck_streak = 0
        if obs.get("dead"):
            reason = obs.get("reason")
            elapsed = time.monotonic() - t0
            print(f"[eval] bot dead after action {len(trail)-1}: {reason}")
            print(f"[eval-done] termination=dead_mid_run elapsed={elapsed:.1f}s "
                  f"actions={len(trail)-1} dead_reason={reason}  n_escapes={n_escapes}")
            return trail, {"termination": "dead_mid_run", "elapsed_s": elapsed,
                           "dead_reason": reason, "dead_at_action": len(trail) - 1,
                           "n_escapes": n_escapes}
    elapsed = time.monotonic() - t0
    print(f"[eval-done] termination=budget_exhausted elapsed={elapsed:.1f}s "
          f"actions={len(trail)-1} dead_reason=None  n_escapes={n_escapes}")
    return trail, {"termination": "budget_exhausted", "elapsed_s": elapsed,
                   "dead_reason": None, "dead_at_action": None,
                   "n_escapes": n_escapes}


def run_oracle_episode(env: Env, seed: int, radius_cells: int,
                       budget_s: float, plan_fn=None) -> tuple[list[dict], int]:
    """Online-replanning oracle. After each hop, re-plan from the bot's
    *actual* landed position (which may differ from the planned target
    when pathfinder fails). The planner skips biomes already physically
    entered, so failures don't cascade — the next hop targets whatever
    is closest *now*.

    Returns (trail, planned_ub_initial):
      - trail: actual obs sequence (subject to pathfinder failures)
      - planned_ub_initial: the offline plan's expected biome count
        from the START position (the theoretical UB at episode start).
    """
    from mdp import oracle  # local: keep numpy out of the policy path
    if plan_fn is None:
        plan_fn = oracle.plan

    obs = env.observe()
    trail = [obs]
    if obs.get("dead"):
        return trail, 0

    visited: set[int] = set()
    start_b = obs.get("biomeId", -1)
    if start_b is not None and start_b >= 0:
        visited.add(start_b)

    # Build the INITIAL plan from start position for reporting the
    # theoretical UB (what the offline planner thinks is achievable).
    start_cell = (obs["cellX"], obs["cellZ"])
    initial_plan = plan_fn(seed=seed, start_cell=start_cell,
                           radius_cells=radius_cells,
                           time_budget_s=budget_s)
    planned_ub_initial = len(initial_plan.expected_biomes)

    t0 = time.monotonic()
    # Online replan loop — re-plan after every step from the current
    # cell, skipping already-visited biomes. Execute only the FIRST hop
    # of each fresh plan.
    while time.monotonic() - t0 < budget_s:
        cur_cell = (obs.get("cellX"), obs.get("cellZ"))
        if cur_cell[0] is None or cur_cell[1] is None:
            break
        remaining_budget = budget_s - (time.monotonic() - t0)
        new_plan = plan_fn(seed=seed, start_cell=cur_cell,
                           radius_cells=radius_cells,
                           time_budget_s=remaining_budget,
                           visited=visited)
        if not new_plan.hops:
            break  # no more reachable biomes within budget
        hop = new_plan.hops[0]
        obs = env.step_raw(hop.theta_deg, hop.distance_blocks)
        trail.append(obs)
        b = obs.get("biomeId", -1)
        if b is not None and b >= 0:
            visited.add(b)
        if obs.get("dead"):
            break
    return trail, planned_ub_initial


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
        x, z = obs.get("x"), obs.get("z")
        if x is None or z is None:
            continue  # bot disconnected; bridge shipped null coords
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
                    choices=["random", "frontier", "frontier_sector",
                             "frontier_sector_grid32", "frontier_sector_grid64",
                             "frontier_sector_penalty",
                             "frontier_sector_short", "frontier_sector_long",
                             "frontier_closest", "frontier_cluster",
                             "frontier_cells", "qlearn", "oracle",
                             "oracle_cluster", "oracle_lookahead"])
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
    ap.add_argument("--weights", type=Path, default=Path("weights/qlearn.npz"),
                    help="path to trained linear-Q weights (for --policy qlearn)")
    args = ap.parse_args()

    view = NpzWorldView(args.seed) if args.mode == "complete" else None
    # Build the policy first so we can read off any hop-distance override
    # before constructing Env (which uses a fixed distance per episode).
    policy_obj = None
    pre_policy = None
    if args.policy not in ("oracle", "oracle_cluster", "oracle_lookahead", "qlearn"):
        pre_policy = make_policy(args.policy, seed=args.seed)
    elif args.policy == "qlearn":
        from mdp.qlearn import LinearQ
        pre_policy = LinearQ.load(args.weights)
        pre_policy.epsilon = 0.0
    distance_override = getattr(pre_policy, "distance", None) if pre_policy else None
    env_kwargs = {"port": 9000 + args.bot_id,
                  "timeout": args.budget_s + 60,
                  "world_view": view}
    if distance_override is not None:
        env_kwargs["distance"] = distance_override
    env = Env(**env_kwargs)
    termination: dict = {"termination": "oracle", "elapsed_s": None,
                         "dead_reason": None, "dead_at_action": None}
    oracle_expected_biomes = None
    try:
        if args.policy == "oracle":
            trail, oracle_expected_biomes = run_oracle_episode(
                env, args.seed, args.radius, args.budget_s)
        elif args.policy == "oracle_cluster":
            trail, oracle_expected_biomes = run_oracle_episode(
                env, args.seed, args.radius, args.budget_s,
                plan_fn=oracle_cluster_mod.plan)
        elif args.policy == "oracle_lookahead":
            trail, oracle_expected_biomes = run_oracle_episode(
                env, args.seed, args.radius, args.budget_s,
                plan_fn=oracle_lookahead_mod.plan)
        elif args.policy == "qlearn":
            policy_obj = pre_policy
            trail, termination = run_policy_episode(env, pre_policy, args.budget_s)
        else:
            policy_obj = pre_policy
            trail, termination = run_policy_episode(env, pre_policy, args.budget_s)
    finally:
        env.close()

    metrics = compute_metrics(trail)
    # Oracle reports the THEORETICAL upper bound from the plan, not the
    # pathfinder-degraded execution. Keep the executed-trail metrics in
    # 'actual_unique_biomes' for diagnostic comparison.
    if oracle_expected_biomes is not None:
        metrics["actual_unique_biomes"] = metrics["unique_biomes"]
        metrics["unique_biomes"] = oracle_expected_biomes
    n_stuck = sum(1 for o in trail if o.get("stuck"))
    # Pull policy-internal diagnostics if the policy exposes a stats Counter.
    policy_stats = {}
    if policy_obj is not None and hasattr(policy_obj, "stats"):
        policy_stats = dict(policy_obj.stats)
    print(f"[eval] {n_stuck} stuck of {metrics['n_actions']} actions  "
          f"policy_stats={policy_stats}")
    out = RESULTS_DIR / f"{args.policy}_{args.seed}_{args.episode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "policy": args.policy,
        "seed": args.seed,
        "episode": args.episode,
        "budget_s": args.budget_s,
        "n_stuck": n_stuck,
        **metrics,
        **termination,
        "policy_stats": policy_stats,
        # Per-step (x,z) path for trajectory plots (top-down maps).
        "trail_xz": [[o.get("x"), o.get("z")] for o in trail],
    }, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()