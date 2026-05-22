"""Locomotion smoke test.

Connects to a running bot bridge on port 9000+id and steps through all
8 compass actions at the configured hop distance, printing position and
biome after each step. Usage:

    python3 tools/test_locomotion.py                 # bot 0, distance 16
    python3 tools/test_locomotion.py --id 3 --dist 32

Requires the Paper server and bot bridge to already be running.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdp.env import Env, NUM_ACTIONS, action_to_theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0, help="bot id (port = 9000+id)")
    ap.add_argument("--dist", type=int, default=16, help="hop distance in blocks")
    ap.add_argument("--timeout", type=float, default=45.0)
    args = ap.parse_args()

    port = 9000 + args.id
    print(f"connecting to bot {args.id} on :{port}, hop = {args.dist} blocks")

    try:
        env = Env(port=port, distance=args.dist, timeout=args.timeout)
    except ConnectionRefusedError:
        print(f"ERROR: nothing listening on :{port}. Start the bridge first.")
        sys.exit(1)

    start = env.step(0)  # use first action also as a baseline readout
    print(f"start  -> x={start['x']:5d} z={start['z']:5d} "
          f"biome={start['biomeName']:20s} stuck={start.get('stuck', False)}")

    n_stuck = 1 if start.get("stuck") else 0
    biomes = set(start["visitedBiomes"])

    for a in range(1, NUM_ACTIONS):
        obs = env.step(a)
        flag = "STUCK" if obs.get("stuck") else "ok"
        print(f"a={a} theta={action_to_theta(a):3.0f} -> "
              f"x={obs['x']:5d} z={obs['z']:5d} "
              f"biome={obs['biomeName']:20s} [{flag}]")
        if obs.get("stuck"):
            n_stuck += 1
        biomes.update(obs["visitedBiomes"])

    env.close()
    print(f"\n{NUM_ACTIONS} actions: {NUM_ACTIONS - n_stuck} clean, {n_stuck} stuck")
    print(f"biomes seen: {sorted(biomes)}")


if __name__ == "__main__":
    main()
