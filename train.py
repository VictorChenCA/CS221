"""Train linear Q-learning on the proposal's 10 train seeds (proposal §6).

Parallel design — 10 worlds × 1 bot each × 3 rounds (Hogwild-ish shared W):

  Outer loop (sequential, R rounds):
    - Spawn 10 fresh bot bridges, one per server (fresh visitedBiomes each)
    - Inner loop (parallel, 10 threads): each bot runs ONE episode and TD(0)
      updates the shared agent.W as it goes. Updates are guarded by a
      threading.Lock; per-step cost is microseconds vs the ~20s pathfinder
      wait, so the lock is negligible.
    - Kill all 10 bots; decay ε; save weights.

  Servers are booted once at the start and reused across all rounds.

Checkpoint / resume:
  weights/qlearn.npz stores W + epsilon + rounds_completed. If present at
  start, training picks up at the next round. --fresh to override.

Usage:
    python3 train.py                         # all 10 train seeds, 3 rounds
    python3 train.py --episodes-per-seed 5   # 5 rounds instead of 3
    python3 train.py --fresh                 # ignore any saved checkpoint
    python3 train.py --seeds 1111,2222       # subset of seeds
"""

import argparse
import atexit
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mdp.env import Env
from mdp.world import NpzWorldView
from mdp.qlearn import LinearQ, compute_reward
from mdp.features import featurize, init_stuck_trace, update_stuck_trace

# ------- config -------------------------------------------------------------

LOGS = ROOT / "logs"
WEIGHTS_OUT = ROOT / "weights" / "qlearn.npz"
TRAIN_PORT_BASE = 25570         # avoid colliding with test-eval's 25565-7
BOT_PORT_BASE = 9000            # bridge.js hardcodes 9000+bot_id
HEAP_GB = 2                     # per Paper server; 10 × 2 GB = 20 GB
SERVER_READY_TIMEOUT_S = 240
BOT_SETTLE_S = 35               # 10 bots/load takes longer than 3 bots/load

PROCS: list[subprocess.Popen] = []


# ------- process lifecycle --------------------------------------------------

def cleanup() -> None:
    for p in PROCS:
        if p.poll() is None:
            p.terminate()
    for p in PROCS:
        try: p.wait(timeout=5)
        except subprocess.TimeoutExpired: p.kill()


atexit.register(cleanup)
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: sys.exit(130))


def spawn(cmd, log: Path, *, env=None, cwd=None) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(cmd, stdout=log.open("w"),
                         stderr=subprocess.STDOUT, env=env,
                         cwd=str(cwd) if cwd else None)
    PROCS.append(p)
    return p


def reap(p: subprocess.Popen) -> None:
    if p.poll() is None:
        p.terminate()
        try: p.wait(timeout=5)
        except subprocess.TimeoutExpired: p.kill()
    if p in PROCS:
        PROCS.remove(p)


# ------- server staging -----------------------------------------------------

def offline_uuid(name: str) -> str:
    h = bytearray(hashlib.md5(f"OfflinePlayer:{name}".encode()).digest())
    h[6] = (h[6] & 0x0F) | 0x30
    h[8] = (h[8] & 0x3F) | 0x80
    s = h.hex()
    return f"{s[0:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}"


def stage_server_dir(seed: int, port: int, bot_id: int) -> Path:
    src = ROOT / "mc-server"
    dst = ROOT / f"mc-server-train{seed}"
    if not dst.exists():
        dst.mkdir()
        shutil.copy(src / "paper.jar", dst / "paper.jar")
        for fname in ("eula.txt", "server.properties", "bukkit.yml",
                      "spigot.yml", "commands.yml", "help.yml",
                      "permissions.yml"):
            if (src / fname).exists():
                shutil.copy(src / fname, dst / fname)
        if (src / "config").exists():
            shutil.copytree(src / "config", dst / "config")
        props = dst / "server.properties"
        text = props.read_text()
        text = re.sub(r"^level-seed=.*$", f"level-seed={seed}", text, flags=re.M)
        text = re.sub(r"^server-port=.*$", f"server-port={port}", text, flags=re.M)
        props.write_text(text)
    ops = [{"uuid": offline_uuid(f"Explorer_{bot_id}"),
            "name": f"Explorer_{bot_id}", "level": 4,
            "bypassesPlayerLimit": False},
           {"uuid": offline_uuid("Raz0rMC"), "name": "Raz0rMC",
            "level": 4, "bypassesPlayerLimit": False}]
    (dst / "ops.json").write_text(json.dumps(ops, indent=2))
    return dst


def wait_for_done(log_path: Path, label: str) -> None:
    deadline = time.time() + SERVER_READY_TIMEOUT_S
    while time.time() < deadline:
        if log_path.exists() and "Done" in log_path.read_text(errors="ignore"):
            print(f"[ready] {label}")
            return
        time.sleep(2)
    raise TimeoutError(f"{label} not ready within {SERVER_READY_TIMEOUT_S}s")


# ------- checkpointing ------------------------------------------------------

def save_checkpoint(agent: LinearQ, rounds_completed: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W=agent.W, alpha=np.float32(agent.alpha),
             gamma=np.float32(agent.gamma), epsilon=np.float32(agent.epsilon),
             rounds_completed=np.int32(rounds_completed))


def load_checkpoint(path: Path, alpha: float, gamma: float,
                    epsilon_decay_episodes: int) -> tuple[LinearQ, int]:
    z = np.load(path)
    agent = LinearQ(alpha=float(z["alpha"]), gamma=float(z["gamma"]),
                    epsilon=float(z["epsilon"]),
                    epsilon_decay_episodes=epsilon_decay_episodes)
    agent.W = z["W"].astype(np.float32)
    rounds_completed = int(z["rounds_completed"]) if "rounds_completed" in z.files else 0
    return agent, rounds_completed


# ------- training -----------------------------------------------------------

def connect_env(port: int, view: NpzWorldView, timeout: float) -> Env:
    deadline = time.time() + 30
    last_exc = None
    while time.time() < deadline:
        try:
            return Env(port=port, world_view=view, timeout=timeout)
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            last_exc = e
            time.sleep(1)
    raise RuntimeError(f"could not connect to bot on :{port}: {last_exc}")


# Hogwild-ish: one lock around `W` updates. Cheap vs the ~20s pathfinder hop;
# guarantees correctness against numpy non-atomic in-place adds.
W_LOCK = threading.Lock()


def locked_update(agent: LinearQ, s: dict, a: int, r: float, s_next: dict) -> None:
    phi_s = featurize(s)
    with W_LOCK:
        q_sa = float(agent.W[a] @ phi_s)
    target = r + agent.gamma * float(np.max(agent.W @ featurize(s_next)))
    td_error = target - q_sa
    with W_LOCK:
        agent.W[a] += agent.alpha * td_error * phi_s


def run_episode_thread(env: Env, agent: LinearQ, budget_s: float,
                       seed: int, results: dict) -> None:
    """One worker thread: drive `env` under ε-greedy for `budget_s` seconds,
    TD-update shared `agent.W` after every step (W_LOCK-guarded)."""
    try:
        prev = env.observe()
        prev["was_stuck"] = False
        trace = init_stuck_trace()
        prev["stuck_dirs"] = trace
        t0 = time.monotonic()
        n_steps = 0; n_stuck = 0; total_r = 0.0
        biomes_at_start = int(prev.get("numVisited", 0))
        # Forced-escape threshold. Historically this was hardcoded to 1
        # (force a random action after a single stuck), but that crutch
        # acts FOR the policy on every stuck step, so the policy never
        # experiences its own consecutive stuck actions and can't learn to
        # escape. Default is now 999 (off) so training is rigorous: the
        # policy must learn directional avoidance via the stuck_dirs
        # feature. A high safety valve can still be set via the env var.
        # Stuck-spam is bounded by the distance-proportional action timeout
        # (~33s/50-block hop), so a wedged bot fires ~9 steps in 300s, not
        # the ~6000 we saw before the iter-9 timeout fix.
        import random as _random
        import math as _math
        STUCK_ESCAPE_STREAK = int(os.environ.get("STUCK_ESCAPE_STREAK", "999"))
        # Count-based exploration bonus (Bellemare 2016 / Tang 2017): a
        # +BETA_COUNT/sqrt(N(cell)) intrinsic reward on the discretized
        # position cell drives *directed* spatial coverage (the proposal's
        # planned alternative to undirected eps-greedy). Training-only —
        # eval just runs the learned policy. Default 0 (off).
        BETA_COUNT = float(os.environ.get("BETA_COUNT", "0"))
        COUNT_CELL = int(os.environ.get("COUNT_CELL", "50"))  # blocks/cell
        visit_counts: dict = {}
        escape_rng = _random.Random(seed)
        stuck_streak = 0
        while time.monotonic() - t0 < budget_s:
            prev["was_stuck"] = bool(stuck_streak > 0)
            prev["stuck_dirs"] = trace
            if stuck_streak >= STUCK_ESCAPE_STREAK:
                a = escape_rng.randrange(8)
                stuck_streak = 0
            else:
                a = agent.act(prev)
            obs = env.step(a)
            # A NaN-killed / kicked bot returns {stuck:true, dead:true}
            # *instantly* (no pathfinder wait), so without this break the
            # loop fires thousands of degenerate gradient updates into the
            # shared W in the remaining budget (one dead bot produced a
            # ~14000-stuck round in iter 2). End the episode on death, as
            # eval.py already does.
            if obs.get("dead"):
                break
            stuck_now = bool(obs.get("stuck"))
            obs["was_stuck"] = stuck_now
            trace = update_stuck_trace(trace, a, stuck_now)
            obs["stuck_dirs"] = trace
            r = compute_reward(prev, obs)
            if BETA_COUNT and obs.get("x") is not None:
                cell = (obs["x"] // COUNT_CELL, obs["z"] // COUNT_CELL)
                visit_counts[cell] = visit_counts.get(cell, 0) + 1
                r += BETA_COUNT / _math.sqrt(visit_counts[cell])
            locked_update(agent, prev, a, r, obs)
            total_r += r
            n_steps += 1
            if stuck_now:
                n_stuck += 1
                stuck_streak += 1
            else:
                stuck_streak = 0
            prev = obs
        results[seed] = {
            "ok": True, "reward": total_r, "steps": n_steps,
            "stuck": n_stuck,
            "unique_biomes": int(prev.get("numVisited", 0)),
        }
    except Exception as e:
        results[seed] = {"ok": False, "error": str(e)}


def load_train_seeds() -> list[int]:
    text = (ROOT / "seeds.txt").read_text()
    out = []
    in_train = False
    for line in text.splitlines():
        s = line.strip()
        if not s: continue
        if s.startswith("#"):
            in_train = s.lower().startswith("# train")
            continue
        if in_train:
            out.append(int(s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per-seed", type=int, default=3,
                    help="number of rounds (each round = one episode in every world)")
    ap.add_argument("--budget-s", type=float, default=300.0)
    ap.add_argument("--weights-out", type=Path, default=WEIGHTS_OUT)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds; overrides seeds.txt")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--cleanup", action="store_true",
                    help="rm -rf staged mc-server-train<seed>/ dirs after the run")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing weights checkpoint; start over")
    args = ap.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else load_train_seeds())
    N = len(seeds)
    n_rounds = args.episodes_per_seed
    n_eps_total = N * n_rounds
    print(f"[train] {N} seeds × {n_rounds} rounds = {n_eps_total} total episodes "
          f"(parallel, 10 worlds × 1 bot)")

    # Prereq: cubiomes-derived npz for every train seed.
    for s in seeds:
        if not (ROOT / "data" / f"biomes_{s}.npz").exists():
            sys.exit(f"missing data/biomes_{s}.npz — run "
                     f"`python3 tools/extract_biomes.py --seed {s}` first")

    LOGS.mkdir(exist_ok=True)

    # Resume from checkpoint if present.
    rounds_completed = 0
    # Decay ε over almost the whole run so the agent doesn't go fully greedy
    # before it has seen enough variety. Keep last ~10% greedy for stable
    # final Q updates.
    eps_decay_eps = max(1, int(n_eps_total * 0.9))

    if args.weights_out.exists() and not args.fresh:
        agent, rounds_completed = load_checkpoint(
            args.weights_out, args.alpha, args.gamma, eps_decay_eps)
        print(f"[resume] loaded checkpoint: ε={agent.epsilon:.3f} "
              f"rounds_completed={rounds_completed}/{n_rounds}")
        if rounds_completed >= n_rounds:
            print(f"[complete] all rounds already done. Use --fresh to retrain.")
            return
    else:
        agent = LinearQ(alpha=args.alpha, gamma=args.gamma,
                        epsilon_decay_episodes=eps_decay_eps)
        print(f"[fresh] new agent ε={agent.epsilon:.3f} "
              f"(decays over {eps_decay_eps} of {n_eps_total} episodes)")

    # Stage + boot all 10 servers (parallel).
    print(f"[stage] {N} server dirs ({HEAP_GB} GB heap each, {N * HEAP_GB} GB total)")
    servers: list[tuple[int, int, Path]] = []
    for i, seed in enumerate(seeds):
        port = TRAIN_PORT_BASE + i
        bot_id = i
        dst = stage_server_dir(seed, port, bot_id=bot_id)
        servers.append((seed, port, dst))

    print(f"[boot] booting {N} Paper servers in parallel")
    for seed, port, dst in servers:
        log = LOGS / f"train_server_{seed}.log"
        spawn(["java", f"-Xmx{HEAP_GB}G", "-Xms512M", "-jar", "paper.jar", "nogui"],
              log, cwd=dst)
    for seed, _, _ in servers:
        wait_for_done(LOGS / f"train_server_{seed}.log", f"server seed={seed}")
    print(f"[ready] all {N} servers up")

    try:
        for round_idx in range(rounds_completed, n_rounds):
            print(f"\n[round {round_idx+1}/{n_rounds}] starting "
                  f"({time.strftime('%H:%M:%S')})")

            # Spawn fresh bots (one per server). Sequential spawn but quick.
            bot_procs = []
            for i, (seed, port, _) in enumerate(servers):
                bot_id = i
                env_vars = os.environ.copy()
                env_vars["MC_PORT"] = str(port)
                env_vars["DISPERSE_N"] = "1"  # single bot per world — no dispersal
                bot_log = LOGS / f"train_bot_{seed}_r{round_idx}.log"
                bot_procs.append(spawn(
                    ["node", "bot/bridge.js", str(bot_id)],
                    bot_log, env=env_vars, cwd=ROOT))
                time.sleep(0.3)
            print(f"[bots] {N} bots spawning; settling {BOT_SETTLE_S}s")
            time.sleep(BOT_SETTLE_S)

            # Connect 10 Envs, start 10 threads (one episode each).
            results: dict[int, dict] = {}
            threads: list[threading.Thread] = []
            envs: list[Env] = []
            for i, (seed, _, _) in enumerate(servers):
                try:
                    view = NpzWorldView(seed)
                    env = connect_env(BOT_PORT_BASE + i, view, args.budget_s + 60)
                    envs.append(env)
                    t = threading.Thread(
                        target=run_episode_thread,
                        args=(env, agent, args.budget_s, seed, results),
                        name=f"ep-seed{seed}")
                    t.start()
                    threads.append(t)
                except Exception as e:
                    print(f"[ep] seed={seed} FAILED to start: {e}")
                    results[seed] = {"ok": False, "error": str(e)}

            for t in threads:
                t.join()
            for env in envs:
                try: env.close()
                except Exception: pass

            # Per-seed result lines.
            for seed in seeds:
                res = results.get(seed, {"ok": False, "error": "no result"})
                if res["ok"]:
                    print(f"[ep] seed={seed} round={round_idx} "
                          f"reward={res['reward']:+.2f} steps={res['steps']:>3} "
                          f"stuck={res['stuck']:>2} biomes={res['unique_biomes']}")
                else:
                    print(f"[ep] seed={seed} round={round_idx} ERROR: {res['error']}")

            # Kill bots; persist; decay ε.
            for p in bot_procs:
                reap(p)
            rounds_completed = round_idx + 1
            agent.decay_epsilon(rounds_completed * N - 1)
            print(f"[checkpoint] round {round_idx+1} done; ε={agent.epsilon:.3f}")
            if not args.no_save:
                save_checkpoint(agent, rounds_completed, args.weights_out)
                print(f"[saved] {args.weights_out}")
    finally:
        for seed, _, dst in servers:
            for p in PROCS[:]:
                if p.poll() is None:
                    reap(p)
            if args.cleanup:
                shutil.rmtree(dst, ignore_errors=True)

    if args.no_save:
        print("[complete] --no-save: weights discarded")
    else:
        print(f"[complete] saved weights to {args.weights_out}")


if __name__ == "__main__":
    main()
