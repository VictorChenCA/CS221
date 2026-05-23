"""Train linear Q-learning on the proposal's 10 train seeds (proposal §6).

Per train seed:
  1. Stage a Paper server dir (mc-server-train<seed>/) and boot it.
  2. For each of EPISODES_PER_SEED episodes:
     - Spawn one bot process (fresh `visitedBiomes` set each time).
     - Connect Env, run the episode under ε-greedy, TD(0)-update Q after
       every step.
     - Kill the bot.
  3. Kill the server.

Weights are persisted to weights/qlearn.npz after every episode (cheap).
At eval time:  python eval.py --policy qlearn --weights weights/qlearn.npz

Prereqs:
  - Cubiomes built (README §1.4)
  - Biome dump for every train seed:
      for s in $(grep -v '^#' seeds.txt | sed -n '/^[0-9]/p' | head -10);
      do python3 tools/extract_biomes.py --seed $s; done

Usage:
    python3 train.py
    python3 train.py --episodes-per-seed 3 --budget-s 300
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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mdp.env import Env
from mdp.world import NpzWorldView
from mdp.qlearn import LinearQ, compute_reward

# ------- config -------------------------------------------------------------

LOGS = ROOT / "logs"
WEIGHTS_OUT = ROOT / "weights" / "qlearn.npz"
TRAIN_PORT = 25570              # avoid colliding with test-eval's 25565-7
BOT_PORT_BASE = 9100            # avoid colliding with test-eval's 9000-14
SERVER_READY_TIMEOUT_S = 180
BOT_SETTLE_S = 25               # /tp dispersal + chunk load + bridge open

PROCS: list[subprocess.Popen] = []


# ------- process lifecycle --------------------------------------------------

def cleanup() -> None:
    for p in PROCS:
        if p.poll() is None:
            p.terminate()
    for p in PROCS:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


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


# ------- server staging (minimal duplicate from run_test_eval) --------------

def offline_uuid(name: str) -> str:
    h = bytearray(hashlib.md5(f"OfflinePlayer:{name}".encode()).digest())
    h[6] = (h[6] & 0x0F) | 0x30
    h[8] = (h[8] & 0x3F) | 0x80
    s = h.hex()
    return f"{s[0:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}"


def stage_server_dir(seed: int, port: int, max_bot_id: int) -> Path:
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
    # Ops list covers max_bot_id + the human ops.
    ops = [{"uuid": offline_uuid(f"Explorer_{i}"), "name": f"Explorer_{i}",
            "level": 4, "bypassesPlayerLimit": False}
           for i in range(max_bot_id + 1)]
    ops += [{"uuid": offline_uuid("Raz0rMC"), "name": "Raz0rMC",
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


def run_train_episode(env: Env, agent: LinearQ, budget_s: float) -> dict:
    prev = env.step(0)  # warmup observation
    t0 = time.monotonic()
    n_steps = 0
    n_stuck = 0
    cumulative_reward = 0.0
    biomes_at_start = int(prev.get("numVisited", 0))

    while time.monotonic() - t0 < budget_s:
        a = agent.act(prev)
        obs = env.step(a)
        r = compute_reward(prev, obs)
        agent.update(prev, a, r, obs, done=False)
        cumulative_reward += r
        n_steps += 1
        if obs.get("stuck"):
            n_stuck += 1
        prev = obs

    return {
        "reward": cumulative_reward,
        "steps": n_steps,
        "stuck": n_stuck,
        "unique_biomes": int(prev.get("numVisited", 0)) - biomes_at_start
                          + (1 if biomes_at_start > 0 else 0),
    }


def load_train_seeds() -> list[int]:
    text = (ROOT / "seeds.txt").read_text()
    # Take seeds until the "# test" header
    out = []
    in_train = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            in_train = s.lower().startswith("# train")
            continue
        if in_train:
            out.append(int(s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per-seed", type=int, default=3)
    ap.add_argument("--budget-s", type=float, default=300.0)
    ap.add_argument("--weights-out", type=Path, default=WEIGHTS_OUT)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.95)
    args = ap.parse_args()

    seeds = load_train_seeds()
    n_eps_total = len(seeds) * args.episodes_per_seed
    print(f"[train] {len(seeds)} seeds × {args.episodes_per_seed} eps = {n_eps_total} total episodes")

    # Prereq check: cubiomes-derived npz for every train seed.
    for s in seeds:
        if not (ROOT / "data" / f"biomes_{s}.npz").exists():
            sys.exit(f"missing data/biomes_{s}.npz — run "
                     f"`python3 tools/extract_biomes.py --seed {s}` first")

    LOGS.mkdir(exist_ok=True)
    agent = LinearQ(alpha=args.alpha, gamma=args.gamma,
                    epsilon_decay_episodes=n_eps_total // 2)

    ep_global = 0
    for seed in seeds:
        dst = stage_server_dir(seed, TRAIN_PORT, max_bot_id=args.episodes_per_seed - 1)
        log = LOGS / f"train_server_{seed}.log"
        print(f"[server] booting seed={seed}")
        server_proc = spawn(
            ["java", "-Xmx4G", "-Xms1G", "-jar", "paper.jar", "nogui"],
            log, cwd=dst)
        try:
            wait_for_done(log, f"train server seed={seed}")
            view = NpzWorldView(seed)

            for ep in range(args.episodes_per_seed):
                bot_id = ep
                bot_port = BOT_PORT_BASE + bot_id
                env_vars = os.environ.copy()
                env_vars["MC_PORT"] = str(TRAIN_PORT)
                env_vars["DISPERSE_N"] = str(args.episodes_per_seed)
                bot_log = LOGS / f"train_bot_{seed}_{ep}.log"
                bot_proc = spawn(
                    ["node", "bot/bridge.js", str(bot_id)],
                    bot_log, env=env_vars, cwd=ROOT)
                print(f"[bot] seed={seed} ep={ep} bot_id={bot_id}, settling {BOT_SETTLE_S}s")
                time.sleep(BOT_SETTLE_S)
                try:
                    env = connect_env(bot_port, view, args.budget_s + 60)
                    metrics = run_train_episode(env, agent, args.budget_s)
                    env.close()
                    print(f"[ep] seed={seed} ep={ep:>2} "
                          f"reward={metrics['reward']:+.2f} steps={metrics['steps']:>3} "
                          f"stuck={metrics['stuck']:>2} biomes={metrics['unique_biomes']} "
                          f"ε={agent.epsilon:.3f}")
                finally:
                    reap(bot_proc)

                agent.decay_epsilon(ep_global)
                ep_global += 1
                agent.save(args.weights_out)
        finally:
            reap(server_proc)

    print(f"[complete] saved weights to {args.weights_out}")


if __name__ == "__main__":
    main()
