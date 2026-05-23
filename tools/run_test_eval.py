"""Cross-platform test eval driver (macOS, Linux, Windows).

Stages one PaperMC server per test seed (ports 25565..25567), spawns
5 bot bridges per server (ports 9000..9014), then runs each policy
sequentially across all 15 bots in parallel. Wall-clock per policy
≈ one episode budget (default 10 min). Total ≈ 30 min for the three
policies (random, frontier, oracle), 45 episodes overall.

Prereqs: cubiomes built (README §1.4) AND a biome dump per test seed
(`python3 tools/extract_biomes.py --seed N` for 123, 456, 789).

Usage:
    python3 tools/run_test_eval.py
"""

import atexit
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [123, 456, 789]
POLICIES = ["frontier", "random"]
BOTS_PER_SERVER = 3
BUDGET_S = 300
BASE_MC_PORT = 25565
SETTLE_S = 35
SERVER_READY_TIMEOUT_S = 180

LOGS = ROOT / "logs"
PROCS: list[subprocess.Popen] = []


def cleanup() -> None:
    print("[cleanup]")
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


def offline_uuid(name: str) -> str:
    """Java's UUID.nameUUIDFromBytes("OfflinePlayer:<name>") = the UUID
    Paper's offline mode assigns. Used so ops.json matches the bots
    that connect."""
    h = bytearray(hashlib.md5(f"OfflinePlayer:{name}".encode()).digest())
    h[6] = (h[6] & 0x0F) | 0x30  # UUID v3
    h[8] = (h[8] & 0x3F) | 0x80  # variant
    s = h.hex()
    return f"{s[0:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}"


EXTRA_OPS = ["Raz0rMC"]  # human accounts (offline-mode UUIDs)


def write_ops_json(dst: Path, n_bots_total: int) -> None:
    names = [f"Explorer_{i}" for i in range(n_bots_total)] + EXTRA_OPS
    ops = [{"uuid": offline_uuid(n), "name": n, "level": 4,
            "bypassesPlayerLimit": False}
           for n in names]
    (dst / "ops.json").write_text(json.dumps(ops, indent=2))


def stage_server_dir(seed: int, port: int, n_bots_total: int) -> Path:
    src = ROOT / "mc-server"
    dst = ROOT / f"mc-server-test{seed}"
    if not dst.exists():
        print(f"[setup] {dst.name} (seed={seed} port={port})")
        dst.mkdir()
        # Copy (not symlink — Windows requires admin or developer mode).
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
    # Always rewrite ops.json — needs all bot UUIDs so they can /tp.
    write_ops_json(dst, n_bots_total)
    return dst


def wait_for_done(log_path: Path, label: str) -> None:
    deadline = time.time() + SERVER_READY_TIMEOUT_S
    while time.time() < deadline:
        if log_path.exists() and "Done" in log_path.read_text(errors="ignore"):
            print(f"[ready] {label}")
            return
        time.sleep(2)
    raise TimeoutError(f"{label} not ready within {SERVER_READY_TIMEOUT_S}s")


def spawn(cmd: list[str], log: Path, *, env=None, cwd=None) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(cmd, stdout=log.open("w"),
                         stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None)
    PROCS.append(p)
    return p


def main() -> None:
    LOGS.mkdir(exist_ok=True)

    for s in SEEDS:
        if not (ROOT / "data" / f"biomes_{s}.npz").exists():
            sys.exit(f"missing data/biomes_{s}.npz — run "
                     f"`python3 tools/extract_biomes.py --seed {s}` first")

    n_bots_total = len(SEEDS) * BOTS_PER_SERVER
    servers = []
    for i, seed in enumerate(SEEDS):
        port = BASE_MC_PORT + i
        d = stage_server_dir(seed, port, n_bots_total)
        servers.append((seed, port, d))

    for seed, _, d in servers:
        log = LOGS / f"server_{seed}.log"
        print(f"[server] booting {d.name}")
        spawn(["java", "-Xmx4G", "-Xms1G", "-jar", "paper.jar", "nogui"],
              log, cwd=d)
    for seed, _, _ in servers:
        wait_for_done(LOGS / f"server_{seed}.log", f"server {seed}")
    print()
    print("=== Server endpoints (connect with vanilla Minecraft client) ===")
    for seed, port, _ in servers:
        print(f"  seed={seed}  ->  localhost:{port}")
    print()

    n_bots = 0
    for s_idx, (_, port, _) in enumerate(servers):
        for b in range(BOTS_PER_SERVER):
            bot_id = s_idx * BOTS_PER_SERVER + b
            env = os.environ.copy()
            env["MC_PORT"] = str(port)
            env["DISPERSE_N"] = str(BOTS_PER_SERVER)
            spawn(["node", "bot/bridge.js", str(bot_id)],
                  LOGS / f"bot_{bot_id}.log", env=env, cwd=ROOT)
            n_bots += 1
            time.sleep(0.5)
    # Each bot takes SPAWN_CHUNK_WAIT_MS + DISPERSE_WAIT_MS (=5s) plus
    # connect time before it can take an action; pad to 15s to be safe.
    print(f"[bots] {n_bots} bridges spawning; settling {SETTLE_S} s")
    time.sleep(SETTLE_S)

    for policy in POLICIES:
        print(f"[run] policy={policy} start={time.strftime('%H:%M:%S')}")
        evals: list[subprocess.Popen] = []
        for s_idx, (seed, _, _) in enumerate(servers):
            for b in range(BOTS_PER_SERVER):
                bot_id = s_idx * BOTS_PER_SERVER + b
                log = LOGS / f"eval_{policy}_{seed}_{b}.log"
                evals.append(spawn(
                    [sys.executable, "eval.py", "--policy", policy,
                     "--seed", str(seed), "--bot-id", str(bot_id),
                     "--episode", str(b), "--budget-s", str(BUDGET_S)],
                    log, cwd=ROOT))
        for p in evals:
            p.wait()
        print(f"[done] policy={policy} end={time.strftime('%H:%M:%S')}")

    n_results = len(list((ROOT / "results").glob("*.json"))) if (ROOT / "results").exists() else 0
    print(f"[complete] {n_results} result files in results/")


if __name__ == "__main__":
    main()
