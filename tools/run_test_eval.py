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
# Test seeds: proposal defaults [123, 456, 789]; override via SEEDS=11111,22222,...
SEEDS = [int(s) for s in os.environ.get("SEEDS", "123,456,789").split(",")]
POLICIES = ["random", "frontier", "oracle"]
BOTS_PER_SERVER = 5
BUDGET_S = 600
# Lets us stack episodes across sequential runs of this driver:
# run 1 with EPISODE_OFFSET=0 writes ep ids 0..(BOTS_PER_SERVER-1),
# run 2 with EPISODE_OFFSET=5 writes ep ids 5..(BOTS_PER_SERVER-1+5), etc.
EPISODE_OFFSET = int(os.environ.get("EPISODE_OFFSET", "0"))
# Number of parallel MC servers per seed. With SEED_INSTANCES=2, each
# of the three test seeds runs on two servers (=6 servers total), giving
# n=10 episodes per seed at BOTS_PER_SERVER=5. Spreads chunk-gen load so
# we don't bunch 10 bots onto one server (keepalive risk).
SEED_INSTANCES = int(os.environ.get("SEED_INSTANCES", "1"))
BASE_MC_PORT = 25565
SETTLE_S = int(os.environ.get("SETTLE_S", "90"))
SERVER_READY_TIMEOUT_S = 360
# Restart MC servers between policies. Default off. Set RESTART_SERVERS=1
# to enable — fixes the 'random's accumulated chunk-gen GC pressure
# kills the next policy's bots at settle time' cascade we saw in v17/v18.
RESTART_SERVERS = os.environ.get("RESTART_SERVERS", "0") == "1"
# JVM heap: lower when we're running many servers in parallel to fit
# in typical 32GB laptop RAM. 6 servers × 4G = 24G.
JVM_XMX = os.environ.get("JVM_XMX", "6G")
JVM_XMS = os.environ.get("JVM_XMS", "2G")

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


def stage_server_dir(seed: int, instance: int, port: int, n_bots_total: int) -> Path:
    src = ROOT / "mc-server"
    # Distinct dir per (seed, instance) so duplicate-seed servers don't
    # share world data. instance=0 keeps the legacy dir name.
    suffix = "" if instance == 0 else f"-i{instance}"
    dst = ROOT / f"mc-server-test{seed}{suffix}"
    fresh = not dst.exists()
    if fresh:
        print(f"[setup] {dst.name} (seed={seed} port={port})")
        dst.mkdir()
        # Copy (not symlink — Windows requires admin or developer mode).
        shutil.copy(src / "paper.jar", dst / "paper.jar")
        for fname in ("eula.txt", "bukkit.yml", "spigot.yml",
                      "commands.yml", "help.yml", "permissions.yml"):
            if (src / fname).exists():
                shutil.copy(src / fname, dst / fname)
        if (src / "config").exists():
            shutil.copytree(src / "config", dst / "config")
    # Always rewrite server.properties so view-distance / sim-distance
    # tweaks in the source propagate to existing test dirs (otherwise an
    # old high view-distance keeps causing keepalive kicks).
    text = (src / "server.properties").read_text()
    text = re.sub(r"^level-seed=.*$", f"level-seed={seed}", text, flags=re.M)
    text = re.sub(r"^server-port=.*$", f"server-port={port}", text, flags=re.M)
    (dst / "server.properties").write_text(text)
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=str, default=",".join(POLICIES),
                    help="comma-separated policies to eval (random/frontier/qlearn/oracle)")
    ap.add_argument("--weights", type=str, default="weights/qlearn.npz",
                    help="qlearn weights path (only used when 'qlearn' is in --policies)")
    ap.add_argument("--budget-s", type=int, default=BUDGET_S,
                    help=f"episode budget seconds (default {BUDGET_S})")
    args = ap.parse_args()
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    LOGS.mkdir(exist_ok=True)

    for s in SEEDS:
        if not (ROOT / "data" / f"biomes_{s}.npz").exists():
            sys.exit(f"missing data/biomes_{s}.npz — run "
                     f"`python3 tools/extract_biomes.py --seed {s}` first")
    if "qlearn" in policies and not (ROOT / args.weights).exists():
        sys.exit(f"--policies includes 'qlearn' but {args.weights} doesn't exist. "
                 f"Train first with `python3 train.py`.")

    # Expand SEEDS into one entry per (seed, instance) — duplicates show
    # up multiple times in this list so the loop body doesn't have to
    # know about instances.
    server_specs = [(s, i) for s in SEEDS for i in range(SEED_INSTANCES)]
    n_bots_total = len(server_specs) * BOTS_PER_SERVER
    servers = []  # list of (seed, instance, port, dir)
    for srv_idx, (seed, inst) in enumerate(server_specs):
        port = BASE_MC_PORT + srv_idx
        d = stage_server_dir(seed, inst, port, n_bots_total)
        servers.append((seed, inst, port, d))

    server_procs: list[subprocess.Popen] = []

    def boot_servers(policy_tag: str = "") -> None:
        """Boot fresh MC servers (or initial boot). Wait for all 'Done'."""
        server_procs.clear()
        suffix = f"_{policy_tag}" if policy_tag else ""
        for seed, inst, _, d in servers:
            log = LOGS / f"server_{seed}_i{inst}{suffix}.log"
            print(f"[server] booting {d.name}")
            p = spawn(["java", f"-Xmx{JVM_XMX}", f"-Xms{JVM_XMS}",
                      "-jar", "paper.jar", "nogui"], log, cwd=d)
            server_procs.append(p)
        for seed, inst, _, _ in servers:
            wait_for_done(LOGS / f"server_{seed}_i{inst}{suffix}.log",
                          f"server {seed}_i{inst}")

    def kill_all_servers() -> None:
        """Stop all MC servers — clears JVM heap, frees chunk-gen state.
        Used between policies to prevent GC-pause keepalive cascades."""
        for p in server_procs:
            if p.poll() is None:
                p.terminate()
        for p in server_procs:
            try:
                p.wait(timeout=30)  # Paper takes a few seconds to save chunks
            except subprocess.TimeoutExpired:
                p.kill()
        time.sleep(5)

    boot_servers()
    print()
    print("=== Server endpoints (connect with vanilla Minecraft client) ===")
    for seed, inst, port, _ in servers:
        print(f"  seed={seed} inst={inst}  ->  localhost:{port}")
    print()

    bot_procs: list[subprocess.Popen] = []

    def spawn_all_bots(policy_tag: str) -> None:
        """Spawn one bot bridge per (server, slot). Tag logs by policy so
        bots from a later policy don't clobber the prior policy's logs."""
        bot_procs.clear()
        for s_idx, (_, _, port, _) in enumerate(servers):
            for b in range(BOTS_PER_SERVER):
                bot_id = s_idx * BOTS_PER_SERVER + b
                env = os.environ.copy()
                env["MC_PORT"] = str(port)
                env["DISPERSE_N"] = str(BOTS_PER_SERVER)
                p = spawn(["node", "bot/bridge.js", str(bot_id)],
                          LOGS / f"bot_{policy_tag}_{bot_id}.log",
                          env=env, cwd=ROOT)
                bot_procs.append(p)
                time.sleep(0.5)
        print(f"[bots] {len(bot_procs)} bridges spawning ({policy_tag}); "
              f"settling {SETTLE_S} s")
        time.sleep(SETTLE_S)

    def kill_all_bots() -> None:
        """Terminate every bridge so the next policy gets fresh bots
        (last policy's kicks don't carry over)."""
        for p in bot_procs:
            if p.poll() is None:
                p.terminate()
        for p in bot_procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        # Bridges hold TCP listeners; give the OS a moment to free ports.
        time.sleep(3)

    for policy_idx, policy in enumerate(policies):
        if RESTART_SERVERS and policy_idx > 0:
            print(f"[restart] killing MC servers for clean state before {policy}")
            kill_all_servers()
            boot_servers(policy)
        spawn_all_bots(policy)
        print(f"[run] policy={policy} start={time.strftime('%H:%M:%S')}")
        evals: list[subprocess.Popen] = []
        for s_idx, (seed, inst, _, _) in enumerate(servers):
            for b in range(BOTS_PER_SERVER):
                bot_id = s_idx * BOTS_PER_SERVER + b
                # Episode id groups together episodes from a given seed
                # across all its instances: inst 0 writes ep 0..4,
                # inst 1 writes ep 5..9, etc. Shifted by EPISODE_OFFSET
                # for stacking multiple sequential driver runs.
                episode = inst * BOTS_PER_SERVER + b + EPISODE_OFFSET
                log = LOGS / f"eval_{policy}_{seed}_{episode}.log"
                cmd = [sys.executable, "eval.py", "--policy", policy,
                       "--seed", str(seed), "--bot-id", str(bot_id),
                       "--episode", str(episode), "--budget-s", str(args.budget_s)]
                if policy == "qlearn":
                    cmd += ["--weights", args.weights]
                evals.append(spawn(cmd, log, cwd=ROOT))
        for p in evals:
            p.wait()
        print(f"[done] policy={policy} end={time.strftime('%H:%M:%S')}")
        kill_all_bots()

    n_results = len(list((ROOT / "results").glob("*.json"))) if (ROOT / "results").exists() else 0
    print(f"[complete] {n_results} result files in results/")
    summarize_results()


def summarize_results() -> None:
    """Aggregate results/*.json into a printed table + results/summary.txt."""
    import json
    from collections import defaultdict
    import statistics as stats

    results_dir = ROOT / "results"
    if not results_dir.exists():
        return
    rows = defaultdict(list)
    for f in sorted(results_dir.glob("*.json")):
        r = json.loads(f.read_text())
        rows[(r["policy"], r["seed"])].append(r)
    if not rows:
        return

    lines = []
    lines.append(f"{'policy':<10} {'seed':>5} {'n':>2} {'ub_mean':>8} {'ub_max':>6} {'n_act':>6}")
    lines.append("-" * 45)
    for (p, s), rs in sorted(rows.items()):
        ubs = [r["unique_biomes"] for r in rs]
        ns = [r["n_actions"] for r in rs]
        lines.append(f"{p:<10} {s:>5} {len(rs):>2} {sum(ubs)/len(ubs):>8.2f} "
                     f"{max(ubs):>6} {sum(ns)/len(ns):>6.1f}")
    lines.append("")
    lines.append("=== aggregate by policy ===")
    by_policy = defaultdict(list)
    for (p, _), rs in rows.items():
        by_policy[p].extend(rs)
    for p, rs in sorted(by_policy.items()):
        ubs = [r["unique_biomes"] for r in rs]
        ent = [r["biome_entropy"] for r in rs]
        cov = [r["position_coverage"] for r in rs]
        sd = stats.stdev(ubs) if len(ubs) > 1 else 0.0
        lines.append(f"  {p:<10} n={len(rs)} ub: mean={sum(ubs)/len(ubs):.2f} "
                     f"sd={sd:.2f} max={max(ubs)} "
                     f"| biome_ent={sum(ent)/len(ent):.2f} "
                     f"| cov={sum(cov)/len(cov):.4f}")

    # Termination breakdown — distinguishes real full-budget episodes
    # from ones cut short by death/laptop-closed/etc. If you see >0
    # dead_mid_run or laptop-closed-style truncated elapsed_s, the
    # ub_mean above is undercounting that policy's true performance.
    lines.append("")
    lines.append("=== termination breakdown ===")
    for p, rs in sorted(by_policy.items()):
        term_counts: dict[str, int] = defaultdict(int)
        elapsed: list[float] = []
        for r in rs:
            term_counts[r.get("termination", "unknown")] += 1
            if r.get("elapsed_s") is not None:
                elapsed.append(r["elapsed_s"])
        mean_elapsed = (sum(elapsed) / len(elapsed)) if elapsed else 0.0
        breakdown = " ".join(f"{k}={v}" for k, v in sorted(term_counts.items()))
        lines.append(f"  {p:<10} elapsed_mean={mean_elapsed:>6.1f}s   {breakdown}")

    summary = "\n".join(lines)
    print()
    print(summary)
    (results_dir / "summary.txt").write_text(summary + "\n")
    print(f"[summary] written to {results_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
