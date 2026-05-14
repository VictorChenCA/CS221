# Setup

We test on Mac first, then Windows. Standardize on Minecraft **1.20.1**
running on **PaperMC** (drop-in replacement for the vanilla server).

## Paper vs vanilla server

Same protocol, same client, same world format — Paper is the vanilla
`server.jar` with performance and config patches. The bots cannot tell
the difference at the wire level.

What we get from Paper that vanilla doesn't give us:

- **Throughput.** Paper rewrites the chunk loader, entity tracker, and
  tick scheduler. With 10 bots loading chunks in parallel, vanilla
  drops to single-digit TPS; Paper holds ~20.
- **Async chunk loading.** Bot spawns and teleports don't stall the
  main thread.
- **`paper.yml` / `spigot.yml`.** Per-world tuning knobs (view distance,
  mob spawn ranges, entity activation range) that vanilla hardcodes.
- **`/timings` and `/mspt`.** Built-in profiling — useful when a run
  starts lagging and we need to know whether it's the agent or the
  server.
- **Watchdog tolerance.** Vanilla kills the server if a tick takes
  >60s; Paper's threshold is configurable. Helpful during heavy
  pathfinder calls.

What we lose: nothing relevant. Paper is plugin-compatible (Bukkit
API), but we don't load plugins. Worldgen, biomes, block behavior, and
mob AI are identical to vanilla 1.20.1, which matters because
`oracle.py` reads biomes straight from the region files.

## 1. Minecraft Java Edition (optional, for visual debugging)

The bot connects to the local server directly, not through a client, so a
client install is only needed if you want to see what the bot sees.

- Mac & Windows: install the Minecraft Launcher from minecraft.net,
  launch 1.20.1 once to download assets.

## 2. Java 21

- Mac: `brew install --cask temurin`
- Windows: install Adoptium Temurin 21 MSI from adoptium.net
  (check "Set JAVA_HOME" and "Add to PATH").
- Verify: `java -version` → 21.x.

## 3. PaperMC server (1.20.1)

- Download `paper-1.20.1-<build>.jar` from papermc.io/downloads.
- Put it in `~/mc-server/paper.jar` (Mac) or `C:\mc-server\paper.jar` (Windows).
- First run:
  - `java -Xmx4G -jar paper.jar nogui`
  - Stops with EULA error. Edit `eula.txt` → `eula=true`.
- Edit `server.properties`:
  - `online-mode=false` (bots can join without auth)
  - `max-players=15` (10 bots + headroom)
  - `level-seed=1111` (change per experiment)
  - `gamemode=adventure`
  - `difficulty=peaceful` (no mobs — keeps bots alive)
  - `view-distance=12`
  - `simulation-distance=10`
  - `spawn-protection=0`
- Restart with 6G heap for the real runs:
  - Mac: `java -Xmx6G -Xms2G -jar paper.jar nogui`
  - Windows: same in PowerShell, or via a `start.bat`.
- Should listen on `localhost:25565`.

## 4. Node.js 20 LTS

- Mac: `brew install node@20` then `brew link --overwrite --force node@20`
- Windows: install Node 20 LTS MSI from nodejs.org.
- Verify: `node -v` (20.x), `npm -v`.
- In repo: `npm install` (pulls `mineflayer`, `mineflayer-pathfinder`,
  `minecraft-data`, `vec3`).

## 5. Python 3.11

- Mac: `brew install python@3.11`
- Windows: install Python 3.11 from python.org ("Add Python to PATH" checked).
- In repo:
  - Mac: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
  - Windows: `python -m venv .venv && .venv\Scripts\Activate.ps1 && pip install -r requirements.txt`
  - Windows note: if activation is blocked, run once:
    `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

`anvil-parser` is for the offline biome extractor used by `oracle.py`.

## 6. Git

- Mac: `brew install git` or `xcode-select --install`.
- Windows: Git for Windows from git-scm.com (includes Git Bash).

## 7. Editor / shell

- VS Code + Python + ESLint extensions on both OSes.
- Windows: prefer PowerShell 7 (`winget install Microsoft.PowerShell`).

## 8. Ports / firewall

- `25565` — Minecraft server.
- `9000`–`9009` — one per bot bridge (we run up to 10 bots).
- Mac: `lsof -iTCP:25565 -sTCP:LISTEN`
- Windows: `Get-NetTCPConnection -LocalPort 25565`
- Allow `java`, `node`, `python` through the firewall when first prompted.

## 9. Launch order (every session)

Three terminals, in order:

1. **Server** — `java -Xmx6G -Xms2G -jar paper.jar nogui` from `~/mc-server`.
   Wait for `Done!`.
2. **Bots** — from the repo: `node bot/spawn.js 10`. Spawns 10 bridges,
   staggered 2s apart, each listening on `9000+id`.
3. **Agent** — from the repo: `python eval.py --policy random --seed 1111`.

## 10. Smoke test before MDP work

In order, on Mac first. Only move to Windows once everything passes here.

1. Server starts, `Done!` appears.
2. `node bot/bridge.js 0` (single bot) — bot joins, prints
   `bot 0 listening on 9000`. Confirm in server logs that `Explorer_0`
   joined.
3. From a Python REPL:
   ```python
   from agent.env import Env
   e = Env(port=9000)
   print(e.step(0))   # send action 0, print observation
   ```
   Should return a dict with `biomeName`, position, etc.
4. Run `node bot/spawn.js 3` — three bots join without crashing the server.

Once all four pass, start filling in `agent/env.py`, `agent/features.py`,
and `agent/qlearn.py`.

## Not installed

- No GPU stack (linear Q-learning runs fine on CPU).
- No Docker.
- No MineRL / MineDojo (no pixel observations).
- No Forge / Fabric mods — Paper is plugin-compatible but we use vanilla
  behavior only.
