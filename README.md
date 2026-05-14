# CS221 project

Minecraft agent that tries to visit as many distinct biomes as possible
within a 10-minute time budget. RL with linear function approximation,
compared against a random walk and a frontier-based baseline.

## world model

The MDP discretizes the world into a 2D grid where **each cell is a 4×4
block section** (the unit Minecraft 1.18+ uses to store biomes). The
agent's known-world state is a sparse map `(cellX, cellZ) → biomeId`,
filled in as the bot walks. Two cells are "the same biome" iff their
biome ids match, so visit-counting and frontier reasoning happen at this
resolution.

## action / locomotion contract

Actions are **8 compass directions × a configurable hop distance**
(typically K cells = 4K blocks). The bot is considered to have arrived
when it lands anywhere inside the target 4×4 cell; the bridge uses
`GoalNear(x, y, z, range=3)` so `goal_reached` fires reliably on a
successful walk, and `stuck=true` only when pathfinding actually fails
or hits the per-action timeout. Block-level locomotion (jumping,
climbing, swimming) is delegated to `mineflayer-pathfinder`; the agent
never reasons below the 4×4 cell level.

## setup

```
npm install
pip install -r requirements.txt
```

Needs a PaperMC 1.20.1 server running locally on `:25565`.

## running

Start the server and spawn bots (one process per bot, ports `9000+id`):

```
./mc-server/start.sh        # or .bat on Windows
./bot/start.sh 10           # or .bat on Windows; arg = N bots, default 10
```

Then in another shell:

```
python eval.py --policy qlearn --seed 1111
```

Policies: `random`, `frontier`, `qlearn`, `oracle`.

## layout

- `bot/` — mineflayer side (Node)
- `agent/` — learning + policies (Python)
- `oracle.py` — offline max-coverage planner
- `eval.py` — runs a policy on a seed, reports metrics
- `seeds.txt` — train/test seeds
