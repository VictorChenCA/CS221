# CS221 project

Minecraft agent that tries to visit as many distinct biomes as possible
within a 10-minute time budget. RL with linear function approximation,
compared against a random walk and a frontier-based baseline.

## setup

```
npm install
pip install -r requirements.txt
```

Needs a PaperMC 1.20.1 server running locally on `:25565`.

## running

Spawn bots (one process per bot, ports `9000+id`):

```
node bot/spawn.js 10
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
