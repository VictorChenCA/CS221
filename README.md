# CS221 project

Minecraft agent that tries to visit as many distinct biomes as possible
within a 10-minute time budget. RL with linear function approximation,
compared against a random walk and a frontier-based baseline.

## setup

```
npm install
pip install -r requirements.txt
```

Needs a Minecraft 1.20 Java server running locally on the default port.

## running

Start the bridge:

```
node bot/bridge.js
```

Then in another shell:

```
python eval.py --policy qlearn --seed 12345
```

Policies: `random`, `frontier`, `qlearn`, `oracle`.

## layout

- `bot/` — mineflayer side (Node)
- `agent/` — learning + policies (Python)
- `oracle.py` — offline max-coverage planner
- `eval.py` — runs a policy on a seed, reports metrics
- `seeds.txt` — train/test seeds
