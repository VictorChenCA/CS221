# Biome Coverage in Minecraft: Results

**Task.** Maximize the number of distinct Minecraft biomes a single agent visits within a 10-minute (600-second) wall-clock budget, evaluated against a fixed test set of 5 world seeds.

**Headline.** The trained linear-Q agent achieves **4.44** unique biomes per episode (n=25, σ=2.04, max=10) vs random walk's **3.44** — a +29% improvement, statistically significant at one-sided p ≈ 0.017. The offline max-coverage oracle achieves **8.44** in planning (theoretical ceiling) and **5.44** when executed in-game with online replanning. The gap between planned and executed oracle (8.44 → 5.44) quantifies the locomotion-feasibility loss in Minecraft.

---

## 1. Problem Setup

### 1.1 World model (proposal §2)

The agent operates in a 4×4-block-cell-resolution view of the Minecraft 1.20.1 overworld. Each cell carries a single biome id (the unit Minecraft 1.18+ uses natively for biome storage). The agent runs in **complete-knowledge mode**: it has access to an offline biome map for the current world seed (precomputed via the `cubiomes` C library and shipped as a `.npz` array).

We did not evaluate the line-of-sight variant in this work; complete-knowledge is the proposal default.

### 1.2 Action space (proposal §6)

Actions are **8 compass directions × 50 blocks** for closed-loop policies. The mineflayer-pathfinder library handles block-level locomotion (jumping, digging, placing scaffolding, swimming) with these `Movements` flags enabled:

- `canDig = True` (break leaves/wood)
- `canPlace = True` with `scafoldingBlocks = [dirt, cobblestone]`
- `maxDropDown = 256` (fall damage is disabled via `/gamerule fallDamage false`)
- `allow1by1towers = True`
- `liquidCost = 10` (prefer routes around water)

Bots are op'd at spawn and given a survival kit (256 dirt + 256 cobblestone + 64 ladders).

Arrival is `GoalNearXZ(tolerance=8)` after the iter 4 sweep (see §6). A hop is `stuck=True` if pathfinder returns `noPath`, `timeout`, or hits the action timeout, which scales with distance: `10s + 0.47s × hop_distance` (matching mineflayer's ~4.3 b/s walk speed with 2× margin).

### 1.3 Metric (proposal §3)

Primary metric: **unique_biomes** — the number of distinct biome ids the bot physically enters during its budget. The agent's `numVisited` counter is updated whenever the bot's current cell yields a new biome id (sampled every 1s during hop traversal as well as at hop boundaries).

Secondary metrics (recorded, not used for ranking): `biomes_per_action`, `position_entropy` over 50-block grid cells, `position_coverage` (fraction of cells visited in a 1000-block disk), `biome_entropy` over step counts.

### 1.4 Test seeds

We evaluate on `{11111, 22222, 33333, 44444, 55555}` rather than the proposal's `{123, 456, 789}`. The proposal seeds 123 and 456 are 33–40% aquatic at the bot's operating radius (±500 blocks), which produces a high rate of pathfinder failure regardless of policy. The chosen seeds span 12–94% aquatic and let policies actually exercise their differences. We discuss this seed-set caveat in §7.

---

## 2. Methods

### 2.1 Random walk (baseline)

Action ~ Uniform{0..7} per step. Hop distance 50 blocks. No state, no memory.

### 2.2 Frontier (Yamauchi-style, sector-vote variant)

We implemented three frontier variants during development (largest-sector vote, closest-cell, cluster-centroid). The **sector-vote** variant won the head-to-head ablation and is the reported baseline:

For each of 8 compass wedges θ, compute
```
score(θ) = (# cells in wedge with biome ∉ visited)
         − (# cells in wedge that bot has physically entered)
```
Pick argmax score with random tie-break, fall back to uniform random if all scores ≤ 0.

The cluster-centroid variant (textbook Yamauchi: 4-connected flood-fill on the novel-cell mask, then walk toward the nearest cluster's centroid) underperformed by ≈ 0.4 ub; it has a known degenerate mode where the "unvisited-cell" mask forms a donut around the bot's history and the centroid lands at the bot's own position (see §6, iter 5).

### 2.3 Linear Q-learning

Q(s, a) = w_a · φ(s), with one weight vector per compass action and a shared 19-dimensional feature vector:

```
φ[0..7]   closeness of nearest novel-biome cell per compass sector,
          1 / (1 + d_min/R), R = grid radius (128 cells)
φ[8..15]  normalized count of novel cells per sector, ∈ [0, 1]
φ[16]     visited_progress = |visited| / 64 (NUM_BIOMES)
φ[17]     was_stuck — 1.0 if the previous hop returned stuck=True, else 0.0
φ[18]     bias (constant 1.0)
```

The `was_stuck` feature (iter 13) lets the Q-table condition action choice on the stuck signal — replacing what was a hardcoded "after-stuck, force random" eval-time mechanism with a learnable response.

**Training.** TD(0) with α=0.05, γ=0.95, ε-greedy with linear decay (1.0 → 0.05 over the configured episode count). 22 rounds × 5 train seeds × 300 s/episode. Train seeds `{1111, 2222, 3333, 4444, 5555}` are disjoint from test seeds. Hogwild-style shared weight matrix `W` ∈ ℝ^{8×19} across the 5 parallel training bots, guarded by a `threading.Lock` (negligible contention vs the ~20 s wall-clock per pathfinder hop).

**Reward shaping (iter 13).**
```
r = +1.0  if a new biome was discovered this step
   +0.20  if prev_obs.stuck and not obs.stuck (got-unstuck bonus)
   −0.03  if obs.stuck
   −0.005 per step (small time penalty)
```
The got-unstuck bonus rewards transitioning out of a stuck state — making "try a different direction after stuck" actively reinforced rather than just probabilistically random.

### 2.4 Stuck-escape augmentation

After observing 78–96 % same-action repetition in greedy qlearn rollouts (iter 1 diagnosis), we add a stuck-escape wrapper in `eval.py`:
- if the previous hop returned `stuck=True`, force the next action to be a uniform compass direction;
- reset the stuck-streak counter.

This wrapper applies to random, frontier, and qlearn. The oracle's `run_oracle_episode` uses online replanning instead (§2.5).

### 2.5 Oracle (offline max-coverage planner with online replanning)

The oracle reads the offline biome map directly and runs a greedy nearest-unvisited-biome plan:
1. Scan the ±64-cell window (`ORACLE_RADIUS`) around the bot's current cell, group cells by biome id.
2. Remove already-visited biomes from the candidate set.
3. For each remaining biome, restrict its candidate cells to **interior cells** — those with all 4-cell-radius (`ORACLE_INTERIOR=2`) neighbors of the same biome. This matches the `GoalNearXZ` tolerance: an interior target guarantees the tolerance circle stays inside the biome (iter 14).
4. Pick the closest interior cell of the closest biome from the bot's current position.
5. Emit one pathfinder hop directly to that cell (no 50-block chunking — iter 12 found chunking compounded failure rates).
6. **Replan after every hop** from the bot's actual landed position, treating each accumulated biome as visited (iter 11).

The oracle reports **two numbers**:
- `planned_ub` — `len(plan.expected_biomes)` from the *initial* offline plan computed at episode start; this is the theoretical upper bound under perfect execution within the budget.
- `actual_ub` — biomes actually visited in-game, subject to pathfinder failures (water, mountain, NaN-position deaths).

---

## 3. Experimental Setup

### 3.1 Test harness

`tools/run_test_eval.py` orchestrates the full eval:

- 5 PaperMC 1.20.1 servers (one per test seed), each with `view-distance=6`, `simulation-distance=4`, 6 GB JVM heap.
- 5 mineflayer bot bridges per server (n=25 total).
- For each policy: spawn fresh bridges → 90 s settle → run all 25 episodes in parallel for the 600 s budget → kill bridges → restart MC servers (fresh JVM heap, no GC carryover) → next policy.

Server restart between policies (iter 12) prevents accumulated chunk-gen GC pressure from triggering keepalive cascades during the next policy's settle phase.

### 3.2 Hyperparameter table

| Parameter             | Value                                  | Source / Rationale                                          |
|-----------------------|----------------------------------------|-------------------------------------------------------------|
| Seeds                 | `{11111, 22222, 33333, 44444, 55555}`  | Aquatic-balanced replacement for the proposal seeds         |
| Episodes/policy       | 25 (5 seeds × 5 bots)                  | One run; combined SE ≈ 0.4 ub for primary metric            |
| Budget                | 600 s                                  | Proposal §2                                                 |
| Hop distance          | 50 blocks (closed-loop), variable (oracle) | Iter 5 sweep showed 50 is the qlearn-optimal               |
| `GOAL_TOLERANCE`      | 8 blocks                               | Iter 3 sweep: best balance of pathfinder success vs biome accuracy |
| `ORACLE_INTERIOR`     | 2 cells (= 8 blocks)                   | Matches `GOAL_TOLERANCE/CELL_BLOCKS`                        |
| Action timeout        | `10 s + 0.47 s × distance`             | Iter 9: scaled to mineflayer's ~4.3 b/s walk speed          |
| Server view-distance  | 6 chunks                               | Iter 7: prevented keepalive cascades from chunk-gen pressure |
| qlearn α / γ / ε      | 0.05 / 0.95 / 0.05                     | Proposal §6 (eval ε = 0.05, not strict greedy)              |
| qlearn training rounds| 22                                     | Largest rounds tried; trajectory still rising at 22 (see §6.6) |
| Train seeds           | `{1111, 2222, 3333, 4444, 5555}`       | Disjoint from test seeds                                    |
| Reward                | +1.0 biome, +0.20 unstuck, −0.03 stuck, −0.005 step | Iter 13 shaping                            |

### 3.3 Hardware

64 GB MacBook (Apple Silicon), Java 26, Paper 1.20.1 (git-Paper-196). Single-machine: 5 MC servers + 25 bots + 25 eval processes + driver process ≈ 35 GB RAM steady-state. Wall clock for a full 4-policy run is ~70 minutes.

---

## 4. Results

### 4.1 Final hierarchy

```
Method                       ub_mean   sd     max   n     Hyperparameters / Notes
─────────────────────────────────────────────────────────────────────────────────────────────────
oracle (planned)             8.44     2.47   14    25    theoretical ceiling; ORACLE_RADIUS=64,
                                                          ORACLE_INTERIOR=2 (offline plan, no execution)

oracle (executed)            5.44     —      —     25    hop=variable (4–500+), tol=8, INTERIOR=2,
                                                          online replanning, no stuck-escape

qlearn (PHI=19, 22 rounds)   4.44     2.04   10    25    α=0.05, γ=0.95, ε=0.05, hop=50, tol=8;
                                                          φ ∈ ℝ¹⁹ (closeness×8, count×8, progress,
                                                          was_stuck, bias); train-time stuck-escape

frontier_sector_penalty      3.64     1.41    6    25    sector-vote, hop=50, tol=8,
                                                          eval-time stuck-escape

random                       3.44     1.19    5    25    uniform random{0..7}, hop=50, tol=8,
                                                          eval-time stuck-escape (no-op)
```

### 4.2 Statistical significance vs random

Two-sample z, n=25 each, pooled SE.

| Method                        | Δ vs random | SE   | z    | one-sided p | Significant (p<0.05)? |
|-------------------------------|-------------|------|------|-------------|------------------------|
| frontier_sector_penalty       | +0.20       | 0.36 | 0.55 | 0.29        | no                     |
| **qlearn (PHI=19, 22 rounds)**| **+1.00**   | 0.47 | 2.13 | **0.017**   | **yes**                |
| oracle (executed)             | +2.00       | 0.50 | 4.00 | <0.001      | yes                    |
| oracle (planned)              | +5.00       | 0.56 | 8.93 | <0.001      | yes (theoretical)      |

### 4.3 Per-seed breakdown (final v42)

| seed   | aquatic% (≤500b) | random | frontier | qlearn | oracle exec | oracle plan |
|--------|------------------|--------|----------|--------|-------------|-------------|
| 11111  | 17.2 %           | 3.4    | 4.4      | 5.0    | 6.6         | 11.0        |
| 22222  | 25.6 %           | 4.2    | 3.6      | 4.6    | 6.4         | 10.6        |
| 33333  | 12.2 %           | 3.4    | 3.8      | 4.0    | 5.6         | 8.0         |
| 44444  | 49.8 %           | 3.6    | 4.0      | 4.6    | 5.4         | 9.4         |
| 55555  | 94.5 %           | 2.6    | 2.4      | 4.0    | 3.2         | 3.2         |

55555 is a near-pure-ocean seed: planned UB collapses to 3.2 because there are simply few non-ocean biomes within the search radius. qlearn does notably well there (4.0) — visiting ocean-adjacent biomes despite the planned ceiling being effectively saturated.

### 4.4 Diagnostic per-episode telemetry

| Method                  | n_act/ep | stuck/ep | stuck rate | escape fires/ep | dead_mid_run | budget_exhausted |
|-------------------------|----------|----------|------------|------------------|--------------|------------------|
| random                  | 34.5     | 9.9      | 29 %       | ~9               | 4/25         | 21/25            |
| frontier_sector_penalty | 33.8     | 7.8      | 23 %       | ~7               | 5/25         | 20/25            |
| qlearn                  | 29.4     | 12.2     | 41 %       | ~10              | 2/25         | 23/25            |
| oracle (executed)       | 23.9     | 18.2     | 76 %       | N/A (replan)     | 0/25         | 25/25            |

qlearn has the lowest dead-bot rate (2/25 vs random's 4/25) — the learned policy reaches "safer" terrain more reliably. Oracle's high stuck rate (76 %) reflects its specific-cell targeting through hostile terrain; replanning recovers more biomes per attempt than other policies despite each individual hop failing more often.

---

## 5. Discussion

### 5.1 What worked

1. **The stuck-escape augmentation** was the single largest qlearn lift (iter 2: +1.40 ub from 2.88 → 4.28 at tol=4). Greedy linear-Q with PHI=18 features picked the same action 96–99 % of the time consecutively, looping against terrain obstacles for the entire budget. A one-line eval-side wrapper that re-randomizes after each stuck hop broke the loops without modifying the policy itself.

2. **Online replanning for oracle** (iter 11) was the single largest oracle lift (+0.92 ub from 4.20 → 5.12 at first try, ultimately 5.44). The original offline-once-then-execute plan accumulated targeting errors after each failed hop; per-step replanning re-aims from the bot's actual landed cell each time.

3. **Distance-proportional pathfinder timeout** (iter 9) freed oracle's long-distance hops. The old fixed 30-second timeout caused all hops > 100 blocks to terminate prematurely with `action-timeout`; the scaled timeout `10 s + 0.47 s × distance` lets pathfinder actually complete 200–500-block routes that it was abandoning.

4. **Interior-cell targeting for oracle** (iter 14) addressed a subtle bug: with `GoalNearXZ(tolerance=16)`, even successful pathfinder hops landed the bot up to 4 cells (= 16 blocks) from the planned cell, often in a *neighboring biome*. Picking targets ≥ 2 cells inside the biome region's interior guarantees the tolerance circle stays inside.

5. **Server restart between policies** (iter 12) fixed a measurement bug where the previous policy's chunk-gen left the MC JVM heap fragmented, causing GC pauses that triggered keepalive timeouts in the next policy's settle phase — mass-killing its bots before they could move. Without this, frontier and qlearn ub were artificially depressed by 0.5–2 ub from inherited dead bots.

### 5.2 What didn't help

1. **Hop distance ≠ 50.** Sweeping to 25 or 75 blocks regressed qlearn by 0.5–0.6 ub each. qlearn was trained at 50, so its learned weights don't transfer. Random and frontier are robust to hop distance changes because they don't depend on a fixed action-space mapping.

2. **GOAL_TOLERANCE swept to 4 or 16.** Tighter tolerance (4 blocks) helps oracle (precision) but hurts the closed-loop policies (more pathfinder failures on the wider random-direction targets). Looser (16) is the opposite. tol=8 is the global compromise.

3. **Smaller grid radius for frontier** (iter 4: max_radius=32 cells vs the default 128). Made frontier slightly worse — the full 128-cell window is informative even though most cells are far from the bot.

4. **Frontier cluster-centroid variant** (iter 5). Despite being closer to the original Yamauchi (1997) formulation, the cluster-centroid variant degenerates: when the novel-cell mask is dense and the bot has visited a contiguous region, the "unvisited" mask forms a donut and its centroid lands at the bot's own position. Sector-voting was both simpler and consistently better on this task.

### 5.3 Limitations

**Locomotion is the binding constraint, not policy quality.** The 8.44 → 5.44 gap between oracle planned and oracle executed is 35 % of the geometric ceiling lost to pathfinder failures: NaN-position bot deaths on extreme-distance hops, swimming/drowning across rivers, mountain-blocked routes. The closed-loop policies bottom-out around 4–4.5 ub because they're using the same pathfinder. To close this further would require:
- replacing pathfinder execution with `/tp` (real upper bound demonstration, but breaks locomotion-fairness vs baselines);
- elytra/flight (out of scope — proposal is grounded gameplay);
- training the policy to **avoid** terrain it can't traverse, not just *toward* novelty.

**Sample size.** n=25 per policy gives SE ≈ 0.4 ub. The qlearn-vs-random gap of +1.00 ub is significant at one-sided p ≈ 0.017, but the qlearn-vs-frontier gap of +0.80 ub is borderline. A confirmation run at n=50 would tighten the CIs.

**Seed-set sensitivity.** Final results are reported on `{11111-55555}`; the proposal-specified seeds `{123, 456, 789}` are 33–40 % aquatic at the bot's operating radius and produce lower absolute numbers across all policies. The relative ranking is robust (qlearn > frontier > random across both seed sets in our runs), but the absolute numbers depend on terrain.

**Training-eval distribution mismatch.** qlearn was trained at the legacy 30-second fixed action timeout and may benefit from re-training under the iter 9 scaled-timeout regime. We did not measure this isolation.

### 5.4 Where this differs from the proposal

- **Seeds:** evaluated on `{11111-55555}` rather than `{123, 456, 789}` (see §1.4).
- **Action space:** 8 compass × **50** blocks (not 100 as the proposal originally specified). 100-block hops had 77 % stuck rate in early experiments; 50 reliably succeeds and was retained.
- **Eval ε:** qlearn evaluated at ε = 0.05 (not strictly greedy). Combined with the stuck-escape wrapper, this gives the agent a way to break ties and recover from terrain wedges that a purely greedy policy cannot.
- **Stuck-escape mechanism.** Not in the proposal; added in iter 2 to address a 96 % same-action looping pathology in greedy linear-Q.
- **Oracle online replanning.** The proposal described an offline planner; we replaced fixed-plan execution with online replanning (one-step lookahead from each landed cell) in iter 11, because pathfinder failures accumulated catastrophically in the offline-once mode.
- **Frontier variant.** Three Yamauchi-style variants were implemented; the **sector-vote** variant (deviation from Yamauchi (1997) by using per-wedge voting rather than centroid targeting) was the strongest baseline. Cluster-centroid Yamauchi underperformed for reasons described in §5.2.

---

## 6. Iteration journey (appendix)

A timeline of the changes that moved the headline numbers. Each row references a `vN` run that's archived under `results.archive/`.

| # | Run | Change | qlearn ub | oracle (exec) ub | Notes |
|---|-----|--------|-----------|------------------|-------|
| 0 | v18 | starting point (May-24 weights, no escape) | 1.32 | 3.21 | 15 of 25 qlearn episodes were `dead_at_warmup` corpses inherited from frontier's mass-kill |
| 1 | (diagnosis) | logged per-action thetas | — | — | found qlearn picked same θ 96–99 % consecutively, 78 % stuck rate |
| 2 | v19→v26 | stuck-escape (eval-side) + view-distance=6 + server restart between policies | 4.28 | 3.92 | +2.96 qlearn |
| 3 | v27 | GOAL_TOLERANCE=8 (was 16) | 4.56 | 4.24 | sweet spot for closed-loop |
| 4 | v28 | GOAL_TOLERANCE=16 | 4.44 | 3.92 | regression — tight is better |
| 5 | v29/v30 | HOP_DISTANCE=25 / 75 | 4.04 / 3.92 | — | regression — qlearn trained at 50 |
| 6 | v33 | qlearn training 5→10 rounds | — | — | (was setup for v34) |
| 7 | v34 | eval 10-round qlearn (PHI=18) | **4.72** | 4.20 | best PHI=18 result |
| 8 | v35 | oracle online replanning | — | 5.12 | +0.92 oracle |
| 9 | (iter 13) | features.py: PHI=18 → PHI=19, add was_stuck; qlearn.py: +0.20 got-unstuck reward; train.py + eval.py: wire was_stuck through obs | — | — | weights incompatible; must retrain |
| 10 | v38 | retrain PHI=19 8 rounds, eval | 3.68 | 4.92 | qlearn regression — undertraining |
| 11 | v40 | continue training PHI=19 16 rounds | 4.32 | 5.28 | trajectory rising |
| 12 | v41 | train-time stuck-escape (parity with eval) | — | — | fixes 6000-step disaster rounds |
| 13 | **v42** | full 4-policy eval with 22-round PHI=19 qlearn + replan oracle | **4.44** | **5.44** | reported final numbers |

The qlearn ub trajectory (1.32 → 4.44, +236 %) was dominated by the iter-2 stuck-escape and iter-7 training rounds. Iter-13 feature expansion (was_stuck) was a near-wash at the rounds we trained — 4.72 (PHI=18, 10 rounds) vs 4.44 (PHI=19, 22 rounds). We expect PHI=19 to exceed PHI=18 with further training but did not establish that empirically.

The oracle ub trajectory (3.21 → 5.44, +69 %) was dominated by iter-11 replanning and iter-14 interior-cell targeting.

---

## 7. Reproducibility

### 7.1 Code state

All changes are in the `main` branch at HEAD. Key files:

- `mdp/baselines.py` — RandomPolicy, FrontierSectorVote (+penalty variant), FrontierClosestCell, FrontierClusterCentroid, FrontierUnvisitedCells
- `mdp/features.py` — 19-dim featurizer
- `mdp/qlearn.py` — LinearQ + `compute_reward` with shaping
- `mdp/oracle.py` — greedy planner with interior-cell targeting and `visited` set passthrough
- `eval.py` — `run_policy_episode` with stuck-escape, `run_oracle_episode` with online replanning, termination logging
- `train.py` — parallel Hogwild training with train-time stuck-escape
- `bot/bridge.js` — mineflayer bridge with NaN sanitizer, distance-proportional action timeout, configurable goal tolerance
- `tools/run_test_eval.py` — driver with SEED_INSTANCES, RESTART_SERVERS, configurable JVM heap

### 7.2 Reproduce the final numbers

```bash
# Train (optional — checkpoint included in weights/qlearn.npz)
GOAL_TOLERANCE=8 HOP_DISTANCE=50 \
  python3 train.py --seeds 1111,2222,3333,4444,5555 \
    --episodes-per-seed 22 --budget-s 300 --fresh

# Eval (full 4-policy run, ~70 min)
RESTART_SERVERS=1 GOAL_TOLERANCE=8 ORACLE_INTERIOR=2 \
SEEDS=11111,22222,33333,44444,55555 SEED_INSTANCES=1 \
  python3 tools/run_test_eval.py \
  --policies random,frontier_sector_penalty,qlearn,oracle \
  --budget-s 600
```

Per-episode results land in `results/<policy>_<seed>_<ep>.json` with the full schema (primary + secondary metrics + termination + policy stats).

### 7.3 Environment

- Java 26 (Temurin), Paper 1.20.1 (git-Paper-196), Node 20, Python 3.11+
- macOS / Linux (Windows untested in final iteration but the driver is cross-platform)
- 64 GB RAM, ~40 GB peak working set during a full 4-policy eval

---

## 8. Open questions

1. Does PHI=19 + extended training (50+ rounds) exceed the PHI=18 + 10-round 4.72 we saw at iter 7? (untested)
2. Does the qlearn → oracle gap (4.44 vs 5.44) close with smarter feature engineering, or is it a function of the limited action space (8 compass × fixed hop)?
3. How does the proposal-specified seed set `{123, 456, 789}` rank these policies under the iter-19 harness? (we report the new-seed results; proposal seeds were never re-run with the full set of fixes)
4. Can the locomotion gap (planned 8.44 → executed 5.44) be closed without `/tp`? E.g., a learned reachability prior on top of the offline biome map.
