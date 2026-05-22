#!/usr/bin/env bash
# Run baselines + oracle on the 3 test seeds with 5 bots per server.
# One Paper server per seed (ports 25565..25567), 15 bot bridges total
# (ports 9000..9014). Policies run sequentially across all 15 bots, so
# wall-clock per policy ≈ one episode budget (default 10 min).
#
# Prereqs: cubiomes built (see README §1.4) AND npz dumps exist for each
# test seed:
#     for s in 123 456 789; do python3 tools/extract_biomes.py --seed $s; done
#
# Usage: ./tools/run_test_eval.sh

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(123 456 789)
POLICIES=(random frontier oracle)
BOTS_PER_SERVER=5
BUDGET_S=600
BASE_MC_PORT=25565
LOGS=logs
mkdir -p "$LOGS"

# Portable in-place sed (BSD vs GNU).
sedi() { if [[ "$(uname)" == "Darwin" ]]; then sed -i '' "$@"; else sed -i "$@"; fi; }

# Prereq: npz dumps for each test seed.
for s in "${SEEDS[@]}"; do
    [[ -f "data/biomes_${s}.npz" ]] || {
        echo "missing data/biomes_${s}.npz — run tools/extract_biomes.py --seed $s"
        exit 1
    }
done

# Per-seed server dir, paper.jar symlinked from the template.
for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    port=$((BASE_MC_PORT + i))
    dir="mc-server-test${seed}"
    if [[ ! -d "$dir" ]]; then
        echo "[setup] $dir (seed=$seed port=$port)"
        mkdir -p "$dir"
        ln -sf "../mc-server/paper.jar" "$dir/paper.jar"
        for f in eula.txt server.properties bukkit.yml spigot.yml \
                 commands.yml help.yml permissions.yml; do
            [[ -e "mc-server/$f" ]] && cp "mc-server/$f" "$dir/"
        done
        [[ -d mc-server/config ]] && cp -r mc-server/config "$dir/"
        sedi -e "s/^level-seed=.*/level-seed=$seed/" \
             -e "s/^server-port=.*/server-port=$port/" \
             "$dir/server.properties"
    fi
done

declare -a SERVERS BOTS
cleanup() {
    echo "[cleanup]"
    [[ ${#BOTS[@]}    -gt 0 ]] && kill "${BOTS[@]}"    2>/dev/null || true
    [[ ${#SERVERS[@]} -gt 0 ]] && kill "${SERVERS[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Boot 3 servers in parallel.
for seed in "${SEEDS[@]}"; do
    dir="mc-server-test${seed}"
    log="$LOGS/server_${seed}.log"
    echo "[server] $dir"
    (cd "$dir" && exec java -Xmx3G -Xms1G -jar paper.jar nogui) > "$log" 2>&1 &
    SERVERS+=($!)
done

# Wait for "Done!" line in every server log.
for seed in "${SEEDS[@]}"; do
    log="$LOGS/server_${seed}.log"
    until grep -q 'Done' "$log" 2>/dev/null; do sleep 2; done
    echo "[ready] server $seed"
done

# Spawn 15 bot bridges, 5 per server. MC_PORT picks which server each joins.
for s in "${!SEEDS[@]}"; do
    mc_port=$((BASE_MC_PORT + s))
    for b in $(seq 0 $((BOTS_PER_SERVER - 1))); do
        id=$((s * BOTS_PER_SERVER + b))
        MC_PORT=$mc_port node bot/bridge.js "$id" > "$LOGS/bot_$id.log" 2>&1 &
        BOTS+=($!)
        sleep 0.5
    done
done
echo "[bots] 15 bridges spawning; settling 10 s"
sleep 10

# Sequential policies, 15 parallel eval.py per policy.
for policy in "${POLICIES[@]}"; do
    echo "[run] policy=$policy start=$(date +%H:%M:%S)"
    declare -a EVALS=()
    for s in "${!SEEDS[@]}"; do
        seed=${SEEDS[$s]}
        for b in $(seq 0 $((BOTS_PER_SERVER - 1))); do
            id=$((s * BOTS_PER_SERVER + b))
            python3 eval.py --policy "$policy" --seed "$seed" \
                --bot-id "$id" --episode "$b" --budget-s "$BUDGET_S" \
                > "$LOGS/eval_${policy}_${seed}_${b}.log" 2>&1 &
            EVALS+=($!)
        done
    done
    wait "${EVALS[@]}"
    echo "[done] policy=$policy end=$(date +%H:%M:%S)"
done

echo "[complete] 45 episodes total; results/ has $(ls results 2>/dev/null | wc -l | tr -d ' ') files"
