#!/usr/bin/env bash
set -euo pipefail

POOL_DIR="./checkpoints/pool"
PPO_CHECKPOINT="./checkpoints/ppo_checkpoint.pt"
SEEDS=(
  "seed_max_base_power"
  "seed_simple_heuristic"
  "seed_fuzzy_heuristic"
)

if [[ ! -d "$POOL_DIR" ]]; then
  echo "Pool dir not found: $POOL_DIR" >&2
  exit 1
fi

shopt -s nullglob dotglob
for path in "$POOL_DIR"/*; do
  name="$(basename "$path")"
  keep=0
  if [[ "$name" == "pool_state.json" ]]; then
    keep=1
  else
    for seed in "${SEEDS[@]}"; do
      if [[ "$name" == "$seed".* ]]; then
        keep=1
        break
      fi
    done
  fi

  if [[ $keep -eq 0 ]]; then
    rm -rf -- "$path"
  fi
done
shopt -u nullglob dotglob

rm -f -- "$PPO_CHECKPOINT"

cat > "$POOL_DIR/pool_state.json" <<'JSON'
{
  "opponent_ids": [
    "seed_max_base_power",
    "seed_simple_heuristic",
    "seed_fuzzy_heuristic"
  ],
  "win_rates": {
    "seed_max_base_power": 0.5,
    "seed_simple_heuristic": 0.5,
    "seed_fuzzy_heuristic": 0.5
  }
}
JSON

echo "Reset complete: $POOL_DIR"
echo "Deleted: $PPO_CHECKPOINT"
