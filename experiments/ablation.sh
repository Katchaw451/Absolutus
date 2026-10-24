#!/usr/bin/env bash
# Ablation: does adding a quantum attention head hurt trainability?
# Sweeps two configs across seeds, measures pass rate and wall time.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="./build/absolutus"
OUT="results/ablation_$(date +%Y%m%d_%H%M%S).csv"
SEEDS=${SEEDS:-10}
EPOCHS=${EPOCHS:-300}
LR=${LR:-0.05}

echo "config,seed,time_ms,final_loss,status" > "$OUT"

run() {
  local cfg=$1 seed=$2 args=$3
  local start end time_ms status loss out
  start=$(date +%s%N)
  if out=$("$BIN" --experiment train_lm --epochs "$EPOCHS" --lr "$LR" \
              --seed "$seed" $args 2>&1 >/dev/null); then
    status=PASS
  else
    status=FAIL
  fi
  end=$(date +%s%N)
  time_ms=$(( (end - start) / 1000000 ))
  loss=$(printf '%s\n' "$out" | awk -F'final: ' '/final:/ {print $2}' | awk '{print $1}')
  echo "$cfg,$seed,$time_ms,${loss:-nan},$status" >> "$OUT"
  printf "%-14s seed=%2d  %5dms  loss=%-12s  %s\n" "$cfg" "$seed" "$time_ms" "${loss:-nan}" "$status"
}

echo "=== classical_2h (baseline) ==="
for seed in $(seq 1 "$SEEDS"); do
  run "classical_2h" "$seed" "--n-heads 2 --qhead -1"
done

echo
echo "=== hybrid_q0 (head 0 = quantum) ==="
for seed in $(seq 1 "$SEEDS"); do
  run "hybrid_q0" "$seed" "--n-heads 2 --qhead 0"
done

echo
echo "=== summary ==="
awk -F, 'NR>1 {
  cfg=$1; n[cfg]++;
  if ($5=="PASS") pass[cfg]++;
  sum_t[cfg]+=$3;
} END {
  for (c in n) {
    printf "%-14s  %2d/%2d pass  mean_time=%.0fms\n",
      c, pass[c]+0, n[c], sum_t[c]/n[c];
  }
}' "$OUT"
echo
echo "raw: $OUT"
