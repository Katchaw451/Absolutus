#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="./build/absolutus"
OUT="results/qxor_sweep_$(date +%Y%m%d_%H%M%S).csv"
echo "seed,n_params,ok" > "$OUT"
ok=0; total=0
for params in 4 6 8 12; do
  for seed in $(seq 1 10); do
    total=$((total+1))
    if "$BIN" --experiment train_qxor --epochs 4000 --lr 0.1 --momentum 0.9 \
              --params "$params" --seed "$seed" >/dev/null 2>&1; then
      ok=$((ok+1)); echo "$seed,$params,1" >> "$OUT"; printf "params=%2d seed=%2d PASS\n" "$params" "$seed"
    else
      echo "$seed,$params,0" >> "$OUT"; printf "params=%2d seed=%2d FAIL\n" "$params" "$seed"
    fi
  done
done
echo "TOTAL: $ok / $total"
