#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="./build/absolutus"
OUT="results/xor_sweep_$(date +%Y%m%d_%H%M%S).csv"
echo "seed,ok" > "$OUT"
ok=0; total=0
for seed in $(seq 1 20); do
    total=$((total+1))
    if "$BIN" --experiment train_xor --epochs 2000 --lr 0.1 --momentum 0.9 \
              --seed "$seed" >/dev/null 2>&1; then
        ok=$((ok+1)); printf "seed=%2d  PASS\n" "$seed"; echo "$seed,1" >> "$OUT"
    else
        printf "seed=%2d  FAIL\n" "$seed"; echo "$seed,0" >> "$OUT"
    fi
done
echo "PASS: $ok / $total"
echo "raw: $OUT"
