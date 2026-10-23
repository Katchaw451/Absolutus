#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="./build/absolutus"
OUT="results/ghz_sweep_$(date +%Y%m%d_%H%M%S).csv"

echo "n,entropy_q0,entropy_mid,purity_q01,zz_mean,x_parity" > "$OUT"
for n in 2 4 6 8 10 12 14 16 18 20; do
    line=$("$BIN" --experiment ghz --qubits "$n" 2>/dev/null | tail -n1)
    vals=$(echo "$line" | cut -d, -f3-8)
    echo "$n,$vals" >> "$OUT"
    printf "n=%2d  %s\n" "$n" "$vals"
done
echo
awk -F, 'NR>1 { if ($2==1 && $3==1 && $4==0.5 && $5==1 && $6==1) ok++; else bad++ }
END { printf "PASS: %d  FAIL: %d\n", ok+0, bad+0 }' "$OUT"
echo "raw: $OUT"
