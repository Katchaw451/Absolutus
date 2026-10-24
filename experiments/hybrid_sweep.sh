#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="./build/absolutus"
OUT="results/hybrid_sweep_$(date +%Y%m%d_%H%M%S).csv"
echo "n_q,q_params,seed,status" > "$OUT"

pass=0; fail=0; skip=0; total=0
for nq in 2 3 4; do
  for qp in 2 4 6 8; do
    if [ $((qp % nq)) -ne 0 ]; then
      printf "nq=%d params=%d  SKIP (invalid: params %% nq != 0)\n" "$nq" "$qp"
      for seed in 1 2 3 4 5; do
        echo "$nq,$qp,$seed,SKIP" >> "$OUT"
        skip=$((skip+1))
      done
      continue
    fi
    for seed in 1 2 3 4 5; do
      total=$((total+1))
      if "$BIN" --experiment train_hybrid --epochs 2000 --lr 0.1 --momentum 0.9 \
                --qubits "$nq" --params "$qp" --seed "$seed" >/dev/null 2>&1; then
        pass=$((pass+1)); echo "$nq,$qp,$seed,PASS" >> "$OUT"
        printf "nq=%d params=%d seed=%d PASS\n" "$nq" "$qp" "$seed"
      else
        fail=$((fail+1)); echo "$nq,$qp,$seed,FAIL" >> "$OUT"
        printf "nq=%d params=%d seed=%d FAIL\n" "$nq" "$qp" "$seed"
      fi
    done
  done
done
echo
echo "Total valid: $total"
echo "PASS: $pass   FAIL: $fail   SKIP (invalid config): $skip"
echo "raw: $OUT"

# Summary table
echo
echo "--- per (nq, params) ---"
awk -F, 'NR>1 && $4!="SKIP" {
  key=$1"/"$2;
  if ($4=="PASS") p[key]++; else f[key]++;
} END {
  for (k in p) printf "  %s: %d/%d\n", k, p[k], p[k]+f[k];
}' "$OUT" | sort
