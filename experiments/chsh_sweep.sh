#!/usr/bin/env bash
# Sweep CHSH over N seeds, collect S_sampled, report mean ± std.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="./build/absolutus"
OUT="results/chsh_sweep_$(date +%Y%m%d_%H%M%S).csv"
N_SEEDS=${N_SEEDS:-20}
TRIALS=${TRIALS:-100000}

echo "seed,S_sampled" > "$OUT"
for seed in $(seq 1 "$N_SEEDS"); do
    S=$("$BIN" --experiment chsh --trials "$TRIALS" --seed "$seed" \
        2>/dev/null | tail -n1 | cut -d, -f9)
    echo "$seed,$S" >> "$OUT"
    printf "seed=%2d  S=%.6f\n" "$seed" "$S"
done

echo
echo "--- statistics (Tsirelson = 2.828427) ---"
awk -F, 'NR>1 {
    n++; sum+=$2; sumsq+=$2*$2;
    if ($2 > 2.828427) over++;
} END {
    mean = sum/n;
    var  = sumsq/n - mean*mean;
    printf "n         = %d\n", n;
    printf "mean S    = %.6f\n", mean;
    printf "std  S    = %.6f   (theory ~ 2/sqrt(N) = %.6f)\n",
           sqrt(var), 2/sqrt(100000);
    printf "runs > 2*sqrt(2): %d/%d\n", over, n;
}' "$OUT"
echo
echo "raw data: $OUT"
