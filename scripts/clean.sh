#!/usr/bin/env bash
# Remove build artifacts and runtime outputs. Keeps sources intact.
set -euo pipefail

cd "$(dirname "$0")/.."

rm -rf build/*
rm -f results/*.csv results/*.dat
rm -f logs/*.log

echo "[clean] done."
