#!/usr/bin/env bash
# One-time setup: verify toolchain, prepare build directory.
set -euo pipefail

echo "[setup] checking toolchain..."
command -v cmake >/dev/null || { echo "missing: cmake"; exit 1; }
command -v g++   >/dev/null || { echo "missing: g++";   exit 1; }

echo "[setup] cmake: $(cmake --version | head -n1)"
echo "[setup] g++:   $(g++ --version | head -n1)"

mkdir -p build
echo "[setup] done. run 'scripts/build.sh' to compile."
