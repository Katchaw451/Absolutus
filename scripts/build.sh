#!/usr/bin/env bash
# Configure and build in Release mode with native CPU optimizations.
set -euo pipefail

cd "$(dirname "$0")/.."

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
