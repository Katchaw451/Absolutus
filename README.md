# Absolutus

A from-scratch quantum simulator written in pure C++17, with no external
frameworks or libraries beyond the C++ standard library.

## Purpose

Build, test, and understand advanced mathematics and quantum physics
locally, on commodity hardware. Every experiment is verified against
known physical results. No black boxes.

## Target hardware

Acer Aspire Go 15 — Intel Core i3-N305, 8 GB RAM, no GPU.
Design constraint: every optimization must pay for itself in qubits or
wall-clock time. 500 MB of freed memory ≈ 1 extra qubit of simulation.

## Rules

- No external frameworks (no PyTorch, no PennyLane, no Eigen, no Qiskit).
- C++17 standard library only.
- Bash orchestrates, C++ computes, nano edits.
- Every experiment has a physics sanity test (unitarity, norm, known limits).

## Layout

- `include/` — headers, one subfolder per domain
- `src/`     — implementation, mirrors `include/`
- `tests/`   — unit tests, each one asserts a physical invariant
- `experiments/` — bash scripts that drive parameter sweeps
- `results/` — CSV and time-series output
- `logs/`    — long-run logs (nohup, cron)
- `docs/`    — derivations, notes, lab notebook
- `configs/` — run parameters, reproducible per experiment
- `scripts/` — setup, cleanup, benchmark helpers
- `build/`   — compiled artifacts (git-ignored)

## Build

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j4

## Status

Early. Foundations being laid.
