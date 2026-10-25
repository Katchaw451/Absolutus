# Absolutus

A from-scratch quantum simulator and hybrid quantum-classical transformer,
written in pure C++17 on commodity hardware. No external frameworks, no
GPU, no cloud. Every layer tested against known physical results.

## Status

The project trains end-to-end. A copy task with a transformer where one
attention head is a parameterized quantum circuit runs on an Acer
Aspire Go 15 (i3-N305, 8 GB RAM) in ~3 seconds:

    Baseline (2 classical heads):      0.354 s,  loss = 0.000000
    Hybrid   (1 quantum + 1 classical): 2.736 s,  loss = 0.000000

The 5.3x slowdown (10-seed mean) is the parameter-shift rule's inherent cost: each
quantum parameter needs 2 forward circuit evaluations per backward pass,
versus 1 analytic gradient for a classical head.

## What's inside

- Quantum primitives: complex arithmetic, dense matrices, statevector,
  gate application, measurement, partial trace, entropy, fidelity.
- Physics experiments: CHSH (Bell violation), GHZ (n up to 20),
  VQE (H2 ground state, checked against exact diagonalization).
- Classical ML: Dense, MLP, softmax, cross-entropy, LayerNorm (with full
  3-term backward), Attention, MultiHeadAttention, TransformerBlock
  (pre-norm + residual), PositionEncoding, MiniLM.
- Hybrid: QuantumLayer (parameterized circuit as an ML component),
  QuantumAttention, MiniLM with a quantum head slot.
- Tests: 23 ctest targets, 2.7 seconds total. Every experiment and every
  layer has an invariant-tested assertion; every gradient is checked
  against finite differences.

## Quickstart

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)
    cd build && ctest --output-on-failure

    # Physics
    ./build/absolutus --experiment chsh
    ./build/absolutus --experiment ghz   --qubits 20
    ./build/absolutus --experiment vqe

    # Classical ML
    ./build/absolutus --experiment train_xor
    ./build/absolutus --experiment train_lm

    # Hybrid
    ./build/absolutus --experiment train_qxor   --params 8
    ./build/absolutus --experiment train_hybrid --qubits 2 --params 4
    ./build/absolutus --experiment train_lm --n-heads 2 --qhead 0

## Rules

- No external frameworks (no PyTorch, no PennyLane, no Eigen, no Qiskit).
- C++17 standard library only.
- Bash orchestrates, C++ computes, nano edits.
- Every experiment has a physics sanity test.
- Every gradient is checked against finite differences in ctest.

## Layout

    include/{math,core,physics,ml,hybrid}/   headers
    src/{math,core,physics,ml,hybrid}/       implementations
    tests/                                   per-layer unit tests
    experiments/                             bash parameter sweeps
    results/                                 CSV output (git-ignored)
    docs/                                    results, design notes

## Documentation

- docs/RESULTS.md  -- measured numbers, all experiments
- docs/TEXT_TRAINING.md -- BPE + language model trajectory (7.44 -> 3.88)
- docs/EXPERIMENTS_LOG.md -- A/B experiments: 1000 epochs vs d_model=32
- docs/DESIGN.md   -- architectural decisions and rationale
- docs/NOTEBOOK.md -- chronological lab notebook

## Hardware

Developed on an Acer Aspire Go 15 (i3-N305, 8 GB RAM, no GPU).
Design constraint: every optimization pays for itself in qubits or
wall-clock time. 500 MB of freed memory ~ 1 extra qubit of simulation.
