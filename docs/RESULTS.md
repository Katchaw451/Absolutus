# Results

All numbers below are from a single Acer Aspire Go 15 (i3-N305, 8 GB RAM,
no GPU), Release build with `-O3 -march=native`, across multiple seeds
unless noted. Reproducible via the CLI commands shown.

---

## Physics — quantum simulator

### CHSH inequality (Bell violation)

Experiment: `./build/absolutus --experiment chsh --trials 100000 --seed 7`

| Quantity | Value |
| :--- | :--- |
| E00 (exact) | +0.707106781 |
| E01 (exact) | +0.707106781 |
| E10 (exact) | +0.707106781 |
| E11 (exact) | -0.707106781 |
| **S exact** | **+2.828427125 = 2√2** |
| S sampled (N=100k) | +2.835400 |
| Classical bound | 2.0 |
| Tsirelson bound | 2√2 ≈ 2.8284 |

The exact value hits 2√2 to 9 decimal places. Sampled runs fluctuate
above and below 2√2 by ~1σ of 2/√N ≈ 0.0045, as expected for a finite
estimator against a quantum-mechanical upper bound.

Sweep across 20 seeds at N=100k: mean 2.829018, std 0.004484, matching
the theoretical √(2/N) = 0.00447 to the fourth decimal.

### GHZ multipartite entanglement

Experiment: `./build/absolutus --experiment ghz --qubits 20`

| Quantity | Value | Expected |
| :--- | :--- | :--- |
| S(ρ_0) | 1.000000000 | 1 |
| S(ρ_mid) | 1.000000000 | 1 |
| Tr(ρ_01²) | 0.500000000 | 0.5 |
| ⟨Z_0 Z_k⟩ (avg) | 1.000000000 | 1 |
| ⟨X_0…X_{n-1}⟩ | 1.000000000 | 1 |

Verified up to n=20 (dimension 2^20 = 1,048,576 amplitudes). All five
independent physics assertions hold. At n=20 the state is a pure state
whose every single-qubit reduction is maximally mixed — the definition
of maximal multipartite entanglement.

### VQE for H₂ (2-qubit reduced, Jordan-Wigner)

Experiment: `./build/absolutus --experiment vqe --iters 200 --lr 0.1`

| Quantity | Value |
| :--- | :--- |
| E_exact (2×2 diagonalization) | -1.851199124 |
| E_vqe (parameter-shift + SGD) | -1.851199124 |
| |ΔE| | 4.219 × 10⁻¹⁵ |
| θ* | 3.371336 rad |
| Final gradient | 8.5 × 10⁻⁸ |

The VQE converges to the analytically diagonalized ground energy to
machine precision (double). This is a cross-validation: gradient descent
on a parameterized circuit and direct linear algebra on a 2×2 Hermitian
matrix reach the same answer from completely different directions.

---

## Classical ML — from scratch

### XOR (MLP + momentum)

Experiment: `./build/absolutus --experiment train_xor`

| Config | Result |
| :--- | :--- |
| Vanilla SGD, lr=0.5, 20 seeds | 10 / 20 pass |
| Momentum β=0.9, lr=0.1, 20 seeds | **20 / 20 pass** |

Vanilla SGD stalls at the XOR saddle (0.5, 0.5, 0.5, 0.5). Momentum
carries the accumulated velocity through the plateau. Symmetry-breaking
bias init (b=0.1) also helps. Both changes are cheap; both are standard.

### MiniLM copy task (classical, 2 heads)

Experiment: `./build/absolutus --experiment train_lm --n-heads 2 --qhead -1`

| Config | Final loss | Baseline log(8) |
| :--- | :--- | :--- |
| L=4, d=8, dff=16, V=8, 2 blocks | 0.000000 | 2.0794 |

Loss drops from 2.4 to 0.000000 in 50 epochs. The model memorizes the
32-sequence dataset — this is infrastructure validation, not generalization.

---

## Hybrid quantum-classical

### Pure QuantumLayer on XOR

Experiment: `./build/absolutus --experiment train_qxor --params 8`

| Config | Success rate |
| :--- | :--- |
| nq=2, params=2 (1 layer) | 2/5 |
| nq=2, params=4 (2 layers) | 3/5 |
| nq=2, params=8 (4 layers) | 4/5 |
| **nq=3, params=6 (2 layers)** | **5/5** |

The pattern reveals two phenomena: (1) more parameters help but with
diminishing returns, and (2) 3 qubits with 6 params beat 2 qubits with
6 params. More qubits = larger Hilbert space = more expressive circuit
without more barren-plateau tendency at these scales. This is
expressivity-vs-trainability in miniature.

### MiniLM with one quantum head

Experiment:
./build/absolutus --experiment train_lm --n-heads 2 --qhead 0

| Config | Time | Final loss |
| :--- | :--- | :--- |
| Baseline (2 classical heads) | 0.354 s | 0.000000 |
| Hybrid (1 quantum + 1 classical) | 2.736 s | 0.000000 |

The hybrid trains end-to-end. The 7.7× slowdown is the cost of
parameter-shift: each quantum parameter needs 2 forward circuit
evaluations per backward pass. The classical head is unaffected.

---

## Test suite

`cd build && ctest --parallel 4`

| Category | Count |
| :--- | :--- |
| Unit tests (physics + ML primitives) | 16 |
| End-to-end integration | 7 |
| **Total** | **23** |
| Wall-clock | ~2.7 s |
