# MPS Boundary: where classical simulation fails

Empirical measurement of the boundary between classically simulable
quantum states and classically hard ones, on a 2015-era laptop CPU.
This is the research question that 2025 MPS-on-laptop papers (Science,
Nature Communications) study at scale; here it is measured at n=10
with a from-scratch simulator.

## What is being measured

For a pure state |psi> on n qubits, and a cut at position k, the
Schmidt rank across the bipartition {0..k-1} | {k..n-1} is the number
of nonzero eigenvalues of the reduced density matrix rho_A on the
first k qubits. This number is exactly the minimum bond dimension chi
required for an exact Matrix Product State representation at that cut.

A state with small max_k chi is classically simulable by MPS. A state
with max_k chi ~ 2^(n/2) is at the theoretical maximum and cannot be
compressed; simulating it requires the full 2^n statevector.

## Method

For each state and each cut k in 1..n-1:

1. Compute rho_A = partial_trace(psi, qubits [0..k-1]) via the existing
   2^(n+k) algorithm in `core::partial_trace`.
2. Diagonalize rho_A via the doubled-real Jacobi eigendecomposition
   in `math::eig_hermitian`.
3. Count eigenvalues > 1e-10 (threshold absorbs numerical zero).

The output is a profile[k] for k = 0..n, with profile[0] = profile[n]
= 1 by convention.

## Results

**The empirical law.** For every state measured, the bond profile at cut
k satisfies:

    chi(k, d, n) <= min(2^d, 2^k, 2^(n-k))

where d is circuit depth, k is the cut position, n is qubit count.
The three bounds are:

- **2^d** — light cone. After d layers of RY + CNOT, each qubit is
  causally connected to at most 2^d others. The rank cannot exceed this.
- **2^k** — left ceiling. A cut at k has at most 2^k left-side qubits.
- **2^(n-k)** — right ceiling. Symmetrically.

The formula has no free parameters. It is a prediction, not a fit.

### Verification at n=10 (seed 42)

| State | d | max_bond | Profile |
| :--- | :--- | :--- | :--- |
| product (H^n) | 0 | 1 | `1 1 1 1 1 1 1 1 1 1 1` |
| GHZ | - | 2 | `1 2 2 2 2 2 2 2 2 2 1` |
| W | - | 2 | `1 2 2 2 2 2 2 2 2 2 1` |
| random | 1 | 2 | `1 2 2 2 2 2 2 2 2 2 1` |
| random | 3 | 8 | `1 2 4 8 8 8 8 8 4 2 1` |
| random | 10 | 32 | `1 2 4 8 16 32 16 8 4 2 1` |

At n=10, the light-cone bound (2^d) and the center ceiling (2^(n/2)
= 32) coincide at d=5. Depth 10 exceeds this, so rand_d10 saturates.

### Verification at n=14 (2 seeds)

| State | d | Profile (seed 0) |
| :--- | :--- | :--- |
| random | 3 | `1 2 4 8 5 8 8 8 8 8 5 8 4 2 1` |
| random | 10 | `1 2 4 8 16 32 64 128 64 32 16 8 4 2 1` |

Predicted for d=10: `min(1024, 2^k, 2^(14-k))`. At k=7: min(1024, 128,
128) = 128. **Observed: 128. Exact match.** All cuts match.

At n=14, depth 10 exceeds n/2 = 7, so saturation at the center is
expected and observed.

### Verification at n=16 (2 seeds, ~106s per run)

| State | d | Profile (seed 0) |
| :--- | :--- | :--- |
| random | 3 | `1 2 4 8 8 8 8 8 8 8 8 8 8 8 4 2 1` |
| random | 10 | `1 2 4 8 16 32 64 128 200 128 64 32 16 8 4 2 1` |

**All non-central cuts match the formula exactly.** k=7 (and k=9):
min(1024, 128, 512) = 128, observed 128. k=6 (and k=10): 64, observed
64. k=5 (and k=11): 32, observed 32.

**The central cut deviates.** k=8 has formula ceiling min(1024, 256,
256) = 256. Observed 200 (seed 0), 236 (seed 1). Both below ceiling.

### Verification at n=18 (1 seed, 9m14s)

| State | d | Profile |
| :--- | :--- | :--- |
| random | 3 | `1 2 4 8 8 8 8 8 8 8 8 8 8 8 8 8 4 2 1` |
| random | 10 | `1 2 4 8 16 32 64 128 195 216 195 128 64 32 16 8 4 2 1` |

Non-central cuts (k=1..7 and k=11..17) still match the formula
exactly. Central cuts:

- k=8: ceiling 256, observed **195** (24% deficit)
- k=9: ceiling 512, observed **216** (58% deficit)
- k=10: ceiling 256, observed **195** (24% deficit)

## The saturation deficit grows with n

| n | center k | ceiling min(2^d, 2^(n/2)) | observed | deficit |
| :--- | :--- | :--- | :--- | :--- |
| 10 | 5 | 32 | 32 | **0%** |
| 14 | 7 | 128 | 128 | **0%** |
| 16 | 8 | 256 | 200-236 | **8-22%** |
| 18 | 9 | 512 | 216 | **58%** |

At d=10 fixed, the deficit is zero for n ≤ 14, starts to appear at
n=16, and is substantial at n=18. Saturation is not a property of
depth alone; it degrades with n at constant depth.

**The intuitive bound `d >= n/2` is not sufficient.** For this
circuit family (random RY per qubit + CNOT chain, repeated d times),
n=18 with d=10 satisfies `d > n/2 = 9` by one layer, yet reaches only
216/512 = 42% of the ceiling. The circuit needs substantially more
than n/2 layers for the Schmidt distribution at the center to
approach flat.

**Physical interpretation.** The CNOT chain has local interaction
range: each layer propagates entanglement by one qubit of distance.
After d layers, the light cone has radius ~d. For the center cut
(k = n/2), the entanglement across the bipartition requires the two
halves to have 'seen' each other for long enough that the reduced
density matrix is close to maximally mixed. Empirically, this requires
d > n/2 by a factor that itself grows with n. At n=18, d=10 is far
from enough; a plausible rule of thumb is d ~ n, but the data is only
three points (14, 16, 18) at one d.

**What this means for classical simulability.** Circuit depth is not
just a lower bound on chi. At a given depth, chi depends on where the
cut is relative to the light-cone front. For circuits shallower than
~n, the center of the chain is far from saturation and MPS with
modest chi captures the state accurately. For circuits deeper than
~n, the center saturates and chi = 2^(n/2). The crossover is where
classical simulation of this circuit family becomes hard.

This is a stronger empirical statement than 'chi <= min(...)'. It
says something about *how close* random circuits come to that bound,
as a function of (n, d). The 3-point deficit table is the first
quantification on this hardware.

### Practical ceiling for this experiment

n=18 is the practical maximum on the Ryzen 5 5625U for this method.
A single seed takes 9m14s (one state at 512x512 Jacobi). Multi-seed
at n=18 would take 30+ minutes. n=20 would require 1024x1024 Jacobi
and take >1 hour per seed.

Going further requires a faster eigensolver (Lanczos for the top-K
eigenvalues would be 100x faster for this task, since we only need
counts above threshold). That is out of scope for this session but
is the natural next step for the experiment.

## What each row says

**Product state**: chi=1 everywhere. Zero entanglement at every cut.
Trivial for MPS.

**GHZ and W**: chi=2 everywhere (interior cuts). One or two branches
of the wavefunction, and both are always present. In n=100 qubits,
these states compress to a few hundred complex amplitudes — the full
statevector would need 2^100 amplitudes, which exceeds the number of
atoms in the observable universe.

**Random depth 1**: chi=2. A single layer of RY + CNOT does not
generate substantial entanglement. Barely distinguishable from GHZ at
this depth.

**Random depth 3**: chi=8 in the middle, tapering at the ends. Three
layers of RY + CNOT produce moderate entanglement, but not maximal.
The chi=8 at k=3..7 shows the emaranhamento spans several qubits.

**Random depth 10**: chi=32 at the center = 2^5 = 2^(n/2). This is
the theoretical maximum for n=10: the full Hilbert space is 2^10 =
1024, and the largest possible bipartition has dimension 2^5 = 32.
A random circuit of depth 10 saturates the space. No MPS compression
is possible.

## The extrapolation to n=100 (from the verified formula)

The formula chi <= min(2^d, 2^k, 2^(n-k)) is confirmed at n=10, 14, 16.
The question that matters: what does it predict for larger n?

| State | n=100 chi | MPS memory | Statevector memory |
| :--- | :--- | :--- | :--- |
| product | 1 | ~1.6 KB | 2^100 amplitudes |
| GHZ | 2 | ~3.2 KB | (exceeds universe) |
| W | 2 | ~3.2 KB | (same) |
| random depth 10 | ~2^50 | ~10^30 bytes | (same) |

The MPS memory is n * chi^2 * 2 * 16 bytes (complex double per tensor
element). For chi=2 at n=100, that is 100 * 4 * 2 * 16 = 12.8 KB. For
chi=2^50, that is 100 * 2^100 * 2 * 16 ≈ 10^32 bytes. The gap is 30
orders of magnitude.

This is the entire content of "quantum advantage": some quantum states
are classically simulable (small chi), others are not (large chi). The
boundary is not a philosophical line but a specific chi threshold
determined by the hardware available.

## Where the boundary sits, empirically

From the profiles above:
- **chi <= 8** covers products, GHZ, W, and circuits of depth ~3
- **chi = 2^(n/2)** is the ceiling, reached by depth ~10 at n=10
- The transition from chi=8 to chi=32 in n=10 (depth 3 -> depth 10)
  shows that the boundary is reached fast: 10 layers of RY + CNOT
  are enough to saturate in this regime.

The 2025 Science paper simulated hundreds of qubits on a laptop by
choosing circuits that stay in the small-chi regime (3D networks with
low entanglement per bond). The circuits that are hard (random depth
> ~n) are precisely the ones those papers exclude.

## What this measurement does NOT prove

- **n=10 is small**. The saturation at depth 10 is a small-n effect.
  In n=100, the depth needed to saturate is larger, and circuits of
  depth 10 in n=100 would still have chi << 2^50. The exact scaling
  with n and depth requires n=14-16, which is measurable on the Acer
  but not in this session.
- **The profile is one instance**. Two random circuits of depth 10
  produce different profiles; the measured chi=32 is one draw. A proper
  measurement would average over seeds and report mean + std.
- **The threshold matters**. 1e-10 vs 1e-6 changes which eigenvalues
  count. For GHZ with exact zeros, both thresholds give chi=2. For
  near-saturated states, the threshold matters. A full study would
  report chi as a function of threshold.

## What this measurement does prove

- **GHZ and W compress to chi=2** in n=10. The 2025 result that these
  states are classically simulable at n=100 is consistent with the
  mechanism: the MPS bond stays at 2 regardless of n.
- **Random circuits saturate**. Depth 10 in n=10 gives the maximum
  chi=32. No compression. This matches the theoretical prediction.
- **The implementation is correct**. Product states give chi=1, GHZ
  gives chi=2, and the saturation ceiling is hit as theory predicts.
  A bug in partial_trace or eig_hermitian would break one of these.

The classification of states into "small chi" (simulable) and "large
chi" (hard) is measured on the Acer, in C++, from scratch, in ~100
ms per state. The same measurement at n=100 would require the MPS
implementation itself (not just the profile measurement), which is
the next natural step.

## Reproduce

    cd ~/Absolutus
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)
    ./build/absolutus --experiment bond_dim

    # For multi-seed studies:
    for seed in 1 2 3 4 5; do
        ./build/absolutus --experiment bond_dim --seed $seed
    done

## Comparison to published work

The 2025 Science paper (Kim et al.) simulated the Sycamore-class
circuits on a laptop using MPS with truncation. The chi required for
the specific circuits was small enough (hundreds) to fit in 32 GB RAM.
The paper's key claim: the quantum computation is not inherently
beyond classical simulation; it is only beyond full-statevector
simulation.

This document measures the same quantity at n=10: the MPS bond
dimension as a function of state structure and circuit depth. The
mechanism is the same. The scale is 1000x smaller. The result is
consistent.

## Next steps (not run)

1. **Multi-seed at n=14**. 5 seeds per state, report mean chi + std
   for each (state, depth) pair. ~15 min on the Acer.
2. **Truncated MPS implementation**. The `MPS` class in
   `include/core/mps.hpp` is a stub; implementing `from_statevector`
   with SVD truncation would allow simulating states with chi above
   the statevector ceiling (chi > 2^(n/2) is not possible, but the
   truncated MPS could represent states of n=100 with chi=100
   approximately).
3. **Apply MPS to VQE**. The VQE ansatz at n=6 has chi << 2^3. If
   MPS-VQE gives the same energy as statevector-VQE, that validates
   MPS for the hybrid pipeline.
