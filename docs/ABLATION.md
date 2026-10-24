# Ablation: does a quantum attention head hurt trainability?

Test the claim in docs/DESIGN.md: adding a parameterized quantum
circuit as one attention head preserves end-to-end training at the
cost of compute (parameter-shift requires 2 forward evaluations per
parameter per backward pass).

## Setup

- `MiniLM(L=4, d_model=8, d_ff=16, vocab=8, blocks=2)`
- Head configuration: `n_heads=2`, `d_k=4` per head
- Two configs:
  - **classical_2h**: both heads classical `Attention`
  - **hybrid_q0**: head 0 is `QuantumAttention`, head 1 classical
- Task: copy (target == input)
- Epochs: 300, lr=0.05, clip=1.0
- Seeds: 1..10
- Pass criterion: `final_loss < 0.5 * log(8) ≈ 1.0397`

## Results

| Config | Pass | Mean time | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| classical_2h | **10/10** | 364 ms | 358 | 376 |
| hybrid_q0 | **10/10** | 1924 ms | 1672 | 2763 |

All 20 runs end with `final_loss = 0.0000` to 4 decimals.
Ratio of means: **5.28×**. Ratio of medians: **4.67×**.

### Per-run times (ms)

    classical_2h:  361 358 368 376 363 366 362 362 363 360
    hybrid_q0:    2763 2337 1702 1693 1701 1698 1672 2280 1692 1697

## Observations

### 1. Trainability is preserved

10/10 vs 10/10, loss identical to 4 decimals. The parameter-shift
gradient composes cleanly with the rest of the backward pass. Nothing
about having a quantum head makes the copy task harder to fit.

### 2. Compute cost is real, and higher than naive theory predicts

First-order estimate: head 0 has 4 parameters (n_qubits=4, 1 layer),
so parameter-shift is 8 circuit evaluations per backward vs 1
analytic evaluation for the classical head. Head 0 is half the
attention. Attention is roughly half the block. Back-of-envelope
whole-model slowdown: (8·0.5 + 1·0.5) on attention × 0.5 of block
+ 0.5 of block unchanged ≈ **2.3×**.

Measured: **5.3×**. The quantum head costs ~2.3× more than the naive
estimate. Where the extra time goes is not obvious from these
measurements — candidates include pre_q Dense re-forward per token,
larger memory traffic on theta, and per-step clip_gradients walks.

### 3. Timing variance is bimodal, not Gaussian

Classical: 358–376 ms, spread 5% — what you'd expect on an i3-N305
with OS scheduling noise.

Hybrid: three seeds (1, 2, 8) at 2280–2763 ms, seven at 1672–1702 ms.
That's ~1.5× difference on identical code, identical epochs, identical
parameter counts. The distribution is bimodal.

**Hypothesis (not verified):** some seeds' training trajectories push
the gradient norm above `clip_max = 1.0` more frequently, triggering
the `dynamic_cast` walk across every head's gradient buffer, and
possibly different mean θ magnitudes that affect nothing computationally
but correlate with slow runs by coincidence.

**Alternative hypothesis:** E-core thermal/power state on the Acer.
8 E-cores, no SMT, package power management may not schedule them
identically. Kernel-level tracing (`perf`, `strace -c`) would settle it.

We do not claim to know. Both hypotheses would produce this pattern.

## Reproduction

    cd ~/Absolutus
    ./experiments/ablation.sh          # ~60 s on the Acer
    cat results/ablation_*.csv         # raw per-run times

## What this ablation does NOT show

- **Generalization.** The copy task memorizes 32 sequences. Loss 0.0000
  means "fit the training set", not "learn the concept of copying".
- **Scalability.** d_model=8. At d_model=64 the classical head's
  O(seq²·d_k) cost dominates parameter-shift's fixed per-param cost,
  and the ratio would invert.
- **Quantum advantage.** Nothing here suggests the quantum head is
  *better*. It's not worse on this task, and it's slower. That's all.
