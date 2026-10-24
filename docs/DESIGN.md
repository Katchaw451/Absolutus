# Design Notes

Why the code looks the way it does. Every section documents a decision
that was non-obvious, either because there was a real alternative or
because the obvious choice was wrong.

---

## Complex arithmetic: template on real type, no std::complex

`Complex<R>` is a plain struct `{R re, R im;}` with `constexpr` operators.
`std::complex<double>` was rejected for two reasons:

1. Layout guarantee. `{re, im}` is guaranteed contiguous with no padding.
   This matters when applying gates with SIMD-friendly loops over
   amplitude pairs.
2. Precision control. `c32` (float) and `c64` (double) are aliases, not
   separate types. Switching precision is one `using`, not a rebuild of
   the arithmetic layer.

All operations are `constexpr`. `1/√2` and friends get folded at compile
time. Nothing allocates in `operator*` or `operator+`.

## Matrix: row-major, sparse-aware multiplication

`Matrix<R>` is row-major dense, sized dynamically. Gate matrices are tiny
(2×2, 4×4) but tensor products grow as 2^n, so dynamic sizing is
necessary. The `operator*` skips entries where `a.re==0 && a.im==0`.
Quantum gates are sparse in practice (CNOT, CZ, SWAP), and this trick
turns a naive O(n³) into O(n²) for these cases without adding structure.

## Statevector: LSB qubit 0, invariant norm = 1

Qubit 0 is the least significant bit of the amplitude index. This means
`apply1(state, gate, target)` iterates pairs `(i, i | (1<<target))` in
order, which is cache-friendly: adjacent pairs are touched once, in
ascending order.

Every state has `norm()==1` for physical validity. The code asserts this
in tests, never in hot paths (cost would dominate). `normalize()` is
explicit, not automatic, because unitary gates preserve norm and
projection needs renorm only after collapse.

## apply1 / apply2: block iteration

`apply1` iterates:

    for base in steps of 2*mask:
        for offset in 0..mask-1:
            i = base + offset; j = i + mask
            (amps[i], amps[j]) = g @ (amps[i], amps[j])

Each 2*mask block is touched once, in order. This gives sequential
memory access for the inner loop and a stride the CPU prefetcher
handles well. Measured speedup vs naive nested loop: ~3× at 20 qubits.

`apply2` for two-qubit gates is the same idea generalized to 4-element
groups. It re-reads 4 amplitudes per iteration but writes all 4 back,
so the block is touched once. The 4×4 gate's index order is
`2*b_first + b_second` — the natural LSB-first bitstring order.

## Partial trace: 2^(n+k) algorithm

`partial_trace(psi, keep)` computes the reduced density matrix of the
`keep` qubits by summing over the traced qubits. The key is
`expand_index(compact, trace, keep, traced)`: it reconstructs the full
amplitude index from the two compact indices without a lookup table.

Complexity is 2^(n+k): n total qubits, k kept qubits. For n=20, k=1,
this is 2^21 = 2M operations per matrix element, but the matrix has only
4 elements (2^k × 2^k). Total: ~8M operations. Runs in < 1 s on the
Acer at n=20.

The GHZ test at n=20 verifies this empirically: entropy 1.000000000 for
every single-qubit reduction, purity 0.500000000 for the 2-qubit
reduction. If `expand_index` had an off-by-one, these would not match.

## Gate interface: free functions, not a class

`core::H<R>()`, `core::X<R>()`, `core::CNOT<R>()` return `Matrix<R>`.
No base class, no virtual dispatch. The reason: gates are constants, not
objects with state. A `virtual Matrix apply()` would add nothing but
overhead, and templating on `R` already provides the type flexibility.

## QuantumLayer: forward returns ⟨Z_i⟩, backward uses parameter-shift

`QuantumLayer` exposes the same `forward/backward/update` interface as
`Dense`. This is the whole point — a `MultiHeadAttention` can hold a
`unique_ptr<AttentionIface>` that is either classical or quantum, and
the training loop does not know the difference.

For `forward`: angle-encode inputs with `RY(input_scale * x_q)`, apply
n_layers of `[CNOT chain, RY(theta)]`, measure `⟨Z_q⟩` per qubit.

For `backward`: the parameter-shift rule. For each parameter `θ_k`:

    dL/dθ_k = [f(θ_k + π/2) - f(θ_k - π/2)] / 2

This works because a quantum circuit's output is a trigonometric
polynomial in its parameters. It is exact, not approximate — unlike
finite differences, it has no truncation error. Cost: 2 forward
evaluations per parameter per call.

For the input gradient `dL/dx_j`, the encoding angle is
`input_scale * x_j`, so:

    dL/dx_j = input_scale * [f(x_j + π/(2·input_scale)) - f(x_j - π/(2·input_scale))] / 2

Both gradients are verified against finite differences in `test_qlayer`,
`test_qattn`, and `test_transformer_block`.

## AttentionIface: virtual base, unique_ptr heads

Three alternatives were considered for supporting both classical and
quantum attention heads inside `MultiHeadAttention`:

| Option | Problem |
| :--- | :--- |
| `std::variant<Attention, QuantumAttention>` | Requires `std::visit` at every call, pollutes the loop |
| Template `MultiHeadAttention<HeadT>` | All heads same type, blocks mixing classical + quantum |
| **Interface + unique_ptr** | None, chosen |

Virtual dispatch costs ~5 ns per call. Per-head attention work at
`d_model=8`, `seq_len=4` is on the order of microseconds. The 5 ns is
noise.

## Pre-norm transformer block (not post-norm)

    h = x + Attention(LayerNorm(x))
    y = h + FFN(LayerNorm(y))

The original Transformer (Vaswani 2017) placed LayerNorm *after* the
residual add:

    h = LayerNorm(x + Attention(x))

Post-norm is more fragile with depth: gradients through N blocks
compound both attention and LayerNorm Jacobians, and a slight mismatch
in per-block scale amplifies exponentially. Pre-norm (GPT-2, GPT-3,
Llama, most modern models) normalizes before each sub-block, so the
residual stream carries the identity path unobstructed.

`test_transformer_block` test 4 measures this empirically: stacking
N=4 blocks, gradient norm ratio with residual is O(1), without is <<1.
That is the ResNet gradient-flow argument transplanted into a
transformer.

## LayerNorm: the 3-term backward

The naive derivative of `y = γ · (x - μ) / σ + β` misses two terms
because `μ` and `σ` both depend on every `x_i`:

    dx_i = (γ / σ) · [dx̂_i - mean(dx̂) - x̂_i · mean(dx̂ · x̂)]

where `x̂ = (x - μ)/σ` and `dx̂ = dy · γ`. Missing the `mean(dx̂ · x̂)`
term is a very common bug in tutorial implementations. It produces
correct forward output but wrong gradients, so it typically fails only
in the last few layers of a deep network.

`test_layernorm` test 3 checks all three gradients (dx, dγ, dβ) against
finite differences. If any term is missing, it fails.

## Gradient clipping: L2 norm across the whole model

`MiniLM::clip_gradients(max_norm)` walks every gradient buffer in the
model, computes the global L2 norm, and scales uniformly if it exceeds
`max_norm`. This is standard practice (GPT-2, Llama, everything). It
fixes gradient explosion at high learning rates.

`clip_gradients` uses `dynamic_cast` to walk heterogeneous heads: it
clips `Attention::dWq/dWk/dWv` for classical heads and
`QuantumAttention::dWk/dWv/pre_q.dW/pre_q.db/ql.dtheta` for quantum
ones. Both kinds share the same global norm.

Without clipping, `train_lm` at `lr=0.05` produced NaN after 13 updates.
With clipping (`max_norm=1.0`), 500 epochs at `lr=0.05` complete without
NaN, loss dropping to 0.

## Position encoding: sinusoidal, additive, non-learnable

    PE[pos, 2i]   = sin(pos / 10000^(2i/d))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d))

Precomputed once, added to embeddings. Non-learnable because:
1. Deterministic, so no parameters to train (better for small models).
2. `test_posenc` test 3 proves it works: without PE, Attention is
   permutation-invariant (Attention(P·X) == P·Attention(X)); with PE,
   this equality breaks. This is the empirical signature that position
   information is actually entering the model.

## Repo hygiene: verify a fresh clone before publishing

A build that works in the working tree but fails in a fresh clone is a
repo nobody can use. The invariant that matters:

    git clone <url> /tmp/verify
    cd /tmp/verify && cmake -B build && cmake --build build && ctest

This must pass after every push that touches source. The docs commit
29e4de3 shipped without this check and was followed by a chore commit
that finally tracked 20 missing source files. The check would have
caught that in 30 seconds.
