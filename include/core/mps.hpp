#pragma once
#include "core/state.hpp"
#include <vector>

namespace absolutus {
namespace core {

// Matrix Product State (MPS).
//
// A pure state |ψ⟩ on n qubits is represented as a chain of tensors
//   A[0], A[1], ..., A[n-1]
// where A[k] has shape (χ_{k-1}, 2, χ_k), with χ_0 = χ_n = 1 and
// bond dimensions χ_k bounded by max_bond.
//
// Memory: O(n * max_bond^2 * 2) complex numbers instead of O(2^n).
// For a product state, max_bond=1 suffices (exact). For GHZ, max_bond=2.
// For a random state, exact representation requires χ = 2^(n/2); below
// that, the MPS is approximate.
//
// The empirical question this class exists to answer: for a given
// circuit (GHZ, Grover, ansatz), how does the required χ grow with
// depth, width, and entanglement? That is the boundary of classical
// simulability, which is what papers on "quantum advantage" measure.
//
// This first version implements:
//   - Conversion from StateVector via successive SVDs.
//   - Reconstruction to StateVector (for verification).
//   - Inspection of achieved bond dimensions and memory.
// Gate application is deferred to the next version.

template <typename R>
class MPS {
public:
    using C = math::Complex<R>;

    MPS() = default;

    // Convert a full StateVector to MPS. If max_bond < 2^(n/2), the MPS
    // is approximate (SVD truncation).
    static MPS from_statevector(const StateVector<R>& psi, size_t max_bond);

    // Reconstruct the full statevector. For exact MPS, matches the
    // original to machine precision. For truncated MPS, gives the
    // best rank-max_bond approximation.
    StateVector<R> to_statevector() const;

    size_t n_qubits() const { return n_qubits_; }
    size_t max_bond() const { return max_bond_; }

    // Bond dimensions actually used: χ_0, χ_1, ..., χ_n.
    const std::vector<size_t>& bond_dims() const { return bond_dims_; }

    // Memory footprint in bytes (for comparison against StateVector).
    size_t memory_bytes() const;

    // Fidelity with another statevector: |⟨ψ_mps|ψ_sv⟩|^2. Used to
    // measure the truncation error for approximate MPS.
    R fidelity(const StateVector<R>& psi) const;

private:
    size_t n_qubits_ = 0;
    size_t max_bond_ = 1;
    std::vector<size_t> bond_dims_;       // χ_0, χ_1, ..., χ_n
    std::vector<std::vector<C>> tensors_; // A[k], size χ_{k-1}*2*χ_k
};

using mps32 = MPS<float>;
using mps64 = MPS<double>;

}}
