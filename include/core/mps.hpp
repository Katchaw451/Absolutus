#pragma once
#include "core/state.hpp"
#include <vector>

namespace absolutus {
namespace core {

// Matrix Product State (MPS).
//
// A pure state |psi> on n qubits is represented as a chain of tensors
//   A[0], A[1], ..., A[n-1]
// where A[k] has shape (chi_{k-1}, 2, chi_k), with chi_0 = chi_n = 1
// and bond dimensions chi_k bounded by max_bond.
//
// Memory: O(n * max_bond^2 * 2) complex numbers instead of O(2^n).
// For a product state, max_bond=1 suffices (exact). For GHZ, max_bond=2.
// For a random state, exact representation requires chi = 2^(n/2); below
// that, the MPS is approximate.
//
// The empirical question this class exists to answer: for a given
// circuit (GHZ, Grover, ansatz), how does the required chi grow with
// depth, width, and entanglement? That is the boundary of classical
// simulability.
//
// This first version implements:
//   - Conversion from StateVector via successive SVDs.
//   - Reconstruction to StateVector (for verification).
//   - Inspection of achieved bond dimensions and memory.
// Gate application is deferred to a later version.

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

    // Bond dimensions actually used: chi_0, chi_1, ..., chi_n.
    const std::vector<size_t>& bond_dims() const { return bond_dims_; }

    // Memory footprint in bytes (for comparison against StateVector).
    size_t memory_bytes() const;

    // Fidelity with another statevector: |<mps|sv>|^2.
    R fidelity(const StateVector<R>& psi) const;

private:
    size_t n_qubits_ = 0;
    size_t max_bond_ = 1;
    std::vector<size_t> bond_dims_;        // chi_0, chi_1, ..., chi_n
    std::vector<std::vector<C>> tensors_;  // A[k], size chi_{k-1}*2*chi_k
};

using mps32 = MPS<float>;
using mps64 = MPS<double>;

}}
