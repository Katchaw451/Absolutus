#pragma once

#include "core/state.hpp"
#include "core/gates.hpp"
#include "math/matrix.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace absolutus {
namespace core {

using math::Complex;
using math::Matrix;

// <psi|O|psi> for a Hermitian operator O of dimension 2^n.
template <typename R>
R expectation(const StateVector<R>& psi, const Matrix<R>& O) {
    if (O.rows != psi.dim() || O.cols != psi.dim())
        throw std::invalid_argument("expectation: dim mismatch");
    Complex<R> acc;
    for (size_t i = 0; i < psi.dim(); ++i) {
        Complex<R> row;
        for (size_t j = 0; j < psi.dim(); ++j)
            row += O.at(i, j) * psi.amp(j);
        acc += psi.amp(i).conj() * row;
    }
    return acc.re;
}

// <psi| (O_a ⊗ O_b ⊗ ...) |psi> with operators on distinct qubits.
// Operators on different qubits commute, so order of application is free.
template <typename R>
R expectation_local(const StateVector<R>& psi,
                    const std::vector<Matrix<R>>& gates,
                    const std::vector<size_t>& qubits) {
    if (gates.size() != qubits.size())
        throw std::invalid_argument("expectation_local: mismatch");
    StateVector<R> tmp = psi;
    for (size_t k = 0; k < gates.size(); ++k)
        apply1(tmp, gates[k], qubits[k]);
    return psi.inner(tmp).re;
}

// Build the full index from a "compact" index over kept qubits
// and a "trace" index over the traced-out qubits.
template <typename R>
size_t expand_index(size_t compact, size_t trace,
                    const std::vector<size_t>& keep,
                    const std::vector<size_t>& traced) {
    size_t full = 0;
    for (size_t b = 0; b < keep.size(); ++b)
        if (compact & (size_t(1) << b))
            full |= size_t(1) << keep[b];
    for (size_t b = 0; b < traced.size(); ++b)
        if (trace & (size_t(1) << b))
            full |= size_t(1) << traced[b];
    return full;
}

// Partial trace: keep `keep` qubits, trace out the rest.
// Returns the reduced density matrix rho of size 2^k × 2^k (row-major).
// Complexity: 2^(n+k) for n total qubits and k kept.
template <typename R>
Matrix<R> partial_trace(const StateVector<R>& psi,
                        const std::vector<size_t>& keep) {
    const size_t n = psi.n_qubits();
    const size_t k = keep.size();
    if (k > n) throw std::invalid_argument("partial_trace: too many qubits");

    std::vector<bool> is_kept(n, false);
    for (auto q : keep) {
        if (q >= n) throw std::out_of_range("partial_trace: qubit range");
        is_kept[q] = true;
    }
    std::vector<size_t> traced;
    for (size_t q = 0; q < n; ++q)
        if (!is_kept[q]) traced.push_back(q);

    const size_t dim_keep  = size_t(1) << k;
    const size_t dim_trace = size_t(1) << traced.size();

    Matrix<R> rho(dim_keep, dim_keep);
    for (size_t ci = 0; ci < dim_keep; ++ci) {
        for (size_t cj = 0; cj < dim_keep; ++cj) {
            Complex<R> acc;
            for (size_t t = 0; t < dim_trace; ++t) {
                const size_t i = expand_index<R>(ci, t, keep, traced);
                const size_t j = expand_index<R>(cj, t, keep, traced);
                acc += psi.amp(i) * psi.amp(j).conj();
            }
            rho.at(ci, cj) = acc;
        }
    }
    return rho;
}

// Von Neumann entropy S = -Σ λᵢ log₂ λᵢ for a 2×2 density matrix.
// λ = (tr ± √(tr² − 4·det)) / 2  (Hermitian, unit trace).
template <typename R>
R entropy_1q(const Matrix<R>& rho) {
    if (rho.rows != 2 || rho.cols != 2)
        throw std::invalid_argument("entropy_1q: need 2x2");
    const R tr = (rho.at(0, 0) + rho.at(1, 1)).re;
    const Complex<R> det = rho.at(0, 0) * rho.at(1, 1)
                         - rho.at(0, 1) * rho.at(1, 0);
    const R disc = tr * tr - R(4) * det.re;
    if (disc < R(-1e-12))
        throw std::runtime_error("entropy_1q: non-Hermitian input");
    const R sq = std::sqrt(disc > R(0) ? disc : R(0));
    const R l1 = (tr + sq) / R(2);
    const R l2 = (tr - sq) / R(2);
    R S = R(0);
    auto add = [&S](R l) { if (l > R(1e-12)) S -= l * std::log2(l); };
    add(l1);
    add(l2);
    return S;
}

// Fidelity |<psi|phi>|² between pure states.
template <typename R>
R fidelity(const StateVector<R>& a, const StateVector<R>& b) {
    if (a.dim() != b.dim())
        throw std::invalid_argument("fidelity: dim mismatch");
    Complex<R> acc;
    for (size_t i = 0; i < a.dim(); ++i)
        acc += a.amp(i).conj() * b.amp(i);
    return acc.norm_sq();
}

} // namespace core
} // namespace absolutus
