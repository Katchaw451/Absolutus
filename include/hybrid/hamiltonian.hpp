#pragma once
#include "core/state.hpp"
#include "core/gates.hpp"
#include <vector>
#include <string>
#include <stdexcept>

namespace absolutus {
namespace hybrid {

// Pauli string in our qubit convention (qubit 0 = LSB in state indexing).
// Example: "ZI" means Z on qubit 0, I on qubit 1.
struct PauliTerm {
    std::string ops;
    double coeff;
};

struct Hamiltonian {
    size_t n_qubits = 0;
    std::vector<PauliTerm> terms;
};

// <psi|H|psi> = sum_t coeff_t * <psi|P_t|psi>
// Applies Pauli strings directly to a copy of psi; O(terms * n * 2^n).
template <typename R>
R expectation(const core::StateVector<R>& psi, const Hamiltonian& H) {
    R acc = R(0);
    for (const auto& term : H.terms) {
        if (term.ops.size() != H.n_qubits)
            throw std::invalid_argument("expectation: term length mismatch");
        core::StateVector<R> tmp = psi;
        for (size_t q = 0; q < H.n_qubits; ++q) {
            char op = term.ops[q];
            if      (op == 'X') core::apply1(tmp, core::X<R>(), q);
            else if (op == 'Y') core::apply1(tmp, core::Y<R>(), q);
            else if (op == 'Z') core::apply1(tmp, core::Z<R>(), q);
            else if (op != 'I') throw std::invalid_argument("expectation: bad Pauli");
        }
        acc += R(term.coeff) * psi.inner(tmp).re;
    }
    return acc;
}

}}  // namespace hybrid / absolutus
