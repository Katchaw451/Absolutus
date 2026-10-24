#pragma once
#include "core/state.hpp"
#include "core/gates.hpp"

namespace absolutus {
namespace hybrid {

// 2-qubit H2 UCCSD-style ansatz:
//   |00> -> X(1) -> |01> -> RY(theta, 0) -> CNOT(0,1)
// This traces the path: cos(theta/2)|01> + sin(theta/2)|10>
// The whole family lives in the {|01>, |10>} subspace,
// which is the particle-number-1 sector where the H2 ground state lives.
template <typename R>
core::StateVector<R> h2_ansatz(R theta) {
    core::StateVector<R> psi(2);
    core::apply1(psi, core::X<R>(), 1);       // |00> -> |01>  (HF state)
    core::apply1(psi, core::RY<R>(theta), 0); // mix |0>, |1> on qubit 0
    core::apply2(psi, core::CNOT<R>(), 0, 1); // entangle: |11> -> |10>
    return psi;
}

}}  // namespace hybrid / absolutus
