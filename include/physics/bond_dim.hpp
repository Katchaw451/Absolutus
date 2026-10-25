#pragma once
#include "core/state.hpp"
#include <vector>

namespace absolutus {
namespace physics {

// For a pure state on n qubits, profile[k] = number of Schmidt values
// above threshold across the bipartition {0..k-1} | {k..n-1}.
// This equals the minimum chi required at cut k for an exact MPS
// representation at that cut. profile[0] = profile[n] = 1 by convention.
template <typename R>
std::vector<size_t> bond_profile(const core::StateVector<R>& psi,
                                 R threshold = R(1e-10));

int run_bond_dim(size_t n, int n_seeds, uint64_t seed);

}}
