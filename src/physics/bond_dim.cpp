#include "physics/bond_dim.hpp"
#include "core/gates.hpp"
#include "core/measure.hpp"
#include "math/eig.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace absolutus {
namespace physics {

using math::Complex;
using math::eig_hermitian;
using core::StateVector;

template <typename R>
std::vector<size_t> bond_profile(const StateVector<R>& psi, R threshold) {
    const size_t n = psi.n_qubits();
    std::vector<size_t> profile(n + 1, 1);
    for (size_t k = 1; k < n; ++k) {
        std::vector<size_t> keep(k);
        for (size_t i = 0; i < k; ++i) keep[i] = i;
        auto rho = core::partial_trace(psi, keep);
        const size_t d = rho.rows;
        std::vector<Complex<double>> H(d * d);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j)
                H[i*d + j] = Complex<double>((double)rho.at(i, j).re,
                                             (double)rho.at(i, j).im);
        std::vector<double> evals;
        std::vector<Complex<double>> evecs;
        eig_hermitian(H, d, evals, evecs);
        size_t cnt = 0;
        for (auto e : evals) if (e > (double)threshold) ++cnt;
        profile[k] = cnt;
    }
    return profile;
}

template std::vector<size_t> bond_profile<double>(const StateVector<double>&, double);
template std::vector<size_t> bond_profile<float>(const StateVector<float>&, float);

static StateVector<double> make_product(size_t n) {
    StateVector<double> s(n);
    for (size_t q = 0; q < n; ++q) core::apply1(s, core::H<double>(), q);
    return s;
}

static StateVector<double> make_ghz(size_t n) {
    StateVector<double> s(n);
    core::apply1(s, core::H<double>(), 0);
    for (size_t q = 1; q < n; ++q)
        core::apply2(s, core::CNOT<double>(), 0, q);
    return s;
}

static StateVector<double> make_w(size_t n) {
    StateVector<double> s(n);
    const double a = 1.0 / std::sqrt((double)n);
    for (size_t q = 0; q < n; ++q)
        s.amp(size_t(1) << q) = Complex<double>(a, 0.0);
    return s;
}

static StateVector<double> make_random_circuit(size_t n, size_t depth,
                                               std::mt19937_64& rng) {
    StateVector<double> s(n);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (size_t d = 0; d < depth; ++d) {
        for (size_t q = 0; q < n; ++q)
            core::apply1(s, core::RY<double>(u(rng) * 6.283185307179586), q);
        for (size_t q = 0; q + 1 < n; ++q)
            core::apply2(s, core::CNOT<double>(), q, q + 1);
    }
    return s;
}

int run_bond_dim(uint64_t seed) {
    std::mt19937_64 rng(seed);
    const size_t n = 10;
    std::printf("state,n,max_bond,profile\n");
    auto report = [&](const char* name, const StateVector<double>& s) {
        auto p = bond_profile(s, 1e-10);
        size_t mb = 0;
        for (auto x : p) if (x > mb) mb = x;
        std::printf("%s,%zu,%zu,\"", name, n, mb);
        for (size_t k = 0; k <= n; ++k) {
            std::printf("%zu%s", p[k], k < n ? " " : "");
        }
        std::printf("\"\n");
    };
    report("product",   make_product(n));
    report("ghz",       make_ghz(n));
    report("w",         make_w(n));
    report("rand_d1",   make_random_circuit(n, 1,  rng));
    report("rand_d3",   make_random_circuit(n, 3,  rng));
    report("rand_d10",  make_random_circuit(n, 10, rng));
    return 0;
}

}}
