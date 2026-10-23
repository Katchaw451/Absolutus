#include "core/gates.hpp"
#include "core/state.hpp"
#include "core/measure.hpp"
#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <cmath>
#include <stdexcept>
#include <string>

namespace absolutus {
namespace physics {

using math::Complex;
using math::Matrix;
using core::StateVector;

// GHZ_n = (|0...0> + |1...1>) / sqrt(2)
template <typename R>
StateVector<R> make_ghz(size_t n) {
    if (n < 2) throw std::invalid_argument("ghz: n >= 2");
    StateVector<R> s(n);
    core::apply1(s, core::H<R>(), 0);
    for (size_t q = 1; q < n; ++q)
        core::apply2(s, core::CNOT<R>(), 0, q);
    return s;
}

int run_ghz(size_t n_qubits, int n_seeds) {
    using R = double;
    if (n_qubits < 2 || n_qubits > 25)
        throw std::invalid_argument("ghz: n in [2, 25]");

    auto psi = make_ghz<R>(n_qubits);

    // Sanity: only |0...0> and |1...1> have amplitude
    const size_t last = psi.dim() - 1;
    const R r = R(1) / std::sqrt(R(2));
    R err = R(0);
    for (size_t i = 0; i < psi.dim(); ++i) {
        R expected = R(0);
        if (i == 0 || i == last) expected = r;
        err += std::abs(psi.amp(i).norm_sq() - expected * expected);
    }
    if (err > R(1e-9)) {
        std::fprintf(stderr, "GHZ build FAIL: amplitude mismatch (err=%.3e)\n", err);
        return 1;
    }

    // Reduced density matrix of qubit 0 must be I/2 => entropy 1
    auto rho0 = core::partial_trace(psi, {0});
    const R S0 = core::entropy_1q(rho0);
    const R entropy_err = std::abs(S0 - R(1));

    // Interior qubit: same result
    auto rho_mid = core::partial_trace(psi, {n_qubits / 2});
    const R S_mid = core::entropy_1q(rho_mid);

    // Two-qubit reduction {0,1}: rank-2 branch, purity 1/2
    auto rho_01 = core::partial_trace(psi, {0, 1});
    Complex<R> tr2;
    for (size_t i = 0; i < rho_01.rows; ++i)
        for (size_t j = 0; j < rho_01.cols; ++j)
            tr2 += rho_01.at(i, j) * rho_01.at(j, i);
    const R purity = tr2.re;
    const R purity_err = std::abs(purity - R(0.5));

    // <Z_0 Z_k> = +1 for every k
    R zz_sum = R(0);
    for (size_t k = 1; k < n_qubits; ++k)
        zz_sum += core::expectation_local(psi,
            {core::Z<R>(), core::Z<R>()}, {0, k});
    const R zz_mean = zz_sum / R(n_qubits - 1);
    const R zz_err = std::abs(zz_mean - R(1));

    // <X_0 X_1 ... X_{n-1}> = +1
    std::vector<Matrix<R>> xs(n_qubits);
    std::vector<size_t> idx(n_qubits);
    for (size_t q = 0; q < n_qubits; ++q) { xs[q] = core::X<R>(); idx[q] = q; }
    const R x_parity = core::expectation_local(psi, xs, idx);
    const R x_err = std::abs(x_parity - R(1));

    std::printf("experiment,n,entropy_q0,entropy_mid,purity_q01,"
                "<ZZ>_mean,<X_all>,entropy_err,purity_err,zz_err,x_err\n");
    std::printf("ghz,%zu,%.9f,%.9f,%.9f,%.9f,%.9f,%.3e,%.3e,%.3e,%.3e\n",
                n_qubits, S0, S_mid, purity, zz_mean, x_parity,
                entropy_err, purity_err, zz_err, x_err);

    std::fprintf(stderr,
        "GHZ_%zu = (|0..0> + |1..1>)/sqrt(2)\n"
        "  S(rho_0)      = %.9f   (expect 1)\n"
        "  S(rho_mid)    = %.9f   (expect 1)\n"
        "  Tr(rho_01^2)  = %.9f   (expect 0.5, rank-2 branch)\n"
        "  <Z_0 Z_k>_avg = %.9f   (expect 1 for all k)\n"
        "  <X_0..X_{n-1}> = %.9f  (expect 1)\n",
        n_qubits, S0, S_mid, purity, zz_mean, x_parity);

    if (entropy_err > R(1e-9) || purity_err > R(1e-9) ||
        zz_err > R(1e-9) || x_err > R(1e-9)) {
        std::fprintf(stderr, "  RESULT: FAIL\n");
        return 1;
    }
    std::fprintf(stderr, "  RESULT: PASS\n");
    (void)n_seeds;
    return 0;
}

} // namespace physics
} // namespace absolutus
