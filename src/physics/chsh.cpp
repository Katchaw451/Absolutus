#include "core/gates.hpp"
#include "core/state.hpp"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <random>

namespace absolutus {
namespace physics {

using math::Complex;
using math::Matrix;
using core::StateVector;

template <typename R>
static R expectation(const StateVector<R>& psi,
                     const Matrix<R>& A, const Matrix<R>& B) {
    auto tmp = psi;
    core::apply1(tmp, A, 0);
    core::apply1(tmp, B, 1);
    return psi.inner(tmp).re;
}

int run_chsh(int n_trials, uint64_t seed) {
    using R = double;
    const R PI = R(3.14159265358979323846);

    // Bell state |Phi+> = (|00> + |11>) / sqrt(2)
    StateVector<R> bell(2);
    core::apply1(bell, core::H<R>(), 0);
    core::apply2(bell, core::CNOT<R>(), 0, 1);

    // Observables in XZ plane
    auto Z = core::Z<R>();
    auto X = core::X<R>();
    const R s = R(1) / std::sqrt(R(2));

    Matrix<R> B0(2, 2), B1(2, 2);
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        B0.at(i,j) = (Z.at(i,j) + X.at(i,j)) * Complex<R>(s);
        B1.at(i,j) = (Z.at(i,j) - X.at(i,j)) * Complex<R>(s);
    }

    // Exact expectation values (direct sandwich)
    const R E00 = expectation(bell, Z, B0);
    const R E01 = expectation(bell, Z, B1);
    const R E10 = expectation(bell, X, B0);
    const R E11 = expectation(bell, X, B1);
    const R S_exact = E00 + E01 + E10 - E11;

    // Sampled: basis rotation (RY(-theta)) + Z measurement
    struct Meas { R theta; };
    const Meas a0{0}, a1{PI/2}, b0{PI/4}, b1{-PI/4};

    std::mt19937_64 rng(seed);

    auto sample_E = [&](Meas A, Meas B) -> R {
        StateVector<R> psi = bell;
        core::apply1(psi, core::RY<R>(-A.theta), 0);
        core::apply1(psi, core::RY<R>(-B.theta), 1);
        auto probs = psi.probabilities();
        std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
        long sum = 0;
        for (int t = 0; t < n_trials; ++t) {
            const size_t idx = dist(rng);
            const int za = (idx & 1) ? -1 : +1;
            const int zb = (idx & 2) ? -1 : +1;
            sum += za * zb;
        }
        return R(sum) / R(n_trials);
    };

    const R sE00 = sample_E(a0, b0);
    const R sE01 = sample_E(a0, b1);
    const R sE10 = sample_E(a1, b0);
    const R sE11 = sample_E(a1, b1);
    const R S_samp = sE00 + sE01 + sE10 - sE11;

    // CSV to stdout (for bash orchestration)
    std::printf("experiment,trials,seed,E00,E01,E10,E11,S_exact,S_sampled\n");
    std::printf("chsh,%d,%" PRIu64 ",%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                n_trials, seed,
                E00, E01, E10, E11, S_exact, S_samp);

    // Human summary to stderr (so stdout stays pure CSV)
    const R tsirelson = R(2) * std::sqrt(R(2));
    std::fprintf(stderr,
        "CHSH experiment on |Phi+>\n"
        "  exact:   E00=%+.6f  E01=%+.6f  E10=%+.6f  E11=%+.6f\n"
        "           S = %+.6f  (Tsirelson = %.6f)\n"
        "  sampled: E00=%+.6f  E01=%+.6f  E10=%+.6f  E11=%+.6f  (N=%d)\n"
        "           S = %+.6f\n"
        "  classical bound: 2.000000\n"
        "  violation (exact):  %+.6f\n"
        "  violation (sampled):%+.6f\n",
        E00, E01, E10, E11, S_exact, tsirelson,
        sE00, sE01, sE10, sE11, n_trials, S_samp,
        std::abs(S_exact) - R(2), std::abs(S_samp) - R(2));

    // Physics assertions
    if (std::abs(std::abs(S_exact) - tsirelson) > R(1e-9)) {
        std::fprintf(stderr, "FAIL: |S_exact| != 2*sqrt(2)\n");
        return 1;
    }
    if (std::abs(S_samp) <= R(2)) {
        std::fprintf(stderr, "FAIL: sampled |S| did not exceed 2\n");
        return 1;
    }
    std::fprintf(stderr, "  RESULT: PASS\n");
    return 0;
}

} // namespace physics
} // namespace absolutus
