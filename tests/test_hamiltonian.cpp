#include "hybrid/hamiltonian.hpp"
#include "hybrid/ansatz.hpp"
#include <cstdio>
#include <cmath>

using namespace absolutus;
using C  = math::Complex<double>;
using SV = core::StateVector<double>;
using hybrid::Hamiltonian;
using hybrid::expectation;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

static Hamiltonian make_h2() {
    Hamiltonian H;
    H.n_qubits = 2;
    H.terms = {
        {"II", -0.4804}, {"ZI",  0.3435}, {"IZ", -0.4347},
        {"ZZ",  0.5716}, {"XX",  0.0910}, {"YY",  0.0910},
    };
    return H;
}

int main() {
    const Hamiltonian H = make_h2();

    // Basis kets: |q0=1,q1=0> = index 1, |q0=0,q1=1> = index 2
    const SV k10({C(0), C(1), C(0), C(0)});
    const SV k01({C(0), C(0), C(1), C(0)});
    const SV k00({C(1), C(0), C(0), C(0)});
    const SV k11({C(0), C(0), C(0), C(1)});

    CHECK(std::abs(expectation(k00, H) - 0.0) < 1e-4);
    CHECK(std::abs(expectation(k11, H) - 0.1824) < 1e-4);
    CHECK(std::abs(expectation(k10, H) - (-1.8302)) < 1e-4);
    CHECK(std::abs(expectation(k01, H) - (-0.2738)) < 1e-4);

    // Superposition |+> = (k10 + k01)/sqrt(2): off-diagonal contribution
    SV plus({C(0), C(1), C(1), C(0)});
    plus.normalize();
    // <+|H|+> = (h10 + h01)/2 + h_offdiag = -1.052 + 0.182 = -0.870
    CHECK(std::abs(expectation(plus, H) - (-0.870)) < 1e-4);

    SV minus({C(0), C(1), C(-1), C(0)});
    minus.normalize();
    CHECK(std::abs(expectation(minus, H) - (-1.234)) < 1e-4);

    // Analytic ground energy of the 2x2 block in (k10, k01)
    const double h10 = -1.8302, h01 = -0.2738, hx = 0.182;
    const double E0 = (h10 + h01)/2 - std::sqrt(std::pow((h10-h01)/2, 2) + hx*hx);
    CHECK(std::abs(E0 - (-1.8512)) < 1e-4);

    // The ansatz at the VQE minimum theta* ~ 3.371336 reaches E0
    const double theta_star = 3.371336;
    CHECK(std::abs(expectation(hybrid::h2_ansatz(theta_star), H) - E0) < 1e-4);

    return 0;
}
