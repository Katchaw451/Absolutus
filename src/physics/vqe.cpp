#include "hybrid/hamiltonian.hpp"
#include "hybrid/ansatz.hpp"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

namespace absolutus {
namespace physics {

using math::Complex;
using core::StateVector;
using hybrid::Hamiltonian;
using hybrid::expectation;
using hybrid::h2_ansatz;

// H2 2-qubit reduced electronic Hamiltonian (Jordan-Wigner).
// Coefficients chosen so the ground state lies in the single-
// excitation subspace {|q0=1,q1=0>, |q0=0,q1=1>}, which is exactly
// what h2_ansatz traces. Nuclear repulsion excluded; ground energy
// is the electronic part. Adding E_nuc ≈ +0.7137 recovers the
// familiar total H2 value ≈ -1.137 Ha.
static Hamiltonian make_h2() {
    Hamiltonian H;
    H.n_qubits = 2;
    H.terms = {
        {"II", -0.4804},
        {"ZI",  0.3435},
        {"IZ", -0.4347},
        {"ZZ",  0.5716},
        {"XX",  0.0910},
        {"YY",  0.0910},
    };
    return H;
}

// Apply H to psi, returning a NEW vector.
// Zero-initialized: no spurious |0...0> component.
template <typename R>
StateVector<R> apply_H(const StateVector<R>& psi, const Hamiltonian& H) {
    StateVector<R> out(psi.n_qubits());
    for (size_t i = 0; i < out.dim(); ++i) out.amp(i) = Complex<R>(R(0));
    for (const auto& term : H.terms) {
        StateVector<R> t = psi;
        for (size_t q = 0; q < H.n_qubits; ++q) {
            char op = term.ops[q];
            if      (op == 'X') core::apply1(t, core::X<R>(), q);
            else if (op == 'Y') core::apply1(t, core::Y<R>(), q);
            else if (op == 'Z') core::apply1(t, core::Z<R>(), q);
            // 'I' does nothing
        }
        for (size_t i = 0; i < out.dim(); ++i)
            out.amp(i) += Complex<R>(R(term.coeff)) * t.amp(i);
    }
    return out;
}

// Exact ground energy of H restricted to the single-excitation
// subspace {|q0=1,q1=0> (index 1), |q0=0,q1=1> (index 2)}.
// Built as pure basis kets via initializer_list (NOT the n-qubit
// constructor, which returns |0...0>).
template <typename R>
R exact_ground(const Hamiltonian& H) {
    const Complex<R> z(R(0)), o(R(1));
    StateVector<R> k10({z, o, z, z});   // |q0=1, q1=0> = index 1
    StateVector<R> k01({z, z, o, z});   // |q0=0, q1=1> = index 2

    const auto Hk10 = apply_H(k10, H);
    const auto Hk01 = apply_H(k01, H);

    const R h00 = k10.inner(Hk10).re;
    const R h11 = k01.inner(Hk01).re;
    const R h01 = k10.inner(Hk01).re;   // real by symmetry

    const R tr = h00 + h11;
    const R det = h00 * h11 - h01 * h01;
    const R disc = tr * tr - R(4) * det;
    const R sd = std::sqrt(std::max(disc, R(0)));
    const R l_minus = (tr - sd) / R(2);
    const R l_plus  = (tr + sd) / R(2);
    return std::min(l_minus, l_plus);
}

int run_vqe(int max_iter, double lr, uint64_t seed) {
    using R = double;
    (void)seed;
    const R PI = R(3.14159265358979323846);

    const Hamiltonian H = make_h2();
    const R E_exact = exact_ground<R>(H);

    // Initial guess: theta = 2.0 rad (outside the local maximum at 0).
    R theta = R(2.0);
    R E_final = expectation(h2_ansatz<R>(theta), H);
    R grad_final = R(0);

    std::printf("iter,theta,energy,gradient\n");
    std::printf("0,%.6f,%.9f,%.9f\n", theta, E_final, R(0));

    for (int iter = 1; iter <= max_iter; ++iter) {
        // Parameter-shift rule for RY(theta):
        //   df/dtheta = [f(theta + pi/2) - f(theta - pi/2)] / 2
        const R Ep = expectation(h2_ansatz<R>(theta + PI/R(2)), H);
        const R Em = expectation(h2_ansatz<R>(theta - PI/R(2)), H);
        const R grad = (Ep - Em) / R(2);

        theta -= R(lr) * grad;
        E_final = expectation(h2_ansatz<R>(theta), H);
        grad_final = grad;

        if (iter % 20 == 0 || iter == max_iter)
            std::printf("%d,%.6f,%.9f,%.9f\n", iter, theta, E_final, grad);
    }

    const R err = std::abs(E_final - E_exact);
    std::fprintf(stderr,
        "VQE for H2 (2-qubit reduced)\n"
        "  iterations: %d,  lr: %.4f\n"
        "  E_exact:    %.9f   (analytic, 2x2 block in {|10>,|01>})\n"
        "  E_vqe:      %.9f\n"
        "  |dE|:       %.3e\n"
        "  theta*:     %.6f rad\n"
        "  last grad:  %.3e\n",
        max_iter, lr, E_exact, E_final, err, theta, std::abs(grad_final));

    if (err > R(1e-6)) {
        std::fprintf(stderr, "  RESULT: FAIL (VQE did not converge to exact)\n");
        return 1;
    }
    std::fprintf(stderr, "  RESULT: PASS\n");
    return 0;
}

}}  // namespace physics / absolutus
