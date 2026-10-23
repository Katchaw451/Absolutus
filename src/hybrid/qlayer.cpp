#include "hybrid/qlayer.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace absolutus {
namespace hybrid {

using math::Complex;
using core::StateVector;

static constexpr double PI = 3.14159265358979323846;

// <Z_q> = sum_i (-1)^{bit_q(i)} |amp_i|^2
static double exp_Z(const StateVector<double>& psi, size_t q) {
    double acc = 0.0;
    for (size_t i = 0; i < psi.dim(); ++i)
        acc += (((i >> q) & 1) ? -1.0 : 1.0) * psi.amp(i).norm_sq();
    return acc;
}

QuantumLayer::QuantumLayer(size_t nq, size_t np, std::mt19937_64& rng,
                           double mom, double iscale)
    : n_qubits(nq), n_params(np), momentum(mom), input_scale(iscale),
      theta(np), dtheta(np, 0.0), vtheta(np, 0.0),
      x_saved(nq, 0.0), y_saved(nq, 0.0) {
    if (np % nq != 0)
        throw std::invalid_argument("QuantumLayer: n_params must be multiple of n_qubits");
    std::uniform_real_distribution<double> dist(0.0, 2.0 * PI);
    for (auto& t : theta) t = dist(rng);
}

std::vector<double> QuantumLayer::run(const std::vector<double>& x,
                                     const std::vector<double>& th) const {
    if (x.size()  != n_qubits) throw std::invalid_argument("QLayer: x size");
    if (th.size() != n_params) throw std::invalid_argument("QLayer: th size");

    StateVector<double> psi(n_qubits);
    for (size_t q = 0; q < n_qubits; ++q)
        core::apply1(psi, core::RY<double>(input_scale * x[q]), q);

    const size_t n_layers = n_params / n_qubits;
    for (size_t l = 0; l < n_layers; ++l) {
        for (size_t q = 0; q + 1 < n_qubits; ++q)
            core::apply2(psi, core::CNOT<double>(), q, q + 1);
        for (size_t q = 0; q < n_qubits; ++q)
            core::apply1(psi, core::RY<double>(th[l * n_qubits + q]), q);
    }

    std::vector<double> out(n_qubits);
    for (size_t q = 0; q < n_qubits; ++q) out[q] = exp_Z(psi, q);
    return out;
}

const std::vector<double>& QuantumLayer::forward(const std::vector<double>& x) {
    x_saved = x;
    y_saved = run(x, theta);
    return y_saved;
}

std::vector<double> QuantumLayer::backward(const std::vector<double>& grad_y) {
    if (grad_y.size() != n_qubits)
        throw std::invalid_argument("QLayer: grad_y size");

    const double SHIFT = PI / 2.0;
    const double HALF  = 0.5;

    // dL/dtheta_k = sum_i grad_y[i] * d<Z_i>/dtheta_k
    std::fill(dtheta.begin(), dtheta.end(), 0.0);
    for (size_t k = 0; k < n_params; ++k) {
        auto th_p = theta; th_p[k] += SHIFT;
        auto th_m = theta; th_m[k] -= SHIFT;
        auto yp = run(x_saved, th_p);
        auto ym = run(x_saved, th_m);
        for (size_t i = 0; i < n_qubits; ++i)
            dtheta[k] += grad_y[i] * HALF * (yp[i] - ym[i]);
    }

    // dL/dx_j = sum_i grad_y[i] * d<Z_i>/dx_j
    // angle_j = input_scale * x_j, so d<Z>/dx = input_scale * d<Z>/dangle
    std::vector<double> grad_x(n_qubits, 0.0);
    const double dx = SHIFT / input_scale;
    for (size_t j = 0; j < n_qubits; ++j) {
        auto xp = x_saved; xp[j] += dx;
        auto xm = x_saved; xm[j] -= dx;
        auto yp = run(xp, theta);
        auto ym = run(xm, theta);
        for (size_t i = 0; i < n_qubits; ++i)
            grad_x[j] += grad_y[i] * input_scale * HALF * (yp[i] - ym[i]);
    }
    return grad_x;
}

void QuantumLayer::update(double lr) {
    for (size_t k = 0; k < n_params; ++k) {
        vtheta[k] = momentum * vtheta[k] + dtheta[k];
        theta[k] -= lr * vtheta[k];
    }
}

}}
