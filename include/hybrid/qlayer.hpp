#pragma once
#include "core/state.hpp"
#include "core/gates.hpp"
#include "math/complex.hpp"
#include <random>
#include <vector>

namespace absolutus {
namespace hybrid {

// Quantum layer as a drop-in ML component.
// Circuit: RY(input_scale * x_q) per qubit, then n_layers of
//   CNOT chain (q -> q+1), RY(theta_{l*n_qubits+q}) per qubit.
// Outputs: <Z_q> for each qubit, in [-1, +1].
// Gradients: parameter-shift on both trainable params and inputs.
class QuantumLayer {
public:
    size_t n_qubits;
    size_t n_params;
    double momentum;
    double input_scale;
    std::vector<double> theta, dtheta, vtheta;

    QuantumLayer(size_t nq, size_t np, std::mt19937_64& rng,
                 double momentum_ = 0.9,
                 double input_scale_ = 3.14159265358979323846);

    const std::vector<double>& forward(const std::vector<double>& x);
    std::vector<double> backward(const std::vector<double>& grad_y);
    void update(double lr);

private:
    std::vector<double> x_saved, y_saved;
    std::vector<double> run(const std::vector<double>& x,
                            const std::vector<double>& th) const;
};

}}
