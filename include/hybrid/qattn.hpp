#pragma once
#include "hybrid/qlayer.hpp"
#include "ml/mlp.hpp"
#include <random>
#include <vector>

namespace absolutus {
namespace hybrid {

// Attention where the Q projection is a quantum circuit.
//   X [seq_len, d_model]
//     -> pre_q (Dense: d_model -> n_qubits, tanh) [seq_len, n_qubits]
//     -> ql (QuantumLayer: n_qubits -> n_qubits)  [seq_len, n_qubits] = Q
//   K, V produced classically: Wk, Wv [d_model, n_qubits]
//   d_k = n_qubits (one <Z> per qubit)
//   Y = softmax(Q K^T / sqrt(d_k)) V
//
// Gradients:
//   - Classical path (K, V, and pre_q's Dense): backprop
//   - Quantum path (ql.theta): parameter-shift via ql.backward
// Both compose in a single backward pass through the attention math.
class QuantumAttention {
public:
    size_t seq_len, d_model, n_qubits;
    double momentum;

    std::vector<double> Wk, Wv, dWk, dWv, vWk, vWv;
    ml::Dense pre_q;
    QuantumLayer ql;

    QuantumAttention(size_t sl, size_t dm, size_t nq,
                     size_t ql_layers, std::mt19937_64& rng,
                     double momentum_ = 0.9);

    const std::vector<double>& forward(const std::vector<double>& X);
    std::vector<double> backward(const std::vector<double>& grad_out);
    void update(double lr);

private:
    std::vector<double> X_, Q_, K_, V_, S_, A_, Y_;
    std::vector<double> pre_q_acts_;   // [seq_len * n_qubits]
};

}}
