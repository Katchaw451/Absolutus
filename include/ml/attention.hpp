#pragma once
#include "ml/attention_iface.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace absolutus {
namespace ml {

// Multi-head attention, classical.
// Shapes: X is [seq_len, d_model]. Q, K, V projections are
// [d_model, d_k]. Scores = Q K^T / sqrt(d_k), softmax per row, * V.
// Output: [seq_len, d_k].
class Attention : public AttentionIface {
public:
    size_t seq_len, d_model, d_k;
    double momentum;
    std::vector<double> Wq, Wk, Wv;   // [d_model * d_k]
    std::vector<double> dWq, dWk, dWv;
    std::vector<double> vWq, vWk, vWv;

    Attention(size_t seq_len_, size_t d_model_, size_t d_k_,
              std::mt19937_64& rng, double momentum_ = 0.9);

    // X: [seq_len, d_model]. Returns [seq_len, d_k].
    const std::vector<double>& forward(const std::vector<double>& X) override;

    // grad_out: [seq_len, d_k]. Returns grad_X: [seq_len, d_model].
    std::vector<double> backward(const std::vector<double>& grad_out) override;

    void update(double lr) override;

private:
    std::vector<double> X_, Q_, K_, V_, S_, A_, Y_;   // cached
};

}}
