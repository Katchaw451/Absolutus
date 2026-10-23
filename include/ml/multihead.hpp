#pragma once
#include "ml/attention.hpp"
#include <random>
#include <vector>

namespace absolutus {
namespace ml {

// Multi-head attention: h independent Attention blocks, each with
// d_k = d_model / h, concatenated and projected with Wo [d_model, d_model].
//
//   per head: y_h = Attention_h(X)         [seq_len, d_k]
//   concat:   C = [y_0 | y_1 | ... | y_{h-1}]  [seq_len, d_model]
//   output:   Y = C Wo^T                    [seq_len, d_model]
//
// Wo initialized to identity so h=1 reproduces single Attention.
class MultiHeadAttention {
public:
    size_t seq_len, d_model, n_heads, d_k;
    double momentum;
    std::vector<Attention> heads;
    std::vector<double> Wo, dWo, vWo;

    MultiHeadAttention(size_t sl, size_t dm, size_t nh, std::mt19937_64& rng,
                       double momentum_ = 0.9);

    const std::vector<double>& forward(const std::vector<double>& X);
    std::vector<double> backward(const std::vector<double>& grad_out);
    void update(double lr);

private:
    std::vector<double> X_, concat_, Y_;
};

}}
