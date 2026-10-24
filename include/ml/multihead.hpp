#pragma once
#include "ml/attention_iface.hpp"
#include <memory>
#include <random>
#include <vector>

namespace absolutus {
namespace ml {

// Multi-head attention with a heterogeneous head set.
// By default all heads are classical Attention. Pass quantum_head_idx=h
// (0 <= h < n_heads) to make head h a QuantumAttention circuit; the rest
// remain classical. Head outputs are concatenated and projected by Wo.
class MultiHeadAttention {
public:
    size_t seq_len, d_model, n_heads, d_k;
    double momentum;
    std::vector<std::unique_ptr<AttentionIface>> heads;
    std::vector<double> Wo, dWo, vWo;

    MultiHeadAttention(size_t sl, size_t dm, size_t nh, std::mt19937_64& rng,
                       double momentum_ = 0.9,
                       int quantum_head_idx = -1,
                       size_t quantum_ql_layers = 1,
                       bool causal = false);

    const std::vector<double>& forward(const std::vector<double>& X);
    std::vector<double> backward(const std::vector<double>& grad_out);
    void update(double lr);

private:
    std::vector<double> X_, concat_, Y_;
};

}}
