#pragma once
#include "ml/attention.hpp"
#include "ml/layernorm.hpp"
#include "ml/mlp.hpp"
#include <random>
#include <vector>

namespace absolutus {
namespace ml {

// Pre-norm transformer block, the unit that stacks.
//   h = x + Attention(LayerNorm(x))
//   y = h + FFN(LayerNorm(h))       where FFN(v) = W2 relu(W1 v)
// Requires d_k == d_model for the residual stream.
// use_residual=false gives the "no residual" ablation for tests only.
class TransformerBlock {
public:
    size_t seq_len, d_model, d_ff;
    bool use_residual;
    double momentum;

    LayerNorm ln1, ln2;
    Attention attn;
    Dense ffn1, ffn2;

    TransformerBlock(size_t sl, size_t dm, size_t dff,
                     std::mt19937_64& rng, bool residual = true,
                     double momentum_ = 0.9);

    const std::vector<double>& forward(const std::vector<double>& x);
    std::vector<double> backward(const std::vector<double>& grad_out);
    void update(double lr);

private:
    std::vector<double> x_, ln1_out_, attn_out_, h_, ln2_out_;
    std::vector<double> ffn1_out_, ffn2_out_, y_;
};

}}
