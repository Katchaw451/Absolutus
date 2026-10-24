#pragma once
#include "ml/transformer_block.hpp"
#include "ml/posenc.hpp"
#include "ml/mlp.hpp"
#include <random>
#include <vector>

namespace absolutus {
namespace ml {

class MiniLM {
public:
    size_t seq_len, d_model, d_ff, vocab_size, n_blocks;
    size_t n_heads;
    int quantum_head_idx;
    double momentum;

    std::vector<double> embed, dembed;
    PositionEncoding pe;
    std::vector<TransformerBlock> blocks;
    Dense head;

    MiniLM(size_t sl, size_t dm, size_t dff, size_t vs,
           size_t nblocks, std::mt19937_64& rng,
           double momentum_ = 0.9,
           size_t n_heads_ = 1,
           int quantum_head_idx_ = -1,
           size_t quantum_ql_layers = 1);

    const std::vector<double>& forward(const std::vector<size_t>& tokens);
    void backward(const std::vector<double>& grad_logits);
    void clip_gradients(double max_norm);
    void update(double lr);

private:
    std::vector<double> h0_, logits_, cur_, grad_h0_;
    std::vector<size_t> tokens_;
};

}}
