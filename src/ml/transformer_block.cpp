#include "ml/transformer_block.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace absolutus {
namespace ml {

static bool finite_ok(const std::vector<double>& v) {
    for (double x : v) if (!std::isfinite(x)) return false;
    return true;
}

TransformerBlock::TransformerBlock(size_t sl, size_t dm, size_t dff,
                                   std::mt19937_64& rng, bool residual,
                                   double mom, size_t nh,
                                   int qidx, size_t q_layers, bool causal)
    : seq_len(sl), d_model(dm), d_ff(dff), n_heads(nh),
      use_residual(residual), momentum(mom),
      ln1(dm, mom), ln2(dm, mom),
      attn(sl, dm, nh, rng, mom, qidx, q_layers, causal),
      ffn1(dm, dff, Activation::ReLU, rng, mom),
      ffn2(dff, dm, Activation::Linear, rng, mom),
      x_(sl*dm, 0.0), ln1_out_(sl*dm, 0.0), attn_out_(sl*dm, 0.0),
      h_(sl*dm, 0.0), ln2_out_(sl*dm, 0.0),
      ffn1_out_(sl*dff, 0.0), ffn2_out_(sl*dm, 0.0), y_(sl*dm, 0.0) {}

const std::vector<double>& TransformerBlock::forward(const std::vector<double>& x) {
    if (x.size() != seq_len * d_model)
        throw std::invalid_argument("TransformerBlock::forward: x shape");
    x_ = x;

    ln1_out_ = ln1.forward(x_);
    if (!finite_ok(ln1_out_)) throw std::runtime_error("block: ln1");

    attn_out_ = attn.forward(ln1_out_);
    if (!finite_ok(attn_out_)) throw std::runtime_error("block: attn");

    for (size_t i = 0; i < seq_len * d_model; ++i)
        h_[i] = use_residual ? (x_[i] + attn_out_[i]) : attn_out_[i];

    ln2_out_ = ln2.forward(h_);
    if (!finite_ok(ln2_out_)) throw std::runtime_error("block: ln2");

    for (size_t t = 0; t < seq_len; ++t) {
        std::vector<double> v(ln2_out_.begin() + t*d_model,
                              ln2_out_.begin() + (t+1)*d_model);
        auto hh = ffn1.forward(v);
        if (!finite_ok(hh)) throw std::runtime_error("block: ffn1");
        std::copy(hh.begin(), hh.end(), ffn1_out_.begin() + t*d_ff);

        auto oo = ffn2.forward(hh);
        if (!finite_ok(oo)) throw std::runtime_error("block: ffn2");
        std::copy(oo.begin(), oo.end(), ffn2_out_.begin() + t*d_model);
    }

    for (size_t i = 0; i < seq_len * d_model; ++i)
        y_[i] = use_residual ? (h_[i] + ffn2_out_[i]) : ffn2_out_[i];

    if (!finite_ok(y_)) throw std::runtime_error("block: y");
    return y_;
}

std::vector<double> TransformerBlock::backward(const std::vector<double>& grad_out) {
    if (grad_out.size() != seq_len * d_model)
        throw std::invalid_argument("TransformerBlock::backward: grad shape");

    std::vector<double> grad_h(seq_len * d_model, 0.0);
    std::vector<double> grad_ffn_out(seq_len * d_model, 0.0);
    if (use_residual) { grad_h = grad_out; grad_ffn_out = grad_out; }
    else              { grad_ffn_out = grad_out; }

    std::fill(ffn1.dW.begin(), ffn1.dW.end(), 0.0);
    std::fill(ffn1.db.begin(), ffn1.db.end(), 0.0);
    std::fill(ffn2.dW.begin(), ffn2.dW.end(), 0.0);
    std::fill(ffn2.db.begin(), ffn2.db.end(), 0.0);
    std::vector<double> acc_ffn1_dW(ffn1.W.size(), 0.0);
    std::vector<double> acc_ffn1_db(ffn1.b.size(), 0.0);
    std::vector<double> acc_ffn2_dW(ffn2.W.size(), 0.0);
    std::vector<double> acc_ffn2_db(ffn2.b.size(), 0.0);
    std::vector<double> grad_ln2_out(seq_len * d_model, 0.0);

    for (size_t t = 0; t < seq_len; ++t) {
        std::vector<double> g_t(grad_ffn_out.begin() + t*d_model,
                                grad_ffn_out.begin() + (t+1)*d_model);
        std::vector<double> hh(ffn1_out_.begin() + t*d_ff,
                               ffn1_out_.begin() + (t+1)*d_ff);
        std::vector<double> v(ln2_out_.begin() + t*d_model,
                              ln2_out_.begin() + (t+1)*d_model);
        ffn1.forward(v);
        ffn2.forward(hh);

        auto g_hh = ffn2.backward(g_t);
        for (size_t i = 0; i < ffn2.dW.size(); ++i) acc_ffn2_dW[i] += ffn2.dW[i];
        for (size_t i = 0; i < ffn2.db.size(); ++i) acc_ffn2_db[i] += ffn2.db[i];

        auto g_v = ffn1.backward(g_hh);
        for (size_t i = 0; i < ffn1.dW.size(); ++i) acc_ffn1_dW[i] += ffn1.dW[i];
        for (size_t i = 0; i < ffn1.db.size(); ++i) acc_ffn1_db[i] += ffn1.db[i];

        for (size_t i = 0; i < d_model; ++i) grad_ln2_out[t*d_model+i] = g_v[i];
    }
    std::copy(acc_ffn1_dW.begin(), acc_ffn1_dW.end(), ffn1.dW.begin());
    std::copy(acc_ffn1_db.begin(), acc_ffn1_db.end(), ffn1.db.begin());
    std::copy(acc_ffn2_dW.begin(), acc_ffn2_dW.end(), ffn2.dW.begin());
    std::copy(acc_ffn2_db.begin(), acc_ffn2_db.end(), ffn2.db.begin());

    auto grad_h_2 = ln2.backward(grad_ln2_out);
    for (size_t i = 0; i < seq_len * d_model; ++i) grad_h[i] += grad_h_2[i];

    std::vector<double> grad_x_1(seq_len * d_model, 0.0);
    std::vector<double> grad_attn_out(seq_len * d_model, 0.0);
    if (use_residual) { grad_x_1 = grad_h; grad_attn_out = grad_h; }
    else              { grad_attn_out = grad_h; }

    auto grad_ln1_out = attn.backward(grad_attn_out);
    auto grad_x_2 = ln1.backward(grad_ln1_out);

    std::vector<double> grad_x(seq_len * d_model);
    for (size_t i = 0; i < seq_len * d_model; ++i)
        grad_x[i] = grad_x_1[i] + grad_x_2[i];
    return grad_x;
}

void TransformerBlock::update(double lr) {
    ln1.update(lr);
    ln2.update(lr);
    attn.update(lr);
    ffn1.update(lr);
    ffn2.update(lr);
}

}}
