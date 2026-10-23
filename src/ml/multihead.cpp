#include "ml/multihead.hpp"
#include <algorithm>
#include <stdexcept>

namespace absolutus {
namespace ml {

MultiHeadAttention::MultiHeadAttention(size_t sl, size_t dm, size_t nh,
                                       std::mt19937_64& rng, double mom)
    : seq_len(sl), d_model(dm), n_heads(nh), momentum(mom),
      Wo(dm * dm, 0.0), dWo(dm * dm, 0.0), vWo(dm * dm, 0.0),
      X_(sl * dm, 0.0), concat_(sl * dm, 0.0), Y_(sl * dm, 0.0) {
    if (nh == 0 || dm % nh != 0)
        throw std::invalid_argument("MultiHeadAttention: d_model % n_heads != 0");
    d_k = dm / nh;
    heads.reserve(nh);
    for (size_t h = 0; h < nh; ++h)
        heads.emplace_back(sl, dm, d_k, rng, mom);
    for (size_t i = 0; i < dm; ++i) Wo[i * dm + i] = 1.0;
}

const std::vector<double>& MultiHeadAttention::forward(const std::vector<double>& X) {
    if (X.size() != seq_len * d_model)
        throw std::invalid_argument("MultiHeadAttention::forward: X shape");
    X_ = X;

    for (size_t h = 0; h < n_heads; ++h) {
        const auto& yh = heads[h].forward(X_);
        for (size_t t = 0; t < seq_len; ++t)
            for (size_t kk = 0; kk < d_k; ++kk)
                concat_[t * d_model + h * d_k + kk] = yh[t * d_k + kk];
    }

    for (size_t t = 0; t < seq_len; ++t)
        for (size_t j = 0; j < d_model; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < d_model; ++i)
                s += Wo[j * d_model + i] * concat_[t * d_model + i];
            Y_[t * d_model + j] = s;
        }
    return Y_;
}

std::vector<double> MultiHeadAttention::backward(const std::vector<double>& grad_out) {
    if (grad_out.size() != seq_len * d_model)
        throw std::invalid_argument("MultiHeadAttention::backward: grad shape");

    std::fill(dWo.begin(), dWo.end(), 0.0);
    std::vector<double> d_concat(seq_len * d_model, 0.0);
    for (size_t t = 0; t < seq_len; ++t)
        for (size_t j = 0; j < d_model; ++j) {
            const double g = grad_out[t * d_model + j];
            for (size_t i = 0; i < d_model; ++i) {
                dWo[j * d_model + i] += g * concat_[t * d_model + i];
                d_concat[t * d_model + i] += g * Wo[j * d_model + i];
            }
        }

    std::vector<double> grad_X(seq_len * d_model, 0.0);
    for (size_t h = 0; h < n_heads; ++h) {
        std::vector<double> g_h(seq_len * d_k);
        for (size_t t = 0; t < seq_len; ++t)
            for (size_t kk = 0; kk < d_k; ++kk)
                g_h[t * d_k + kk] = d_concat[t * d_model + h * d_k + kk];
        auto gx_h = heads[h].backward(g_h);
        for (size_t i = 0; i < seq_len * d_model; ++i) grad_X[i] += gx_h[i];
    }
    return grad_X;
}

void MultiHeadAttention::update(double lr) {
    for (auto& h : heads) h.update(lr);
    for (size_t i = 0; i < Wo.size(); ++i) {
        vWo[i] = momentum * vWo[i] + dWo[i];
        Wo[i] -= lr * vWo[i];
    }
}

}}
