#include "ml/attention.hpp"
#include <algorithm>
#include <stdexcept>

namespace absolutus {
namespace ml {

static void matvec(const std::vector<double>& W, size_t n_out, size_t n_in,
                   const std::vector<double>& x, std::vector<double>& y) {
    for (size_t i = 0; i < n_out; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < n_in; ++j) s += W[i * n_in + j] * x[j];
        y[i] = s;
    }
}

// W^T * v, for [n_out, n_in] * [n_out] -> [n_in]
static void matTvec(const std::vector<double>& W, size_t n_out, size_t n_in,
                    const std::vector<double>& v, std::vector<double>& y) {
    std::fill(y.begin(), y.end(), 0.0);
    for (size_t i = 0; i < n_out; ++i)
        for (size_t j = 0; j < n_in; ++j)
            y[j] += W[i * n_in + j] * v[i];
}

Attention::Attention(size_t sl, size_t dm, size_t dk,
                     std::mt19937_64& rng, double mom, bool causal)
    : seq_len(sl), d_model(dm), d_k(dk), momentum(mom), causal_(causal),
      Wq(dm * dk), Wk(dm * dk), Wv(dm * dk),
      dWq(dm * dk, 0.0), dWk(dm * dk, 0.0), dWv(dm * dk, 0.0),
      vWq(dm * dk, 0.0), vWk(dm * dk, 0.0), vWv(dm * dk, 0.0),
      X_(sl * dm, 0.0), Q_(sl * dk, 0.0), K_(sl * dk, 0.0),
      V_(sl * dk, 0.0), S_(sl * sl, 0.0), A_(sl * sl, 0.0),
      Y_(sl * dk, 0.0) {
    const double scale = std::sqrt(2.0 / dm);
    std::normal_distribution<double> dist(0.0, scale);
    for (auto& w : Wq) w = dist(rng);
    for (auto& w : Wk) w = dist(rng);
    for (auto& w : Wv) w = dist(rng);
}

const std::vector<double>& Attention::forward(const std::vector<double>& X) {
    if (X.size() != seq_len * d_model)
        throw std::invalid_argument("Attention::forward: X shape");
    X_ = X;

    std::vector<double> q(d_k), k(d_k), v(d_k);
    for (size_t t = 0; t < seq_len; ++t) {
        std::vector<double> xt(X_.begin() + t * d_model,
                               X_.begin() + (t + 1) * d_model);
        matvec(Wq, d_k, d_model, xt, q);
        matvec(Wk, d_k, d_model, xt, k);
        matvec(Wv, d_k, d_model, xt, v);
        for (size_t i = 0; i < d_k; ++i) {
            Q_[t * d_k + i] = q[i];
            K_[t * d_k + i] = k[i];
            V_[t * d_k + i] = v[i];
        }
    }

    const double inv_sqrt_dk = 1.0 / std::sqrt(double(d_k));
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            double s = 0.0;
            for (size_t kk = 0; kk < d_k; ++kk)
                s += Q_[i * d_k + kk] * K_[j * d_k + kk];
            S_[i * seq_len + j] = s * inv_sqrt_dk;
            if (causal_ && j > i) S_[i * seq_len + j] = -1e30;
        }
        // softmax row i
        double m = *std::max_element(S_.begin() + i * seq_len,
                                     S_.begin() + (i + 1) * seq_len);
        double sum = 0.0;
        for (size_t j = 0; j < seq_len; ++j) {
            A_[i * seq_len + j] = std::exp(S_[i * seq_len + j] - m);
            sum += A_[i * seq_len + j];
        }
        for (size_t j = 0; j < seq_len; ++j) A_[i * seq_len + j] /= sum;
    }

    for (size_t i = 0; i < seq_len; ++i)
        for (size_t kk = 0; kk < d_k; ++kk) {
            double s = 0.0;
            for (size_t j = 0; j < seq_len; ++j)
                s += A_[i * seq_len + j] * V_[j * d_k + kk];
            Y_[i * d_k + kk] = s;
        }
    return Y_;
}

std::vector<double> Attention::backward(const std::vector<double>& grad_out) {
    if (grad_out.size() != seq_len * d_k)
        throw std::invalid_argument("Attention::backward: grad shape");

    std::vector<double> dA(seq_len * seq_len, 0.0);
    std::vector<double> dV(seq_len * d_k, 0.0);
    for (size_t i = 0; i < seq_len; ++i)
        for (size_t j = 0; j < seq_len; ++j) {
            double s = 0.0;
            for (size_t kk = 0; kk < d_k; ++kk)
                s += grad_out[i * d_k + kk] * V_[j * d_k + kk];
            dA[i * seq_len + j] = s;
        }
    for (size_t j = 0; j < seq_len; ++j)
        for (size_t kk = 0; kk < d_k; ++kk) {
            double s = 0.0;
            for (size_t i = 0; i < seq_len; ++i)
                s += A_[i * seq_len + j] * grad_out[i * d_k + kk];
            dV[j * d_k + kk] = s;
        }

    // dS = A * (dA - sum(dA*A)) per row (softmax backward)
    std::vector<double> dS(seq_len * seq_len, 0.0);
    for (size_t i = 0; i < seq_len; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < seq_len; ++j)
            row_sum += dA[i * seq_len + j] * A_[i * seq_len + j];
        for (size_t j = 0; j < seq_len; ++j)
            dS[i * seq_len + j] = A_[i * seq_len + j]
                                * (dA[i * seq_len + j] - row_sum);
    }

    const double inv_sqrt_dk = 1.0 / std::sqrt(double(d_k));
    for (size_t i = 0; i < seq_len * seq_len; ++i) dS[i] *= inv_sqrt_dk;

    // dQ = dS * K, dK = dS^T * Q
    std::vector<double> dQ(seq_len * d_k, 0.0);
    std::vector<double> dK(seq_len * d_k, 0.0);
    for (size_t i = 0; i < seq_len; ++i)
        for (size_t kk = 0; kk < d_k; ++kk) {
            double s = 0.0;
            for (size_t j = 0; j < seq_len; ++j)
                s += dS[i * seq_len + j] * K_[j * d_k + kk];
            dQ[i * d_k + kk] = s;
        }
    for (size_t j = 0; j < seq_len; ++j)
        for (size_t kk = 0; kk < d_k; ++kk) {
            double s = 0.0;
            for (size_t i = 0; i < seq_len; ++i)
                s += dS[i * seq_len + j] * Q_[i * d_k + kk];
            dK[j * d_k + kk] = s;
        }

    // Weight grads
    std::fill(dWq.begin(), dWq.end(), 0.0);
    std::fill(dWk.begin(), dWk.end(), 0.0);
    std::fill(dWv.begin(), dWv.end(), 0.0);
    std::vector<double> xt(d_model), dq(d_k), dk(d_k), dv(d_k);
    for (size_t t = 0; t < seq_len; ++t) {
        std::copy(X_.begin() + t * d_model, X_.begin() + (t + 1) * d_model,
                  xt.begin());
        std::copy(dQ.begin() + t * d_k, dQ.begin() + (t + 1) * d_k, dq.begin());
        std::copy(dK.begin() + t * d_k, dK.begin() + (t + 1) * d_k, dk.begin());
        std::copy(dV.begin() + t * d_k, dV.begin() + (t + 1) * d_k, dv.begin());
        for (size_t i = 0; i < d_k; ++i)
            for (size_t j = 0; j < d_model; ++j) {
                dWq[i * d_model + j] += dq[i] * xt[j];
                dWk[i * d_model + j] += dk[i] * xt[j];
                dWv[i * d_model + j] += dv[i] * xt[j];
            }
    }

    // grad_X
    std::vector<double> grad_X(seq_len * d_model, 0.0);
    std::vector<double> gx(d_model);
    for (size_t t = 0; t < seq_len; ++t) {
        std::copy(dQ.begin() + t * d_k, dQ.begin() + (t + 1) * d_k, dq.begin());
        std::copy(dK.begin() + t * d_k, dK.begin() + (t + 1) * d_k, dk.begin());
        std::copy(dV.begin() + t * d_k, dV.begin() + (t + 1) * d_k, dv.begin());
        std::fill(gx.begin(), gx.end(), 0.0);
        std::vector<double> tmp(d_model, 0.0);
        matTvec(Wq, d_k, d_model, dq, tmp);
        for (size_t i = 0; i < d_model; ++i) gx[i] += tmp[i];
        matTvec(Wk, d_k, d_model, dk, tmp);
        for (size_t i = 0; i < d_model; ++i) gx[i] += tmp[i];
        matTvec(Wv, d_k, d_model, dv, tmp);
        for (size_t i = 0; i < d_model; ++i) gx[i] += tmp[i];
        for (size_t i = 0; i < d_model; ++i)
            grad_X[t * d_model + i] = gx[i];
    }
    return grad_X;
}

void Attention::update(double lr) {
    auto step = [&](std::vector<double>& W, std::vector<double>& dW,
                    std::vector<double>& vW) {
        for (size_t i = 0; i < W.size(); ++i) {
            vW[i] = momentum * vW[i] + dW[i];
            W[i] -= lr * vW[i];
        }
    };
    step(Wq, dWq, vWq);
    step(Wk, dWk, vWk);
    step(Wv, dWv, vWv);
}

}}
