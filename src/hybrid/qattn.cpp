#include "hybrid/qattn.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace absolutus {
namespace hybrid {

using ml::Activation;

static void matvec(const std::vector<double>& W, size_t n_out, size_t n_in,
                   const std::vector<double>& x, std::vector<double>& y) {
    for (size_t i = 0; i < n_out; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < n_in; ++j) s += W[i * n_in + j] * x[j];
        y[i] = s;
    }
}

static void matTvec(const std::vector<double>& W, size_t n_out, size_t n_in,
                    const std::vector<double>& v, std::vector<double>& y) {
    std::fill(y.begin(), y.end(), 0.0);
    for (size_t i = 0; i < n_out; ++i)
        for (size_t j = 0; j < n_in; ++j)
            y[j] += W[i * n_in + j] * v[i];
}

QuantumAttention::QuantumAttention(size_t sl, size_t dm, size_t nq,
                                   size_t ql_layers, std::mt19937_64& rng,
                                   double mom)
    : seq_len(sl), d_model(dm), n_qubits(nq), momentum(mom),
      Wk(dm * nq), Wv(dm * nq), dWk(dm * nq, 0.0), dWv(dm * nq, 0.0),
      vWk(dm * nq, 0.0), vWv(dm * nq, 0.0),
      pre_q(dm, nq, Activation::Tanh, rng, mom),
      ql(nq, nq * ql_layers, rng, mom, 3.14159265358979323846),
      X_(sl * dm, 0.0), Q_(sl * nq, 0.0), K_(sl * nq, 0.0), V_(sl * nq, 0.0),
      S_(sl * sl, 0.0), A_(sl * sl, 0.0), Y_(sl * nq, 0.0),
      pre_q_acts_(sl * nq, 0.0) {
    const double scale = std::sqrt(2.0 / dm);
    std::normal_distribution<double> dist(0.0, scale);
    for (auto& w : Wk) w = dist(rng);
    for (auto& w : Wv) w = dist(rng);
}

const std::vector<double>& QuantumAttention::forward(const std::vector<double>& X) {
    if (X.size() != seq_len * d_model)
        throw std::invalid_argument("QuantumAttention::forward: X shape");
    X_ = X;

    for (size_t t = 0; t < seq_len; ++t) {
        std::vector<double> xt(X_.begin() + t * d_model,
                               X_.begin() + (t + 1) * d_model);
        auto a = pre_q.forward(xt);
        auto q = ql.forward(a);
        for (size_t i = 0; i < n_qubits; ++i) {
            pre_q_acts_[t * n_qubits + i] = a[i];
            Q_[t * n_qubits + i] = q[i];
        }
    }

    std::vector<double> k(n_qubits), v(n_qubits);
    for (size_t t = 0; t < seq_len; ++t) {
        std::vector<double> xt(X_.begin() + t * d_model,
                               X_.begin() + (t + 1) * d_model);
        matvec(Wk, n_qubits, d_model, xt, k);
        matvec(Wv, n_qubits, d_model, xt, v);
        for (size_t i = 0; i < n_qubits; ++i) {
            K_[t * n_qubits + i] = k[i];
            V_[t * n_qubits + i] = v[i];
        }
    }

    const double inv_sqrt_dk = 1.0 / std::sqrt(double(n_qubits));
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            double s = 0.0;
            for (size_t kk = 0; kk < n_qubits; ++kk)
                s += Q_[i * n_qubits + kk] * K_[j * n_qubits + kk];
            S_[i * seq_len + j] = s * inv_sqrt_dk;
        }
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
        for (size_t kk = 0; kk < n_qubits; ++kk) {
            double s = 0.0;
            for (size_t j = 0; j < seq_len; ++j)
                s += A_[i * seq_len + j] * V_[j * n_qubits + kk];
            Y_[i * n_qubits + kk] = s;
        }
    return Y_;
}

std::vector<double> QuantumAttention::backward(const std::vector<double>& grad_out) {
    if (grad_out.size() != seq_len * n_qubits)
        throw std::invalid_argument("QuantumAttention::backward: grad shape");

    // dA, dV
    std::vector<double> dA(seq_len * seq_len, 0.0);
    std::vector<double> dV(seq_len * n_qubits, 0.0);
    for (size_t i = 0; i < seq_len; ++i)
        for (size_t j = 0; j < seq_len; ++j) {
            double s = 0.0;
            for (size_t kk = 0; kk < n_qubits; ++kk)
                s += grad_out[i * n_qubits + kk] * V_[j * n_qubits + kk];
            dA[i * seq_len + j] = s;
        }
    for (size_t j = 0; j < seq_len; ++j)
        for (size_t kk = 0; kk < n_qubits; ++kk) {
            double s = 0.0;
            for (size_t i = 0; i < seq_len; ++i)
                s += A_[i * seq_len + j] * grad_out[i * n_qubits + kk];
            dV[j * n_qubits + kk] = s;
        }

    // dS from softmax backward
    std::vector<double> dS(seq_len * seq_len, 0.0);
    for (size_t i = 0; i < seq_len; ++i) {
        double rs = 0.0;
        for (size_t j = 0; j < seq_len; ++j)
            rs += dA[i * seq_len + j] * A_[i * seq_len + j];
        for (size_t j = 0; j < seq_len; ++j)
            dS[i * seq_len + j] = A_[i * seq_len + j]
                                * (dA[i * seq_len + j] - rs);
    }
    const double inv_sqrt_dk = 1.0 / std::sqrt(double(n_qubits));
    for (auto& s : dS) s *= inv_sqrt_dk;

    // dQ, dK
    std::vector<double> dQ(seq_len * n_qubits, 0.0);
    std::vector<double> dK(seq_len * n_qubits, 0.0);
    for (size_t i = 0; i < seq_len; ++i)
        for (size_t kk = 0; kk < n_qubits; ++kk) {
            double s = 0.0;
            for (size_t j = 0; j < seq_len; ++j)
                s += dS[i * seq_len + j] * K_[j * n_qubits + kk];
            dQ[i * n_qubits + kk] = s;
        }
    for (size_t j = 0; j < seq_len; ++j)
        for (size_t kk = 0; kk < n_qubits; ++kk) {
            double s = 0.0;
            for (size_t i = 0; i < seq_len; ++i)
                s += dS[i * seq_len + j] * Q_[i * n_qubits + kk];
            dK[j * n_qubits + kk] = s;
        }

    // dWk, dWv, and classical path grad_X (from K, V)
    std::fill(dWk.begin(), dWk.end(), 0.0);
    std::fill(dWv.begin(), dWv.end(), 0.0);
    std::vector<double> grad_X(seq_len * d_model, 0.0);
    std::vector<double> xt(d_model), dk(n_qubits), dv(n_qubits), gx(d_model), tmp(d_model);
    for (size_t t = 0; t < seq_len; ++t) {
        std::copy(X_.begin() + t * d_model, X_.begin() + (t + 1) * d_model, xt.begin());
        std::copy(dK.begin() + t * n_qubits, dK.begin() + (t + 1) * n_qubits, dk.begin());
        std::copy(dV.begin() + t * n_qubits, dV.begin() + (t + 1) * n_qubits, dv.begin());
        for (size_t i = 0; i < n_qubits; ++i)
            for (size_t j = 0; j < d_model; ++j) {
                dWk[i * d_model + j] += dk[i] * xt[j];
                dWv[i * d_model + j] += dv[i] * xt[j];
            }
        std::fill(gx.begin(), gx.end(), 0.0);
        matTvec(Wk, n_qubits, d_model, dk, tmp);
        for (size_t i = 0; i < d_model; ++i) gx[i] += tmp[i];
        matTvec(Wv, n_qubits, d_model, dv, tmp);
        for (size_t i = 0; i < d_model; ++i) gx[i] += tmp[i];
        for (size_t i = 0; i < d_model; ++i)
            grad_X[t * d_model + i] = gx[i];
    }

    // Quantum path: accumulate dtheta for ql and dW/db for pre_q
    std::vector<double> acc_pre_dW(pre_q.W.size(), 0.0);
    std::vector<double> acc_pre_db(pre_q.b.size(), 0.0);
    std::vector<double> acc_ql_dtheta(ql.theta.size(), 0.0);

    for (size_t t = 0; t < seq_len; ++t) {
        std::vector<double> xt(X_.begin() + t * d_model, X_.begin() + (t + 1) * d_model);
        std::vector<double> dQ_t(dQ.begin() + t * n_qubits, dQ.begin() + (t + 1) * n_qubits);

        auto a = pre_q.forward(xt);
        ql.forward(a);

        auto d_a = ql.backward(dQ_t);
        for (size_t i = 0; i < ql.dtheta.size(); ++i)
            acc_ql_dtheta[i] += ql.dtheta[i];

        auto g_t = pre_q.backward(d_a);
        for (size_t i = 0; i < pre_q.W.size(); ++i) acc_pre_dW[i] += pre_q.dW[i];
        for (size_t i = 0; i < pre_q.b.size(); ++i) acc_pre_db[i] += pre_q.db[i];

        for (size_t i = 0; i < d_model; ++i)
            grad_X[t * d_model + i] += g_t[i];
    }

    std::copy(acc_pre_dW.begin(), acc_pre_dW.end(), pre_q.dW.begin());
    std::copy(acc_pre_db.begin(), acc_pre_db.end(), pre_q.db.begin());
    std::copy(acc_ql_dtheta.begin(), acc_ql_dtheta.end(), ql.dtheta.begin());

    return grad_X;
}

void QuantumAttention::update(double lr) {
    auto step = [&](std::vector<double>& W, std::vector<double>& dW,
                    std::vector<double>& vW) {
        for (size_t i = 0; i < W.size(); ++i) {
            vW[i] = momentum * vW[i] + dW[i];
            W[i] -= lr * vW[i];
        }
    };
    step(Wk, dWk, vWk);
    step(Wv, dWv, vWv);
    pre_q.update(lr);
    ql.update(lr);
}

}}
