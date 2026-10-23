#pragma once
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace absolutus {
namespace ml {

enum class Activation { Linear, ReLU, Tanh, Sigmoid };

inline double activate(Activation a, double z) {
    switch (a) {
        case Activation::Linear:  return z;
        case Activation::ReLU:    return z > 0.0 ? z : 0.0;
        case Activation::Tanh:    return std::tanh(z);
        case Activation::Sigmoid: return 1.0 / (1.0 + std::exp(-z));
    }
    return z;
}

inline double activate_deriv(Activation a, double z, double y) {
    switch (a) {
        case Activation::Linear:  return 1.0;
        case Activation::ReLU:    return z > 0.0 ? 1.0 : 0.0;
        case Activation::Tanh:    return 1.0 - y * y;
        case Activation::Sigmoid: return y * (1.0 - y);
    }
    return 1.0;
}

class Dense {
public:
    size_t n_in, n_out;
    Activation act;
    double momentum = 0.9;
    std::vector<double> W, b, dW, db, vW, vb;

    Dense(size_t n_in_, size_t n_out_, Activation act_, std::mt19937_64& rng,
          double momentum_ = 0.9)
        : n_in(n_in_), n_out(n_out_), act(act_), momentum(momentum_),
          W(n_in_ * n_out_), b(n_out_),
          dW(n_in_ * n_out_, 0.0), db(n_out_, 0.0),
          vW(n_in_ * n_out_, 0.0), vb(n_out_, 0.0),
          x_(n_in_, 0.0), z_(n_out_, 0.0), y_(n_out_, 0.0) {
        const double scale = std::sqrt(1.0 / n_in);
        std::normal_distribution<double> dist(0.0, scale);
        for (auto& w : W) w = dist(rng);
        // small positive bias breaks the (0.5,0.5) saddle at init
        for (auto& bi : b) bi = 0.1;
    }

    const std::vector<double>& forward(const std::vector<double>& x) {
        if (x.size() != n_in) throw std::invalid_argument("Dense::forward");
        x_ = x;
        for (size_t i = 0; i < n_out; ++i) {
            double s = b[i];
            for (size_t j = 0; j < n_in; ++j) s += W[i * n_in + j] * x_[j];
            z_[i] = s;
            y_[i] = activate(act, s);
        }
        return y_;
    }

    std::vector<double> backward(const std::vector<double>& grad_y) {
        std::vector<double> grad_x(n_in, 0.0);
        std::fill(dW.begin(), dW.end(), 0.0);
        std::fill(db.begin(), db.end(), 0.0);
        for (size_t i = 0; i < n_out; ++i) {
            const double dz = grad_y[i] * activate_deriv(act, z_[i], y_[i]);
            db[i] = dz;
            for (size_t j = 0; j < n_in; ++j) {
                dW[i * n_in + j] = dz * x_[j];
                grad_x[j] += W[i * n_in + j] * dz;
            }
        }
        return grad_x;
    }

    void update(double lr) {
        for (size_t i = 0; i < W.size(); ++i) {
            vW[i] = momentum * vW[i] + dW[i];
            W[i] -= lr * vW[i];
        }
        for (size_t i = 0; i < b.size(); ++i) {
            vb[i] = momentum * vb[i] + db[i];
            b[i] -= lr * vb[i];
        }
    }

private:
    std::vector<double> x_, z_, y_;
};

inline std::vector<double> softmax(const std::vector<double>& z) {
    const double m = *std::max_element(z.begin(), z.end());
    std::vector<double> e(z.size());
    double s = 0.0;
    for (size_t i = 0; i < z.size(); ++i) { e[i] = std::exp(z[i] - m); s += e[i]; }
    for (auto& v : e) v /= s;
    return e;
}

inline double cross_entropy(const std::vector<double>& p,
                            const std::vector<double>& y) {
    double l = 0.0;
    for (size_t i = 0; i < p.size(); ++i)
        if (y[i] > 0.0) l -= y[i] * std::log(p[i] + 1e-12);
    return l;
}

class MLP {
public:
    std::vector<Dense> layers;

    std::vector<double> forward(const std::vector<double>& x) {
        std::vector<double> cur = x;
        for (auto& l : layers) cur = l.forward(cur);
        return cur;
    }

    void backward(const std::vector<double>& grad_logits) {
        std::vector<double> g = grad_logits;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            g = it->backward(g);
    }

    void update(double lr) { for (auto& l : layers) l.update(lr); }
};

}}
