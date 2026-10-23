#include "ml/layernorm.hpp"
#include <cmath>
#include <stdexcept>

namespace absolutus {
namespace ml {

LayerNorm::LayerNorm(size_t dm, double mom)
    : d_model(dm), momentum(mom),
      gamma(dm, 1.0), beta(dm, 0.0),
      dgamma(dm, 0.0), dbeta(dm, 0.0),
      vgamma(dm, 0.0), vbeta(dm, 0.0) {}

const std::vector<double>& LayerNorm::forward(const std::vector<double>& x) {
    if (x.size() % d_model != 0)
        throw std::invalid_argument("LayerNorm::forward: x size");
    n_tokens_ = x.size() / d_model;
    x_saved_ = x;
    x_hat_.assign(x.size(), 0.0);
    inv_sigma_.assign(n_tokens_, 0.0);

    for (size_t t = 0; t < n_tokens_; ++t) {
        const size_t base = t * d_model;
        double mu = 0.0;
        for (size_t i = 0; i < d_model; ++i) mu += x[base + i];
        mu /= double(d_model);

        double var = 0.0;
        for (size_t i = 0; i < d_model; ++i) {
            double d = x[base + i] - mu;
            var += d * d;
        }
        var /= double(d_model);

        const double inv_sigma = 1.0 / std::sqrt(var + eps);
        inv_sigma_[t] = inv_sigma;

        for (size_t i = 0; i < d_model; ++i) {
            const double xh = (x[base + i] - mu) * inv_sigma;
            x_hat_[base + i] = xh;
            x_saved_[base + i] = gamma[i] * xh + beta[i];
        }
    }
    return x_saved_;
}

std::vector<double> LayerNorm::backward(const std::vector<double>& grad_out) {
    if (grad_out.size() != x_saved_.size())
        throw std::invalid_argument("LayerNorm::backward: grad shape");

    std::fill(dgamma.begin(), dgamma.end(), 0.0);
    std::fill(dbeta.begin(), dbeta.end(), 0.0);
    std::vector<double> grad_x(grad_out.size(), 0.0);

    for (size_t t = 0; t < n_tokens_; ++t) {
        const size_t base = t * d_model;

        // Accumulate dgamma, dbeta
        for (size_t i = 0; i < d_model; ++i) {
            dgamma[i] += grad_out[base + i] * x_hat_[base + i];
            dbeta[i]  += grad_out[base + i];
        }

        // dx_hat = dy * gamma
        std::vector<double> dxh(d_model);
        for (size_t i = 0; i < d_model; ++i)
            dxh[i] = grad_out[base + i] * gamma[i];

        // Standard LayerNorm backward:
        // dx_i = (1/sigma) * (dxh_i - mean(dxh) - x_hat_i * mean(dxh * x_hat))
        double mean_dxh = 0.0, mean_dxh_xhat = 0.0;
        for (size_t i = 0; i < d_model; ++i) {
            mean_dxh += dxh[i];
            mean_dxh_xhat += dxh[i] * x_hat_[base + i];
        }
        mean_dxh /= double(d_model);
        mean_dxh_xhat /= double(d_model);

        for (size_t i = 0; i < d_model; ++i) {
            grad_x[base + i] = inv_sigma_[t]
                * (dxh[i] - mean_dxh - x_hat_[base + i] * mean_dxh_xhat);
        }
    }
    return grad_x;
}

void LayerNorm::update(double lr) {
    for (size_t i = 0; i < d_model; ++i) {
        vgamma[i] = momentum * vgamma[i] + dgamma[i];
        vbeta[i]  = momentum * vbeta[i]  + dbeta[i];
        gamma[i] -= lr * vgamma[i];
        beta[i]  -= lr * vbeta[i];
    }
}

}}
