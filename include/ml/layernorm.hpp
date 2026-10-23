#pragma once
#include <random>
#include <vector>

namespace absolutus {
namespace ml {

// LayerNorm over the feature dim, applied per token.
//   y_t = gamma * (x_t - mean(x_t)) / sqrt(var(x_t) + eps) + beta
// gamma, beta are [d_model], shared across tokens.
//
// Input/output shape: [n_tokens * d_model], flat row-major.
class LayerNorm {
public:
    size_t d_model;
    double eps = 1e-5;
    double momentum;
    std::vector<double> gamma, beta, dgamma, dbeta, vgamma, vbeta;

    LayerNorm(size_t d_model_, double momentum_ = 0.9);

    const std::vector<double>& forward(const std::vector<double>& x);
    std::vector<double> backward(const std::vector<double>& grad_out);
    void update(double lr);

private:
    size_t n_tokens_ = 0;
    std::vector<double> x_saved_;
    std::vector<double> x_hat_;      // [n_tokens * d_model]
    std::vector<double> inv_sigma_;  // [n_tokens]
};

}}
