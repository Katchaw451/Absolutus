#include "ml/layernorm.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using namespace absolutus::ml;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    std::mt19937_64 rng(11);

    // --- 1. Normalization: with gamma=1, beta=0, each token has mean 0, var 1 ---
    {
        LayerNorm ln(4);
        std::vector<double> x = {
            1.0, 2.0, 3.0, 4.0,          // token 0
            -5.0, 0.5, 2.5, 7.0          // token 1
        };
        auto y = ln.forward(x);
        for (size_t t = 0; t < 2; ++t) {
            double mu = 0.0, var = 0.0;
            for (size_t i = 0; i < 4; ++i) mu += y[t*4 + i];
            mu /= 4.0;
            for (size_t i = 0; i < 4; ++i) {
                double d = y[t*4 + i] - mu;
                var += d*d;
            }
            var /= 4.0;
            CHECK(std::abs(mu) < 1e-9);
            // eps=1e-5 means variance is slightly under 1
            CHECK(std::abs(var - 1.0) < 1e-4);
        }
    }

    // --- 2. gamma, beta shift output as expected ---
    {
        LayerNorm ln(3);
        ln.gamma = {2.0, 1.0, 0.5};
        ln.beta  = {1.0, -1.0, 0.0};
        std::vector<double> x = {1.0, 2.0, 3.0};
        auto y = ln.forward(x);
        // x_hat with d=3, mu=2, var=2/3: (-1, 0, 1) / sqrt(2/3) ~ (-1.2247, 0, 1.2247)
        // y = gamma * x_hat + beta
        const double s = std::sqrt(2.0/3.0 + 1e-5);
        CHECK(std::abs(y[0] - (2.0 * (-1.0/s) + 1.0)) < 1e-4);
        CHECK(std::abs(y[1] - (1.0 * 0.0        - 1.0)) < 1e-4);
        CHECK(std::abs(y[2] - (0.5 * (1.0/s)  + 0.0)) < 1e-4);
    }

    // --- 3. Gradient vs finite differences (all outputs combined) ---
    {
        LayerNorm ln(3);
        ln.gamma = {1.3, 0.7, 1.1};
        ln.beta  = {0.1, -0.2, 0.3};
        std::vector<double> x = {0.5, -0.2, 0.8,  1.1, 0.4, -0.9};
        std::vector<double> gy = {1.0, -0.5, 0.3,  0.2, 0.7, -0.1};

        auto y = ln.forward(x);
        (void)y;
        auto gx = ln.backward(gy);

        // Save analytic gradients
        auto analytic_dgamma = ln.dgamma;
        auto analytic_dbeta  = ln.dbeta;

        // FD on a few entries of x
        const double EPS = 1e-5;
        for (size_t idx : {size_t(0), size_t(2), size_t(4)}) {
            auto xp = x; xp[idx] += EPS;
            auto yp = ln.forward(xp);
            double Lp = 0.0;
            for (size_t i = 0; i < gy.size(); ++i) Lp += gy[i] * yp[i];

            auto xm = x; xm[idx] -= EPS;
            auto ym = ln.forward(xm);
            double Lm = 0.0;
            for (size_t i = 0; i < gy.size(); ++i) Lm += gy[i] * ym[i];

            double fd = (Lp - Lm) / (2.0 * EPS);
            CHECK(std::abs(gx[idx] - fd) < 1e-4);
        }

        // FD on gamma[0]
        auto saved_gamma = ln.gamma;
        ln.gamma[0] = saved_gamma[0] + EPS;
        auto yp = ln.forward(x);
        double Lp = 0.0;
        for (size_t i = 0; i < gy.size(); ++i) Lp += gy[i] * yp[i];
        ln.gamma[0] = saved_gamma[0] - EPS;
        auto ym = ln.forward(x);
        double Lm = 0.0;
        for (size_t i = 0; i < gy.size(); ++i) Lm += gy[i] * ym[i];
        ln.gamma[0] = saved_gamma[0];
        double fd_gamma = (Lp - Lm) / (2.0 * EPS);
        CHECK(std::abs(analytic_dgamma[0] - fd_gamma) < 1e-4);

        // FD on beta[1]
        auto saved_beta = ln.beta;
        ln.beta[1] = saved_beta[1] + EPS;
        auto ypb = ln.forward(x);
        double Lpb = 0.0;
        for (size_t i = 0; i < gy.size(); ++i) Lpb += gy[i] * ypb[i];
        ln.beta[1] = saved_beta[1] - EPS;
        auto ymb = ln.forward(x);
        double Lmb = 0.0;
        for (size_t i = 0; i < gy.size(); ++i) Lmb += gy[i] * ymb[i];
        ln.beta[1] = saved_beta[1];
        double fd_beta = (Lpb - Lmb) / (2.0 * EPS);
        CHECK(std::abs(analytic_dbeta[1] - fd_beta) < 1e-4);
    }

    // --- 4. Pre-norm residual pattern: y = x + LayerNorm(x) ---
    // Checks the composition (LayerNorm in a residual stream) is well-formed.
    {
        LayerNorm ln(3);
        std::vector<double> x = {0.5, -0.2, 0.8,  1.1, 0.4, -0.9};
        auto ln_y = ln.forward(x);
        std::vector<double> residual_y(x.size());
        for (size_t i = 0; i < x.size(); ++i) residual_y[i] = x[i] + ln_y[i];
        for (double v : residual_y) CHECK(std::isfinite(v));
    }

    // --- 5. Update keeps weights finite ---
    {
        LayerNorm ln(3);
        std::vector<double> x = {0.5, -0.2, 0.8};
        ln.forward(x);
        ln.backward({1.0, -1.0, 0.5});
        ln.update(0.1);
        for (double g : ln.gamma) CHECK(std::isfinite(g));
        for (double b : ln.beta)  CHECK(std::isfinite(b));
    }

    return 0;
}
