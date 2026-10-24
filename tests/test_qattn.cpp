#include "hybrid/qattn.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using namespace absolutus::hybrid;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    std::mt19937_64 rng(7);

    // --- 1. Forward shape and finiteness ---
    {
        QuantumAttention A(2, 3, 2, 1, rng);
        std::vector<double> X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
        auto Y = A.forward(X);
        CHECK(Y.size() == 2 * 2);
        for (double y : Y) CHECK(std::isfinite(y));
    }

    // --- 2. Gradient dL/dtheta_ql[k] vs finite differences ---
    {
        QuantumAttention A(2, 3, 2, 1, rng);
        std::vector<double> X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
        std::vector<double> gy = {1.0, -0.5, 0.7, -0.3};

        A.forward(X);
        A.backward(gy);
        // Store analytic dtheta for the ql path
        auto analytic_dtheta = A.ql.dtheta;

        // For each theta index, compute numeric via FD
        const double EPS = 1e-5;
        for (size_t k = 0; k < A.ql.theta.size(); ++k) {
            auto saved = A.ql.theta;
            A.ql.theta[k] = saved[k] + EPS;
            auto Yp = A.forward(X);
            double Lp = 0.0;
            for (size_t i = 0; i < gy.size(); ++i) Lp += gy[i] * Yp[i];

            A.ql.theta[k] = saved[k] - EPS;
            auto Ym = A.forward(X);
            double Lm = 0.0;
            for (size_t i = 0; i < gy.size(); ++i) Lm += gy[i] * Ym[i];

            A.ql.theta[k] = saved[k];
            double fd = (Lp - Lm) / (2.0 * EPS);
            CHECK(std::abs(analytic_dtheta[k] - fd) < 1e-3);
        }
        // Restore to a valid cached state (re-run forward)
        A.forward(X);
    }

    // --- 3. Update keeps everything finite ---
    {
        QuantumAttention A(3, 2, 2, 1, rng);
        std::vector<double> X = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
        A.forward(X);
        A.backward({1.0, -1.0, 0.5, -0.5, 0.2, 0.0});
        A.update(0.01);
        for (double w : A.Wk)  CHECK(std::isfinite(w));
        for (double w : A.Wv)  CHECK(std::isfinite(w));
        for (double w : A.pre_q.W) CHECK(std::isfinite(w));
        for (double t : A.ql.theta) CHECK(std::isfinite(t));
    }

    return 0;
}
