#include "hybrid/qlayer.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using namespace absolutus;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    std::mt19937_64 rng(7);
    hybrid::QuantumLayer ql(2, 4, rng, 0.9);

    // Fix theta to reproducible values
    ql.theta[0] = 0.5; ql.theta[1] = 1.2;
    ql.theta[2] = -0.7; ql.theta[3] = 2.0;

    const std::vector<double> x = {0.3, 0.7};

    // Forward: 2 outputs in [-1, +1]
    auto y = ql.forward(x);
    CHECK(y.size() == 2);
    CHECK(std::abs(y[0]) <= 1.0 + 1e-9);
    CHECK(std::abs(y[1]) <= 1.0 + 1e-9);

    // Parameter-shift gradient vs finite differences, for theta_0
    {
        auto saved = ql.theta;
        const double EPS = 1e-5;
        const std::vector<double> gy = {1.0, 0.0};

        // Analytic
        ql.forward(x);
        ql.backward(gy);
        const double analytic = ql.dtheta[0];

        // Numeric
        ql.theta = saved;
        ql.theta[0] += EPS;
        auto yp = ql.forward(x);
        ql.theta = saved;
        ql.theta[0] -= EPS;
        auto ym = ql.forward(x);
        ql.theta = saved;
        const double numeric = (yp[0] - ym[0]) / (2.0 * EPS);

        CHECK(std::abs(analytic - numeric) < 1e-4);
    }

    // Input gradient vs finite differences, for x_0
    {
        auto saved_x = x;
        const double EPS = 1e-5;
        const std::vector<double> gy = {1.0, 0.0};

        auto gx = ql.backward(gy);  // uses x_saved from last forward
        const double analytic = gx[0];

        auto xp = saved_x; xp[0] += EPS;
        auto yp = ql.forward(xp);
        auto xm = saved_x; xm[0] -= EPS;
        auto ym = ql.forward(xm);
        ql.forward(saved_x);  // restore internal state

        const double numeric = (yp[0] - ym[0]) / (2.0 * EPS);
        CHECK(std::abs(analytic - numeric) < 1e-4);
    }

    // Update keeps theta in bounded range (no NaN / Inf)
    ql.forward(x);
    ql.backward({1.0, 0.0});
    ql.update(0.1);
    for (double t : ql.theta) {
        CHECK(std::isfinite(t));
    }

    return 0;
}
