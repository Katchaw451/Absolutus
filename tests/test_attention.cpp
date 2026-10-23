#include "ml/attention.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using namespace absolutus::ml;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    std::mt19937_64 rng(7);

    // --- Test 1: identity projections => uniform attention ---
    // With Wq = Wk = 0, all scores = 0, softmax = uniform 1/T.
    // With Wv = I, output = mean over rows of X (broadcast per dim).
    {
        Attention A(3, 2, 2, rng);
        A.Wq = {0,0,0,0};
        A.Wk = {0,0,0,0};
        A.Wv = {1,0, 0,1};

        std::vector<double> X = {
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        };
        auto Y = A.forward(X);
        // Mean per dim: (1+3+5)/3 = 3.0, (2+4+6)/3 = 4.0
        for (size_t t = 0; t < 3; ++t) {
            CHECK(std::abs(Y[t*2 + 0] - 3.0) < 1e-12);
            CHECK(std::abs(Y[t*2 + 1] - 4.0) < 1e-12);
        }
    }

    // --- Test 2: one-hot attention picks a specific row ---
    // Set Wq = I, Wk = I, Wv = I. X with rows [1,0], [0,1], [0,0].
    // Scores: q_i · k_j = x_i · x_j. Row 0 has high score with itself.
    {
        Attention A(3, 2, 2, rng);
        A.Wq = {1,0, 0,1};
        A.Wk = {1,0, 0,1};
        A.Wv = {1,0, 0,1};

        std::vector<double> X = {
            10.0, 0.0,
             0.0, 10.0,
             0.0,  0.0
        };
        auto Y = A.forward(X);
        // Large scores: row 0 attends to itself (10*10 = 100), row 1 to itself.
        // The softmax becomes very peaked with d_k=2 and these magnitudes.
        // Row 0 output should be close to [10, 0].
        CHECK(Y[0] > 9.5);        // close to 10
        CHECK(std::abs(Y[1]) < 0.5);
        // Row 1 output should be close to [0, 10].
        CHECK(std::abs(Y[2]) < 0.5);
        CHECK(Y[3] > 9.5);
    }

    // --- Test 3: backward grad against numerical finite differences ---
    {
        Attention A(2, 3, 2, rng);
        // Random initial weights from constructor. Fix X.
        std::vector<double> X = {0.5, -0.2, 0.1, 0.3, 0.4, -0.6};

        auto Y = A.forward(X);
        std::vector<double> gy = {1.0, 0.5, -0.5, 0.2};
        auto gX = A.backward(gy);

        // Numeric check on one component of X
        const double EPS = 1e-5;
        const size_t idx = 1;
        auto Xp = X; Xp[idx] += EPS;
        auto Yp = A.forward(Xp);
        double numeric = 0.0;
        for (size_t i = 0; i < gy.size(); ++i) numeric += gy[i] * Yp[i];

        auto Xm = X; Xm[idx] -= EPS;
        auto Ym = A.forward(Xm);
        double numeric_m = 0.0;
        for (size_t i = 0; i < gy.size(); ++i) numeric_m += gy[i] * Ym[i];

        A.forward(X); // restore
        double fd = (numeric - numeric_m) / (2.0 * EPS);
        CHECK(std::abs(gX[idx] - fd) < 1e-4);
    }

    // --- Test 4: update does not blow up weights ---
    {
        Attention A(2, 2, 2, rng);
        std::vector<double> X = {0.1, 0.2, 0.3, 0.4};
        A.forward(X);
        A.backward({1.0, -1.0, 0.5, -0.5});
        A.update(0.01);
        for (double w : A.Wq) CHECK(std::isfinite(w));
        for (double w : A.Wk) CHECK(std::isfinite(w));
        for (double w : A.Wv) CHECK(std::isfinite(w));
    }

    return 0;
}
