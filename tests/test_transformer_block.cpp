#include "ml/transformer_block.hpp"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <random>

using namespace absolutus::ml;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    std::mt19937_64 rng(123);

    // 1. Forward shape and finiteness (classical, 1 head)
    {
        TransformerBlock b(2, 4, 8, rng);
        std::vector<double> x = {0.5, 0.3, -0.2, 0.7, 0.1, -0.4, 0.6, -0.5};
        auto y = b.forward(x);
        CHECK(y.size() == 8);
        for (double v : y) CHECK(std::isfinite(v));
    }

    // 2. FD gradient check on input x
    {
        TransformerBlock b(2, 3, 6, rng);
        std::vector<double> x = {0.5, 0.3, -0.2, 0.7, 0.1, -0.4};
        std::vector<double> gy = {1.0, -0.5, 0.3, 0.2, 0.7, -0.1};

        b.forward(x);
        auto gx = b.backward(gy);

        const double EPS = 1e-5;
        for (size_t idx : {size_t(0), size_t(3), size_t(5)}) {
            auto xp = x; xp[idx] += EPS;
            auto yp = b.forward(xp);
            double Lp = 0.0;
            for (size_t i = 0; i < gy.size(); ++i) Lp += gy[i] * yp[i];

            auto xm = x; xm[idx] -= EPS;
            auto ym = b.forward(xm);
            double Lm = 0.0;
            for (size_t i = 0; i < gy.size(); ++i) Lm += gy[i] * ym[i];

            b.forward(x);
            double fd = (Lp - Lm) / (2.0 * EPS);
            CHECK(std::abs(gx[idx] - fd) < 1e-3);
        }
    }

    // 3. Update keeps weights finite
    {
        TransformerBlock b(2, 3, 6, rng);
        std::vector<double> x = {0.5, 0.3, -0.2, 0.7, 0.1, -0.4};
        b.forward(x);
        b.backward({1.0, -1.0, 0.5, 0.2, 0.3, -0.4});
        b.update(0.01);
        CHECK(std::isfinite(b.ln1.gamma[0]));
        CHECK(std::isfinite(b.ffn1.W[0]));
    }

    // 4. Residual preserves gradient scale across depth
    {
        auto measure = [&](size_t n_blocks, bool residual) {
            std::mt19937_64 rng_local(999);
            std::vector<TransformerBlock> blocks;
            for (size_t i = 0; i < n_blocks; ++i)
                blocks.emplace_back(2, 4, 8, rng_local, residual);
            std::vector<double> x = {0.5, 0.3, -0.2, 0.7, 0.1, -0.4, 0.6, -0.5};
            std::vector<double> cur = x;
            for (auto& b : blocks) cur = b.forward(cur);
            std::vector<double> gy(cur.size(), 1.0);
            std::vector<double> g = gy;
            for (auto it = blocks.rbegin(); it != blocks.rend(); ++it)
                g = it->backward(g);
            double n2 = 0.0;
            for (double v : g) n2 += v*v;
            return std::sqrt(n2);
        };

        const double g1r  = measure(1, true);
        const double g4r  = measure(4, true);
        const double g1nr = measure(1, false);
        const double g4nr = measure(4, false);

        const double ratio_r  = g4r / g1r;
        const double ratio_nr = g4nr / g1nr;

        std::fprintf(stderr,
            "  gradient scale (||g_N=4|| / ||g_N=1||):\n"
            "    with residual:    %.4f\n"
            "    without residual: %.4f\n", ratio_r, ratio_nr);

        CHECK(ratio_r > 0.05 && ratio_r < 20.0);
        const double dev_r  = std::abs(std::log(ratio_r));
        const double dev_nr = std::abs(std::log(ratio_nr));
        CHECK(dev_r <= dev_nr + 1.0);
    }

    // 5. Hybrid block: n_heads=2, quantum_head_idx=0 (one quantum head).
    //    Forward finite, FD gradient on dX matches to 1e-2.
    {
        TransformerBlock b(2, 4, 8, rng, true, 0.9, /*n_heads=*/2, /*qidx=*/0);
        std::vector<double> x = {0.5, 0.3, -0.2, 0.7, 0.1, -0.4, 0.6, -0.5};
        std::vector<double> gy = {1.0, -0.5, 0.3, 0.2, 0.7, -0.1, -0.3, 0.4};

        auto y = b.forward(x);
        CHECK(y.size() == 8);
        for (double v : y) CHECK(std::isfinite(v));

        auto gx = b.backward(gy);
        const double EPS = 1e-4;
        for (size_t idx : {size_t(0), size_t(4)}) {
            auto xp = x; xp[idx] += EPS;
            auto yp = b.forward(xp);
            double Lp = 0.0;
            for (size_t i = 0; i < gy.size(); ++i) Lp += gy[i] * yp[i];

            auto xm = x; xm[idx] -= EPS;
            auto ym = b.forward(xm);
            double Lm = 0.0;
            for (size_t i = 0; i < gy.size(); ++i) Lm += gy[i] * ym[i];

            b.forward(x);
            double fd = (Lp - Lm) / (2.0 * EPS);
            CHECK(std::abs(gx[idx] - fd) < 1e-2);
        }
    }

    return 0;
}
