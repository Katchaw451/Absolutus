#include "ml/multihead.hpp"
#include "ml/attention.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using namespace absolutus::ml;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    std::mt19937_64 rng(2025);

    // 1. h=1 with Wo=I reduces to a single Attention exactly.
    {
        MultiHeadAttention m(2, 4, 1, rng);
        std::vector<double> X = {0.1, 0.2, 0.3, 0.4,  0.5, -0.2, 0.3, -0.1};
        auto mh_out = m.forward(X);
        std::vector<double> saved(mh_out.begin(), mh_out.end());
        auto sh_out = m.heads[0]->forward(X);
        for (size_t i = 0; i < saved.size(); ++i)
            CHECK(std::abs(saved[i] - sh_out[i]) < 1e-12);
    }

    // 2. h=2 forward: shape and finiteness
    {
        MultiHeadAttention m(3, 4, 2, rng);
        std::vector<double> X = {0.1, 0.2, 0.3, 0.4,
                                 0.5, -0.2, 0.3, -0.1,
                                 -0.3, 0.6, -0.4, 0.2};
        auto Y = m.forward(X);
        CHECK(Y.size() == 3 * 4);
        for (double v : Y) CHECK(std::isfinite(v));
    }

    // 3. h=2 FD gradient on dX
    {
        MultiHeadAttention m(2, 4, 2, rng);
        std::vector<double> X = {0.1, 0.2, 0.3, 0.4,  0.5, -0.2, 0.3, -0.1};
        std::vector<double> gy = {1.0, -0.5, 0.3, 0.2,  -0.4, 0.7, -0.1, 0.6};

        m.forward(X);
        auto gX = m.backward(gy);

        const double EPS = 1e-5;
        for (size_t idx : {size_t(0), size_t(4), size_t(7)}) {
            auto Xp = X; Xp[idx] += EPS;
            auto Yp = m.forward(Xp);
            double Lp = 0.0;
            for (size_t i = 0; i < gy.size(); ++i) Lp += gy[i] * Yp[i];

            auto Xm = X; Xm[idx] -= EPS;
            auto Ym = m.forward(Xm);
            double Lm = 0.0;
            for (size_t i = 0; i < gy.size(); ++i) Lm += gy[i] * Ym[i];

            m.forward(X);
            double fd = (Lp - Lm) / (2.0 * EPS);
            CHECK(std::abs(gX[idx] - fd) < 1e-3);
        }
    }

    // 4. update keeps weights finite (all-classical case)
    {
        MultiHeadAttention m(2, 4, 2, rng);
        std::vector<double> X = {0.1, 0.2, 0.3, 0.4,  0.5, -0.2, 0.3, -0.1};
        m.forward(X);
        m.backward({1.0, -1.0, 0.5, 0.2, -0.4, 0.7, -0.1, 0.6});
        m.update(0.01);
        for (double v : m.Wo) CHECK(std::isfinite(v));
        for (auto& h : m.heads) {
            auto* a = dynamic_cast<Attention*>(h.get());
            CHECK(a != nullptr);
            if (a) for (double v : a->Wq) CHECK(std::isfinite(v));
        }
    }

    // 5. invalid config throws
    {
        bool threw = false;
        try { MultiHeadAttention bad(2, 5, 2, rng); }
        catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);

        bool threw_q = false;
        try { MultiHeadAttention bad(2, 4, 2, rng, 0.9, /*qidx=*/2); }
        catch (const std::invalid_argument&) { threw_q = true; }
        CHECK(threw_q);
    }

    // 6. Quantum head: forward finite, FD gradient on dX matches.
    //    Head 0 is QuantumAttention, head 1 is classical Attention.
    {
        MultiHeadAttention m(2, 4, 2, rng, 0.9, /*quantum_head_idx=*/0);
        std::vector<double> X = {0.1, 0.2, 0.3, 0.4,  0.5, -0.2, 0.3, -0.1};
        std::vector<double> gy = {1.0, -0.5, 0.3, 0.2,  -0.4, 0.7, -0.1, 0.6};

        auto Y = m.forward(X);
        for (double v : Y) CHECK(std::isfinite(v));

        auto gX = m.backward(gy);

        const double EPS = 1e-4;
        for (size_t idx : {size_t(0), size_t(4)}) {
            auto Xp = X; Xp[idx] += EPS;
            auto Yp = m.forward(Xp);
            double Lp = 0.0;
            for (size_t i = 0; i < gy.size(); ++i) Lp += gy[i] * Yp[i];

            auto Xm = X; Xm[idx] -= EPS;
            auto Ym = m.forward(Xm);
            double Lm = 0.0;
            for (size_t i = 0; i < gy.size(); ++i) Lm += gy[i] * Ym[i];

            m.forward(X);
            double fd = (Lp - Lm) / (2.0 * EPS);
            CHECK(std::abs(gX[idx] - fd) < 1e-2);
        }
    }

    return 0;
}
