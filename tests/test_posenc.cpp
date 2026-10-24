#include "ml/posenc.hpp"
#include "ml/attention.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using namespace absolutus::ml;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    // 1. Known values: pos=0 => [sin(0), cos(0), sin(0), cos(0)] = [0,1,0,1]
    {
        PositionEncoding pe(3, 4);
        for (size_t i = 0; i < 4; ++i) {
            const double expected = (i % 2 == 0) ? 0.0 : 1.0;
            CHECK(std::abs(pe.table[i] - expected) < 1e-12);
        }
    }

    // 2. Distinct rows for distinct positions
    {
        PositionEncoding pe(4, 6);
        bool all_distinct = true;
        for (size_t i = 1; i < 4; ++i) {
            double diff = 0.0;
            for (size_t j = 0; j < 6; ++j)
                diff += std::abs(pe.table[i*6+j] - pe.table[(i-1)*6+j]);
            if (diff < 1e-6) all_distinct = false;
        }
        CHECK(all_distinct);
    }

    // 3. Adding PE breaks permutation invariance of Attention.
    //    With PE: Attention(P X) != P Attention(X + PE).
    //    Without PE: Attention(P X) == P Attention(X) (up to numeric).
    {
        std::mt19937_64 rng(42);
        Attention A(3, 4, 4, rng);
        PositionEncoding pe(3, 4);

        std::vector<double> X = {0.1, 0.2, 0.3, 0.4,
                                 0.5, -0.2, 0.3, -0.1,
                                 -0.3, 0.6, -0.4, 0.2};
        std::vector<double> PX = {X[8], X[9], X[10], X[11],
                                  X[0], X[1], X[2], X[3],
                                  X[4], X[5], X[6], X[7]};

        // Without PE
        auto Yn = A.forward(X);
        auto Yp_perm = A.forward(PX);
        std::vector<double> Yn_perm = {Yn[8], Yn[9], Yn[10], Yn[11],
                                       Yn[0], Yn[1], Yn[2], Yn[3],
                                       Yn[4], Yn[5], Yn[6], Yn[7]};
        double diff_no_pe = 0.0;
        for (size_t i = 0; i < 12; ++i) diff_no_pe += std::abs(Yp_perm[i] - Yn_perm[i]);
        CHECK(diff_no_pe < 1e-9);   // permutation-invariant without PE

        // With PE
        auto Xpe  = pe.add(X);
        auto PXpe = pe.add(PX);
        auto Y1 = A.forward(Xpe);
        auto Y2 = A.forward(PXpe);
        double diff_pe = 0.0;
        for (size_t i = 0; i < 12; ++i) diff_pe += std::abs(Y2[i] - Y1[i]);
        CHECK(diff_pe > 1e-3);      // permutation-broken with PE
    }

    return 0;
}
