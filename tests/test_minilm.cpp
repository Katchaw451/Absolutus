#include "ml/minilm.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using namespace absolutus::ml;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

static bool all_finite(const std::vector<double>& v) {
    for (double x : v) if (!std::isfinite(x)) return false;
    return true;
}

int main() {
    // Bisect: with 0, 1, and 2 blocks, at the exact MiniLM sizes.
    const size_t L = 4, DM = 8, DFF = 16, VS = 8;

    for (size_t nb : {size_t(0), size_t(1), size_t(2)}) {
        std::mt19937_64 rng(42);
        MiniLM lm(L, DM, DFF, VS, nb, rng);
        std::vector<size_t> tokens = {0, 3, 5, 7};
        const auto& logits = lm.forward(tokens);

        if (!all_finite(logits)) {
            std::fprintf(stderr, "FAIL: NaN with n_blocks=%zu\n", nb);
            for (size_t i = 0; i < logits.size(); ++i)
                if (!std::isfinite(logits[i]))
                    std::fprintf(stderr, "  logits[%zu]=%g\n", i, logits[i]);
            return 1;
        }
    }
    std::printf("ok\n");
    return 0;
}
