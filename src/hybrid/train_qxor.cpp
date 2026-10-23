#include "hybrid/qlayer.hpp"
#include "ml/mlp.hpp"
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace absolutus {
namespace hybrid {

int run_train_qxor(int epochs, double lr, double momentum,
                   size_t n_params, uint64_t seed) {
    std::mt19937_64 rng(seed);

    struct Sample { std::vector<double> x, y; };
    const std::vector<Sample> data = {
        {{0.0, 0.0}, {1.0, 0.0}},
        {{0.0, 1.0}, {0.0, 1.0}},
        {{1.0, 0.0}, {0.0, 1.0}},
        {{1.0, 1.0}, {1.0, 0.0}},
    };

    QuantumLayer ql(2, n_params, rng, momentum, 3.14159265358979323846);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        for (const auto& s : data) {
            auto z = ql.forward(s.x);
            auto p = ml::softmax(z);
            std::vector<double> g(p.size());
            for (size_t i = 0; i < p.size(); ++i) g[i] = p[i] - s.y[i];
            ql.backward(g);
            ql.update(lr);
        }
    }

    std::fprintf(stderr,
        "QuantumLayer on XOR (epochs=%d, lr=%.3f, momentum=%.2f, "
        "n_params=%zu, seed=%llu):\n",
        epochs, lr, momentum, n_params, (unsigned long long)seed);

    int correct = 0;
    for (const auto& s : data) {
        auto z = ql.forward(s.x);
        auto p = ml::softmax(z);
        int pred  = (p[0] > p[1]) ? 0 : 1;
        int truth = (s.y[0] > s.y[1]) ? 0 : 1;
        if (pred == truth) ++correct;
        std::fprintf(stderr,
            "  in=(%.0f,%.0f)  truth=%d  pred=%d  "
            "<Z>=[%+.4f, %+.4f]  p=[%.4f, %.4f]\n",
            s.x[0], s.x[1], truth, pred, z[0], z[1], p[0], p[1]);
    }

    if (correct != 4) {
        std::fprintf(stderr, "  RESULT: FAIL (%d/4)\n", correct);
        return 1;
    }
    std::fprintf(stderr, "  RESULT: PASS (4/4)\n");
    return 0;
}

}}
