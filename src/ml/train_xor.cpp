#include "ml/mlp.hpp"
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace absolutus {
namespace ml {

int run_train_xor(int epochs, double lr, double momentum, uint64_t seed) {
    std::mt19937_64 rng(seed);

    struct Sample { std::vector<double> x, y; };
    const std::vector<Sample> data = {
        {{0.0, 0.0}, {1.0, 0.0}},
        {{0.0, 1.0}, {0.0, 1.0}},
        {{1.0, 0.0}, {0.0, 1.0}},
        {{1.0, 1.0}, {1.0, 0.0}},
    };

    MLP net;
    net.layers.emplace_back(2, 4, Activation::Tanh,   rng, momentum);
    net.layers.emplace_back(4, 2, Activation::Linear, rng, momentum);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        for (const auto& s : data) {
            const auto logits = net.forward(s.x);
            const auto probs  = softmax(logits);
            std::vector<double> g(probs.size());
            for (size_t i = 0; i < probs.size(); ++i) g[i] = probs[i] - s.y[i];
            net.backward(g);
            net.update(lr);
        }
    }

    std::fprintf(stderr, "XOR after %d epochs (lr=%.3f, momentum=%.2f, seed=%llu):\n",
                 epochs, lr, momentum, (unsigned long long)seed);
    int correct = 0;
    for (const auto& s : data) {
        const auto probs = softmax(net.forward(s.x));
        const int pred  = (probs[0] > probs[1]) ? 0 : 1;
        const int truth = (s.y[0]  > s.y[1])    ? 0 : 1;
        if (pred == truth) ++correct;
        std::fprintf(stderr, "  in=(%.0f,%.0f)  truth=%d  pred=%d  p=[%.4f, %.4f]\n",
                     s.x[0], s.x[1], truth, pred, probs[0], probs[1]);
    }
    if (correct != 4) { std::fprintf(stderr, "  RESULT: FAIL (%d/4)\n", correct); return 1; }
    std::fprintf(stderr, "  RESULT: PASS (4/4)\n");
    return 0;
}

}}
