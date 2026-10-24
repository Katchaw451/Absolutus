#include "hybrid/qlayer.hpp"
#include "ml/mlp.hpp"
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace absolutus {
namespace hybrid {

// Architecture:
//   x in R^2  --> Dense(2 -> n_q, tanh) --> QuantumLayer(n_q qubits)
//             --> Dense(n_q -> 2, linear) --> softmax --> CE loss
//
// Two gradient mechanisms compose in a single backward pass:
//   - Dense layers: standard backprop (analytic)
//   - QuantumLayer: parameter-shift (circuit re-evaluations)
// The MLP infrastructure doesn't notice the difference: both expose
// forward/backward/update.

int run_train_hybrid(int epochs, double lr, double momentum,
                     size_t n_q, size_t q_params, uint64_t seed) {
    std::mt19937_64 rng(seed);

    struct Sample { std::vector<double> x, y; };
    const std::vector<Sample> data = {
        {{0.0, 0.0}, {1.0, 0.0}},
        {{0.0, 1.0}, {0.0, 1.0}},
        {{1.0, 0.0}, {0.0, 1.0}},
        {{1.0, 1.0}, {1.0, 0.0}},
    };

    ml::Dense     d1(2,    n_q, ml::Activation::Tanh,   rng, momentum);
    QuantumLayer  ql(n_q, q_params, rng, momentum, 3.14159265358979323846);
    ml::Dense     d2(n_q, 2,   ml::Activation::Linear, rng, momentum);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        for (const auto& s : data) {
            // Forward (three stages, two worlds)
            auto h      = d1.forward(s.x);       // classical
            auto zq     = ql.forward(h);         // quantum: <Z_q>(h)
            auto logits = d2.forward(zq);        // classical
            auto probs  = ml::softmax(logits);

            // dL/dlogits = probs - one_hot
            std::vector<double> g_logits(probs.size());
            for (size_t i = 0; i < probs.size(); ++i)
                g_logits[i] = probs[i] - s.y[i];

            // Backward (three stages, two mechanisms)
            auto g_zq = d2.backward(g_logits);   // backprop
            auto g_h  = ql.backward(g_zq);       // parameter-shift on input
            d1.backward(g_h);                    // backprop

            // Update (each layer's own optimizer)
            d1.update(lr);
            ql.update(lr);
            d2.update(lr);
        }
    }

    std::fprintf(stderr,
        "Hybrid XOR (epochs=%d, lr=%.3f, momentum=%.2f, "
        "n_q=%zu, q_params=%zu, seed=%llu):\n",
        epochs, lr, momentum, n_q, q_params, (unsigned long long)seed);

    int correct = 0;
    for (const auto& s : data) {
        auto h      = d1.forward(s.x);
        auto zq     = ql.forward(h);
        auto logits = d2.forward(zq);
        auto probs  = ml::softmax(logits);
        int pred  = (probs[0] > probs[1]) ? 0 : 1;
        int truth = (s.y[0] > s.y[1])    ? 0 : 1;
        if (pred == truth) ++correct;
        std::fprintf(stderr,
            "  in=(%.0f,%.0f)  truth=%d  pred=%d  "
            "<Z_q>=[%+.3f,%+.3f]  p=[%.4f, %.4f]\n",
            s.x[0], s.x[1], truth, pred, zq[0], zq[1], probs[0], probs[1]);
    }
    if (correct != 4) {
        std::fprintf(stderr, "  RESULT: FAIL (%d/4)\n", correct);
        return 1;
    }
    std::fprintf(stderr, "  RESULT: PASS (4/4)\n");
    return 0;
}

}}
