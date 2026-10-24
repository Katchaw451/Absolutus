#include "ml/mlp.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using namespace absolutus::ml;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    std::mt19937_64 rng(7);

    // Dense: identity weights => y = x
    {
        Dense d(2, 2, Activation::Linear, rng);
        d.W = {1.0, 0.0, 0.0, 1.0};
        d.b = {0.0, 0.0};
        auto y = d.forward({3.0, 4.0});
        CHECK(std::abs(y[0] - 3.0) < 1e-12);
        CHECK(std::abs(y[1] - 4.0) < 1e-12);
    }

    // Dense with tanh: forward is bounded, gradient respects activation
    {
        Dense d(2, 2, Activation::Tanh, rng);
        d.W = {1.0, 0.0, 0.0, 1.0};
        d.b = {0.0, 0.0};
        auto y = d.forward({0.5, -0.5});
        CHECK(std::abs(y[0] - std::tanh(0.5)) < 1e-12);
        auto gx = d.backward({1.0, 0.0});
        // d/dx tanh(x) at x=0.5 = 1 - tanh(0.5)^2
        const double expected = 1.0 - std::tanh(0.5) * std::tanh(0.5);
        CHECK(std::abs(gx[0] - expected) < 1e-12);
        CHECK(std::abs(gx[1]) < 1e-12);
    }

    // softmax sums to 1, monotonically increasing
    {
        auto p = softmax({1.0, 2.0, 3.0});
        CHECK(std::abs(p[0] + p[1] + p[2] - 1.0) < 1e-12);
        CHECK(p[2] > p[1]);
        CHECK(p[1] > p[0]);
    }

    // cross-entropy = 0 when prediction is one-hot correct
    {
        CHECK(cross_entropy({1.0, 0.0}, {1.0, 0.0}) < 1e-9);
        CHECK(cross_entropy({0.0, 1.0}, {0.0, 1.0}) < 1e-9);
    }

    // cross-entropy > 0 when prediction is wrong
    {
        CHECK(cross_entropy({0.0, 1.0}, {1.0, 0.0}) > 10.0);
    }

    // MLP: sum of gradients flow from output to input
    {
        MLP net;
        net.layers.emplace_back(3, 4, Activation::Tanh, rng);
        net.layers.emplace_back(4, 2, Activation::Linear, rng);
        auto y = net.forward({0.1, 0.2, 0.3});
        CHECK(y.size() == 2);
        // Doesn't crash, no NaNs
        net.backward({1.0, -1.0});
        net.update(0.01);
        auto y2 = net.forward({0.1, 0.2, 0.3});
        CHECK(std::isfinite(y2[0]));
        CHECK(std::isfinite(y2[1]));
    }

    return 0;
}
