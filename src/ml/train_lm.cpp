#include "ml/minilm.hpp"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>

namespace absolutus {
namespace ml {

int run_train_lm(int epochs, double lr, uint64_t seed) {
    const size_t L = 4, DM = 8, DFF = 16, VS = 8, NB = 2;
    std::mt19937_64 rng(seed);
    MiniLM lm(L, DM, DFF, VS, NB, rng);

    std::uniform_int_distribution<size_t> tok(0, VS - 1);
    const size_t n_seqs = 32;
    std::vector<std::vector<size_t>> dataset(n_seqs, std::vector<size_t>(L));
    for (auto& s : dataset) for (auto& x : s) x = tok(rng);

    std::printf("epoch,loss\n");
    double first_loss = 0.0, last_loss = 0.0;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t si = 0; si < dataset.size(); ++si) {
            const auto& tokens = dataset[si];
            std::vector<double> logits_copy;
            try {
                logits_copy = lm.forward(tokens);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "epoch %d seq %zu tokens = {", epoch, si);
                for (size_t t = 0; t < tokens.size(); ++t)
                    std::fprintf(stderr, "%zu%s", tokens[t],
                                 t + 1 < tokens.size() ? "," : "");
                std::fprintf(stderr, "}  error: %s\n", e.what());
                return 1;
            }

            std::vector<double> grad_logits(L * VS, 0.0);
            for (size_t t = 0; t < L; ++t) {
                double m = -1e30;
                for (size_t v = 0; v < VS; ++v) if (logits_copy[t*VS+v] > m) m = logits_copy[t*VS+v];
                double sum = 0.0;
                for (size_t v = 0; v < VS; ++v) sum += std::exp(logits_copy[t*VS+v] - m);
                for (size_t v = 0; v < VS; ++v) {
                    double p = std::exp(logits_copy[t*VS+v] - m) / sum;
                    const double target = (v == tokens[t]) ? 1.0 : 0.0;
                    if (target > 0.0) total_loss -= std::log(p + 1e-12);
                    grad_logits[t*VS+v] = p - target;
                }
            }

            lm.backward(grad_logits);
            lm.update(lr);
        }
        const double avg_loss = total_loss / double(n_seqs * L);
        if (epoch == 1) first_loss = avg_loss;
        last_loss = avg_loss;
        if (epoch % 50 == 0 || epoch == 1 || epoch == epochs)
            std::printf("%d,%.6f\n", epoch, avg_loss);
    }

    const double log_vocab = std::log(double(VS));
    std::fprintf(stderr,
        "MiniLM copy: L=%zu, d=%zu, dff=%zu, V=%zu, blocks=%zu\n"
        "  baseline log(V): %.4f\n"
        "  first: %.4f\n"
        "  final: %.4f\n",
        L, DM, DFF, VS, NB, log_vocab, first_loss, last_loss);

    if (!std::isfinite(last_loss) || last_loss > 0.5 * log_vocab) {
        std::fprintf(stderr, "  RESULT: FAIL\n");
        return 1;
    }
    std::fprintf(stderr, "  RESULT: PASS\n");
    return 0;
}

}}
