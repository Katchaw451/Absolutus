#include "ml/minilm.hpp"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>

namespace absolutus {
namespace ml {

int run_train_lm(int epochs, double lr, uint64_t seed,
                 size_t n_heads, int qhead) {
    const size_t L = 4, DM = 8, DFF = 16, VS = 8, NB = 2;
    const double CLIP = 1.0;

    std::mt19937_64 rng(seed);
    MiniLM lm(L, DM, DFF, VS, NB, rng, 0.9, n_heads, qhead);

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
            std::vector<double> logits;
            try {
                logits = lm.forward(tokens);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "epoch %d seq %zu: %s\n", epoch, si, e.what());
                return 1;
            }

            std::vector<double> grad_logits(L * VS, 0.0);
            for (size_t t = 0; t < L; ++t) {
                double m = -1e30;
                for (size_t v = 0; v < VS; ++v)
                    if (logits[t*VS+v] > m) m = logits[t*VS+v];
                double sum = 0.0;
                for (size_t v = 0; v < VS; ++v) sum += std::exp(logits[t*VS+v] - m);
                for (size_t v = 0; v < VS; ++v) {
                    double p = std::exp(logits[t*VS+v] - m) / sum;
                    const double tgt = (v == tokens[t]) ? 1.0 : 0.0;
                    if (tgt > 0.0) total_loss -= std::log(p + 1e-12);
                    grad_logits[t*VS+v] = p - tgt;
                }
            }
            lm.backward(grad_logits);
            lm.clip_gradients(CLIP);
            lm.update(lr);
        }
        const double avg = total_loss / double(n_seqs * L);
        if (epoch == 1) first_loss = avg;
        last_loss = avg;
        if (epoch % 50 == 0 || epoch == 1 || epoch == epochs)
            std::printf("%d,%.6f\n", epoch, avg);
    }

    const double baseline = std::log(double(VS));
    std::fprintf(stderr,
        "MiniLM copy: L=%zu d=%zu dff=%zu V=%zu blocks=%zu\n"
        "  n_heads=%zu  quantum_head_idx=%d  lr=%.4f  clip=%.2f\n"
        "  baseline: %.4f  first: %.4f  final: %.4f\n",
        L, DM, DFF, VS, NB, n_heads, qhead, lr, CLIP,
        baseline, first_loss, last_loss);

    if (!std::isfinite(last_loss) || last_loss > 0.5 * baseline) {
        std::fprintf(stderr, "  RESULT: FAIL\n");
        return 1;
    }
    std::fprintf(stderr, "  RESULT: PASS\n");
    return 0;
}

}}
