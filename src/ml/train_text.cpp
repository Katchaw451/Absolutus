#include "ml/train_text.hpp"
#include "ml/bpe.hpp"
#include "ml/minilm.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace absolutus {
namespace ml {

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Split text into words (same logic as BPE::train).
static std::vector<std::string> split_words(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Tokenize a raw string using the trained BPE.
static std::vector<size_t> tokenize(const BPE& bpe, const std::string& text,
                                    size_t vocab_size) {
    std::vector<size_t> ids;
    for (auto& w : split_words(text)) {
        auto toks = bpe.encode_word(w);
        for (int id : toks) {
            // Clamp unknown ids to unk (0..3 specials never appear in output)
            if (id < 0 || (size_t)id >= vocab_size) ids.push_back((size_t)bpe.id_unk);
            else ids.push_back((size_t)id);
        }
    }
    return ids;
}

int run_train_text(int epochs, double lr, uint64_t seed,
                   const std::string& corpus_path,
                   size_t n_merges,
                   size_t d_model, size_t d_ff, size_t n_blocks,
                   size_t seq_len, size_t n_sample) {
    const std::string corpus = read_file(corpus_path);
    if (corpus.size() < 1000) throw std::runtime_error("corpus too small");

    std::fprintf(stderr, "training BPE (%zu merges) on %zu bytes\n",
                 n_merges, corpus.size());
    BPE bpe = BPE::train(corpus, n_merges);
    const size_t vocab_size = bpe.vocab.size();
    std::fprintf(stderr, "  vocab: %zu tokens\n", vocab_size);

    auto stream = tokenize(bpe, corpus, vocab_size);
    std::fprintf(stderr, "  corpus: %zu tokens\n", stream.size());

    if (stream.size() < seq_len * 2) throw std::runtime_error("corpus too short after tokenization");

    // Build windows: input = stream[i..i+seq_len), target = stream[i+1..i+seq_len+1)
    const size_t n_windows = (stream.size() - seq_len - 1) / seq_len;
    std::fprintf(stderr, "  windows: %zu (seq_len=%zu)\n", n_windows, seq_len);

    std::mt19937_64 rng(seed);
    MiniLM lm(seq_len, d_model, d_ff, vocab_size, n_blocks, rng, 0.9, 1, -1, 1, true);
    const double CLIP = 1.0;

    std::printf("epoch,loss\n");
    double first = 0.0, last = 0.0;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total = 0.0;
        size_t n_batches = 0;
        const size_t windows_per_epoch = std::min<size_t>(n_windows, 500);
        const size_t epoch_start = ((size_t)epoch * windows_per_epoch) % n_windows;
        for (size_t w = 0; w < windows_per_epoch; ++w) {
            const size_t off = ((epoch_start + w) % n_windows) * seq_len;
            std::vector<size_t> tokens(seq_len);
            std::vector<size_t> targets(seq_len);
            for (size_t t = 0; t < seq_len; ++t) {
                tokens[t]  = stream[off + t];
                targets[t] = stream[off + t + 1];
            }
            std::vector<double> logits;
            try {
                logits = lm.forward(tokens);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "forward fail at epoch %d window %zu: %s\n",
                             epoch, w, e.what());
                return 1;
            }
            std::vector<double> grad_logits(seq_len * vocab_size, 0.0);
            for (size_t t = 0; t < seq_len; ++t) {
                double m = -1e30;
                for (size_t v = 0; v < vocab_size; ++v)
                    if (logits[t*vocab_size+v] > m) m = logits[t*vocab_size+v];
                double sum = 0.0;
                for (size_t v = 0; v < vocab_size; ++v)
                    sum += std::exp(logits[t*vocab_size+v] - m);
                for (size_t v = 0; v < vocab_size; ++v) {
                    double p = std::exp(logits[t*vocab_size+v] - m) / sum;
                    const double tgt = (v == targets[t]) ? 1.0 : 0.0;
                    if (tgt > 0.0) total -= std::log(p + 1e-12);
                    grad_logits[t*vocab_size+v] = p - tgt;
                }
            }
            lm.backward(grad_logits);
            lm.clip_gradients(CLIP);
            lm.update(lr);
            ++n_batches;
        }
        const double avg = total / double(n_batches * seq_len);
        if (epoch == 1) first = avg;
        last = avg;
        if (epoch % 5 == 0 || epoch == 1 || epoch == epochs)
            std::printf("%d,%.6f\n", epoch, avg); std::fflush(stdout);
    }

    const double baseline = std::log(double(vocab_size));
    std::fprintf(stderr,
        "MiniLM on text: V=%zu d=%zu dff=%zu blocks=%zu L=%zu windows=%zu\n"
        "  baseline log(V): %.4f  first: %.4f  final: %.4f\n",
        vocab_size, d_model, d_ff, n_blocks, seq_len, n_windows,
        baseline, first, last);

    // Greedy generation from the first window's prefix
    if (n_sample > 0) {
        std::vector<size_t> prompt(stream.begin(), stream.begin() + std::min(seq_len, stream.size()));
        auto ids = prompt;
        std::fprintf(stderr, "\n--- generated %zu tokens ---\n", n_sample);
        std::fprintf(stderr, "prompt: '%s'\n", bpe.decode(std::vector<int>(ids.begin(), ids.end())).c_str());
        std::vector<size_t> gen;
        for (size_t step = 0; step < n_sample; ++step) {
            std::vector<size_t> win(seq_len);
            size_t start = ids.size() > seq_len ? ids.size() - seq_len : 0;
            for (size_t t = 0; t < seq_len; ++t) {
                win[t] = start + t < ids.size() ? ids[start + t] : (size_t)bpe.id_pad;
            }
            auto logits = lm.forward(win);
            size_t last_pos = ids.size() > seq_len ? seq_len - 1 : ids.size() - 1;

            std::vector<double> work(vocab_size);
            for (size_t v = 0; v < vocab_size; ++v)
                work[v] = logits[last_pos * vocab_size + v];

            const double REP_PENALTY = 1.3;
            const size_t REP_WINDOW = 20;
            const size_t rep_start = gen.size() > REP_WINDOW ? gen.size() - REP_WINDOW : 0;
            for (size_t g = rep_start; g < gen.size(); ++g) {
                const size_t id = gen[g];
                if (work[id] > 0) work[id] /= REP_PENALTY;
                else work[id] *= REP_PENALTY;
            }

            const size_t TOP_K = 40;
            const size_t kk = std::min(TOP_K, vocab_size);
            std::vector<double> sorted = work;
            std::partial_sort(sorted.begin(), sorted.begin() + kk, sorted.end(),
                              std::greater<double>());
            const double threshold = sorted[kk - 1];

            const double T = 0.7;
            double mx = -1e30;
            for (size_t v = 0; v < vocab_size; ++v)
                if (work[v] >= threshold && work[v] > mx) mx = work[v];
            std::vector<double> probs(vocab_size, 0.0);
            double zsum = 0.0;
            for (size_t v = 0; v < vocab_size; ++v) {
                if (work[v] >= threshold) {
                    probs[v] = std::exp((work[v] - mx) / T);
                    zsum += probs[v];
                }
            }
            for (size_t v = 0; v < vocab_size; ++v) probs[v] /= zsum;
            std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
            size_t best = dist(rng);
            ids.push_back(best);
            gen.push_back(best);
        }
        std::vector<int> gen_ids(gen.begin(), gen.end());
        std::fprintf(stderr, "output:  '%s'\n", bpe.decode(gen_ids).c_str());
    }

    if (!std::isfinite(last) || last > 0.9 * baseline) {
        std::fprintf(stderr, "  RESULT: FAIL (loss %.4f vs baseline %.4f)\n", last, baseline);
        return 1;
    }
    std::fprintf(stderr, "  RESULT: PASS\n");
    return 0;
}

}}
