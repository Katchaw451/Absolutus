#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <exception>

namespace absolutus {
namespace physics {
int run_chsh(int n_trials, uint64_t seed);
int run_ghz(size_t n_qubits, int n_seeds);
int run_vqe(int max_iter, double lr, uint64_t seed);
int run_bond_dim(uint64_t seed);
}
namespace ml {
int run_train_xor(int epochs, double lr, double momentum, uint64_t seed);
int run_train_lm(int epochs, double lr, uint64_t seed,
                 size_t n_heads, int qhead);
int run_train_text(int epochs, double lr, uint64_t seed,
                   const std::string& corpus_path, size_t n_merges,
                   size_t d_model, size_t d_ff, size_t n_blocks,
                   size_t seq_len, size_t n_sample);
}
namespace hybrid {
int run_train_qxor(int epochs, double lr, double momentum,
                   size_t n_params, uint64_t seed);
int run_train_hybrid(int epochs, double lr, double momentum,
                     size_t n_q, size_t q_params, uint64_t seed);
}
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --experiment <name> [options]\n"
        "  names:\n"
        "    chsh, ghz, vqe, train_xor, train_lm, train_qxor, train_hybrid\n"
        "  train_lm options:\n"
        "    [--epochs N] [--lr L] [--seed S] [--n-heads N] [--qhead H]\n"
        "  train_text options:\n"
        "    [--lm-epochs N] [--lr L] [--seed S] [--corpus PATH]\n"
        "    [--merges N] [--d-model N] [--d-ff N] [--blocks N]\n"
        "    [--seq-len N] [--samples N]\n",
        prog);
}

int main(int argc, char** argv) {
    std::string experiment;
    int n_trials = 10000;
    uint64_t seed = 42;
    size_t n_qubits = 10;
    int n_seeds = 1;
    int vqe_iters    = 200;
    int train_epochs = 3000;
    size_t qxor_params = 8;
    size_t hy_q = 2, hy_params = 4;
    size_t lm_heads = 1;
    int lm_qhead = -1;
    size_t lm_epochs = 200;
    size_t lm_merges = 128;
    size_t lm_dmodel = 16;
    size_t lm_dff = 32;
    size_t lm_blocks = 2;
    size_t lm_seq_len = 8;
    size_t lm_samples = 30;
    std::string lm_corpus = "data/shakespeare.txt";
    double lr = 0.1;
    double momentum = 0.9;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--experiment") && i + 1 < argc) experiment = argv[++i];
        else if (!std::strcmp(argv[i], "--trials") && i + 1 < argc) n_trials = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed")   && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--qubits") && i + 1 < argc) n_qubits = (size_t)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seeds")  && i + 1 < argc) n_seeds = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters")  && i + 1 < argc) vqe_iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--epochs") && i + 1 < argc) train_epochs = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--params") && i + 1 < argc) {
            if (experiment == "train_hybrid") hy_params = (size_t)std::atoi(argv[++i]);
            else qxor_params = (size_t)std::atoi(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "--n-heads") && i + 1 < argc) lm_heads = (size_t)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--qhead")   && i + 1 < argc) lm_qhead = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--lm-epochs") && i + 1 < argc) lm_epochs = (size_t)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--merges")    && i + 1 < argc) lm_merges = (size_t)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--d-model")   && i + 1 < argc) lm_dmodel = (size_t)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--d-ff")      && i + 1 < argc) lm_dff = (size_t)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--blocks")    && i + 1 < argc) lm_blocks = (size_t)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seq-len")   && i + 1 < argc) lm_seq_len = (size_t)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--samples")   && i + 1 < argc) lm_samples = (size_t)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--corpus")    && i + 1 < argc) lm_corpus = argv[++i];
        else if (!std::strcmp(argv[i], "--lr")       && i + 1 < argc) lr = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--momentum") && i + 1 < argc) momentum = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else { usage(argv[0]); return 2; }
    }

    if (experiment.empty()) { usage(argv[0]); return 2; }

    try {
        if (experiment == "chsh")         return absolutus::physics::run_chsh(n_trials, seed);
        if (experiment == "bond_dim")     return absolutus::physics::run_bond_dim(seed);
        if (experiment == "ghz")          return absolutus::physics::run_ghz(n_qubits, n_seeds);
        if (experiment == "vqe")          return absolutus::physics::run_vqe(vqe_iters, lr, seed);
        if (experiment == "train_xor")    return absolutus::ml::run_train_xor(train_epochs, lr, momentum, seed);
        if (experiment == "train_lm")     return absolutus::ml::run_train_lm(train_epochs, lr, seed, lm_heads, lm_qhead);
        if (experiment == "train_text")   return absolutus::ml::run_train_text(
            (int)lm_epochs, lr, seed, lm_corpus, lm_merges, lm_dmodel, lm_dff,
            lm_blocks, lm_seq_len, lm_samples);
        if (experiment == "train_qxor")   return absolutus::hybrid::run_train_qxor(train_epochs, lr, momentum, qxor_params, seed);
        if (experiment == "train_hybrid") {
            hy_q = n_qubits;
            return absolutus::hybrid::run_train_hybrid(train_epochs, lr, momentum, hy_q, hy_params, seed);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    std::fprintf(stderr, "unknown experiment: %s\n", experiment.c_str());
    return 2;
}
