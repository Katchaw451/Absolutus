#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <string>

namespace absolutus { namespace physics {
int run_chsh(int n_trials, uint64_t seed);
}}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --experiment <name> [options]\n"
        "  names:\n"
        "    chsh       [--trials N] [--seed S]\n", prog);
}

int main(int argc, char** argv) {
    std::string experiment;
    int n_trials = 10000;
    uint64_t seed = 42;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--experiment") && i + 1 < argc) {
            experiment = argv[++i];
        } else if (!std::strcmp(argv[i], "--trials") && i + 1 < argc) {
            n_trials = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (experiment.empty()) { usage(argv[0]); return 2; }

    if (experiment == "chsh")
        return absolutus::physics::run_chsh(n_trials, seed);

    std::fprintf(stderr, "unknown experiment: %s\n", experiment.c_str());
    return 2;
}
