#include "core/state.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using namespace absolutus;
using C  = math::Complex<double>;
using SV = core::StateVector<double>;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    SV zero(1);
    CHECK(zero.n_qubits() == 1);
    CHECK(std::abs(zero.norm_sq() - 1.0) < 1e-12);
    CHECK(math::approx_eq(zero.amp(0), C(1.0)));
    CHECK(math::approx_eq(zero.amp(1), C(0.0)));

    const double s = 1.0 / std::sqrt(2.0);
    SV bell({C(s), C(0), C(0), C(s)});
    CHECK(bell.n_qubits() == 2);
    CHECK(std::abs(bell.norm_sq() - 1.0) < 1e-12);

    auto p = bell.probabilities();
    CHECK(std::abs(p[0] - 0.5) < 1e-12);
    CHECK(std::abs(p[3] - 0.5) < 1e-12);
    CHECK(std::abs(p[1]) < 1e-12);
    CHECK(std::abs(p[2]) < 1e-12);

    std::mt19937_64 rng(42);
    int c0 = 0, c3 = 0;
    for (int i = 0; i < 10000; ++i) {
        auto out = bell.sample(rng);
        if (out == 0) ++c0;
        if (out == 3) ++c3;
    }
    CHECK(std::abs(c0 - 5000) < 200);
    CHECK(std::abs(c3 - 5000) < 200);

    SV a(1), b(1);
    auto prod = a.tensor(b);
    CHECK(prod.n_qubits() == 2);
    CHECK(math::approx_eq(prod.amp(0), C(1.0)));

    auto ip = bell.inner(bell);
    CHECK(std::abs(ip.re - 1.0) < 1e-12);
    CHECK(std::abs(ip.im) < 1e-12);

    SV collapsed = bell;
    collapsed.collapse(0);
    CHECK(math::approx_eq(collapsed.amp(0), C(1.0)));
    CHECK(math::approx_eq(collapsed.amp(3), C(0.0)));

    return 0;
}
