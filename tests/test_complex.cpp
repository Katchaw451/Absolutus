#include "math/complex.hpp"
#include <cstdio>
#include <cmath>

using namespace absolutus::math;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    c64 a(3.0, 4.0);
    CHECK(std::abs(a.norm() - 5.0) < 1e-12);
    CHECK(std::abs(a.norm_sq() - 25.0) < 1e-12);

    c64 b(1.0, -2.0);
    CHECK(approx_eq(a * b, c64(11.0, -2.0)));

    // (3+4i)/(1-2i) = (3+4i)(1+2i)/5 = (-5+10i)/5 = -1 + 2i
    CHECK(approx_eq(a / b, c64(-1.0, 2.0), 1e-12));

    // Division undoes multiplication (round-trip)
    CHECK(approx_eq((a / b) * b, a, 1e-12));

    c64 i(0.0, 1.0);
    CHECK(approx_eq(i * i, c64(-1.0, 0.0)));
    CHECK(approx_eq(a + a.conj(), c64(6.0, 0.0)));
    CHECK(approx_eq(a * a.conj(), c64(25.0, 0.0)));

    c64 z;
    z += a; z -= a;
    CHECK(approx_eq(z, c64(0.0, 0.0)));
    CHECK(approx_eq(2.0 * a, c64(6.0, 8.0)));
    CHECK(approx_eq(a * 2.0, c64(6.0, 8.0)));

    CHECK(approx_eq(c64(-1.0, 0.0), c64(-1.0, 0.0)));
    CHECK(c64(1.0, 2.0) != c64(1.0, 3.0));

    return 0;
}
