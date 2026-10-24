#include "math/matrix.hpp"
#include <cstdio>
#include <cmath>

using namespace absolutus::math;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    using M = Matrix<double>;
    using C = Complex<double>;

    auto I = M::identity(2);
    CHECK(is_unitary(I));

    const double s = 1.0 / std::sqrt(2.0);
    M H(2, 2);
    H.at(0,0) = C(s);  H.at(0,1) = C(s);
    H.at(1,0) = C(s);  H.at(1,1) = C(-s);
    CHECK(is_unitary(H));

    auto H2 = H * H;
    CHECK(approx_eq(H2.at(0,0), C(1.0)));
    CHECK(approx_eq(H2.at(0,1), C(0.0)));

    M X(2, 2);
    X.at(0,1) = C(1.0);
    X.at(1,0) = C(1.0);
    CHECK(is_unitary(X));

    std::vector<C> ket0 = {C(1.0), C(0.0)};
    auto ket1 = X * ket0;
    CHECK(approx_eq(ket1[0], C(0.0)));
    CHECK(approx_eq(ket1[1], C(1.0)));

    auto XX = kron(X, X);
    CHECK(XX.rows == 4 && XX.cols == 4);
    CHECK(is_unitary(XX));

    auto XI = kron(X, I);
    std::vector<C> ket00 = {C(1.0), C(0.0), C(0.0), C(0.0)};
    auto out = XI * ket00;
    CHECK(approx_eq(out[2], C(1.0)));
    CHECK(approx_eq(out[0], C(0.0)));

    return 0;
}
