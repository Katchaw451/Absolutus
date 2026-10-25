#include "math/eig.hpp"
#include <cstdio>
#include <cmath>

using namespace absolutus;
using math::eig_hermitian;
using math::eig_sym_real;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    // Test 1: real symmetric 2x2, known eigenvalues (-1, 3)
    {
        std::vector<double> M = {1, 2, 2, 1};
        std::vector<double> evals, evecs;
        eig_sym_real(M, 2, evals, evecs);
        CHECK(std::abs(evals[0] - (-1.0)) < 1e-9);
        CHECK(std::abs(evals[1] -   3.0)  < 1e-9);
    }

    // Test 2: complex Hermitian 2x2, known eigenvalues (0, 3)
    {
        std::vector<math::Complex<double>> H = {
            math::Complex<double>(1, 0), math::Complex<double>(1, 1),
            math::Complex<double>(1, -1), math::Complex<double>(2, 0)
        };
        std::vector<double> evals;
        std::vector<math::Complex<double>> evecs;
        eig_hermitian(H, 2, evals, evecs);
        CHECK(std::abs(evals[0]) < 1e-9);
        CHECK(std::abs(evals[1] - 3.0) < 1e-9);

        for (size_t k = 0; k < 2; ++k) {
            math::Complex<double> hv0 = H[0]*evecs[0*2+k] + H[1]*evecs[1*2+k];
            math::Complex<double> hv1 = H[2]*evecs[0*2+k] + H[3]*evecs[1*2+k];
            math::Complex<double> lv0 = evecs[0*2+k] * math::Complex<double>(evals[k], 0);
            math::Complex<double> lv1 = evecs[1*2+k] * math::Complex<double>(evals[k], 0);
            CHECK(std::abs((hv0 - lv0).re) < 1e-9);
            CHECK(std::abs((hv0 - lv0).im) < 1e-9);
            CHECK(std::abs((hv1 - lv1).re) < 1e-9);
            CHECK(std::abs((hv1 - lv1).im) < 1e-9);
        }
    }

    // Test 3: Z (x) Z, diagonal(1, -1, -1, 1): eigenvalues (-1, -1, 1, 1)
    {
        std::vector<math::Complex<double>> H(16, math::Complex<double>(0, 0));
        H[0]  = math::Complex<double>( 1, 0);
        H[5]  = math::Complex<double>(-1, 0);
        H[10] = math::Complex<double>(-1, 0);
        H[15] = math::Complex<double>( 1, 0);
        std::vector<double> evals;
        std::vector<math::Complex<double>> evecs;
        eig_hermitian(H, 4, evals, evecs);
        CHECK(std::abs(evals[0] - (-1.0)) < 1e-9);
        CHECK(std::abs(evals[1] - (-1.0)) < 1e-9);
        CHECK(std::abs(evals[2] -   1.0)  < 1e-9);
        CHECK(std::abs(evals[3] -   1.0)  < 1e-9);
    }

    // Test 4: Bell state reduced rho = I/2, eigenvalues (0.5, 0.5)
    {
        std::vector<math::Complex<double>> rho = {
            math::Complex<double>(0.5, 0), math::Complex<double>(0, 0),
            math::Complex<double>(0, 0),   math::Complex<double>(0.5, 0)
        };
        std::vector<double> evals;
        std::vector<math::Complex<double>> evecs;
        eig_hermitian(rho, 2, evals, evecs);
        CHECK(std::abs(evals[0] - 0.5) < 1e-9);
        CHECK(std::abs(evals[1] - 0.5) < 1e-9);
    }

    std::printf("ok\n");
    return 0;
}
