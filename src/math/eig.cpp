#include "math/eig.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace absolutus {
namespace math {

void eig_sym_real(std::vector<double>& M, size_t n,
                  std::vector<double>& evals,
                  std::vector<double>& evecs) {
    evecs.assign(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) evecs[i * n + i] = 1.0;

    const double eps = 1e-13;
    const size_t max_sweeps = 100;

    for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        double off2 = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                off2 += M[i*n + j] * M[i*n + j];
        if (off2 < eps * eps) break;

        for (size_t p = 0; p + 1 < n; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                if (std::abs(M[p*n + q]) < eps) continue;
                const double app = M[p*n + p];
                const double aqq = M[q*n + q];
                const double apq = M[p*n + q];
                const double theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
                const double c = std::cos(theta);
                const double s = std::sin(theta);

                for (size_t k = 0; k < n; ++k) {
                    const double akp = M[k*n + p];
                    const double akq = M[k*n + q];
                    M[k*n + p] = c * akp - s * akq;
                    M[k*n + q] = s * akp + c * akq;
                }
                for (size_t k = 0; k < n; ++k) {
                    const double apk = M[p*n + k];
                    const double aqk = M[q*n + k];
                    M[p*n + k] = c * apk - s * aqk;
                    M[q*n + k] = s * apk + c * aqk;
                }
                for (size_t k = 0; k < n; ++k) {
                    const double vkp = evecs[k*n + p];
                    const double vkq = evecs[k*n + q];
                    evecs[k*n + p] = c * vkp - s * vkq;
                    evecs[k*n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    evals.resize(n);
    for (size_t i = 0; i < n; ++i) evals[i] = M[i*n + i];

    std::vector<size_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = i;
    std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
        return evals[a] < evals[b];
    });
    std::vector<double> ev(n);
    std::vector<double> evec_sorted(n * n);
    for (size_t i = 0; i < n; ++i) {
        ev[i] = evals[perm[i]];
        for (size_t k = 0; k < n; ++k)
            evec_sorted[k*n + i] = evecs[k*n + perm[i]];
    }
    evals = ev;
    evecs = evec_sorted;
}

void eig_hermitian(const std::vector<Complex<double>>& H, size_t n,
                   std::vector<double>& evals,
                   std::vector<Complex<double>>& evecs) {
    const size_t N = 2 * n;
    std::vector<double> Ht(N * N, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const auto h = H[i*n + j];
            Ht[i*N + j]         = h.re;
            Ht[i*N + j + n]     = -h.im;
            Ht[(i+n)*N + j]     = h.im;
            Ht[(i+n)*N + j + n] = h.re;
        }
    }

    std::vector<double> eval_t, evec_t;
    eig_sym_real(Ht, N, eval_t, evec_t);
    // eig_sym_real now sorts ascending, so eval_t pairs (2j, 2j+1)
    // are the two degenerate copies of the j-th eigenvalue of H.
    // Average them to absorb the (should-be-zero) split.

    evals.resize(n);
    for (size_t j = 0; j < n; ++j)
        evals[j] = 0.5 * (eval_t[2*j] + eval_t[2*j + 1]);

    evecs.assign(n * n, Complex<double>(0, 0));
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
            const double re = evec_t[i*N + 2*j];
            const double im = evec_t[(i+n)*N + 2*j];
            evecs[i*n + j] = Complex<double>(re, im);
        }
    }
}

}}
