#pragma once
#include "math/complex.hpp"
#include <vector>

namespace absolutus {
namespace math {

// Jacobi eigendecomposition for real symmetric matrix (row-major, n x n).
// Destroys the input M. Outputs eigenvalues sorted ascending, and
// eigenvectors as columns of evecs (n x n row-major).
void eig_sym_real(std::vector<double>& M, size_t n,
                  std::vector<double>& evals,
                  std::vector<double>& evecs);

// Eigendecomposition for complex Hermitian matrix.
// Implementation: doubled-real trick (H_r + i H_i -> real symmetric),
// classical Jacobi on the doubled matrix, then reconstruct.
void eig_hermitian(const std::vector<Complex<double>>& H, size_t n,
                   std::vector<double>& evals,
                   std::vector<Complex<double>>& evecs);

}}
