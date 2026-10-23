#pragma once

#include "math/complex.hpp"
#include <vector>
#include <stdexcept>

namespace absolutus {
namespace math {

// Row-major dense matrix. Sized dynamically because gate matrices
// are small (2x2, 4x4) but tensor products grow as 2^n.
template <typename R>
class Matrix {
public:
    size_t rows = 0;
    size_t cols = 0;
    std::vector<Complex<R>> data;

    Matrix() = default;
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c) {}
    Matrix(size_t r, size_t c, Complex<R> fill)
        : rows(r), cols(c), data(r * c, fill) {}

    Complex<R>&       at(size_t i, size_t j)       { return data[i * cols + j]; }
    const Complex<R>& at(size_t i, size_t j) const { return data[i * cols + j]; }

    static Matrix identity(size_t n) {
        Matrix m(n, n);
        for (size_t i = 0; i < n; ++i) m.at(i, i) = Complex<R>(R(1));
        return m;
    }

    Matrix conj_transpose() const {
        Matrix out(cols, rows);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                out.at(j, i) = at(i, j).conj();
        return out;
    }
};

// --- matrix * matrix (skip zero entries for sparse gates) ---------

template <typename R>
Matrix<R> operator*(const Matrix<R>& a, const Matrix<R>& b) {
    if (a.cols != b.rows)
        throw std::invalid_argument("matrix dim mismatch");
    Matrix<R> out(a.rows, b.cols);
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t k = 0; k < a.cols; ++k) {
            const auto aik = a.at(i, k);
            if (aik.re == R(0) && aik.im == R(0)) continue;
            for (size_t j = 0; j < b.cols; ++j)
                out.at(i, j) += aik * b.at(k, j);
        }
    return out;
}

// --- matrix * vector ---------------------------------------------

template <typename R>
std::vector<Complex<R>> operator*(const Matrix<R>& a,
                                  const std::vector<Complex<R>>& v) {
    if (a.cols != v.size())
        throw std::invalid_argument("matrix-vector dim mismatch");
    std::vector<Complex<R>> out(a.rows);
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            out[i] += a.at(i, j) * v[j];
    return out;
}

// --- Kronecker product A ⊗ B -------------------------------------
// This is how single-qubit gates become multi-qubit operators.

template <typename R>
Matrix<R> kron(const Matrix<R>& a, const Matrix<R>& b) {
    Matrix<R> out(a.rows * b.rows, a.cols * b.cols);
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j) {
            const auto aij = a.at(i, j);
            for (size_t k = 0; k < b.rows; ++k)
                for (size_t l = 0; l < b.cols; ++l)
                    out.at(i * b.rows + k, j * b.cols + l) = aij * b.at(k, l);
        }
    return out;
}

// --- unitarity check: U†U = I ------------------------------------

template <typename R>
bool is_unitary(const Matrix<R>& m, R tol = R(1e-6)) {
    if (m.rows != m.cols) return false;
    const auto p = m * m.conj_transpose();
    for (size_t i = 0; i < m.rows; ++i)
        for (size_t j = 0; j < m.rows; ++j) {
            const Complex<R> expected = (i == j) ? Complex<R>(R(1))
                                                 : Complex<R>(R(0));
            if (!approx_eq(p.at(i, j), expected, tol)) return false;
        }
    return true;
}

} // namespace math
} // namespace absolutus
