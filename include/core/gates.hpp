#pragma once

#include "core/state.hpp"
#include "math/matrix.hpp"
#include <cmath>
#include <stdexcept>

namespace absolutus {
namespace core {

using math::Complex;
using math::Matrix;

// --- single-qubit gates ------------------------------------------
// Return 2x2 unitaries. Qubit 0 is the LSB in state indexing.

template <typename R>
Matrix<R> I1() {
    auto m = Matrix<R>::identity(2);
    return m;
}

template <typename R>
Matrix<R> X() {
    Matrix<R> m(2, 2);
    m.at(0, 1) = Complex<R>(R(1));
    m.at(1, 0) = Complex<R>(R(1));
    return m;
}

template <typename R>
Matrix<R> Y() {
    Matrix<R> m(2, 2);
    m.at(0, 1) = Complex<R>(R(0), R(-1));
    m.at(1, 0) = Complex<R>(R(0),  R(1));
    return m;
}

template <typename R>
Matrix<R> Z() {
    Matrix<R> m(2, 2);
    m.at(0, 0) = Complex<R>(R(1));
    m.at(1, 1) = Complex<R>(R(-1));
    return m;
}

template <typename R>
Matrix<R> H() {
    const R s = R(1) / std::sqrt(R(2));
    Matrix<R> m(2, 2);
    m.at(0, 0) = Complex<R>(s);  m.at(0, 1) = Complex<R>(s);
    m.at(1, 0) = Complex<R>(s);  m.at(1, 1) = Complex<R>(-s);
    return m;
}

template <typename R>
Matrix<R> S() {
    Matrix<R> m(2, 2);
    m.at(0, 0) = Complex<R>(R(1));
    m.at(1, 1) = Complex<R>(R(0), R(1));
    return m;
}

template <typename R>
Matrix<R> T() {
    const R inv_sqrt2 = R(1) / std::sqrt(R(2));
    Matrix<R> m(2, 2);
    m.at(0, 0) = Complex<R>(R(1));
    m.at(1, 1) = Complex<R>(inv_sqrt2, inv_sqrt2);
    return m;
}

// Rotation gates: parameter theta in radians.
template <typename R>
Matrix<R> RX(R theta) {
    const R c = std::cos(theta / R(2));
    const R s = std::sin(theta / R(2));
    Matrix<R> m(2, 2);
    m.at(0, 0) = Complex<R>(c);
    m.at(0, 1) = Complex<R>(R(0), -s);
    m.at(1, 0) = Complex<R>(R(0), -s);
    m.at(1, 1) = Complex<R>(c);
    return m;
}

template <typename R>
Matrix<R> RY(R theta) {
    const R c = std::cos(theta / R(2));
    const R s = std::sin(theta / R(2));
    Matrix<R> m(2, 2);
    m.at(0, 0) = Complex<R>( c);
    m.at(0, 1) = Complex<R>(-s);
    m.at(1, 0) = Complex<R>( s);
    m.at(1, 1) = Complex<R>( c);
    return m;
}

template <typename R>
Matrix<R> RZ(R theta) {
    const R c = std::cos(theta / R(2));
    const R s = std::sin(theta / R(2));
    Matrix<R> m(2, 2);
    m.at(0, 0) = Complex<R>(c, -s);
    m.at(1, 1) = Complex<R>(c,  s);
    return m;
}

// --- two-qubit gates ---------------------------------------------
// Indexing: 4x4 matrix, index = 2*b_first + b_second.
// For CNOT(control, target), row/col order is
//   0: |c=0,t=0>  1: |c=0,t=1>  2: |c=1,t=0>  3: |c=1,t=1>

template <typename R>
Matrix<R> CNOT() {
    Matrix<R> m(4, 4);
    m.at(0, 0) = Complex<R>(R(1));
    m.at(1, 1) = Complex<R>(R(1));
    m.at(2, 3) = Complex<R>(R(1));
    m.at(3, 2) = Complex<R>(R(1));
    return m;
}

template <typename R>
Matrix<R> CZ() {
    Matrix<R> m = Matrix<R>::identity(4);
    m.at(3, 3) = Complex<R>(R(-1));
    return m;
}

template <typename R>
Matrix<R> SWAP() {
    Matrix<R> m(4, 4);
    m.at(0, 0) = Complex<R>(R(1));
    m.at(1, 2) = Complex<R>(R(1));
    m.at(2, 1) = Complex<R>(R(1));
    m.at(3, 3) = Complex<R>(R(1));
    return m;
}

// --- apply single-qubit gate to `target` qubit -------------------
// Cache-friendly: the inner loop walks contiguous pairs (i, i+mask),
// and the outer loop jumps by 2*mask, so each 2*mask block of the
// amplitude vector is touched exactly once, in order.

template <typename R>
void apply1(StateVector<R>& psi, const Matrix<R>& g, size_t target) {
    if (g.rows != 2 || g.cols != 2)
        throw std::invalid_argument("apply1: gate must be 2x2");
    if (target >= psi.n_qubits())
        throw std::out_of_range("apply1: target out of range");

    auto& amps = psi.data();
    const size_t mask  = size_t(1) << target;
    const size_t block = mask << 1;
    const size_t dim   = amps.size();

    for (size_t base = 0; base < dim; base += block) {
        for (size_t off = 0; off < mask; ++off) {
            const size_t i = base + off;
            const size_t j = i + mask;
            const Complex<R> a = amps[i];
            const Complex<R> b = amps[j];
            amps[i] = g.at(0, 0) * a + g.at(0, 1) * b;
            amps[j] = g.at(1, 0) * a + g.at(1, 1) * b;
        }
    }
}

// --- apply two-qubit gate to (q_first, q_second) -----------------
// The 4x4 gate's index order is 2*b_first + b_second.
// q_first and q_second must be distinct.

template <typename R>
void apply2(StateVector<R>& psi, const Matrix<R>& g,
            size_t q_first, size_t q_second) {
    if (g.rows != 4 || g.cols != 4)
        throw std::invalid_argument("apply2: gate must be 4x4");
    if (q_first == q_second)
        throw std::invalid_argument("apply2: qubits must differ");
    if (q_first  >= psi.n_qubits() ||
        q_second >= psi.n_qubits())
        throw std::out_of_range("apply2: qubit out of range");

    auto& amps = psi.data();
    const size_t m1 = size_t(1) << q_first;
    const size_t m2 = size_t(1) << q_second;
    const size_t dim = amps.size();

    for (size_t i = 0; i < dim; ++i) {
        if ((i & m1) || (i & m2)) continue;
        const size_t i00 = i;
        const size_t i01 = i | m2;       // second qubit = 1
        const size_t i10 = i | m1;       // first  qubit = 1
        const size_t i11 = i | m1 | m2;

        const Complex<R> a00 = amps[i00];
        const Complex<R> a01 = amps[i01];
        const Complex<R> a10 = amps[i10];
        const Complex<R> a11 = amps[i11];

        amps[i00] = g.at(0,0)*a00 + g.at(0,1)*a01 + g.at(0,2)*a10 + g.at(0,3)*a11;
        amps[i01] = g.at(1,0)*a00 + g.at(1,1)*a01 + g.at(1,2)*a10 + g.at(1,3)*a11;
        amps[i10] = g.at(2,0)*a00 + g.at(2,1)*a01 + g.at(2,2)*a10 + g.at(2,3)*a11;
        amps[i11] = g.at(3,0)*a00 + g.at(3,1)*a01 + g.at(3,2)*a10 + g.at(3,3)*a11;
    }
}

} // namespace core
} // namespace absolutus
