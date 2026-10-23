#pragma once

#include <cmath>

namespace absolutus {
namespace math {

// Minimal complex number, templated on real type.
// Layout is {re, im} — no vtable, no padding surprises.
// c32 = 8 bytes, c64 = 16 bytes. Cache-friendly.
template <typename R>
struct Complex {
    R re;
    R im;

    constexpr Complex()           : re(R(0)), im(R(0)) {}
    constexpr Complex(R r)        : re(r),    im(R(0)) {}
    constexpr Complex(R r, R i)   : re(r),    im(i)    {}

    constexpr Complex conj() const noexcept { return {re, -im}; }
    constexpr R norm_sq()    const noexcept { return re * re + im * im; }
    R norm() const noexcept { return std::sqrt(norm_sq()); }
    R arg()  const noexcept { return std::atan2(im, re); }
};

using c32 = Complex<float>;
using c64 = Complex<double>;

// --- arithmetic ---------------------------------------------------

template <typename R>
constexpr Complex<R> operator+(Complex<R> a, Complex<R> b) noexcept {
    return {a.re + b.re, a.im + b.im};
}

template <typename R>
constexpr Complex<R> operator-(Complex<R> a, Complex<R> b) noexcept {
    return {a.re - b.re, a.im - b.im};
}

template <typename R>
constexpr Complex<R> operator-(Complex<R> a) noexcept {
    return {-a.re, -a.im};
}

template <typename R>
constexpr Complex<R> operator*(Complex<R> a, Complex<R> b) noexcept {
    return {a.re * b.re - a.im * b.im,
            a.re * b.im + a.im * b.re};
}

template <typename R>
constexpr Complex<R> operator/(Complex<R> a, Complex<R> b) noexcept {
    R d = b.norm_sq();
    return {(a.re * b.re + a.im * b.im) / d,
            (a.im * b.re - a.re * b.im) / d};
}

// --- compound assignment ------------------------------------------

template <typename R>
constexpr Complex<R>& operator+=(Complex<R>& a, Complex<R> b) noexcept {
    a.re += b.re; a.im += b.im; return a;
}

template <typename R>
constexpr Complex<R>& operator-=(Complex<R>& a, Complex<R> b) noexcept {
    a.re -= b.re; a.im -= b.im; return a;
}

template <typename R>
constexpr Complex<R>& operator*=(Complex<R>& a, Complex<R> b) noexcept {
    R r = a.re * b.re - a.im * b.im;
    R i = a.re * b.im + a.im * b.re;
    a.re = r; a.im = i; return a;
}

template <typename R>
constexpr Complex<R>& operator/=(Complex<R>& a, Complex<R> b) noexcept {
    R d = b.norm_sq();
    R r = (a.re * b.re + a.im * b.im) / d;
    R i = (a.im * b.re - a.re * b.im) / d;
    a.re = r; a.im = i; return a;
}

// --- equality (exact) ---------------------------------------------

template <typename R>
constexpr bool operator==(Complex<R> a, Complex<R> b) noexcept {
    return a.re == b.re && a.im == b.im;
}

template <typename R>
constexpr bool operator!=(Complex<R> a, Complex<R> b) noexcept {
    return !(a == b);
}

// --- scalar helpers ----------------------------------------------

template <typename R>
constexpr Complex<R> operator*(R s, Complex<R> a) noexcept {
    return {s * a.re, s * a.im};
}

template <typename R>
constexpr Complex<R> operator*(Complex<R> a, R s) noexcept {
    return {a.re * s, a.im * s};
}

// --- approximate comparison (for tests) --------------------------

template <typename R>
inline bool approx_eq(Complex<R> a, Complex<R> b, R tol = R(1e-6)) noexcept {
    return std::abs(a.re - b.re) <= tol && std::abs(a.im - b.im) <= tol;
}

} // namespace math
} // namespace absolutus
