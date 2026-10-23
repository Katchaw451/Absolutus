#pragma once

#include "math/complex.hpp"
#include <vector>
#include <random>
#include <stdexcept>
#include <cmath>

namespace absolutus {
namespace core {

using math::Complex;

// Pure quantum state: vector of 2^n complex amplitudes.
// Convention: index i encodes bitstring with qubit 0 as LSB.
// Invariant: norm() == 1 for any physically valid state.
template <typename R>
class StateVector {
public:
    StateVector() = default;

    // |0...0> with n_qubits qubits
    explicit StateVector(size_t n_qubits)
        : n_qubits_(n_qubits),
          amps_(size_t(1) << n_qubits, Complex<R>(R(0))) {
        if (n_qubits > 30)
            throw std::invalid_argument("state: more than 30 qubits");
        amps_[0] = Complex<R>(R(1));
    }

    // From explicit amplitudes; dimension must be a power of two.
    StateVector(std::initializer_list<Complex<R>> init) : amps_(init) {
        const size_t d = amps_.size();
        if (d == 0 || (d & (d - 1)) != 0)
            throw std::invalid_argument("state: dim must be power of 2");
        n_qubits_ = 0;
        while ((size_t(1) << n_qubits_) < d) ++n_qubits_;
    }

    size_t n_qubits() const { return n_qubits_; }
    size_t dim()      const { return amps_.size(); }

    Complex<R>&       amp(size_t i)       { return amps_[i]; }
    const Complex<R>& amp(size_t i) const { return amps_[i]; }

    std::vector<Complex<R>>&       data()       { return amps_; }
    const std::vector<Complex<R>>& data() const { return amps_; }

    // --- norm -----------------------------------------------------

    R norm_sq() const {
        R s = R(0);
        for (const auto& a : amps_) s += a.norm_sq();
        return s;
    }
    R norm() const { return std::sqrt(norm_sq()); }

    void normalize() {
        const R n = norm();
        if (n == R(0))
            throw std::runtime_error("state: cannot normalize null vector");
        const R inv = R(1) / n;
        for (auto& a : amps_) { a.re *= inv; a.im *= inv; }
    }

    // --- inner product <this|other> -------------------------------

    Complex<R> inner(const StateVector& other) const {
        if (amps_.size() != other.amps_.size())
            throw std::invalid_argument("state: inner dim mismatch");
        Complex<R> s;
        for (size_t i = 0; i < amps_.size(); ++i)
            s += amps_[i].conj() * other.amps_[i];
        return s;
    }

    // --- measurement ---------------------------------------------

    std::vector<R> probabilities() const {
        std::vector<R> p(amps_.size());
        for (size_t i = 0; i < amps_.size(); ++i)
            p[i] = amps_[i].norm_sq();
        return p;
    }

    // Sample the measurement outcome. Returns bitstring index.
    size_t sample(std::mt19937_64& rng) const {
        const auto p = probabilities();
        std::discrete_distribution<size_t> dist(p.begin(), p.end());
        return dist(rng);
    }

    // Project onto outcome and renormalize.
    void collapse(size_t outcome) {
        if (outcome >= amps_.size())
            throw std::out_of_range("state: collapse out of range");
        for (size_t i = 0; i < amps_.size(); ++i)
            if (i != outcome) amps_[i] = Complex<R>(R(0));
        normalize();
    }

    // --- tensor product |this> ⊗ |other> -------------------------

    StateVector tensor(const StateVector& other) const {
        StateVector out;
        out.n_qubits_ = n_qubits_ + other.n_qubits_;
        const size_t d1 = amps_.size();
        const size_t d2 = other.amps_.size();
        out.amps_.resize(d1 * d2);
        for (size_t i = 0; i < d1; ++i)
            for (size_t j = 0; j < d2; ++j)
                out.amps_[i * d2 + j] = amps_[i] * other.amps_[j];
        return out;
    }

private:
    size_t n_qubits_ = 0;
    std::vector<Complex<R>> amps_;
};

using sv32 = StateVector<float>;
using sv64 = StateVector<double>;

} // namespace core
} // namespace absolutus
