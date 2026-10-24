#include "core/measure.hpp"
#include "core/gates.hpp"
#include <cstdio>
#include <cmath>

using namespace absolutus;
using C  = math::Complex<double>;
using SV = core::StateVector<double>;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    { SV s(2);
      auto rho = core::partial_trace(s, {0});
      CHECK(std::abs(core::entropy_1q(rho) - 0.0) < 1e-9); }

    // Superposition is NOT entanglement
    { SV s(2);
      core::apply1(s, core::H<double>(), 0);
      CHECK(std::abs(core::entropy_1q(core::partial_trace(s, {0})) - 0.0) < 1e-9);
      CHECK(std::abs(core::entropy_1q(core::partial_trace(s, {1})) - 0.0) < 1e-9); }

    // Bell has entropy 1 on each qubit
    { SV s(2);
      core::apply1(s, core::H<double>(), 0);
      core::apply2(s, core::CNOT<double>(), 0, 1);
      CHECK(std::abs(core::entropy_1q(core::partial_trace(s, {0})) - 1.0) < 1e-9);
      CHECK(std::abs(core::entropy_1q(core::partial_trace(s, {1})) - 1.0) < 1e-9); }

    { SV s(1);
      CHECK(std::abs(core::expectation(s, core::Z<double>()) - 1.0) < 1e-12); }

    { SV s(1);
      core::apply1(s, core::H<double>(), 0);
      CHECK(std::abs(core::expectation(s, core::X<double>()) - 1.0) < 1e-12);
      CHECK(std::abs(core::expectation(s, core::Z<double>()) - 0.0) < 1e-12); }

    { SV s(2);
      core::apply1(s, core::H<double>(), 0);
      core::apply2(s, core::CNOT<double>(), 0, 1);
      auto ZZ = core::expectation_local(s, {core::Z<double>(), core::Z<double>()}, {0, 1});
      auto XX = core::expectation_local(s, {core::X<double>(), core::X<double>()}, {0, 1});
      auto ZX = core::expectation_local(s, {core::Z<double>(), core::X<double>()}, {0, 1});
      CHECK(std::abs(ZZ - 1.0) < 1e-12);
      CHECK(std::abs(XX - 1.0) < 1e-12);
      CHECK(std::abs(ZX - 0.0) < 1e-12); }

    { SV a(1), b(1);
      core::apply1(b, core::X<double>(), 0);
      CHECK(std::abs(core::fidelity(a, a) - 1.0) < 1e-12);
      CHECK(std::abs(core::fidelity(a, b) - 0.0) < 1e-12); }

    return 0;
}
