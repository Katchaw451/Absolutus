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
    CHECK(math::is_unitary(core::H<double>()));
    CHECK(math::is_unitary(core::X<double>()));
    CHECK(math::is_unitary(core::Y<double>()));
    CHECK(math::is_unitary(core::Z<double>()));
    CHECK(math::is_unitary(core::S<double>()));
    CHECK(math::is_unitary(core::T<double>()));
    CHECK(math::is_unitary(core::RX<double>(0.7)));
    CHECK(math::is_unitary(core::RY<double>(-1.3)));
    CHECK(math::is_unitary(core::RZ<double>(2.1)));
    CHECK(math::is_unitary(core::CNOT<double>()));
    CHECK(math::is_unitary(core::CZ<double>()));
    CHECK(math::is_unitary(core::SWAP<double>()));

    { SV s(1); core::apply1(s, core::X<double>(), 0);
      CHECK(math::approx_eq(s.amp(1), C(1.0))); }

    { SV s(1); core::apply1(s, core::H<double>(), 0);
      const double r = 1.0/std::sqrt(2.0);
      CHECK(math::approx_eq(s.amp(0), C(r), 1e-12));
      CHECK(math::approx_eq(s.amp(1), C(r), 1e-12)); }

    { SV s(1); core::apply1(s, core::H<double>(), 0);
               core::apply1(s, core::H<double>(), 0);
      CHECK(math::approx_eq(s.amp(0), C(1.0), 1e-12)); }

    { SV s(2); core::apply1(s, core::H<double>(), 0);
               core::apply2(s, core::CNOT<double>(), 0, 1);
      const double r = 1.0/std::sqrt(2.0);
      CHECK(math::approx_eq(s.amp(0), C(r), 1e-12));
      CHECK(math::approx_eq(s.amp(3), C(r), 1e-12));
      CHECK(math::approx_eq(s.amp(1), C(0.0), 1e-12));
      CHECK(math::approx_eq(s.amp(2), C(0.0), 1e-12)); }

    // Anticommutation {X, Z} = 0: applying XZ then ZX gives zero net sum
    { SV a(1), b(1);
      core::apply1(a, core::Z<double>(), 0);
      core::apply1(a, core::X<double>(), 0);
      core::apply1(b, core::X<double>(), 0);
      core::apply1(b, core::Z<double>(), 0);
      for (size_t k = 0; k < 2; ++k) {
          C sum = a.amp(k);
          sum += b.amp(k);
          CHECK(math::approx_eq(sum, C(0.0), 1e-12));
      } }

    return 0;
}
