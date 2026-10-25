#include "physics/bond_dim.hpp"
#include "core/state.hpp"
#include "core/gates.hpp"
#include <cstdio>
#include <cmath>

using namespace absolutus;
using core::StateVector;

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while(0)

int main() {
    // Product state H^n: every cut has bond 1
    {
        StateVector<double> s(6);
        for (size_t q = 0; q < 6; ++q) core::apply1(s, core::H<double>(), q);
        auto p = physics::bond_profile(s, 1e-10);
        CHECK(p.size() == 7);
        for (size_t k = 0; k <= 6; ++k) CHECK(p[k] == 1);
    }

    // GHZ_6: bond 2 at every interior cut
    {
        StateVector<double> s(6);
        core::apply1(s, core::H<double>(), 0);
        for (size_t q = 1; q < 6; ++q) core::apply2(s, core::CNOT<double>(), 0, q);
        auto p = physics::bond_profile(s, 1e-10);
        CHECK(p[0] == 1);
        CHECK(p[6] == 1);
        for (size_t k = 1; k < 6; ++k) CHECK(p[k] == 2);
    }

    // Bell: bond 2 at the interior cut
    {
        StateVector<double> s(2);
        core::apply1(s, core::H<double>(), 0);
        core::apply2(s, core::CNOT<double>(), 0, 1);
        auto p = physics::bond_profile(s, 1e-10);
        CHECK(p[1] == 2);
    }

    // W_4: bond 2 at every interior cut (two-branch structure)
    {
        StateVector<double> s(4);
        const double a = 1.0 / std::sqrt(4.0);
        for (size_t q = 0; q < 4; ++q)
            s.amp(size_t(1) << q) = math::Complex<double>(a, 0.0);
        auto p = physics::bond_profile(s, 1e-10);
        CHECK(p[1] == 2);
        CHECK(p[2] == 2);
        CHECK(p[3] == 2);
    }

    std::printf("ok\n");
    return 0;
}
