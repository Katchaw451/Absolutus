#include "ml/posenc.hpp"
#include <cmath>
#include <stdexcept>

namespace absolutus {
namespace ml {

PositionEncoding::PositionEncoding(size_t sl, size_t dm)
    : seq_len(sl), d_model(dm), table(sl * dm, 0.0) {
    if (dm % 2 != 0)
        throw std::invalid_argument("PositionEncoding: d_model must be even");
    for (size_t pos = 0; pos < sl; ++pos) {
        for (size_t i = 0; i < dm / 2; ++i) {
            const double denom = std::pow(10000.0, 2.0 * double(i) / double(dm));
            const double angle = double(pos) / denom;
            table[pos * dm + 2 * i]     = std::sin(angle);
            table[pos * dm + 2 * i + 1] = std::cos(angle);
        }
    }
}

std::vector<double> PositionEncoding::add(const std::vector<double>& x) const {
    if (x.size() != seq_len * d_model)
        throw std::invalid_argument("PositionEncoding::add: x shape");
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] + table[i];
    return y;
}

}}
