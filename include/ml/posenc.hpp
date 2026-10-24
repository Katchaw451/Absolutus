#pragma once
#include <vector>

namespace absolutus {
namespace ml {

// Sinusoidal position encoding (Vaswani et al. 2017).
//   PE[pos, 2i]   = sin(pos / 10000^(2i/d))
//   PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
// Precomputed once for [seq_len, d_model]. Additive, not learnable.
class PositionEncoding {
public:
    size_t seq_len, d_model;
    std::vector<double> table;   // [seq_len * d_model]

    PositionEncoding(size_t sl, size_t dm);

    // h = x + table  (returns a new vector)
    std::vector<double> add(const std::vector<double>& x) const;
};

}}
