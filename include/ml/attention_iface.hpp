#pragma once
#include <vector>

namespace absolutus {
namespace ml {

class AttentionIface {
public:
    virtual ~AttentionIface() = default;
    virtual const std::vector<double>& forward(const std::vector<double>& X) = 0;
    virtual std::vector<double> backward(const std::vector<double>& grad) = 0;
    virtual void update(double lr) = 0;
};

}}
