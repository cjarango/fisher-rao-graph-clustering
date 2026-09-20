#include "frgc/distributions/bernoulli_distribution.hpp"

#include <cmath>
#include <limits>

#include "frgc/exceptions/invalid_parameter_error.hpp"

namespace frgc::distributions {

namespace {
// Returns the outcome (0 or 1) encoded by x, or -1 if x is not an integer
// in {0, 1}.
int outcome_of(double x) {
    const int rounded = static_cast<int>(x);
    if (static_cast<double>(rounded) != x || (rounded != 0 && rounded != 1)) {
        return -1;
    }
    return rounded;
}
}  // namespace

BernoulliDistribution::BernoulliDistribution(double p) : ParametricDistribution(Vector{p}), p_(p) {
    if (!is_valid_parameter_space()) {
        throw frgc::exceptions::InvalidParameterError(
            "BernoulliDistribution requires 0 < p < 1, got p=" + std::to_string(p));
    }
}

void BernoulliDistribution::set_p(double p) {
    const double previous = p_;
    p_ = p;
    if (!is_valid_parameter_space()) {
        p_ = previous;
        throw frgc::exceptions::InvalidParameterError(
            "BernoulliDistribution requires 0 < p < 1, got p=" + std::to_string(p));
    }
    set_params({p_});
}

double BernoulliDistribution::mean() const { return p_; }

double BernoulliDistribution::variance() const { return p_ * (1.0 - p_); }

double BernoulliDistribution::skewness() const {
    return (1.0 - 2.0 * p_) / std::sqrt(p_ * (1.0 - p_));
}

double BernoulliDistribution::kurtosis() const {
    return (1.0 - 6.0 * p_ * (1.0 - p_)) / (p_ * (1.0 - p_));
}

BernoulliDistribution::Vector BernoulliDistribution::pdf(const Vector& x) const {
    Vector result(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const int outcome = outcome_of(x[i]);
        result[i] = outcome == 1 ? p_ : outcome == 0 ? 1.0 - p_ : 0.0;
    }
    return result;
}

BernoulliDistribution::Vector BernoulliDistribution::log_pdf(const Vector& x) const {
    Vector result(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const int outcome = outcome_of(x[i]);
        result[i] = outcome == 1   ? std::log(p_)
                    : outcome == 0 ? std::log(1.0 - p_)
                                   : -std::numeric_limits<double>::infinity();
    }
    return result;
}

bool BernoulliDistribution::is_valid_parameter_space() const { return p_ > 0.0 && p_ < 1.0; }

}  // namespace frgc::distributions