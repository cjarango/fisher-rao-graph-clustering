#include "frgc/distributions/beta_distribution.hpp"

#include <cmath>
#include <limits>

#include "frgc/exceptions/invalid_parameter_error.hpp"

namespace frgc::distributions {

BetaDistribution::BetaDistribution(double alpha, double beta)
    : ParametricDistribution(Vector{alpha, beta}), alpha_(alpha), beta_(beta) {
    if (!is_valid_parameter_space()) {
        throw frgc::exceptions::InvalidParameterError(
            "BetaDistribution requires alpha > 0 and beta > 0, got alpha=" +
            std::to_string(alpha) + ", beta=" + std::to_string(beta));
    }
}

void BetaDistribution::set_alpha(double alpha) {
    const double previous = alpha_;
    alpha_ = alpha;
    if (!is_valid_parameter_space()) {
        alpha_ = previous;
        throw frgc::exceptions::InvalidParameterError(
            "BetaDistribution requires alpha > 0, got alpha=" + std::to_string(alpha));
    }
    set_params({alpha_, beta_});
}

void BetaDistribution::set_beta(double beta) {
    const double previous = beta_;
    beta_ = beta;
    if (!is_valid_parameter_space()) {
        beta_ = previous;
        throw frgc::exceptions::InvalidParameterError(
            "BetaDistribution requires beta > 0, got beta=" + std::to_string(beta));
    }
    set_params({alpha_, beta_});
}

double BetaDistribution::mean() const { return alpha_ / (alpha_ + beta_); }

double BetaDistribution::variance() const {
    const double sum = alpha_ + beta_;
    return (alpha_ * beta_) / (sum * sum * (sum + 1.0));
}

double BetaDistribution::skewness() const {
    const double sum = alpha_ + beta_;
    return 2.0 * (beta_ - alpha_) * std::sqrt(sum + 1.0) /
           ((sum + 2.0) * std::sqrt(alpha_ * beta_));
}

double BetaDistribution::kurtosis() const {
    const double sum = alpha_ + beta_;
    const double diff = alpha_ - beta_;
    const double numerator = diff * diff * (sum + 1.0) - alpha_ * beta_ * (sum + 2.0);
    const double denominator = alpha_ * beta_ * (sum + 2.0) * (sum + 3.0);
    return 6.0 * numerator / denominator;
}

BetaDistribution::Vector BetaDistribution::pdf(const Vector& x) const {
    const double log_norm_const = std::lgamma(alpha_) + std::lgamma(beta_) - std::lgamma(alpha_ + beta_);

    Vector result(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] <= 0.0 || x[i] >= 1.0) {
            result[i] = 0.0;
            continue;
        }
        result[i] = std::exp((alpha_ - 1.0) * std::log(x[i]) + (beta_ - 1.0) * std::log(1.0 - x[i]) -
                              log_norm_const);
    }
    return result;
}

BetaDistribution::Vector BetaDistribution::log_pdf(const Vector& x) const {
    const double log_norm_const = std::lgamma(alpha_) + std::lgamma(beta_) - std::lgamma(alpha_ + beta_);

    Vector result(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] <= 0.0 || x[i] >= 1.0) {
            result[i] = -std::numeric_limits<double>::infinity();
            continue;
        }
        result[i] = (alpha_ - 1.0) * std::log(x[i]) + (beta_ - 1.0) * std::log(1.0 - x[i]) - log_norm_const;
    }
    return result;
}

bool BetaDistribution::is_valid_parameter_space() const { return alpha_ > 0.0 && beta_ > 0.0; }

}  // namespace frgc::distributions