#include "frgc/distributions/gaussian_distribution.hpp"

#include <cmath>

#include "frgc/exceptions/invalid_parameter_error.hpp"

namespace frgc::distributions {

namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
}

GaussianDistribution::GaussianDistribution(double mu, double sigma)
    : ParametricDistribution(Vector{mu, sigma}), mu_(mu), sigma_(sigma) {
    if (!is_valid_parameter_space()) {
        throw frgc::exceptions::InvalidParameterError(
            "GaussianDistribution requires sigma > 0, got sigma=" + std::to_string(sigma));
    }
}

void GaussianDistribution::set_mu(double mu) {
    mu_ = mu;
    set_params({mu_, sigma_});
}

void GaussianDistribution::set_sigma(double sigma) {
    const double previous = sigma_;
    sigma_ = sigma;
    if (!is_valid_parameter_space()) {
        sigma_ = previous;
        throw frgc::exceptions::InvalidParameterError(
            "GaussianDistribution requires sigma > 0, got sigma=" + std::to_string(sigma));
    }
    set_params({mu_, sigma_});
}

double GaussianDistribution::mean() const { return mu_; }
double GaussianDistribution::variance() const { return sigma_ * sigma_; }
double GaussianDistribution::skewness() const { return 0.0; }
double GaussianDistribution::kurtosis() const { return 0.0; }

GaussianDistribution::Vector GaussianDistribution::pdf(const Vector& x) const {
    Vector result(x.size());
    const double norm_const = 1.0 / (sigma_ * std::sqrt(kTwoPi));
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double z = (x[i] - mu_) / sigma_;
        result[i] = norm_const * std::exp(-0.5 * z * z);
    }
    return result;
}

GaussianDistribution::Vector GaussianDistribution::log_pdf(const Vector& x) const {
    Vector result(x.size());
    const double log_norm_const = -std::log(sigma_) - 0.5 * std::log(kTwoPi);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double z = (x[i] - mu_) / sigma_;
        result[i] = log_norm_const - 0.5 * z * z;
    }
    return result;
}

bool GaussianDistribution::is_valid_parameter_space() const { return sigma_ > 0.0; }

}  // namespace frgc::distributions