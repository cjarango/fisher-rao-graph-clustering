#include "frgc/distributions/gamma_distribution.hpp"

#include <cmath>

#include "frgc/exceptions/invalid_parameter_error.hpp"

namespace frgc::distributions {

GammaDistribution::GammaDistribution(double gamma, double kappa)
    : ParametricDistribution(Vector{gamma, kappa}), gamma_(gamma), kappa_(kappa) {
    if (!is_valid_parameter_space()) {
        throw frgc::exceptions::InvalidParameterError(
            "GammaDistribution requires gamma > 0 and kappa > 0, got gamma=" +
            std::to_string(gamma) + ", kappa=" + std::to_string(kappa));
    }
}

void GammaDistribution::set_gamma(double gamma) {
    const double previous = gamma_;
    gamma_ = gamma;
    if (!is_valid_parameter_space()) {
        gamma_ = previous;
        throw frgc::exceptions::InvalidParameterError(
            "GammaDistribution requires gamma > 0, got gamma=" + std::to_string(gamma));
    }
    set_params({gamma_, kappa_});
}

void GammaDistribution::set_kappa(double kappa) {
    const double previous = kappa_;
    kappa_ = kappa;
    if (!is_valid_parameter_space()) {
        kappa_ = previous;
        throw frgc::exceptions::InvalidParameterError(
            "GammaDistribution requires kappa > 0, got kappa=" + std::to_string(kappa));
    }
    set_params({gamma_, kappa_});
}

GammaDistribution::Vector GammaDistribution::get_scale_rate() const {
    return {gamma_ / kappa_, kappa_ / gamma_};
}

double GammaDistribution::mean() const { return gamma_; }
double GammaDistribution::variance() const { return (gamma_ * gamma_) / kappa_; }
double GammaDistribution::skewness() const { return 2.0 / std::sqrt(kappa_); }
double GammaDistribution::kurtosis() const { return 6.0 / kappa_; }

GammaDistribution::Vector GammaDistribution::pdf(const Vector& x) const {
    const double rate = kappa_ / gamma_;
    const double log_norm_const = kappa_ * std::log(rate) - std::lgamma(kappa_);

    Vector result(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] <= 0.0) {
            result[i] = 0.0;
            continue;
        }
        result[i] = std::exp(log_norm_const + (kappa_ - 1.0) * std::log(x[i]) - rate * x[i]);
    }
    return result;
}

GammaDistribution::Vector GammaDistribution::log_pdf(const Vector& x) const {
    const double rate = kappa_ / gamma_;
    const double log_norm_const = kappa_ * std::log(rate) - std::lgamma(kappa_);

    Vector result(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] <= 0.0) {
            result[i] = -std::numeric_limits<double>::infinity();
            continue;
        }
        result[i] = log_norm_const + (kappa_ - 1.0) * std::log(x[i]) - rate * x[i];
    }
    return result;
}

bool GammaDistribution::is_valid_parameter_space() const { return gamma_ > 0.0 && kappa_ > 0.0; }

}  // namespace frgc::distributions