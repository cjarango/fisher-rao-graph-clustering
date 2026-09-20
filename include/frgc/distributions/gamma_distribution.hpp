#pragma once

#include "frgc/distributions/parametric_distribution.hpp"

namespace frgc::distributions {

// Gamma distribution in mean-shape parametrization: gamma is the mean,
// kappa is the shape. Scale = gamma / kappa, rate = kappa / gamma.
class GammaDistribution final : public ParametricDistribution {
public:
    GammaDistribution(double gamma, double kappa);

    [[nodiscard]] double gamma() const noexcept { return gamma_; }
    [[nodiscard]] double kappa() const noexcept { return kappa_; }

    void set_gamma(double gamma);
    void set_kappa(double kappa);

    // Returns {scale, rate} = {gamma / kappa, kappa / gamma}.
    [[nodiscard]] Vector get_scale_rate() const;

    [[nodiscard]] double mean() const override;
    [[nodiscard]] double variance() const override;
    [[nodiscard]] double skewness() const override;
    [[nodiscard]] double kurtosis() const override;
    [[nodiscard]] Vector pdf(const Vector& x) const override;
    [[nodiscard]] Vector log_pdf(const Vector& x) const override;
    [[nodiscard]] bool is_valid_parameter_space() const override;

private:
    double gamma_;
    double kappa_;
};

}  // namespace frgc::distributions