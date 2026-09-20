#pragma once

#include "frgc/distributions/parametric_distribution.hpp"

namespace frgc::distributions {

// Univariate Gaussian N(mu, sigma^2), a point (mu, sigma) on the
// two-parameter Gaussian statistical manifold.
class GaussianDistribution final : public ParametricDistribution {
public:

    GaussianDistribution(double mu, double sigma);

    [[nodiscard]] double mu() const noexcept { return mu_; }
    [[nodiscard]] double sigma() const noexcept { return sigma_; }

    void set_mu(double mu);
    void set_sigma(double sigma);

    [[nodiscard]] double mean() const override;
    [[nodiscard]] double variance() const override;
    [[nodiscard]] double skewness() const override;
    [[nodiscard]] double kurtosis() const override;
    [[nodiscard]] Vector pdf(const Vector& x) const override;
    [[nodiscard]] Vector log_pdf(const Vector& x) const override;
    [[nodiscard]] bool is_valid_parameter_space() const override;

private:
    double mu_;
    double sigma_;
};

}  // namespace frgc::distributions