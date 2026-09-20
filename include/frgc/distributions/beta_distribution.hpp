#pragma once

#include "frgc/distributions/parametric_distribution.hpp"

namespace frgc::distributions {

// Beta distribution with shape parameters alpha, beta, supported on (0, 1).
class BetaDistribution final : public ParametricDistribution {
public:
    BetaDistribution(double alpha, double beta);

    [[nodiscard]] double alpha() const noexcept { return alpha_; }
    [[nodiscard]] double beta() const noexcept { return beta_; }

    void set_alpha(double alpha);
    void set_beta(double beta);

    [[nodiscard]] double mean() const override;
    [[nodiscard]] double variance() const override;
    [[nodiscard]] double skewness() const override;
    [[nodiscard]] double kurtosis() const override;
    [[nodiscard]] Vector pdf(const Vector& x) const override;
    [[nodiscard]] Vector log_pdf(const Vector& x) const override;
    [[nodiscard]] bool is_valid_parameter_space() const override;

private:
    double alpha_;
    double beta_;
};

}  // namespace frgc::distributions