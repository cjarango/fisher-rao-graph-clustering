#pragma once

#include "frgc/distributions/parametric_distribution.hpp"

namespace frgc::distributions {

// Bernoulli distribution with success probability p, p in (0, 1).
class BernoulliDistribution final : public ParametricDistribution {
public:
    explicit BernoulliDistribution(double p);

    [[nodiscard]] double p() const noexcept { return p_; }

    void set_p(double p);

    [[nodiscard]] double mean() const override;
    [[nodiscard]] double variance() const override;
    [[nodiscard]] double skewness() const override;
    [[nodiscard]] double kurtosis() const override;
    [[nodiscard]] Vector pdf(const Vector& x) const override;
    [[nodiscard]] Vector log_pdf(const Vector& x) const override;
    [[nodiscard]] bool is_valid_parameter_space() const override;

private:
    double p_;
};

}  // namespace frgc::distributions