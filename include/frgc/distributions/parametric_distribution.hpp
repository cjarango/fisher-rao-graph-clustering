#pragma once

#include <vector>

namespace frgc::distributions {

// Abstract base for a parametric probability distribution: a point on a
// statistical manifold once its parameter vector theta is fixed.

class ParametricDistribution {
public:
    using Vector = std::vector<double>;

    virtual ~ParametricDistribution() = default;

    ParametricDistribution(const ParametricDistribution&) = default;
    ParametricDistribution(ParametricDistribution&&) noexcept = default;
    ParametricDistribution& operator=(const ParametricDistribution&) = default;
    ParametricDistribution& operator=(ParametricDistribution&&) noexcept = default;

    [[nodiscard]] const Vector& get_params() const noexcept { return params_; }

    [[nodiscard]] virtual double mean() const = 0;
    [[nodiscard]] virtual double variance() const = 0;
    [[nodiscard]] virtual double skewness() const = 0;
    [[nodiscard]] virtual double kurtosis() const = 0;
    [[nodiscard]] virtual Vector pdf(const Vector& x) const = 0;

    // Not derived as log(pdf(x)): lets subclasses use a stable closed form.
    [[nodiscard]] virtual Vector log_pdf(const Vector& x) const = 0;

    [[nodiscard]] virtual bool is_valid_parameter_space() const = 0;

protected:
    explicit ParametricDistribution(Vector params) : params_(std::move(params)) {}

    // For subclass setters to resync params_ after validating named fields.
    void set_params(Vector params) noexcept { params_ = std::move(params); }

    Vector params_;
};

}  // namespace frgc::distributions