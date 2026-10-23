#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "frgc/stats/point_stats.hpp"

namespace frgc::stats {

struct FinalAggregate {
    // rho_expected: mean over the R replicates of the per-replicate rho_mean.
    // rho_var_hat:  variance of the mean, computed as
    //               (1/(R(R-1))) * sum_r (x_r - x_bar)^2.
    // Same for rk_rel_err, per k.
    // If R_eff < 2 the variance is NaN (undefined).
    double rho_expected;
    double rho_var_hat;
    std::array<double, K_NEIGHBORS> rk_rel_err_expected;
    std::array<double, K_NEIGHBORS> rk_rel_err_var_hat;
};

namespace detail {

// Mean and variance-of-the-mean over the finite entries of x.
inline void mean_and_var_of_mean(const std::vector<double>& x,
                                 double& mean_out, double& var_out) {
    double sum = 0.0;
    std::size_t n = 0;
    for (double v : x) {
        if (std::isfinite(v)) {
            sum += v;
            ++n;
        }
    }
    if (n == 0) {
        mean_out = std::numeric_limits<double>::quiet_NaN();
        var_out  = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    const double mean = sum / static_cast<double>(n);
    mean_out = mean;

    if (n < 2) {
        var_out = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    double ssd = 0.0;
    for (double v : x) {
        if (std::isfinite(v)) {
            const double d = v - mean;
            ssd += d * d;
        }
    }
    const double R = static_cast<double>(n);
    var_out = ssd / (R * (R - 1.0));
}

}  // namespace detail

inline FinalAggregate aggregate_replicates(const std::vector<CellStats>& cells) {
    FinalAggregate out;

    std::vector<double> rho_vals;
    rho_vals.reserve(cells.size());
    for (const auto& c : cells) rho_vals.push_back(c.rho_mean);

    detail::mean_and_var_of_mean(rho_vals, out.rho_expected, out.rho_var_hat);

    for (std::size_t k = 0; k < K_NEIGHBORS; ++k) {
        std::vector<double> rk_vals;
        rk_vals.reserve(cells.size());
        for (const auto& c : cells) rk_vals.push_back(c.rk_rel_err_mean[k]);

        detail::mean_and_var_of_mean(rk_vals,
                                     out.rk_rel_err_expected[k],
                                     out.rk_rel_err_var_hat[k]);
    }
    return out;
}

}  // namespace frgc::stats