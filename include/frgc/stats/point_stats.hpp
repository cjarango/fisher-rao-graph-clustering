#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "frgc/stats/pointwise.hpp"

namespace frgc::stats {

using Matrix = std::vector<std::vector<double>>;

// k = 1..9, matching the experimental design: the largest neighborhood
// is smaller than the smallest sample size (N = 10).
constexpr std::size_t K_NEIGHBORS = 9;

struct PointStats {
    // rho: spearman rank correlation at point i, comparing the criterion
    //      ordering against the FR ordering on the row i.
    // r_hat[k-1]: k-th nearest neighbor of i under the criterion.
    // r_fr[k-1]:  k-th nearest neighbor of i under FR.
    // Entries beyond min(K_NEIGHBORS, N-1) are NaN. The covering radius
    // corresponds to k = 1: r_hat[0] = delta_hat, r_fr[0] = delta_fr.
    double rho;
    std::array<double, K_NEIGHBORS> r_hat;
    std::array<double, K_NEIGHBORS> r_fr;
};

inline PointStats compute_point_stats(const Matrix& D_crit,
                                      const Matrix& D_fr,
                                      std::size_t i) {
    PointStats out;
    out.r_hat.fill(std::numeric_limits<double>::quiet_NaN());
    out.r_fr.fill(std::numeric_limits<double>::quiet_NaN());

    out.rho = spearman_row(D_crit, D_fr, i);

    auto row_crit = row_without_self(D_crit, i);
    auto row_fr   = row_without_self(D_fr,   i);
    std::sort(row_crit.begin(), row_crit.end());
    std::sort(row_fr.begin(),   row_fr.end());

    const std::size_t kmax = std::min(K_NEIGHBORS, row_crit.size());
    for (std::size_t k = 1; k <= kmax; ++k) {
        out.r_hat[k - 1] = row_crit[k - 1];
        out.r_fr[k - 1]  = row_fr[k - 1];
    }
    return out;
}

struct CellStats {
    // rho_mean: mean over the N points of rho_i. NaN entries (constant
    // rows or undefined rho) are skipped; if all entries are NaN,
    // rho_mean is NaN.
    // rk_rel_err_mean[k-1]: mean over the N points of |r_hat_k - r_fr_k| / r_fr_k.
    // NaN entries are skipped per k. If no valid pairs remain, that entry
    // is NaN.
    double rho_mean;
    std::array<double, K_NEIGHBORS> rk_rel_err_mean;
};

inline CellStats aggregate_points(const std::vector<PointStats>& pts) {
    CellStats out;

    double rho_sum = 0.0;
    std::size_t rho_count = 0;
    for (const auto& p : pts) {
        if (!std::isnan(p.rho)) {
            rho_sum += p.rho;
            ++rho_count;
        }
    }
    out.rho_mean = rho_count > 0
        ? rho_sum / static_cast<double>(rho_count)
        : std::numeric_limits<double>::quiet_NaN();

    for (std::size_t k = 0; k < K_NEIGHBORS; ++k) {
        double err_sum = 0.0;
        std::size_t err_count = 0;
        for (const auto& p : pts) {
            const double rh = p.r_hat[k];
            const double rf = p.r_fr[k];
            if (std::isfinite(rh) && std::isfinite(rf) && rf > 0.0) {
                err_sum += std::abs(rh - rf) / rf;
                ++err_count;
            }
        }
        out.rk_rel_err_mean[k] = err_count > 0
            ? err_sum / static_cast<double>(err_count)
            : std::numeric_limits<double>::quiet_NaN();
    }
    return out;
}

}  // namespace frgc::stats