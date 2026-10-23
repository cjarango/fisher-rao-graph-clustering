#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace frgc::stats {

// Average ranks for ties, standard convention for Spearman's rho.
inline std::vector<double> ranks(const std::vector<double>& x) {
    const std::size_t n = x.size();
    if (n == 0) return {};

    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&x](std::size_t a, std::size_t b) { return x[a] < x[b]; });

    // Build r with push_back instead of default-constructing a fixed-size
    // vector. The latter triggers a false positive in GCC 15 with
    // -O3 -Warray-bounds when the vector is inlined into callers.
    std::vector<double> r;
    r.reserve(n);
    for (std::size_t k = 0; k < n; ++k) r.push_back(0.0);

    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        while (j + 1 < n && x[idx[j + 1]] == x[idx[i]]) ++j;
        const double avg = 0.5 * (static_cast<double>(i) +
                                  static_cast<double>(j)) + 1.0;
        for (std::size_t k = i; k <= j; ++k) r[idx[k]] = avg;
        i = j + 1;
    }
    return r;
}

// Spearman rank correlation. Returns NaN if either input has zero variance
// (all values equal), which is the mathematically correct behavior.
inline double spearman(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto ra = ranks(a);
    const auto rb = ranks(b);

    const double n    = static_cast<double>(a.size());
    const double mean = 0.5 * (n + 1.0);

    double num = 0.0, da2 = 0.0, db2 = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double da = ra[i] - mean;
        const double db = rb[i] - mean;
        num += da * db;
        da2 += da * da;
        db2 += db * db;
    }
    if (da2 == 0.0 || db2 == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return num / std::sqrt(da2 * db2);
}

inline std::vector<double> row_without_self(
        const std::vector<std::vector<double>>& D, std::size_t self_i) {
    const std::size_t n = D.size();
    std::vector<double> r;
    r.reserve(n - 1);
    for (std::size_t j = 0; j < n; ++j) {
        if (j == self_i) continue;
        r.push_back(D[self_i][j]);
    }
    return r;
}

inline double kth_neighbor_radius(const std::vector<double>& row, std::size_t k) {
    if (k == 0 || k > row.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> copy = row;
    std::nth_element(copy.begin(), copy.begin() + (k - 1), copy.end());
    return copy[k - 1];
}

inline double covering_radius(const std::vector<double>& row) {
    return kth_neighbor_radius(row, 1);
}

inline double spearman_row(const std::vector<std::vector<double>>& D_a,
                           const std::vector<std::vector<double>>& D_b,
                           std::size_t self_i) {
    const auto a = row_without_self(D_a, self_i);
    const auto b = row_without_self(D_b, self_i);
    return spearman(a, b);
}

}  // namespace frgc::stats