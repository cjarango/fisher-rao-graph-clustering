#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "frgc/stats/point_stats.hpp"

using namespace frgc::stats;

namespace {

bool is_nan(double x) { return std::isnan(x); }

// Small symmetric distance matrix with explicit values, N = 5.
Matrix make_matrix_A() {
    return {
        {0.0, 1.0, 2.0, 3.0, 4.0},
        {1.0, 0.0, 1.5, 2.5, 3.5},
        {2.0, 1.5, 0.0, 1.2, 2.2},
        {3.0, 2.5, 1.2, 0.0, 1.1},
        {4.0, 3.5, 2.2, 1.1, 0.0},
    };
}

Matrix make_matrix_B() {
    return {
        {0.0, 1.2, 1.9, 3.1, 3.8},
        {1.2, 0.0, 1.6, 2.4, 3.4},
        {1.9, 1.6, 0.0, 1.3, 2.1},
        {3.1, 2.4, 1.3, 0.0, 1.0},
        {3.8, 3.4, 2.1, 1.0, 0.0},
    };
}

// Number of valid neighbors for a given N: min(K_NEIGHBORS, N-1).
std::size_t valid_k(std::size_t N) {
    return std::min(K_NEIGHBORS, N - 1);
}

}  // namespace

TEST_CASE("compute_point_stats: identical matrices give rho = 1, errors = 0") {
    const Matrix D = make_matrix_A();
    const auto ps = compute_point_stats(D, D, 0);

    CHECK(ps.rho == doctest::Approx(1.0).epsilon(1e-12));
    for (std::size_t k = 0; k < valid_k(D.size()); ++k) {
        CHECK(ps.r_hat[k] == ps.r_fr[k]);
    }
    // Beyond valid_k, both are NaN.
    for (std::size_t k = valid_k(D.size()); k < K_NEIGHBORS; ++k) {
        CHECK(is_nan(ps.r_hat[k]));
        CHECK(is_nan(ps.r_fr[k]));
    }
}

TEST_CASE("compute_point_stats: r_hat and r_fr are sorted") {
    const Matrix D_a = make_matrix_A();
    const Matrix D_b = make_matrix_B();
    const auto ps = compute_point_stats(D_a, D_b, 2);

    const std::size_t kmax = valid_k(D_a.size());
    for (std::size_t k = 1; k < kmax; ++k) {
        CHECK(ps.r_hat[k] >= ps.r_hat[k - 1]);
        CHECK(ps.r_fr[k]  >= ps.r_fr[k - 1]);
    }
}

TEST_CASE("compute_point_stats: r_hat[0] equals the covering radius") {
    const Matrix D_a = make_matrix_A();
    const Matrix D_b = make_matrix_B();
    const auto ps = compute_point_stats(D_a, D_b, 3);

    const auto row_a = row_without_self(D_a, 3);
    const auto row_b = row_without_self(D_b, 3);
    CHECK(ps.r_hat[0] == doctest::Approx(covering_radius(row_a)));
    CHECK(ps.r_fr[0]  == doctest::Approx(covering_radius(row_b)));
}

TEST_CASE("compute_point_stats: kth radius matches direct sort") {
    const Matrix D_a = make_matrix_A();
    const Matrix D_b = make_matrix_B();
    const auto ps = compute_point_stats(D_a, D_b, 1);

    auto row_a = row_without_self(D_a, 1);
    std::sort(row_a.begin(), row_a.end());

    const std::size_t kmax = valid_k(D_a.size());
    for (std::size_t k = 1; k <= kmax; ++k) {
        CHECK(ps.r_hat[k - 1] == doctest::Approx(row_a[k - 1]));
    }
}

TEST_CASE("aggregate_points: identical matrices give rho_mean = 1 and zero errors") {
    const Matrix D = make_matrix_A();
    std::vector<PointStats> pts;
    for (std::size_t i = 0; i < D.size(); ++i) {
        pts.push_back(compute_point_stats(D, D, i));
    }

    const auto cell = aggregate_points(pts);
    CHECK(cell.rho_mean == doctest::Approx(1.0).epsilon(1e-12));

    const std::size_t kmax = valid_k(D.size());
    for (std::size_t k = 0; k < kmax; ++k) {
        CHECK(cell.rk_rel_err_mean[k] == doctest::Approx(0.0).epsilon(1e-12));
    }
    for (std::size_t k = kmax; k < K_NEIGHBORS; ++k) {
        CHECK(is_nan(cell.rk_rel_err_mean[k]));
    }
}

TEST_CASE("aggregate_points: perturbed matrices give positive errors") {
    const Matrix D_a = make_matrix_A();
    const Matrix D_b = make_matrix_B();
    std::vector<PointStats> pts;
    for (std::size_t i = 0; i < D_a.size(); ++i) {
        pts.push_back(compute_point_stats(D_a, D_b, i));
    }

    const auto cell = aggregate_points(pts);
    CHECK(cell.rho_mean > 0.0);
    CHECK(cell.rho_mean <= 1.0);
    for (std::size_t k = 0; k < K_NEIGHBORS; ++k) {
        if (!is_nan(cell.rk_rel_err_mean[k])) {
            CHECK(cell.rk_rel_err_mean[k] >= 0.0);
        }
    }
}

TEST_CASE("aggregate_points: empty input gives NaN") {
    const std::vector<PointStats> empty;
    const auto cell = aggregate_points(empty);
    CHECK(is_nan(cell.rho_mean));
    for (std::size_t k = 0; k < K_NEIGHBORS; ++k) {
        CHECK(is_nan(cell.rk_rel_err_mean[k]));
    }
}

TEST_CASE("aggregate_points: NaN rho values are skipped") {
    std::vector<PointStats> pts(3);
    pts[0].rho = 1.0;
    pts[1].rho = std::numeric_limits<double>::quiet_NaN();
    pts[2].rho = 0.5;
    for (auto& p : pts) {
        p.r_hat.fill(std::numeric_limits<double>::quiet_NaN());
        p.r_fr.fill(std::numeric_limits<double>::quiet_NaN());
    }
    const auto cell = aggregate_points(pts);
    CHECK(cell.rho_mean == doctest::Approx(0.75).epsilon(1e-12));
}