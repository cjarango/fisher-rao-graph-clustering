#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "frgc/stats/pointwise.hpp"

using namespace frgc::stats;

namespace {

bool is_nan(double x) { return std::isnan(x); }

}  // namespace

TEST_CASE("ranks: no ties") {
    const std::vector<double> x = {3.0, 1.0, 4.0, 1.5, 5.0};
    const auto r = ranks(x);
    // Sorted order: 1.0 (idx1), 1.5 (idx3), 3.0 (idx0), 4.0 (idx2), 5.0 (idx4)
    CHECK(r[0] == 3.0);
    CHECK(r[1] == 1.0);
    CHECK(r[2] == 4.0);
    CHECK(r[3] == 2.0);
    CHECK(r[4] == 5.0);
}

TEST_CASE("ranks: ties get average ranks") {
    const std::vector<double> x = {1.0, 1.0, 2.0, 3.0, 3.0, 3.0};
    const auto r = ranks(x);
    CHECK(r[0] == 1.5);
    CHECK(r[1] == 1.5);
    CHECK(r[2] == 3.0);
    CHECK(r[3] == 5.0);
    CHECK(r[4] == 5.0);
    CHECK(r[5] == 5.0);
}

TEST_CASE("spearman: perfect positive correlation") {
    const std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> b = {2.0, 4.0, 6.0, 8.0, 10.0};
    CHECK(spearman(a, b) == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("spearman: perfect negative correlation") {
    const std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> b = {5.0, 4.0, 3.0, 2.0, 1.0};
    CHECK(spearman(a, b) == doctest::Approx(-1.0).epsilon(1e-12));
}

TEST_CASE("spearman: known value -0.5") {
    // a = [1, 2, 3], b = [3, 1, 2]
    // ranks_a = [1, 2, 3], ranks_b = [3, 1, 2]
    // Pearson on ranks gives -0.5.
    const std::vector<double> a = {1.0, 2.0, 3.0};
    const std::vector<double> b = {3.0, 1.0, 2.0};
    CHECK(spearman(a, b) == doctest::Approx(-0.5).epsilon(1e-12));
}

TEST_CASE("spearman: two points always gives ±1") {
    const std::vector<double> a = {0.3, 0.7};
    const std::vector<double> b = {0.1, 0.9};
    CHECK(spearman(a, b) == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(spearman(b, a) == doctest::Approx(1.0).epsilon(1e-12));

    const std::vector<double> c = {0.9, 0.1};
    CHECK(spearman(a, c) == doctest::Approx(-1.0).epsilon(1e-12));
}

TEST_CASE("spearman: constant vector returns NaN") {
    const std::vector<double> a = {1.0, 1.0, 1.0, 1.0};
    const std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    CHECK(is_nan(spearman(a, b)));
    CHECK(is_nan(spearman(b, a)));
}

TEST_CASE("spearman: size mismatch returns NaN") {
    const std::vector<double> a = {1.0, 2.0, 3.0};
    const std::vector<double> b = {1.0, 2.0};
    CHECK(is_nan(spearman(a, b)));
}

TEST_CASE("spearman: symmetric under swap") {
    const std::vector<double> a = {1.0, 3.0, 2.0, 5.0, 4.0};
    const std::vector<double> b = {2.0, 1.0, 4.0, 3.0, 5.0};
    CHECK(spearman(a, b) == doctest::Approx(spearman(b, a)).epsilon(1e-12));
}

TEST_CASE("row_without_self: drops the diagonal entry") {
    const std::vector<std::vector<double>> D = {
        {0.0, 1.0, 2.0},
        {1.0, 0.0, 3.0},
        {2.0, 3.0, 0.0},
    };
    const auto r0 = row_without_self(D, 0);
    CHECK(r0.size() == 2);
    CHECK(r0[0] == 1.0);
    CHECK(r0[1] == 2.0);

    const auto r1 = row_without_self(D, 1);
    CHECK(r1.size() == 2);
    CHECK(r1[0] == 1.0);
    CHECK(r1[1] == 3.0);

    const auto r2 = row_without_self(D, 2);
    CHECK(r2.size() == 2);
    CHECK(r2[0] == 2.0);
    CHECK(r2[1] == 3.0);
}

TEST_CASE("kth_neighbor_radius: basic") {
    const std::vector<double> row = {0.7, 0.3, 0.5, 0.1, 0.9};
    CHECK(kth_neighbor_radius(row, 1) == doctest::Approx(0.1));
    CHECK(kth_neighbor_radius(row, 2) == doctest::Approx(0.3));
    CHECK(kth_neighbor_radius(row, 3) == doctest::Approx(0.5));
    CHECK(kth_neighbor_radius(row, 4) == doctest::Approx(0.7));
    CHECK(kth_neighbor_radius(row, 5) == doctest::Approx(0.9));
}

TEST_CASE("kth_neighbor_radius: out of range returns NaN") {
    const std::vector<double> row = {0.1, 0.2, 0.3};
    CHECK(is_nan(kth_neighbor_radius(row, 0)));
    CHECK(is_nan(kth_neighbor_radius(row, 4)));
}

TEST_CASE("covering_radius: matches kth with k=1") {
    const std::vector<double> row = {0.4, 0.2, 0.9, 0.6};
    CHECK(covering_radius(row) == doctest::Approx(0.2));
}

TEST_CASE("spearman_row: identical matrices give rho = 1") {
    const std::vector<std::vector<double>> D = {
        {0.0, 1.0, 2.0, 3.0},
        {1.0, 0.0, 2.5, 3.5},
        {2.0, 2.5, 0.0, 4.0},
        {3.0, 3.5, 4.0, 0.0},
    };
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(spearman_row(D, D, i) == doctest::Approx(1.0).epsilon(1e-12));
    }
}

TEST_CASE("spearman_row: rank-reversed matrix gives rho = -1") {
    // 3x3 so each row has exactly two off-diagonal entries. With two
    // entries, reversing the vector reverses the ranks exactly.
    const std::vector<std::vector<double>> D_ref = {
        {0.0, 1.0, 2.0},
        {1.0, 0.0, 3.0},
        {2.0, 3.0, 0.0},
    };
    const std::vector<std::vector<double>> D_rev = {
        {0.0, 2.0, 1.0},
        {3.0, 0.0, 1.0},
        {3.0, 2.0, 0.0},
    };

    for (std::size_t i = 0; i < 3; ++i) {
        CHECK(spearman_row(D_ref, D_rev, i) == doctest::Approx(-1.0).epsilon(1e-12));
    }
}