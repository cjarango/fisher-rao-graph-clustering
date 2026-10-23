#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <utility>
#include <vector>

#include "frgc/experiment/sampling.hpp"

using namespace frgc::experiment;

namespace {

bool same_points(const std::vector<Point2D>& a, const std::vector<Point2D>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i][0] != b[i][0] || a[i][1] != b[i][1]) return false;
    }
    return true;
}

bool inside_box(const Point2D& p, const FamilyBox& box) {
    return p[0] >= box.a_min && p[0] <= box.a_max
        && p[1] >= box.b_min && p[1] <= box.b_max;
}

bool inside_safe(const Point2D& p, const FamilyBox& box) {
    const double da = (box.a_max - box.a_min) * box.margin_frac;
    const double db = (box.b_max - box.b_min) * box.margin_frac;
    return p[0] >= box.a_min + da && p[0] <= box.a_max - da
        && p[1] >= box.b_min + db && p[1] <= box.b_max - db;
}

double chebyshev(const Point2D& a, const Point2D& b) {
    return std::max(std::abs(a[0] - b[0]), std::abs(a[1] - b[1]));
}

SampleOptions default_options(Regime reg) {
    SampleOptions opts;
    if (reg == Regime::M2 || reg == Regime::L2) {
        opts.proportions = {0.5, 0.5};
    } else if (reg == Regime::M3 || reg == Regime::L3) {
        opts.proportions = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    }
    opts.s_min = 0.05;
    opts.s_max = 0.20;
    return opts;
}

}  // namespace

TEST_CASE("calibrate_family_scale: W is positive and finite") {
    for (Family f : {Family::Gaussian, Family::Gamma, Family::Beta}) {
        const auto s = calibrate_family_scale(f);
        CHECK(s.W > 0.0);
        CHECK(std::isfinite(s.W));
    }
}

TEST_CASE("calibrate_family_scale: cached calls return identical values") {
    const auto s1 = calibrate_family_scale(Family::Gaussian);
    const auto s2 = calibrate_family_scale(Family::Gaussian);
    CHECK(s1.W == s2.W);
}

TEST_CASE("replicate_seed: deterministic and injective enough") {
    const auto a = replicate_seed(42, Family::Gaussian, Regime::M2, 3);
    const auto b = replicate_seed(42, Family::Gaussian, Regime::M2, 3);
    CHECK(a == b);

    const auto c = replicate_seed(42, Family::Gaussian, Regime::M2, 4);
    const auto d = replicate_seed(42, Family::Gaussian, Regime::L2, 3);
    const auto e = replicate_seed(42, Family::Gamma,    Regime::M2, 3);
    CHECK(a != c);
    CHECK(a != d);
    CHECK(a != e);
}

TEST_CASE("generate_sample: uniform regime returns N points in the full box") {
    const std::uint64_t seed = replicate_seed(0, Family::Gaussian, Regime::U, 0);
    const auto r = generate_sample(Family::Gaussian, Regime::U, 50,
                                   SampleOptions{}, seed);

    CHECK(r.points.size() == 50);
    CHECK(r.labels.empty());
    CHECK(r.cluster_sizes.empty());
    CHECK(r.foci.empty());
    CHECK(r.radii.empty());

    const FamilyBox box = get_family_box(Family::Gaussian);
    for (const auto& p : r.points) {
        CHECK(inside_box(p, box));
    }
}

TEST_CASE("generate_sample: mixture returns N points in the full box") {
    const FamilyBox box = get_family_box(Family::Gaussian);

    for (Regime reg : {Regime::M2, Regime::M3, Regime::L2, Regime::L3}) {
        const std::uint64_t seed = replicate_seed(0, Family::Gaussian, reg, 0);
        const auto opts = default_options(reg);
        const auto r = generate_sample(Family::Gaussian, reg, 200, opts, seed);

        CHECK(r.points.size() == 200);
        CHECK(r.labels.size() == r.points.size());

        const std::size_t k = opts.proportions.size();
        CHECK(r.cluster_sizes.size() == k);
        CHECK(r.foci.size() == k);
        CHECK(r.radii.size() == k);

        std::size_t total = 0;
        for (auto s : r.cluster_sizes) total += s;
        CHECK(total == 200);

        for (const auto& p : r.points) {
            CHECK(inside_box(p, box));
        }
    }
}

TEST_CASE("generate_sample: foci lie inside the safe-zone") {
    const FamilyBox box = get_family_box(Family::Gamma);

    for (Regime reg : {Regime::M2, Regime::M3, Regime::L2, Regime::L3}) {
        const std::uint64_t seed = replicate_seed(3, Family::Gamma, reg, 1);
        const auto opts = default_options(reg);
        const auto r = generate_sample(Family::Gamma, reg, 100, opts, seed);
        for (const auto& p : r.foci) {
            CHECK(inside_safe(p, box));
        }
    }
}

TEST_CASE("generate_sample: foci are Chebyshev-separated by at least r_i") {
    for (Regime reg : {Regime::M2, Regime::M3, Regime::L2, Regime::L3}) {
        const std::uint64_t seed = replicate_seed(7, Family::Gaussian, reg, 2);
        const auto opts = default_options(reg);
        const auto r = generate_sample(Family::Gaussian, reg, 100, opts, seed);

        for (std::size_t i = 0; i < r.foci.size(); ++i) {
            for (std::size_t j = i + 1; j < r.foci.size(); ++j) {
                CHECK(chebyshev(r.foci[i], r.foci[j]) >= r.radii[i] - 1e-12);
                CHECK(chebyshev(r.foci[i], r.foci[j]) >= r.radii[j] - 1e-12);
            }
        }
    }
}

TEST_CASE("generate_sample: labels are consistent with cluster sizes") {
    const std::uint64_t seed = replicate_seed(11, Family::Gaussian, Regime::M3, 0);
    const auto opts = default_options(Regime::M3);
    const auto r = generate_sample(Family::Gaussian, Regime::M3, 150, opts, seed);

    std::vector<std::size_t> counts(r.cluster_sizes.size(), 0);
    for (auto lab : r.labels) {
        REQUIRE(lab < counts.size());
        ++counts[lab];
    }
    CHECK(counts == r.cluster_sizes);
}

TEST_CASE("generate_sample: deterministic for same seed") {
    const auto opts = default_options(Regime::M2);
    const std::uint64_t s1 = replicate_seed(7, Family::Gaussian, Regime::M2, 0);
    const std::uint64_t s2 = replicate_seed(7, Family::Gaussian, Regime::M2, 0);
    const auto r1 = generate_sample(Family::Gaussian, Regime::M2, 50, opts, s1);
    const auto r2 = generate_sample(Family::Gaussian, Regime::M2, 50, opts, s2);

    CHECK(same_points(r1.points, r2.points));
    CHECK(r1.labels == r2.labels);
    CHECK(r1.cluster_sizes == r2.cluster_sizes);
    CHECK(r1.radii == r2.radii);
}

TEST_CASE("generate_sample: different replicates give different points") {
    const auto opts = default_options(Regime::M2);
    const auto r1 = generate_sample(Family::Gaussian, Regime::M2, 50, opts,
                                    replicate_seed(0, Family::Gaussian, Regime::M2, 0));
    const auto r2 = generate_sample(Family::Gaussian, Regime::M2, 50, opts,
                                    replicate_seed(0, Family::Gaussian, Regime::M2, 1));
    CHECK(!same_points(r1.points, r2.points));
}

TEST_CASE("sort_by_coordinates: preserves the multiset of points") {
    const auto opts = default_options(Regime::M2);
    const auto r = generate_sample(Family::Gaussian, Regime::M2, 30, opts,
                                   replicate_seed(0, Family::Gaussian, Regime::M2, 0));
    auto pts = r.points;
    sort_by_coordinates(pts);

    CHECK(pts.size() == r.points.size());

    for (std::size_t i = 1; i < pts.size(); ++i) {
        const bool ordered = pts[i - 1][0] < pts[i][0]
            || (pts[i - 1][0] == pts[i][0] && pts[i - 1][1] <= pts[i][1]);
        CHECK(ordered);
    }

    std::multiset<std::pair<double,double>> before, after;
    for (const auto& p : r.points) before.insert(std::make_pair(p[0], p[1]));
    for (const auto& p : pts)      after.insert(std::make_pair(p[0], p[1]));
    CHECK(before == after);
}