#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "frgc/distributions/gaussian.hpp"
#include "frgc/manifolds/gaussian_manifold.hpp"

using namespace frgc;
using namespace frgc::manifolds;

namespace {

constexpr double kTol = 1e-10;

double rel_err(double a, double b) {
    const double s = std::max(std::abs(a), std::abs(b));
    return s > 0.0 ? std::abs(a - b) / s : 0.0;
}

}  // namespace

TEST_CASE("Gaussian metric tensor matches the closed form") {
    const std::vector<Gaussian> pts = {
        Gaussian(0.0, 1.0),
        Gaussian(2.5, 0.5),
        Gaussian(-1.0, 3.0),
        Gaussian(10.0, 0.1),
    };

    for (const auto& p : pts) {
        const auto g = gaussian::metric_tensor(p);
        const double s2 = p.variance();
        const double g11_expected = 1.0 / s2;
        const double g22_expected = 2.0 / s2;

        CHECK(rel_err(g[0][0], g11_expected) < kTol);
        CHECK(rel_err(g[1][1], g22_expected) < kTol);
        CHECK(g[0][1] == 0.0);
        CHECK(g[1][0] == 0.0);
        CHECK(g[0][0] > 0.0);
        CHECK(g[1][1] > 0.0);
    }
}

TEST_CASE("Gaussian Hellinger vanishes on the diagonal and is positive off it") {
    const Gaussian p(0.0, 1.0);
    const Gaussian q(1.5, 2.0);

    CHECK(gaussian::hellinger(p, p).value == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(gaussian::hellinger(p, q).value > 0.0);

    const auto d_pq = gaussian::hellinger(p, q).value;
    const auto d_qp = gaussian::hellinger(q, p).value;
    CHECK(d_pq == doctest::Approx(d_qp).epsilon(1e-12));
}

TEST_CASE("Gaussian Hellinger matches the closed form from the appendix") {
    // d_H = sqrt(0.5 * [dmu^2 (s1^2 + s2^2) + (s1^2 - s2^2)^2] / (s1^2 s2^2))
    const std::vector<std::pair<Gaussian, Gaussian>> pairs = {
        {Gaussian(0.0, 1.0), Gaussian(1.0, 1.0)},
        {Gaussian(0.0, 1.0), Gaussian(0.0, 2.0)},
        {Gaussian(-2.0, 0.5), Gaussian(3.0, 1.5)},
    };

    for (const auto& pq : pairs) {
        const auto p = pq.first;
        const auto q = pq.second;

        const double dmu2 = (p.mu() - q.mu()) * (p.mu() - q.mu());
        const double s1sq = p.variance();
        const double s2sq = q.variance();
        const double expected = std::sqrt(
            0.5 * (dmu2 * (s1sq + s2sq) + (s1sq - s2sq) * (s1sq - s2sq))
            / (s1sq * s2sq));

        CHECK(rel_err(gaussian::hellinger(p, q).value, expected) < kTol);
    }
}

TEST_CASE("Gaussian distances are zero on the diagonal and positive off it") {
    const Gaussian p(0.0, 1.0);
    const Gaussian q(1.5, 2.0);

    CHECK(gaussian::hellinger(p, p).value    == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(gaussian::tangent_norm(p, p).value == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(gaussian::segment(p, p).value      == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(gaussian::euclidean(p, p).value    == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(gaussian::fisher_rao(p, p).value   == doctest::Approx(0.0).epsilon(1e-8));

    CHECK(gaussian::hellinger(p, q).value    > 0.0);
    CHECK(gaussian::tangent_norm(p, q).value > 0.0);
    CHECK(gaussian::segment(p, q).value      > 0.0);
    CHECK(gaussian::euclidean(p, q).value    > 0.0);
    CHECK(gaussian::fisher_rao(p, q).value   > 0.0);
}

TEST_CASE("Gaussian Fisher-Rao matches the closed form") {
    const std::vector<std::pair<Gaussian, Gaussian>> pairs = {
        {Gaussian(0.0, 1.0), Gaussian(1.0, 1.0)},
        {Gaussian(0.0, 1.0), Gaussian(0.0, 2.0)},
        {Gaussian(-2.0, 0.5), Gaussian(3.0, 1.5)},
        {Gaussian(5.0, 0.2), Gaussian(5.0, 5.0)},
    };

    for (const auto& pq : pairs) {
        const auto p = pq.first;
        const auto q = pq.second;

        const double dmu2  = (p.mu() - q.mu()) * (p.mu() - q.mu());
        const double dsig2 = (p.sigma() - q.sigma()) * (p.sigma() - q.sigma());
        const double arg   = 1.0
            + (dmu2 + 2.0 * dsig2) / (4.0 * p.sigma() * q.sigma());
        const double expected = std::sqrt(2.0) * std::acosh(arg);

        CHECK(rel_err(gaussian::fisher_rao(p, q).value, expected) < kTol);
    }
}

TEST_CASE("Gaussian Fisher-Rao is symmetric") {
    const std::vector<std::pair<Gaussian, Gaussian>> pairs = {
        {Gaussian(0.0, 1.0), Gaussian(2.0, 0.5)},
        {Gaussian(-1.0, 3.0), Gaussian(1.0, 3.0)},
        {Gaussian(10.0, 0.1), Gaussian(0.0, 5.0)},
    };

    for (const auto& pq : pairs) {
        const double d_pq = gaussian::fisher_rao(pq.first,  pq.second).value;
        const double d_qp = gaussian::fisher_rao(pq.second, pq.first).value;
        CHECK(rel_err(d_pq, d_qp) < 1e-12);
    }
}

TEST_CASE("Gaussian FR satisfies the triangle inequality") {
    const Gaussian a(0.0, 1.0);
    const Gaussian b(1.0, 2.0);
    const Gaussian c(3.0, 1.5);

    const double d_ac = gaussian::fisher_rao(a, c).value;
    const double d_ab = gaussian::fisher_rao(a, b).value;
    const double d_bc = gaussian::fisher_rao(b, c).value;

    CHECK(d_ac <= d_ab + d_bc + 1e-10);
}

TEST_CASE("Gaussian segment closed form matches quadrature") {
    const std::vector<std::pair<Gaussian, Gaussian>> pairs = {
        {Gaussian(0.0, 1.0), Gaussian(1.0, 2.0)},
        {Gaussian(-2.0, 0.5), Gaussian(3.0, 1.5)},
        {Gaussian(0.0, 1.0), Gaussian(0.0, 5.0)},
    };

    gaussian::SegmentOptions opts_cf;
    opts_cf.use_closed_form = true;

    gaussian::SegmentOptions opts_quad;
    opts_quad.use_closed_form = false;
    opts_quad.n_quad = 30;

    for (const auto& pq : pairs) {
        const double d_cf   = gaussian::segment(pq.first, pq.second, opts_cf).value;
        const double d_quad = gaussian::segment(pq.first, pq.second, opts_quad).value;
        CHECK(rel_err(d_cf, d_quad) < 1e-9);
    }
}

TEST_CASE("Gaussian criteria agree with FR on nearby points") {
    const Gaussian p(1.0, 2.0);
    const double eps = 1e-4;
    const Gaussian q(1.0 + eps, 2.0 + eps);

    const double d_fr = gaussian::fisher_rao(p, q).value;

    const double d_tan = gaussian::tangent_norm(p, q).value;
    CHECK(rel_err(d_tan, d_fr) < 1e-6);

    const double d_kl = gaussian::hellinger(p, q).value;
    CHECK(rel_err(d_kl, d_fr) < 1e-6);

    const double d_seg = gaussian::segment(p, q).value;
    CHECK(rel_err(d_seg, d_fr) < 1e-6);
}

TEST_CASE("Gaussian geodesic is the affine segment along constant mu") {
    const std::vector<std::pair<Gaussian, Gaussian>> pairs = {
        {Gaussian(0.5, 1.0), Gaussian(0.5, 3.0)},
        {Gaussian(-2.0, 0.5), Gaussian(-2.0, 0.8)},
        {Gaussian(10.0, 0.1), Gaussian(10.0, 5.0)},
    };

    gaussian::SegmentOptions opts_cf;
    opts_cf.use_closed_form = true;

    for (const auto& pq : pairs) {
        const double d_fr  = gaussian::fisher_rao(pq.first, pq.second).value;
        const double d_seg = gaussian::segment(pq.first, pq.second, opts_cf).value;
        CHECK(rel_err(d_seg, d_fr) < 1e-12);
    }
}

TEST_CASE("Gaussian geodesic is shorter than the affine segment along constant sigma") {
    const std::vector<std::pair<Gaussian, Gaussian>> pairs = {
        {Gaussian(0.0, 1.0), Gaussian(1.0, 1.0)},
        {Gaussian(-1.0, 2.0), Gaussian(1.0, 2.0)},
        {Gaussian(0.0, 5.0), Gaussian(3.0, 5.0)},
    };

    gaussian::SegmentOptions opts_cf;
    opts_cf.use_closed_form = true;

    for (const auto& pq : pairs) {
        const double d_fr  = gaussian::fisher_rao(pq.first, pq.second).value;
        const double d_seg = gaussian::segment(pq.first, pq.second, opts_cf).value;
        CHECK(d_seg > d_fr);
    }
}

TEST_CASE("Gaussian FR is invariant under simultaneous rescaling") {
    const double mu1 = 0.0;
    const double mu2 = 2.0;
    const double sigma_base = 1.0;

    const Gaussian a1(mu1, sigma_base);
    const Gaussian a2(mu2, sigma_base);

    for (double lambda : {0.5, 1.0, 2.0, 5.0}) {
        const Gaussian b1(mu1 * lambda, sigma_base * lambda);
        const Gaussian b2(mu2 * lambda, sigma_base * lambda);

        const double d_a = gaussian::fisher_rao(a1, a2).value;
        const double d_b = gaussian::fisher_rao(b1, b2).value;
        CHECK(rel_err(d_a, d_b) < 1e-12);
    }
}