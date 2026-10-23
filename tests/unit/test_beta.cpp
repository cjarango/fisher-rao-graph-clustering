#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <boost/math/special_functions/trigamma.hpp>

#include "frgc/distributions/beta.hpp"
#include "frgc/manifolds/beta_manifold.hpp"

using namespace frgc;
using namespace frgc::manifolds;

namespace {

constexpr double kTol = 1e-10;

double rel_err(double a, double b) {
    const double s = std::max(std::abs(a), std::abs(b));
    return s > 0.0 ? std::abs(a - b) / s : 0.0;
}

beta::ShootingOptions tight_shooting() {
    beta::ShootingOptions o;
    o.max_iter   = 100;
    o.tol        = 1e-12;
    o.rk4_steps  = 500;
    o.fd_epsilon = 1e-8;
    return o;
}

}  // namespace

TEST_CASE("Beta metric tensor at symmetric point mu=0.5, phi=2") {
    const Beta p(0.5, 2.0);
    const auto g = beta::metric_tensor(p);

    const double pi2_6 = M_PI * M_PI / 6.0;
    const double g11_exp = 8.0 * pi2_6;
    const double g22_exp = 0.5 * pi2_6 - (pi2_6 - 1.0);

    CHECK(rel_err(g[0][0], g11_exp) < kTol);
    CHECK(rel_err(g[1][1], g22_exp) < kTol);
    CHECK(std::abs(g[0][1]) < kTol);
    CHECK(g[0][0] > 0.0);
    CHECK(g[1][1] > 0.0);
}

TEST_CASE("Beta metric tensor matches the appendix at generic points") {
    const std::vector<Beta> pts = {
        Beta(0.3, 2.0),
        Beta(0.7, 5.0),
        Beta(0.5, 1.0),
        Beta(0.2, 10.0),
    };

    for (const auto& p : pts) {
        const double mu  = p.mu();
        const double phi = p.phi();
        const double a   = mu * phi;
        const double b   = (1.0 - mu) * phi;
        const double ab  = a + b;

        const double tga  = boost::math::trigamma(a);
        const double tgb  = boost::math::trigamma(b);
        const double tgab = boost::math::trigamma(ab);

        const double g11_exp = phi * phi * (tga + tgb);
        const double g22_exp = mu * mu * tga
                             + (1.0 - mu) * (1.0 - mu) * tgb
                             - tgab;
        const double g12_exp = phi * (mu * tga - (1.0 - mu) * tgb);

        const auto g = beta::metric_tensor(p);
        CHECK(rel_err(g[0][0], g11_exp) < kTol);
        CHECK(rel_err(g[1][1], g22_exp) < kTol);
        CHECK(rel_err(g[0][1], g12_exp) < kTol);
        CHECK(rel_err(g[1][0], g12_exp) < kTol);
    }
}

TEST_CASE("Beta Hellinger vanishes on the diagonal and is positive off it") {
    const Beta p(0.3, 5.0);
    const Beta q(0.6, 8.0);

    CHECK(beta::hellinger(p, p).value == doctest::Approx(0.0).epsilon(1e-12));

    const auto d_pq = beta::hellinger(p, q).value;
    const auto d_qp = beta::hellinger(q, p).value;
    CHECK(d_pq > 0.0);
    CHECK(d_pq == doctest::Approx(d_qp).epsilon(1e-12));
}

TEST_CASE("Beta distances are zero on the diagonal") {
    const Beta p(0.4, 3.0);

    CHECK(beta::hellinger(p, p).value    == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(beta::tangent_norm(p, p).value == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(beta::segment(p, p).value      == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(beta::euclidean(p, p).value    == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(beta::fisher_rao(p, p).value   == doctest::Approx(0.0).epsilon(1e-8));
}

TEST_CASE("Beta Fisher-Rao is symmetric") {
    const std::vector<std::pair<Beta, Beta>> pairs = {
        {Beta(0.3, 2.0), Beta(0.7, 4.0)},
        {Beta(0.5, 1.0), Beta(0.5, 10.0)},
        {Beta(0.2, 5.0), Beta(0.8, 5.0)},
    };

    const auto opts = tight_shooting();
    for (const auto& pq : pairs) {
        const double d_pq = beta::fisher_rao(pq.first,  pq.second, opts).value;
        const double d_qp = beta::fisher_rao(pq.second, pq.first,  opts).value;
        CHECK(rel_err(d_pq, d_qp) < 1e-6);
    }
}

TEST_CASE("Beta shooting converges on nearby points") {
    const std::vector<std::pair<Beta, Beta>> pairs = {
        {Beta(0.5, 2.0), Beta(0.501, 2.001)},
        {Beta(0.3, 5.0), Beta(0.31, 5.05)},
        {Beta(0.5, 1.0), Beta(0.55, 1.2)},
    };

    for (const auto& pq : pairs) {
        const auto r = beta::fisher_rao(pq.first, pq.second);
        CHECK(r.converged);
        CHECK(r.iterations > 0);
    }
}

TEST_CASE("Beta criteria agree with FR on nearby points") {
    const Beta p(0.4, 3.0);
    const double eps = 1e-3;
    const Beta q(0.4 + eps, 3.0 + eps);

    const auto opts = tight_shooting();
    const double d_fr = beta::fisher_rao(p, q, opts).value;

    const double d_tan = beta::tangent_norm(p, q).value;
    CHECK(rel_err(d_tan, d_fr) < 1e-4);

    const double d_kl = beta::hellinger(p, q).value;
    CHECK(rel_err(d_kl, d_fr) < 1e-4);

    const double d_seg = beta::segment(p, q).value;
    CHECK(rel_err(d_seg, d_fr) < 1e-4);
}

TEST_CASE("Beta FR satisfies the triangle inequality") {
    const Beta a(0.3, 2.0);
    const Beta b(0.5, 5.0);
    const Beta c(0.7, 3.0);

    const auto opts = tight_shooting();
    const double d_ac = beta::fisher_rao(a, c, opts).value;
    const double d_ab = beta::fisher_rao(a, b, opts).value;
    const double d_bc = beta::fisher_rao(b, c, opts).value;

    CHECK(d_ac <= d_ab + d_bc + 1e-6);
}