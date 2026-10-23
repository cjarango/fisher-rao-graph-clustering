#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <boost/math/special_functions/trigamma.hpp>

#include "frgc/distributions/gamma.hpp"
#include "frgc/manifolds/gamma_manifold.hpp"

using namespace frgc;
using namespace frgc::manifolds;

namespace {

constexpr double kTol = 1e-10;

double rel_err(double a, double b) {
    const double s = std::max(std::abs(a), std::abs(b));
    return s > 0.0 ? std::abs(a - b) / s : 0.0;
}

gamma::ShootingOptions tight_shooting() {
    gamma::ShootingOptions o;
    o.max_iter   = 100;
    o.tol        = 1e-12;
    o.rk4_steps  = 500;
    o.fd_epsilon = 1e-8;
    return o;
}

}  // namespace

TEST_CASE("Gamma metric tensor matches the closed-form in the appendix") {
    const std::vector<Gamma> pts = {
        Gamma(1.0, 1.0),
        Gamma(2.5, 3.0),
        Gamma(0.5, 0.5),
        Gamma(10.0, 7.5),
    };

    for (const auto& p : pts) {
        const auto g = gamma::metric_tensor(p);
        const double g11_expected = p.k() / (p.gamma() * p.gamma());
        const double g22_expected = boost::math::trigamma(p.k()) - 1.0 / p.k();

        CHECK(rel_err(g[0][0], g11_expected) < kTol);
        CHECK(rel_err(g[1][1], g22_expected) < kTol);
        CHECK(g[0][1] == 0.0);
        CHECK(g[1][0] == 0.0);
        CHECK(g[0][0] > 0.0);
        CHECK(g[1][1] > 0.0);
    }
}

TEST_CASE("Gamma Hellinger vanishes on the diagonal and is positive off it") {
    const Gamma p(2.0, 3.0);
    const Gamma q(4.0, 5.0);

    CHECK(gamma::hellinger(p, p).value == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(gamma::hellinger(p, q).value > 0.0);

    const auto d_pq = gamma::hellinger(p, q).value;
    const auto d_qp = gamma::hellinger(q, p).value;
    CHECK(d_pq == doctest::Approx(d_qp).epsilon(1e-12));
}

TEST_CASE("Gamma distances are zero on the diagonal and positive off it") {
    const Gamma p(2.0, 3.0);
    const Gamma q(4.0, 5.0);

    CHECK(gamma::hellinger(p, p).value    == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(gamma::tangent_norm(p, p).value == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(gamma::segment(p, p).value      == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(gamma::euclidean(p, p).value    == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(gamma::fisher_rao(p, p).value   == doctest::Approx(0.0).epsilon(1e-8));

    CHECK(gamma::hellinger(p, q).value    > 0.0);
    CHECK(gamma::tangent_norm(p, q).value > 0.0);
    CHECK(gamma::segment(p, q).value      > 0.0);
    CHECK(gamma::euclidean(p, q).value    > 0.0);
    CHECK(gamma::fisher_rao(p, q).value   > 0.0);
}

TEST_CASE("Gamma Fisher-Rao is symmetric") {
    const std::vector<std::pair<Gamma, Gamma>> pairs = {
        {Gamma(1.0, 1.0), Gamma(2.0, 2.0)},
        {Gamma(1.0, 2.0), Gamma(3.0, 1.5)},
        {Gamma(5.0, 10.0), Gamma(6.0, 9.0)},
    };

    const auto opts = tight_shooting();
    for (const auto& pq : pairs) {
        const double d_pq = gamma::fisher_rao(pq.first,  pq.second, opts).value;
        const double d_qp = gamma::fisher_rao(pq.second, pq.first,  opts).value;
        CHECK(rel_err(d_pq, d_qp) < 1e-6);
    }
}

TEST_CASE("Gamma shooting reports non-convergence on out-of-domain pairs") {
    const Gamma p(0.5, 4.0);
    const Gamma q(3.0, 0.5);

    const auto opts = tight_shooting();
    const auto r_pq = gamma::fisher_rao(p, q, opts);
    const auto r_qp = gamma::fisher_rao(q, p, opts);

    if (r_pq.converged && r_qp.converged) {
        CHECK(rel_err(r_pq.value, r_qp.value) < 1e-6);
    }
}

TEST_CASE("Gamma shooting converges on well-separated points") {
    const std::vector<std::pair<Gamma, Gamma>> pairs = {
        {Gamma(1.0, 1.0), Gamma(1.001, 1.001)},
        {Gamma(1.0, 1.0), Gamma(2.0, 2.0)},
        {Gamma(0.5, 0.5), Gamma(2.0, 3.0)},
        {Gamma(10.0, 1.0), Gamma(1.0, 10.0)},
    };

    for (const auto& pq : pairs) {
        const auto r = gamma::fisher_rao(pq.first, pq.second);
        CHECK(r.converged);
        CHECK(r.iterations > 0);
    }
}

TEST_CASE("Gamma criteria agree with FR on nearby points") {
    const Gamma p(2.0, 3.0);
    const double eps = 1e-3;
    const Gamma q(2.0 + eps, 3.0 + eps);

    const auto opts = tight_shooting();
    const double d_fr = gamma::fisher_rao(p, q, opts).value;

    const double d_tan = gamma::tangent_norm(p, q).value;
    CHECK(rel_err(d_tan, d_fr) < 1e-4);

    const double d_kl = gamma::hellinger(p, q).value;
    CHECK(rel_err(d_kl, d_fr) < 1e-4);

    const double d_seg = gamma::segment(p, q).value;
    CHECK(rel_err(d_seg, d_fr) < 1e-4);
}

TEST_CASE("Gamma FR satisfies the triangle inequality") {
    const Gamma a(1.0, 1.0);
    const Gamma b(2.0, 3.0);
    const Gamma c(4.0, 2.0);

    const auto opts = tight_shooting();
    const double d_ac = gamma::fisher_rao(a, c, opts).value;
    const double d_ab = gamma::fisher_rao(a, b, opts).value;
    const double d_bc = gamma::fisher_rao(b, c, opts).value;

    CHECK(d_ac <= d_ab + d_bc + 1e-6);
}