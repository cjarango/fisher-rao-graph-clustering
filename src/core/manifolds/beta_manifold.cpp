#include "frgc/manifolds/beta_manifold.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

#include "frgc/manifolds/quadrature.hpp"

namespace frgc::manifolds::beta {

namespace {

using Clock = std::chrono::steady_clock;

inline std::uint64_t elapsed_nanos(Clock::time_point t0, Clock::time_point t1) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

namespace bmp = boost::math::policies;
using quiet_policy = bmp::policy<
    bmp::domain_error<bmp::ignore_error>,
    bmp::pole_error<bmp::ignore_error>,
    bmp::overflow_error<bmp::ignore_error>,
    bmp::underflow_error<bmp::ignore_error>,
    bmp::rounding_error<bmp::ignore_error>,
    bmp::evaluation_error<bmp::ignore_error>
>;

inline double nan_value() {
    return std::numeric_limits<double>::quiet_NaN();
}

inline double digamma(double x) {
    try {
        return boost::math::digamma(x, quiet_policy{});
    } catch (...) {
        return nan_value();
    }
}

inline double trigamma(double x) {
    try {
        return boost::math::trigamma(x, quiet_policy{});
    } catch (...) {
        return nan_value();
    }
}

inline double tetragamma(double x) {
    try {
        return boost::math::polygamma(2, x, quiet_policy{});
    } catch (...) {
        return nan_value();
    }
}

inline bool all_finite(const std::array<double, 4>& y) {
    return std::isfinite(y[0]) && std::isfinite(y[1])
        && std::isfinite(y[2]) && std::isfinite(y[3]);
}

struct BetaMetric {
    double g11, g12, g22;
    double p, q, r, s, u, v;
};

BetaMetric beta_metric_full(double mu, double phi) {
    const double a  = mu * phi;
    const double b  = (1.0 - mu) * phi;
    const double ab = a + b;

    const double tga  = trigamma(a);
    const double tgb  = trigamma(b);
    const double tgab = trigamma(ab);

    const double fga  = tetragamma(a);
    const double fgb  = tetragamma(b);
    const double fgab = tetragamma(ab);

    BetaMetric M;
    M.g11 = phi * phi * (tga + tgb);
    M.g22 = mu * mu * tga + (1.0 - mu) * (1.0 - mu) * tgb - tgab;
    M.g12 = phi * (mu * tga - (1.0 - mu) * tgb);

    M.p = phi * phi * phi * (fga - fgb);
    M.r = 2.0 * mu * tga + mu * mu * phi * fga
        - 2.0 * (1.0 - mu) * tgb - (1.0 - mu) * (1.0 - mu) * phi * fgb;
    M.u = phi * (tga + tgb + phi * (mu * fga + (1.0 - mu) * fgb));

    M.q = 2.0 * phi * (tga + tgb)
        + phi * phi * (mu * fga + (1.0 - mu) * fgb);
    M.s = mu * mu * mu * fga
        + (1.0 - mu) * (1.0 - mu) * (1.0 - mu) * fgb
        - fgab;
    M.v = mu * tga - (1.0 - mu) * tgb
        + phi * (mu * mu * fga - (1.0 - mu) * (1.0 - mu) * fgb);

    return M;
}

std::array<double, 4> geodesic_rhs(const std::array<double, 4>& y) {
    const double mu  = y[0];
    const double phi = y[1];
    const double v_mu  = y[2];
    const double v_phi = y[3];

    const BetaMetric M = beta_metric_full(mu, phi);

    const double det  = M.g11 * M.g22 - M.g12 * M.g12;
    const double gu11 =  M.g22 / det;
    const double gu22 =  M.g11 / det;
    const double gu12 = -M.g12 / det;

    const double G1_11 = 0.5 * (gu11 * M.p + gu12 * (2.0 * M.u - M.q));
    const double G1_12 = 0.5 * (gu11 * M.q + gu12 * M.r);
    const double G1_22 = 0.5 * (gu11 * (2.0 * M.v - M.r) + gu12 * M.s);
    const double G2_11 = 0.5 * (gu12 * M.p + gu22 * (2.0 * M.u - M.q));
    const double G2_12 = 0.5 * (gu12 * M.q + gu22 * M.r);
    const double G2_22 = 0.5 * (gu12 * (2.0 * M.v - M.r) + gu22 * M.s);

    const double acc_mu = -(G1_11 * v_mu * v_mu
                          + 2.0 * G1_12 * v_mu * v_phi
                          + G1_22 * v_phi * v_phi);
    const double acc_phi = -(G2_11 * v_mu * v_mu
                           + 2.0 * G2_12 * v_mu * v_phi
                           + G2_22 * v_phi * v_phi);

    return { v_mu, v_phi, acc_mu, acc_phi };
}

std::array<double, 4> rk4_integrate(std::array<double, 4> y,
                                    double t_end, int n_steps) {
    const double h = t_end / n_steps;
    for (int step = 0; step < n_steps; ++step) {
        const auto k1 = geodesic_rhs(y);

        std::array<double, 4> y_a;
        for (int j = 0; j < 4; ++j) y_a[j] = y[j] + 0.5 * h * k1[j];
        const auto k2 = geodesic_rhs(y_a);

        std::array<double, 4> y_b;
        for (int j = 0; j < 4; ++j) y_b[j] = y[j] + 0.5 * h * k2[j];
        const auto k3 = geodesic_rhs(y_b);

        std::array<double, 4> y_c;
        for (int j = 0; j < 4; ++j) y_c[j] = y[j] + h * k3[j];
        const auto k4 = geodesic_rhs(y_c);

        for (int j = 0; j < 4; ++j) {
            y[j] += (h / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        }

        if (!all_finite(y)) return y;
    }
    return y;
}

std::array<double, 2> solve_2x2(const std::array<std::array<double, 2>, 2>& J,
                                const std::array<double, 2>& b) {
    const double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    return {
        ( J[1][1] * b[0] - J[0][1] * b[1]) / det,
        (-J[1][0] * b[0] + J[0][0] * b[1]) / det
    };
}

struct ShootOutcome {
    std::array<double, 2> v0;
    bool   converged  = false;
    int    iterations = 0;
    double residual   = 0.0;
};

ShootOutcome shoot(const std::array<double, 2>& x_A,
                   const std::array<double, 2>& x_B,
                   std::array<double, 2> v0,
                   const ShootingOptions& opts) {
    ShootOutcome out;
    out.v0 = v0;

    const double scale = std::max(1.0,
        std::sqrt(x_B[0] * x_B[0] + x_B[1] * x_B[1]));

    for (int iter = 0; iter < opts.max_iter; ++iter) {
        std::array<double, 4> y0 = { x_A[0], x_A[1], v0[0], v0[1] };
        const auto y1 = rk4_integrate(y0, 1.0, opts.rk4_steps);

        out.iterations = iter + 1;

        if (!std::isfinite(y1[0]) || !std::isfinite(y1[1])) {
            out.converged = false;
            out.v0 = v0;
            return out;
        }

        const double r0 = y1[0] - x_B[0];
        const double r1 = y1[1] - x_B[1];
        const double res = std::sqrt(r0 * r0 + r1 * r1);

        out.residual = res;

        if (res / scale < opts.tol) {
            out.v0        = v0;
            out.converged = true;
            return out;
        }

        std::array<std::array<double, 2>, 2> J;
        bool jac_ok = true;
        for (int j = 0; j < 2; ++j) {
            auto v_pert = v0;
            v_pert[j] += opts.fd_epsilon;
            std::array<double, 4> y0_pert = { x_A[0], x_A[1], v_pert[0], v_pert[1] };
            const auto y1_pert = rk4_integrate(y0_pert, 1.0, opts.rk4_steps);
            if (!std::isfinite(y1_pert[0]) || !std::isfinite(y1_pert[1])) {
                jac_ok = false;
                break;
            }
            J[0][j] = (y1_pert[0] - y1[0]) / opts.fd_epsilon;
            J[1][j] = (y1_pert[1] - y1[1]) / opts.fd_epsilon;
        }

        if (!jac_ok) {
            out.converged = false;
            out.v0 = v0;
            return out;
        }

        const auto dv = solve_2x2(J, { -r0, -r1 });
        v0[0] += dv[0];
        v0[1] += dv[1];
    }

    out.v0 = v0;
    return out;
}

double segment_integrand(const Beta& p, const Beta& q, double t) {
    const double dmu  = q.mu()  - p.mu();
    const double dphi = q.phi() - p.phi();
    const double mu_t  = p.mu()  + t * dmu;
    const double phi_t = p.phi() + t * dphi;

    const BetaMetric M = beta_metric_full(mu_t, phi_t);
    return std::sqrt(M.g11 * dmu * dmu
                   + 2.0 * M.g12 * dmu * dphi
                   + M.g22 * dphi * dphi);
}

}  // namespace

DistanceResult hellinger(const Beta& p, const Beta& q) {
    const auto t0 = Clock::now();

    const double mu1  = p.mu(),  phi1 = p.phi();
    const double mu2  = q.mu(),  phi2 = q.phi();
    const double a1 = mu1 * phi1,          b1 = (1.0 - mu1) * phi1;
    const double a2 = mu2 * phi2,          b2 = (1.0 - mu2) * phi2;

    const auto kl = [](double a1, double b1, double a2, double b2) {
        const double logB_diff =
              (std::lgamma(a2) + std::lgamma(b2) - std::lgamma(a2 + b2))
            - (std::lgamma(a1) + std::lgamma(b1) - std::lgamma(a1 + b1));
        return logB_diff
             + (a1 - a2) * digamma(a1)
             + (b1 - b2) * digamma(b1)
             + (a2 - a1 + b2 - b1) * digamma(a1 + b1);
    };

    // D_sym = 0.5 * [KL(p||q) + KL(q||p)]
    // Hellinger = sqrt(2 * D_sym) = sqrt(KL(p||q) + KL(q||p))
    const double value = std::sqrt(kl(a1, b1, a2, b2) + kl(a2, b2, a1, b1));

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1), true, 0 };
}

DistanceResult tangent_norm(const Beta& p, const Beta& q) {
    const auto t0 = Clock::now();

    const double dmu  = q.mu()  - p.mu();
    const double dphi = q.phi() - p.phi();

    const BetaMetric Mp = beta_metric_full(p.mu(), p.phi());
    const double d_p = std::sqrt(Mp.g11 * dmu * dmu
                               + 2.0 * Mp.g12 * dmu * dphi
                               + Mp.g22 * dphi * dphi);

    const BetaMetric Mq = beta_metric_full(q.mu(), q.phi());
    const double d_q = std::sqrt(Mq.g11 * dmu * dmu
                               + 2.0 * Mq.g12 * dmu * dphi
                               + Mq.g22 * dphi * dphi);

    const double value = 0.5 * (d_p + d_q);

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1), true, 0 };
}

DistanceResult segment(const Beta& p, const Beta& q,
                       const SegmentOptions& opts) {
    const auto t0 = Clock::now();

    const double value = quadrature::integrate_gl(
        [&](double t) { return segment_integrand(p, q, t); },
        0.0, 1.0, opts.n_quad);

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1), true, 0 };
}

DistanceResult euclidean(const Beta& p, const Beta& q) {
    const auto t0 = Clock::now();

    const double dmu  = q.mu()  - p.mu();
    const double dphi = q.phi() - p.phi();
    const double value = std::sqrt(dmu * dmu + dphi * dphi);

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1), true, 0 };
}

DistanceResult fisher_rao(const Beta& p, const Beta& q,
                          const ShootingOptions& opts) {
    const auto t0 = Clock::now();

    const std::array<double, 2> x_A = { p.mu(), p.phi() };
    const std::array<double, 2> x_B = { q.mu(), q.phi() };

    std::array<double, 2> v0_guess;
    if (opts.has_seed
        && std::isfinite(opts.v0_seed[0])
        && std::isfinite(opts.v0_seed[1])) {
        v0_guess = opts.v0_seed;
    } else {
        v0_guess = { x_B[0] - x_A[0], x_B[1] - x_A[1] };
    }

    const auto outcome = shoot(x_A, x_B, v0_guess, opts);

    if (outcome.converged) {
        opts.last_v0    = outcome.v0;
        opts.seed_valid = true;
    } else {
        opts.seed_valid = false;
    }

    const BetaMetric M = beta_metric_full(x_A[0], x_A[1]);
    const double value = std::sqrt(
          M.g11 * outcome.v0[0] * outcome.v0[0]
        + 2.0 * M.g12 * outcome.v0[0] * outcome.v0[1]
        + M.g22 * outcome.v0[1] * outcome.v0[1]);

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1),
             outcome.converged, outcome.iterations };
}

std::array<std::array<double, 2>, 2> metric_tensor(const Beta& p) {
    const BetaMetric M = beta_metric_full(p.mu(), p.phi());
    return {{{ M.g11, M.g12 }, { M.g12, M.g22 }}};
}

}  // namespace frgc::manifolds::beta