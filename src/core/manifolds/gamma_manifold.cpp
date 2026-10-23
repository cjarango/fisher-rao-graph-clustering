#include "frgc/manifolds/gamma_manifold.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

#include "frgc/manifolds/quadrature.hpp"

namespace frgc::manifolds::gamma {

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

std::array<double, 4> geodesic_rhs(const std::array<double, 4>& y) {
    const double gamma = y[0];
    const double k     = y[1];
    const double v_g   = y[2];
    const double v_k   = y[3];

    const double g22 = trigamma(k) - 1.0 / k;

    const double Gamma1_11 = -1.0 / gamma;
    const double Gamma1_12 =  1.0 / (2.0 * k);
    const double Gamma2_11 = -1.0 / (2.0 * gamma * gamma * g22);
    const double Gamma2_22 = (tetragamma(k) + 1.0 / (k * k)) / (2.0 * g22);

    const double acc_g = -(
        Gamma1_11 * v_g * v_g +
        2.0 * Gamma1_12 * v_g * v_k);

    const double acc_k = -(
        Gamma2_11 * v_g * v_g +
        Gamma2_22 * v_k * v_k);

    return { v_g, v_k, acc_g, acc_k };
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

double segment_integrand(const Gamma& p, const Gamma& q, double t) {
    const double dg = q.gamma() - p.gamma();
    const double dk = q.k()     - p.k();
    const double gamma_t = p.gamma() + t * dg;
    const double k_t     = p.k()     + t * dk;

    const double g11 = k_t / (gamma_t * gamma_t);
    const double g22 = trigamma(k_t) - 1.0 / k_t;

    return std::sqrt(dg * dg * g11 + dk * dk * g22);
}

}  // namespace

DistanceResult hellinger(const Gamma& p, const Gamma& q) {
    const auto t0 = Clock::now();

    const double g1 = p.gamma();
    const double k1 = p.k();
    const double g2 = q.gamma();
    const double k2 = q.k();

    const auto kl = [](double g1, double k1, double g2, double k2) {
        return k2 * std::log((k1 * g2) / (k2 * g1))
             + std::lgamma(k2) - std::lgamma(k1)
             + (k1 - k2) * digamma(k1)
             - k1
             + k2 * g1 / g2;
    };

    // D_sym = 0.5 * [KL(p||q) + KL(q||p)]
    // Hellinger = sqrt(2 * D_sym) = sqrt(KL(p||q) + KL(q||p))
    const double value = std::sqrt(kl(g1, k1, g2, k2) + kl(g2, k2, g1, k1));

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1), true, 0 };
}

DistanceResult tangent_norm(const Gamma& p, const Gamma& q) {
    const auto t0 = Clock::now();

    const double dg = q.gamma() - p.gamma();
    const double dk = q.k()     - p.k();

    const double g11_p = p.k() / (p.gamma() * p.gamma());
    const double g22_p = trigamma(p.k()) - 1.0 / p.k();
    const double d_p = std::sqrt(dg * dg * g11_p + dk * dk * g22_p);

    const double g11_q = q.k() / (q.gamma() * q.gamma());
    const double g22_q = trigamma(q.k()) - 1.0 / q.k();
    const double d_q = std::sqrt(dg * dg * g11_q + dk * dk * g22_q);

    const double value = 0.5 * (d_p + d_q);

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1), true, 0 };
}

DistanceResult segment(const Gamma& p, const Gamma& q,
                       const SegmentOptions& opts) {
    const auto t0 = Clock::now();

    const double value = quadrature::integrate_gl(
        [&](double t) { return segment_integrand(p, q, t); },
        0.0, 1.0, opts.n_quad);

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1), true, 0 };
}

DistanceResult euclidean(const Gamma& p, const Gamma& q) {
    const auto t0 = Clock::now();

    const double dg = q.gamma() - p.gamma();
    const double dk = q.k()     - p.k();
    const double value = std::sqrt(dg * dg + dk * dk);

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1), true, 0 };
}

DistanceResult fisher_rao(const Gamma& p, const Gamma& q,
                          const ShootingOptions& opts) {
    const auto t0 = Clock::now();

    const std::array<double, 2> x_A = { p.gamma(), p.k() };
    const std::array<double, 2> x_B = { q.gamma(), q.k() };

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

    const double g11 = x_A[1] / (x_A[0] * x_A[0]);
    const double g22 = trigamma(x_A[1]) - 1.0 / x_A[1];
    const double value = std::sqrt(
        outcome.v0[0] * outcome.v0[0] * g11 +
        outcome.v0[1] * outcome.v0[1] * g22);

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1),
             outcome.converged, outcome.iterations };
}

std::array<std::array<double, 2>, 2> metric_tensor(const Gamma& p) {
    const double g = p.gamma();
    const double k = p.k();
    const double g11 = k / (g * g);
    const double g22 = trigamma(k) - 1.0 / k;
    return {{{ g11, 0.0 }, { 0.0, g22 }}};
}

}  // namespace frgc::manifolds::gamma