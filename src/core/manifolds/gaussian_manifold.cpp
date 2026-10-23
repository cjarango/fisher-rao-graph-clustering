#include "frgc/manifolds/gaussian_manifold.hpp"

#include <chrono>
#include <cmath>

#include "frgc/manifolds/quadrature.hpp"

namespace frgc::manifolds::gaussian {

namespace {

using Clock = std::chrono::steady_clock;

inline std::uint64_t elapsed_nanos(Clock::time_point t0, Clock::time_point t1) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

double segment_integrand(const Gaussian& p, const Gaussian& q, double t) {
    const double dmu  = q.mu()    - p.mu();
    const double dsig = q.sigma() - p.sigma();
    const double sigma_t = p.sigma() + t * dsig;
    return std::sqrt((dmu * dmu + 2.0 * dsig * dsig) / (sigma_t * sigma_t));
}

}  // namespace

DistanceResult hellinger(const Gaussian& p, const Gaussian& q) {
    const auto t0 = Clock::now();

    const double dmu2 = (p.mu() - q.mu()) * (p.mu() - q.mu());
    const double s1sq = p.variance();
    const double s2sq = q.variance();
    const double num  = dmu2 * (s1sq + s2sq) + (s1sq - s2sq) * (s1sq - s2sq);

    // D_sym = 0.25 * num / (s1sq * s2sq)
    // Hellinger = sqrt(2 * D_sym) = sqrt(0.5 * num / (s1sq * s2sq))
    const double value = std::sqrt(0.5 * num / (s1sq * s2sq));

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1), true, 0 };
}

DistanceResult tangent_norm(const Gaussian& p, const Gaussian& q) {
    const auto t0 = Clock::now();

    const double dmu  = q.mu()    - p.mu();
    const double dsig = q.sigma() - p.sigma();
    const double num  = dmu * dmu + 2.0 * dsig * dsig;

    const double d_p = std::sqrt(num) / p.sigma();
    const double d_q = std::sqrt(num) / q.sigma();

    const double value = 0.5 * (d_p + d_q);

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1), true, 0 };
}

DistanceResult segment(const Gaussian& p, const Gaussian& q,
                       const SegmentOptions& opts) {
    const auto t0 = Clock::now();

    const double dmu  = q.mu()    - p.mu();
    const double dsig = q.sigma() - p.sigma();
    const double num  = std::sqrt(dmu * dmu + 2.0 * dsig * dsig);

    double value;
    if (opts.use_closed_form) {
        if (std::abs(dsig) < 1e-8 * p.sigma()) {
            value = num / p.sigma();
        } else {
            value = num * std::log(q.sigma() / p.sigma()) / dsig;
        }
    } else {
        value = quadrature::integrate_gl(
            [&](double t) { return segment_integrand(p, q, t); },
            0.0, 1.0, opts.n_quad);
    }

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1), true, 0 };
}

DistanceResult euclidean(const Gaussian& p, const Gaussian& q) {
    const auto t0 = Clock::now();

    const double dmu  = q.mu()    - p.mu();
    const double dsig = q.sigma() - p.sigma();
    const double value = std::sqrt(dmu * dmu + dsig * dsig);

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1), true, 0 };
}

DistanceResult fisher_rao(const Gaussian& p, const Gaussian& q) {
    const auto t0 = Clock::now();

    const double dmu2  = (p.mu()    - q.mu())    * (p.mu()    - q.mu());
    const double dsig2 = (p.sigma() - q.sigma()) * (p.sigma() - q.sigma());
    const double arg   = 1.0 + (dmu2 + 2.0 * dsig2) / (4.0 * p.sigma() * q.sigma());
    const double value = std::sqrt(2.0) * std::acosh(arg);

    const auto t1 = Clock::now();
    return { value, elapsed_nanos(t0, t1), true, 0 };
}

std::array<std::array<double, 2>, 2> metric_tensor(const Gaussian& p) {
    const double s2 = p.variance();
    return {{{ 1.0 / s2, 0.0 },
             { 0.0, 2.0 / s2 }}};
}

}  // namespace frgc::manifolds::gaussian