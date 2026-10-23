#include "frgc/experiment/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#include "frgc/distributions/beta.hpp"
#include "frgc/distributions/gamma.hpp"
#include "frgc/distributions/gaussian.hpp"
#include "frgc/manifolds/beta_manifold.hpp"
#include "frgc/manifolds/gamma_manifold.hpp"
#include "frgc/manifolds/gaussian_manifold.hpp"

namespace frgc::experiment {
namespace {

using manifolds::DistanceResult;

constexpr double kPi = 3.14159265358979323846;

struct BoxBounds {
    double a_lo, a_hi, b_lo, b_hi;
};

BoxBounds full_bounds(const FamilyBox& box) {
    return {box.a_min, box.a_max, box.b_min, box.b_max};
}

BoxBounds safe_bounds(const FamilyBox& box) {
    const double da = (box.a_max - box.a_min) * box.margin_frac;
    const double db = (box.b_max - box.b_min) * box.margin_frac;
    return {box.a_min + da, box.a_max - da,
            box.b_min + db, box.b_max - db};
}

std::uint64_t splitmix64(std::uint64_t& x) {
    x += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

double u01(std::uint64_t& state) {
    return (splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

double uniform(std::uint64_t& state, double a, double b) {
    return a + (b - a) * u01(state);
}

double standard_normal(std::uint64_t& state) {
    const double u1 = std::max(u01(state), 1e-300);
    const double u2 = u01(state);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
}

// Fisher-Rao distance. Returns NaN if the shooting solver fails to
// converge or if the result is not finite. NaN-safe downstream checks
// treat NaN comparisons as "reject".
double fr_distance(Family fam, const Point2D& x, const Point2D& y) {
    DistanceResult r;
    switch (fam) {
    case Family::Gaussian:
        r = manifolds::gaussian::fisher_rao(Gaussian(x[0], x[1]),
                                             Gaussian(y[0], y[1]));
        break;
    case Family::Gamma:
        r = manifolds::gamma::fisher_rao(Gamma(x[0], x[1]),
                                          Gamma(y[0], y[1]));
        break;
    case Family::Beta:
        r = manifolds::beta::fisher_rao(Beta(x[0], x[1]),
                                         Beta(y[0], y[1]));
        break;
    default:
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!r.converged || !std::isfinite(r.value) || r.value < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return r.value;
}

// Hellinger distance sqrt(2 * D_sym), closed form for the three families.
// Approximates the Fisher-Rao distance up to third order. Used as a
// fallback when the shooting fails.
double hellinger(Family fam, const Point2D& x, const Point2D& y) {
    DistanceResult r;
    switch (fam) {
    case Family::Gaussian:
        r = manifolds::gaussian::hellinger(Gaussian(x[0], x[1]),
                                            Gaussian(y[0], y[1]));
        break;
    case Family::Gamma:
        r = manifolds::gamma::hellinger(Gamma(x[0], x[1]),
                                         Gamma(y[0], y[1]));
        break;
    case Family::Beta:
        r = manifolds::beta::hellinger(Beta(x[0], x[1]),
                                        Beta(y[0], y[1]));
        break;
    default:
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!std::isfinite(r.value) || r.value < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return r.value;
}

// Single-shot placement: k foci, each rejected if its Fisher-Rao
// distance to any already-placed focus is below the corresponding
// radius r_i. Shooting failures (NaN distance) are treated as rejection.
std::vector<Point2D> place_foci_once(Family fam,
                                     const BoxBounds& safe,
                                     const std::vector<double>& radii,
                                     int k, std::uint64_t& rng) {
    std::vector<Point2D> foci;
    foci.reserve(k);

    for (int j = 0; j < k; ++j) {
        Point2D chosen{0.0, 0.0};
        double best_min_dist = -1.0;
        Point2D best_cand{0.0, 0.0};
        bool placed = false;
        constexpr int kMaxAttempts = 2000;

        for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
            const Point2D cand = {
                uniform(rng, safe.a_lo, safe.a_hi),
                uniform(rng, safe.b_lo, safe.b_hi),
            };

            double min_dist = std::numeric_limits<double>::infinity();
            bool ok = true;
            for (std::size_t i = 0; i < foci.size(); ++i) {
                const double d = fr_distance(fam, cand, foci[i]);
                // NaN-safe: any non-finite d is treated as a rejection.
                if (!(d >= radii[i])) { ok = false; break; }
                min_dist = std::min(min_dist, d);
            }
            if (foci.empty()) min_dist = std::numeric_limits<double>::infinity();

            if (ok) { chosen = cand; placed = true; break; }
            if (min_dist > best_min_dist) {
                best_min_dist = min_dist;
                best_cand = cand;
            }
        }
        if (!placed) chosen = best_cand;
        foci.push_back(chosen);
    }
    return foci;
}

// Wraps place_foci_once with a global balance check: the ratio between
// the largest and smallest pairwise FR distances must be below
// kBalanceRatio. Retries the whole configuration if not.
std::vector<Point2D> place_foci(Family fam,
                                const BoxBounds& safe,
                                const std::vector<double>& radii,
                                int k, std::uint64_t& rng) {
    if (k < 2) return place_foci_once(fam, safe, radii, k, rng);

    constexpr int    kMaxConfigAttempts = 20;
    constexpr double kBalanceRatio      = 2.0;

    std::vector<Point2D> best_foci;
    double best_ratio = std::numeric_limits<double>::infinity();

    for (int cfg = 0; cfg < kMaxConfigAttempts; ++cfg) {
        auto foci = place_foci_once(fam, safe, radii, k, rng);

        double d_min = std::numeric_limits<double>::infinity();
        double d_max = 0.0;
        bool all_finite = true;
        for (std::size_t i = 0; i < foci.size() && all_finite; ++i) {
            for (std::size_t j = i + 1; j < foci.size(); ++j) {
                const double d = fr_distance(fam, foci[i], foci[j]);
                if (!std::isfinite(d)) { all_finite = false; break; }
                d_min = std::min(d_min, d);
                d_max = std::max(d_max, d);
            }
        }
        if (!all_finite) continue;

        const double ratio = d_max / d_min;
        if (ratio <= kBalanceRatio) return foci;
        if (ratio < best_ratio) {
            best_ratio = ratio;
            best_foci  = foci;
        }
    }

    std::fprintf(stderr,
        "[sampling] balanced foci fallback: best ratio %.2f\n",
        best_ratio);
    return best_foci;
}

// Build a 2x2 SPD covariance from four uniform draws. Rejects until
// det > 0. Returns the Cholesky factor L such that L L^T = Sigma.
void sample_covariance(std::uint64_t& rng, double s_min, double s_max,
                       double& L00, double& L01, double& L10, double& L11) {
    for (int attempt = 0; attempt < 100; ++attempt) {
        const double s1 = uniform(rng, s_min, s_max);
        const double s2 = uniform(rng, s_min, s_max);
        const double s3 = uniform(rng, s_min, s_max);
        const double s4 = uniform(rng, s_min, s_max);

        const double a11 = s1 * s1;
        const double a22 = s2 * s2;
        const double a12 = s3 * s4;

        const double det = a11 * a22 - a12 * a12;
        if (det <= 0.0) continue;

        L00 = std::sqrt(a11);
        L10 = a12 / L00;
        L11 = std::sqrt(det / a11);
        L01 = 0.0;
        return;
    }
    const double s = 0.5 * (s_min + s_max);
    L00 = s; L01 = 0.0; L10 = 0.0; L11 = s;
}

// Two-Gaussian mixture: a narrow core plus a wide halo. Halo weight and
// scale are passed in, sampled per focus.
void sample_blob(const Point2D& focus,
                 double L00, double L01, double L10, double L11,
                 double w_halo, double k_halo,
                 std::size_t N, std::uint64_t& rng,
                 std::vector<Point2D>& out, const BoxBounds& full) {
    const std::size_t max_attempts = 1000 * N + 1000;
    std::size_t attempts = 0;
    std::size_t accepted = 0;

    while (accepted < N && attempts < max_attempts) {
        ++attempts;

        const double scale = (u01(rng) < w_halo) ? k_halo : 1.0;

        const double z0 = standard_normal(rng) * scale;
        const double z1 = standard_normal(rng) * scale;

        const double x = focus[0] + L00 * z0 + L01 * z1;
        const double y = focus[1] + L10 * z0 + L11 * z1;

        if (x >= full.a_lo && x <= full.a_hi
            && y >= full.b_lo && y <= full.b_hi) {
            out.push_back({x, y});
            ++accepted;
        }
    }
    if (accepted < N) {
        std::fprintf(stderr,
            "[sampling] blob truncated: accepted %zu of %zu\n",
            accepted, N);
    }
}

std::vector<std::size_t> distribute_counts(std::size_t N,
                                           const std::vector<double>& props,
                                           std::uint64_t& rng) {
    const std::size_t k = props.size();
    std::vector<std::size_t> counts(k);
    std::size_t sum = 0;
    for (std::size_t j = 0; j < k; ++j) {
        const double mean = props[j] * static_cast<double>(N);
        const double sigma = 0.05 * mean;
        double v = mean + sigma * standard_normal(rng);
        long long c = static_cast<long long>(std::llround(v));
        if (c < 1) c = 1;
        counts[j] = static_cast<std::size_t>(c);
        sum += counts[j];
    }
    while (sum > N) {
        std::size_t idx = 0;
        for (std::size_t j = 1; j < k; ++j)
            if (counts[j] > counts[idx]) idx = j;
        if (counts[idx] <= 1) break;
        --counts[idx]; --sum;
    }
    while (sum < N) {
        std::size_t idx = 0;
        for (std::size_t j = 1; j < k; ++j)
            if (counts[j] < counts[idx]) idx = j;
        ++counts[idx]; ++sum;
    }
    return counts;
}

constexpr std::uint64_t TAG_FOCI   = 0xA1;
constexpr std::uint64_t TAG_COUNTS = 0xA2;
constexpr std::uint64_t TAG_RADII  = 0xA3;
constexpr std::uint64_t TAG_COV    = 0xA4;
constexpr std::uint64_t TAG_UNIF   = 0xA5;
constexpr std::uint64_t TAG_HALO   = 0xA6;

}  // namespace

FamilyScale calibrate_family_scale(Family fam) {
    static thread_local std::array<FamilyScale, 3> cache{};
    static thread_local std::array<bool, 3> cached = {false, false, false};

    const std::size_t idx = static_cast<std::size_t>(fam);
    if (cached[idx]) return cache[idx];

    const FamilyBox box = get_family_box(fam);
    const BoxBounds safe = safe_bounds(box);
    const double a_mid = 0.5 * (safe.a_lo + safe.a_hi);
    const double b_mid = 0.5 * (safe.b_lo + safe.b_hi);

    const Point2D p_a0 = {safe.a_lo, b_mid};
    const Point2D p_a1 = {safe.a_hi, b_mid};
    const Point2D p_b0 = {a_mid, safe.b_lo};
    const Point2D p_b1 = {a_mid, safe.b_hi};

    double Wa = fr_distance(fam, p_a0, p_a1);
    double Wb = fr_distance(fam, p_b0, p_b1);

    // Hellinger fallback: same scale as FR, third-order approximation,
    // no shooting required. Only used when the geodesic shooting fails.
    if (!std::isfinite(Wa)) Wa = hellinger(fam, p_a0, p_a1);
    if (!std::isfinite(Wb)) Wb = hellinger(fam, p_b0, p_b1);

    // Last-resort fallback: chart-side length. Only reached if both FR
    // and Hellinger fail, which should not happen for the families in use.
    if (!std::isfinite(Wa) && !std::isfinite(Wb)) {
        Wa = safe.a_hi - safe.a_lo;
        Wb = safe.b_hi - safe.b_lo;
    } else if (!std::isfinite(Wa)) {
        Wa = Wb;
    } else if (!std::isfinite(Wb)) {
        Wb = Wa;
    }

    FamilyScale out;
    out.W = std::max(Wa, Wb);

    cache[idx] = out;
    cached[idx] = true;
    return out;
}

std::uint64_t replicate_seed(std::uint64_t base_seed, Family fam,
                             Regime reg, int replicate) {
    std::uint64_t s = base_seed;
    s ^= (static_cast<std::uint64_t>(fam) << 48);
    s ^= (static_cast<std::uint64_t>(reg) << 40);
    s ^= (static_cast<std::uint64_t>(replicate) << 16);
    return splitmix64(s);
}

SampleResult generate_sample(Family fam, Regime reg, std::size_t N,
                             const SampleOptions& opts,
                             std::uint64_t seed) {
    const FamilyBox box = get_family_box(fam);
    const BoxBounds full = full_bounds(box);
    const BoxBounds safe = safe_bounds(box);

    SampleResult result;

    if (reg == Regime::U) {
        result.points.reserve(N);
        std::uint64_t rng = seed ^ TAG_UNIF;
        for (std::size_t i = 0; i < N; ++i) {
            result.points.push_back({
                uniform(rng, full.a_lo, full.a_hi),
                uniform(rng, full.b_lo, full.b_hi),
            });
        }
        return result;
    }

    const FamilyScale scale = calibrate_family_scale(fam);

    const int k = static_cast<int>(opts.proportions.size());

    const bool is_moderate = (reg == Regime::M2 || reg == Regime::M3);
    const double r_lo = is_moderate ? 0.20 * scale.W : 0.40 * scale.W;
    const double r_hi = is_moderate ? 0.41 * scale.W : 0.51 * scale.W;

    std::uint64_t r_rng = seed ^ TAG_RADII;
    result.radii.reserve(k);
    for (int j = 0; j < k; ++j)
        result.radii.push_back(uniform(r_rng, r_lo, r_hi));

    std::uint64_t foci_rng = seed ^ TAG_FOCI;
    result.foci = place_foci(fam, safe, result.radii, k, foci_rng);

    std::uint64_t count_rng = seed ^ TAG_COUNTS;
    result.cluster_sizes = distribute_counts(N, opts.proportions, count_rng);

    std::uint64_t cov_rng  = seed ^ TAG_COV;
    std::uint64_t halo_rng = seed ^ TAG_HALO;

    result.halo_weights.reserve(k);
    result.halo_scales.reserve(k);

    for (int j = 0; j < k; ++j) {
        const std::size_t N_j = result.cluster_sizes[j];

        double L00, L01, L10, L11;
        sample_covariance(cov_rng, opts.s_min, opts.s_max,
                          L00, L01, L10, L11);

        const double w_halo = uniform(halo_rng, opts.halo_w_min, opts.halo_w_max);
        const double k_halo = uniform(halo_rng, opts.halo_k_min, opts.halo_k_max);

        result.halo_weights.push_back(w_halo);
        result.halo_scales.push_back(k_halo);

        const std::size_t before = result.points.size();
        sample_blob(result.foci[j], L00, L01, L10, L11,
                    w_halo, k_halo, N_j,
                    cov_rng, result.points, full);
        const std::size_t added = result.points.size() - before;

        result.labels.insert(result.labels.end(), added,
                             static_cast<std::size_t>(j));
    }
    return result;
}

void sort_by_coordinates(std::vector<Point2D>& pts) {
    std::sort(pts.begin(), pts.end(), [](const Point2D& p, const Point2D& q) {
        if (p[0] != q[0]) return p[0] < q[0];
        return p[1] < q[1];
    });
}

}  // namespace frgc::experiment