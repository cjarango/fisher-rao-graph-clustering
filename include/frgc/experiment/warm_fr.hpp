#pragma once

#include <cstddef>
#include <type_traits>

#include "frgc/distributions/beta.hpp"
#include "frgc/distributions/gamma.hpp"
#include "frgc/distributions/gaussian.hpp"
#include "frgc/manifolds/beta_manifold.hpp"
#include "frgc/manifolds/distance_matrix.hpp"
#include "frgc/manifolds/gamma_manifold.hpp"
#include "frgc/manifolds/gaussian_manifold.hpp"

namespace frgc::experiment {

// Warm-start functor for the Fisher-Rao distance. Carries the state
// needed to seed the shooting from the previous pair in the same row.
// The seed is propagated only when the previous solve converged; on
// failure the next call falls back to a cold start.
//
// Must be passed by reference to distance_matrix_warm. Passing a
// temporary loses the state.
template <typename Dist>
struct WarmFR;

template <>
struct WarmFR<frgc::Gaussian> {
    using Dist = frgc::Gaussian;

    // Gaussian FR is closed-form. Nothing to warm start.
    void reset_row() {}

    frgc::manifolds::DistanceResult operator()(const Dist& p, const Dist& q) const {
        return frgc::manifolds::gaussian::fisher_rao(p, q);
    }
};

template <>
struct WarmFR<frgc::Gamma> {
    using Dist = frgc::Gamma;

    frgc::manifolds::gamma::ShootingOptions opts;

    void reset_row() {
        opts.has_seed   = false;
        opts.seed_valid = false;
    }

    frgc::manifolds::DistanceResult operator()(const Dist& p, const Dist& q) {
        auto r = frgc::manifolds::gamma::fisher_rao(p, q, opts);
        // Propagate the last velocity for the next pair in the same row,
        // but only if the current solve converged. If it failed, the
        // seed is invalid and the next call must start cold.
        opts.has_seed = r.converged;
        opts.v0_seed  = opts.last_v0;
        return r;
    }
};

template <>
struct WarmFR<frgc::Beta> {
    using Dist = frgc::Beta;

    frgc::manifolds::beta::ShootingOptions opts;

    void reset_row() {
        opts.has_seed   = false;
        opts.seed_valid = false;
    }

    frgc::manifolds::DistanceResult operator()(const Dist& p, const Dist& q) {
        auto r = frgc::manifolds::beta::fisher_rao(p, q, opts);
        opts.has_seed = r.converged;
        opts.v0_seed  = opts.last_v0;
        return r;
    }
};

}  // namespace frgc::experiment