#pragma once

#include <array>

#include "frgc/distributions/gamma.hpp"
#include "frgc/manifolds/distance_matrix.hpp"
#include "frgc/manifolds/shooting_options.hpp"

namespace frgc::manifolds::gamma {

using ShootingOptions = frgc::manifolds::ShootingOptions;

struct SegmentOptions {
    int n_quad = 20;
};

DistanceResult hellinger(const Gamma& p, const Gamma& q);
DistanceResult tangent_norm(const Gamma& p, const Gamma& q);
DistanceResult segment(const Gamma& p, const Gamma& q,
                       const SegmentOptions& opts = {});
DistanceResult euclidean(const Gamma& p, const Gamma& q);
DistanceResult fisher_rao(const Gamma& p, const Gamma& q,
                          const ShootingOptions& opts = {});

struct FisherRao {
    ShootingOptions opts{};

    DistanceResult operator()(const Gamma& p, const Gamma& q) const {
        return fisher_rao(p, q, opts);
    }
};

std::array<std::array<double, 2>, 2> metric_tensor(const Gamma& p);

}  // namespace frgc::manifolds::gamma