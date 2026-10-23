#pragma once

#include <array>

#include "frgc/distributions/beta.hpp"
#include "frgc/manifolds/distance_matrix.hpp"
#include "frgc/manifolds/shooting_options.hpp"

namespace frgc::manifolds::beta {

using ShootingOptions = frgc::manifolds::ShootingOptions;

struct SegmentOptions {
    int n_quad = 20;
};

DistanceResult hellinger(const Beta& p, const Beta& q);
DistanceResult tangent_norm(const Beta& p, const Beta& q);
DistanceResult segment(const Beta& p, const Beta& q,
                       const SegmentOptions& opts = {});
DistanceResult euclidean(const Beta& p, const Beta& q);
DistanceResult fisher_rao(const Beta& p, const Beta& q,
                          const ShootingOptions& opts = {});

struct FisherRao {
    ShootingOptions opts{};

    DistanceResult operator()(const Beta& p, const Beta& q) const {
        return fisher_rao(p, q, opts);
    }
};

std::array<std::array<double, 2>, 2> metric_tensor(const Beta& p);

}  // namespace frgc::manifolds::beta