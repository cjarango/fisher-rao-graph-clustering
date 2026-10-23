#pragma once

#include <array>

#include "frgc/distributions/gaussian.hpp"
#include "frgc/manifolds/distance_matrix.hpp"

namespace frgc::manifolds::gaussian {

struct SegmentOptions {
    int  n_quad          = 20;
    bool use_closed_form = true;
};

DistanceResult hellinger(const Gaussian& p, const Gaussian& q);
DistanceResult tangent_norm(const Gaussian& p, const Gaussian& q);
DistanceResult segment(const Gaussian& p, const Gaussian& q,
                       const SegmentOptions& opts = {});
DistanceResult euclidean(const Gaussian& p, const Gaussian& q);
DistanceResult fisher_rao(const Gaussian& p, const Gaussian& q);

std::array<std::array<double, 2>, 2> metric_tensor(const Gaussian& p);

}  // namespace frgc::manifolds::gaussian