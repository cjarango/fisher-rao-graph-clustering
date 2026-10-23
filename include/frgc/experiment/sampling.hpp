#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "frgc/experiment/config.hpp"

namespace frgc::experiment {

using Point2D = std::array<double, 2>;

struct FamilyScale {
    double W;
};

FamilyScale calibrate_family_scale(Family);

std::uint64_t replicate_seed(std::uint64_t base_seed, Family, Regime, int replicate);

struct SampleOptions {
    std::vector<double> proportions;

    // Covariance parameters of the Gaussian core.
    double s_min = 0.05;
    double s_max = 0.20;

    // Halo parameters, sampled per focus.
    double halo_w_min = 0.15;
    double halo_w_max = 0.40;
    double halo_k_min = 2.5;
    double halo_k_max = 6.0;
};

struct SampleResult {
    std::vector<Point2D>      points;
    std::vector<std::size_t>  labels;
    std::vector<std::size_t>  cluster_sizes;
    std::vector<Point2D>      foci;
    std::vector<double>       radii;
    std::vector<double>       halo_weights;
    std::vector<double>       halo_scales;
};

SampleResult generate_sample(Family, Regime, std::size_t N,
                             const SampleOptions& opts,
                             std::uint64_t seed);

void sort_by_coordinates(std::vector<Point2D>&);

}  // namespace frgc::experiment