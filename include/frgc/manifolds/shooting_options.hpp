#pragma once

#include <array>

namespace frgc::manifolds {

struct ShootingOptions {
    int    max_iter   = 50;
    double tol        = 1e-6;
    int    rk4_steps  = 100;
    double fd_epsilon = 1e-7;

    bool                  has_seed   = false;
    std::array<double, 2> v0_seed    = {0.0, 0.0};
    mutable bool          seed_valid = false;
    mutable std::array<double, 2> last_v0 = {0.0, 0.0};
};

}  // namespace frgc::manifolds