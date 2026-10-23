#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "frgc/experiment/warm_fr.hpp"
#include "frgc/manifolds/distance_matrix.hpp"

using namespace frgc;
using namespace frgc::manifolds;
using namespace frgc::experiment;

TEST_CASE("WarmFR<Gamma>: state propagates after a successful call") {
    WarmFR<Gamma> fr;
    fr.reset_row();
    CHECK(fr.opts.has_seed == false);

    const Gamma a(1.0, 1.0);
    const Gamma b(1.1, 1.1);
    const auto r = fr(a, b);

    CHECK(r.converged);
    CHECK(fr.opts.has_seed == true);
    CHECK(fr.opts.seed_valid == true);
    CHECK(std::isfinite(fr.opts.v0_seed[0]));
    CHECK(std::isfinite(fr.opts.v0_seed[1]));
}

TEST_CASE("WarmFR<Gamma>: reset_row clears the seed") {
    WarmFR<Gamma> fr;
    fr.reset_row();

    const auto r = fr(Gamma(1.0, 1.0), Gamma(1.1, 1.1));
    REQUIRE(r.converged);
    CHECK(fr.opts.has_seed == true);

    fr.reset_row();
    CHECK(fr.opts.has_seed == false);
    CHECK(fr.opts.seed_valid == false);
}

TEST_CASE("WarmFR<Beta>: state propagates after a successful call") {
    WarmFR<Beta> fr;
    fr.reset_row();

    const auto r = fr(Beta(0.4, 3.0), Beta(0.41, 3.05));
    CHECK(r.converged);
    CHECK(fr.opts.has_seed == true);
    CHECK(fr.opts.seed_valid == true);
}

TEST_CASE("distance_matrix_warm: warm and cold agree on the values") {
    std::vector<Gamma> sample;
    sample.reserve(20);
    for (int i = 0; i < 20; ++i) {
        sample.emplace_back(1.0 + 0.05 * i, 1.0 + 0.03 * i);
    }
    std::sort(sample.begin(), sample.end(),
              [](const Gamma& a, const Gamma& b) {
                  if (a.gamma() != b.gamma()) return a.gamma() < b.gamma();
                  return a.k() < b.k();
              });

    WarmFR<Gamma> fr_fn;
    const auto D_warm = distance_matrix_warm(sample, fr_fn);

    const auto D_cold = distance_matrix(sample,
        [](const Gamma& p, const Gamma& q) { return gamma::fisher_rao(p, q); });

    const std::size_t n = sample.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            CHECK(D_warm.matrix[i][j] ==
                  doctest::Approx(D_cold.matrix[i][j]).epsilon(1e-6));
            CHECK(D_warm.matrix[i][j] == D_warm.matrix[j][i]);
        }
    }

    CHECK(D_warm.n_failed == 0);
    CHECK(D_cold.n_failed == 0);
}

TEST_CASE("distance_matrix_warm: n_failed is zero with warm + sorted") {
    std::vector<Gamma> sample;
    sample.reserve(30);
    for (int i = 0; i < 30; ++i) {
        sample.emplace_back(0.5 + 0.1 * i, 0.5 + 0.05 * i);
    }
    std::sort(sample.begin(), sample.end(),
              [](const Gamma& a, const Gamma& b) {
                  if (a.gamma() != b.gamma()) return a.gamma() < b.gamma();
                  return a.k() < b.k();
              });

    WarmFR<Gamma> fr_fn;
    const auto D = distance_matrix_warm(sample, fr_fn);
    CHECK(D.n_failed == 0);
}