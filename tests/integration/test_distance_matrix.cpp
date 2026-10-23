#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "frgc/distributions/gamma.hpp"
#include "frgc/distributions/gaussian.hpp"
#include "frgc/manifolds/distance_matrix.hpp"
#include "frgc/manifolds/gamma_manifold.hpp"
#include "frgc/manifolds/gaussian_manifold.hpp"

using namespace frgc;
using namespace frgc::manifolds;

namespace {

std::vector<Gaussian> gaussian_sample() {
    return {
        Gaussian(0.0, 1.0),
        Gaussian(1.0, 1.5),
        Gaussian(-1.0, 2.0),
        Gaussian(2.0, 0.5),
        Gaussian(0.5, 3.0),
    };
}

std::vector<Gamma> gamma_sample() {
    return {
        Gamma(1.0, 1.0),
        Gamma(1.5, 1.2),
        Gamma(2.0, 0.8),
        Gamma(0.8, 2.5),
        Gamma(3.0, 1.5),
    };
}

void check_matrices_equal(const std::vector<std::vector<double>>& a,
                          const std::vector<std::vector<double>>& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(a[i].size() == b[i].size());
        for (std::size_t j = 0; j < a[i].size(); ++j) {
            CHECK(a[i][j] == b[i][j]);
        }
    }
}

}  // namespace

TEST_CASE("distance_matrix diagonal is zero") {
    const auto sample = gaussian_sample();
    const auto result = distance_matrix(sample, gaussian::fisher_rao);

    const std::size_t n = sample.size();
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(result.matrix[i][i] == 0.0);
    }
}

TEST_CASE("distance_matrix reflects the upper triangle") {
    const auto sample = gaussian_sample();
    const auto result = distance_matrix(sample, gaussian::fisher_rao);

    const std::size_t n = sample.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            CHECK(result.matrix[i][j] == result.matrix[j][i]);
            CHECK(result.matrix[i][j] > 0.0);
        }
    }
}

TEST_CASE("distance_matrix timing is non-zero and n_failed is zero") {
    const auto sample = gaussian_sample();
    const auto result = distance_matrix(sample, gaussian::fisher_rao);

    CHECK(result.total_nanos > 0);

    const std::size_t n = sample.size();
    const std::size_t expected_pairs = n * (n - 1) / 2;
    CHECK(expected_pairs == 10);
    CHECK(result.n_failed == 0);
}

TEST_CASE("distance_matrix with tangent_norm is symmetric") {
    // After symmetrizing the tangent norm, its matrix must be symmetric
    // like the other criteria.
    const std::vector<Gaussian> sample = {
        Gaussian(0.0, 1.0),
        Gaussian(2.0, 4.0),
    };

    const auto result = distance_matrix(sample, gaussian::tangent_norm);

    CHECK(result.matrix[0][1] == result.matrix[1][0]);
    CHECK(result.matrix[0][1] > 0.0);
}

TEST_CASE("distance_matrix is deterministic across thread counts (Gaussian)") {
    // Fast criterion, no shooting. Checks that the write pattern and the
    // reductions are stable under parallelism.
    const auto sample = gaussian_sample();

#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
#else
    const int saved = 1;
#endif

    const auto r_serial = distance_matrix(sample, gaussian::fisher_rao);

#ifdef _OPENMP
    omp_set_num_threads(saved);
#endif

    const auto r_par = distance_matrix(sample, gaussian::fisher_rao);

    check_matrices_equal(r_serial.matrix, r_par.matrix);
    CHECK(r_serial.n_failed == r_par.n_failed);
}

TEST_CASE("distance_matrix is deterministic across thread counts (Gamma)") {
    // Slow criterion, shooting involved. This is the real test: the
    // BVP solver runs inside the parallel loop, and its results must
    // not depend on the thread scheduling.
    const auto sample = gamma_sample();

#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
#else
    const int saved = 1;
#endif

    const auto r_serial = distance_matrix(sample, gamma::FisherRao{});

#ifdef _OPENMP
    omp_set_num_threads(saved);
#endif

    const auto r_par = distance_matrix(sample, gamma::FisherRao{});

    check_matrices_equal(r_serial.matrix, r_par.matrix);
    CHECK(r_serial.n_failed == r_par.n_failed);
}

TEST_CASE("distance_matrix n_failed reduction is exact under threads") {
    // Custom functor that marks pairs as failed based on coordinates,
    // not indices. This makes the expected count computable from the
    // sample alone, and the parallel reduction verifiable.
    const auto sample = gaussian_sample();

    auto sometimes_fails = [](const Gaussian& p, const Gaussian& q) {
        DistanceResult r = gaussian::fisher_rao(p, q);
        if (p.mu() < 0.0) r.converged = false;
        return r;
    };

    std::size_t expected = 0;
    const std::size_t n = sample.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (sample[i].mu() < 0.0) ++expected;
        }
    }
    REQUIRE(expected > 0);

    const auto result = distance_matrix(sample, sometimes_fails);

    CHECK(result.n_failed == expected);
}