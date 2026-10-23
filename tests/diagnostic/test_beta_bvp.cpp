#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "frgc/distributions/beta.hpp"
#include "frgc/manifolds/beta_manifold.hpp"

using namespace frgc;
using namespace frgc::manifolds;

TEST_CASE("Beta BVP failure rate diagnostic") {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> mu_dist(0.05, 0.95);
    std::uniform_real_distribution<double> log_phi(-1.0, 2.0);

    constexpr int N = 50;
    std::vector<Beta> sample;
    sample.reserve(N);
    for (int i = 0; i < N; ++i) {
        sample.emplace_back(mu_dist(rng), std::exp(log_phi(rng)));
    }

    const std::size_t total = N * (N - 1) / 2;

    auto report = [&](const char* label, std::size_t n_failed,
                      std::chrono::steady_clock::time_point t0,
                      std::chrono::steady_clock::time_point t1) {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            t1 - t0).count();
        const double ms = ns / 1.0e6;
        const double us_per_pair = ns / 1000.0 / total;
        std::cout << "[" << label << "] "
                  << n_failed << " / " << total
                  << "  (" << 100.0 * n_failed / total << "%)  "
                  << ms << " ms total  "
                  << us_per_pair << " us/pair\n";
    };

    // Pass 1: default options, cold start, original order.
    {
        beta::ShootingOptions opts;
        std::size_t n_failed = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                const auto r = beta::fisher_rao(sample[i], sample[j], opts);
                if (!r.converged) ++n_failed;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        report("Beta default cold", n_failed, t0, t1);
    }

    // Pass 2: tight options, cold start, original order.
    {
        beta::ShootingOptions opts;
        opts.max_iter  = 100;
        opts.tol       = 1e-10;
        opts.rk4_steps = 500;
        std::size_t n_failed = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                const auto r = beta::fisher_rao(sample[i], sample[j], opts);
                if (!r.converged) ++n_failed;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        report("Beta tight cold", n_failed, t0, t1);
    }

    // Pass 3: default options, warm start, original order.
    {
        beta::ShootingOptions opts;
        std::size_t n_failed = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < N; ++i) {
            opts.has_seed = false;
            opts.v0_seed  = {0.0, 0.0};
            for (std::size_t j = i + 1; j < N; ++j) {
                const auto r = beta::fisher_rao(sample[i], sample[j], opts);
                if (!r.converged) ++n_failed;
                opts.has_seed = opts.seed_valid;
                opts.v0_seed  = opts.last_v0;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        report("Beta default warm", n_failed, t0, t1);
    }

    // Prepare the sorted sample once, used by passes 3b and 4.
    auto sorted = sample;
    std::sort(sorted.begin(), sorted.end(),
              [](const Beta& a, const Beta& b) {
                  if (a.mu() != b.mu()) return a.mu() < b.mu();
                  return a.phi() < b.phi();
              });

    // Pass 3b: default options, cold start, sorted order.
    {
        beta::ShootingOptions opts;
        std::size_t n_failed = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                const auto r = beta::fisher_rao(sorted[i], sorted[j], opts);
                if (!r.converged) ++n_failed;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        report("Beta default cold sorted", n_failed, t0, t1);
    }

    // Pass 4: default options, warm start, sorted order.
    {
        beta::ShootingOptions opts;
        std::size_t n_failed = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < N; ++i) {
            opts.has_seed = false;
            opts.v0_seed  = {0.0, 0.0};
            for (std::size_t j = i + 1; j < N; ++j) {
                const auto r = beta::fisher_rao(sorted[i], sorted[j], opts);
                if (!r.converged) ++n_failed;
                opts.has_seed = opts.seed_valid;
                opts.v0_seed  = opts.last_v0;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        report("Beta default warm sorted", n_failed, t0, t1);
    }
}