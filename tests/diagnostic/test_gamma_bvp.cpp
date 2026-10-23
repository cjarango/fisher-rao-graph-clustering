#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "frgc/distributions/gamma.hpp"
#include "frgc/manifolds/gamma_manifold.hpp"

using namespace frgc;
using namespace frgc::manifolds;

TEST_CASE("Gamma BVP failure rate diagnostic") {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> log_gamma(-1.0, 2.0);
    std::uniform_real_distribution<double> log_k(-1.0, 2.0);

    constexpr int N = 50;
    std::vector<Gamma> sample;
    sample.reserve(N);
    for (int i = 0; i < N; ++i) {
        sample.emplace_back(std::exp(log_gamma(rng)),
                            std::exp(log_k(rng)));
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
        gamma::ShootingOptions opts;
        std::size_t n_failed = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                const auto r = gamma::fisher_rao(sample[i], sample[j], opts);
                if (!r.converged) ++n_failed;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        report("Gamma default cold", n_failed, t0, t1);
    }

    // Pass 2: tight options, cold start, original order.
    {
        gamma::ShootingOptions opts;
        opts.max_iter  = 100;
        opts.tol       = 1e-10;
        opts.rk4_steps = 500;
        std::size_t n_failed = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                const auto r = gamma::fisher_rao(sample[i], sample[j], opts);
                if (!r.converged) ++n_failed;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        report("Gamma tight cold", n_failed, t0, t1);
    }

    // Pass 3: default options, warm start, original order.
    {
        gamma::ShootingOptions opts;
        std::size_t n_failed = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < N; ++i) {
            opts.has_seed = false;
            opts.v0_seed  = {0.0, 0.0};
            for (std::size_t j = i + 1; j < N; ++j) {
                const auto r = gamma::fisher_rao(sample[i], sample[j], opts);
                if (!r.converged) ++n_failed;
                opts.has_seed = opts.seed_valid;
                opts.v0_seed  = opts.last_v0;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        report("Gamma default warm", n_failed, t0, t1);
    }

    // Prepare the sorted sample once, used by passes 3b and 4.
    auto sorted = sample;
    std::sort(sorted.begin(), sorted.end(),
              [](const Gamma& a, const Gamma& b) {
                  if (a.gamma() != b.gamma()) return a.gamma() < b.gamma();
                  return a.k() < b.k();
              });

    // Pass 3b: default options, cold start, sorted order.
    {
        gamma::ShootingOptions opts;
        std::size_t n_failed = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                const auto r = gamma::fisher_rao(sorted[i], sorted[j], opts);
                if (!r.converged) ++n_failed;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        report("Gamma default cold sorted", n_failed, t0, t1);
    }

    // Pass 4: default options, warm start, sorted order.
    {
        gamma::ShootingOptions opts;
        std::size_t n_failed = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < N; ++i) {
            opts.has_seed = false;
            opts.v0_seed  = {0.0, 0.0};
            for (std::size_t j = i + 1; j < N; ++j) {
                const auto r = gamma::fisher_rao(sorted[i], sorted[j], opts);
                if (!r.converged) ++n_failed;
                opts.has_seed = opts.seed_valid;
                opts.v0_seed  = opts.last_v0;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        report("Gamma default warm sorted", n_failed, t0, t1);
    }
}