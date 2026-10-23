#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace frgc::manifolds {

struct DistanceResult {
    double value;
    std::uint64_t nanos;
    bool converged  = true;
    int  iterations = 0;
};

struct DistanceMatrixResult {
    std::vector<std::vector<double>> matrix;
    std::uint64_t total_nanos = 0;
    std::size_t   n_failed    = 0;
};

namespace detail {

template <typename T, typename = void>
struct has_reset_row : std::false_type {};

template <typename T>
struct has_reset_row<T, std::void_t<decltype(std::declval<T&>().reset_row())>>
    : std::true_type {};

}  // namespace detail

// Stateless builder. Functor passed by value, safe to parallelize.
template <typename Point, typename Func>
DistanceMatrixResult distance_matrix(const std::vector<Point>& sample,
                                     Func f) {
    const std::size_t n = sample.size();
    DistanceMatrixResult out;
    out.matrix.assign(n, std::vector<double>(n, 0.0));

    std::uint64_t total_nanos = 0;
    std::size_t   n_failed    = 0;

    #pragma omp parallel for schedule(guided) \
        reduction(+:total_nanos, n_failed)
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const DistanceResult r = f(sample[i], sample[j]);
            out.matrix[i][j] = r.value;
            out.matrix[j][i] = r.value;
            total_nanos += r.nanos;
            if (!r.converged) ++n_failed;
        }
    }

    out.total_nanos = total_nanos;
    out.n_failed    = n_failed;
    return out;
}

// Warm-start builder. Functor passed by reference; state persists across
// calls. If the functor provides reset_row(), it is called at the start
// of each row. Serial by design: warm start is incompatible with
// row-parallel execution unless each thread keeps its own functor.
template <typename Point, typename Func>
DistanceMatrixResult distance_matrix_warm(const std::vector<Point>& sample,
                                          Func& f) {
    const std::size_t n = sample.size();
    DistanceMatrixResult out;
    out.matrix.assign(n, std::vector<double>(n, 0.0));

    std::uint64_t total_nanos = 0;
    std::size_t   n_failed    = 0;

    for (std::size_t i = 0; i < n; ++i) {
        if constexpr (detail::has_reset_row<Func>::value) {
            f.reset_row();
        }
        for (std::size_t j = i + 1; j < n; ++j) {
            const DistanceResult r = f(sample[i], sample[j]);
            out.matrix[i][j] = r.value;
            out.matrix[j][i] = r.value;
            total_nanos += r.nanos;
            if (!r.converged) ++n_failed;
        }
    }

    out.total_nanos = total_nanos;
    out.n_failed    = n_failed;
    return out;
}

}  // namespace frgc::manifolds