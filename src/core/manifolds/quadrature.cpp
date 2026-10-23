#include "frgc/manifolds/quadrature.hpp"

#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace frgc::manifolds::quadrature {

namespace {

constexpr double kPi = 3.14159265358979323846;

void compute_gl(int n, std::vector<double>& nodes, std::vector<double>& weights) {
    nodes.resize(n);
    weights.resize(n);

    for (int i = 0; i < n; ++i) {
        // Initial guess (Chebyshev-like).
        double x = std::cos(kPi * (i + 0.75) / (n + 0.5));

        // Newton-Raphson to find the i-th Legendre root.
        for (int iter = 0; iter < 100; ++iter) {
            double p0 = 1.0;
            double p1 = x;
            for (int k = 2; k <= n; ++k) {
                const double p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            const double dp = n * (x * p1 - p0) / (x * x - 1.0);
            const double dx = p1 / dp;
            x -= dx;
            if (std::abs(dx) < 1e-15) break;
        }
        nodes[i] = x;

        double p0 = 1.0;
        double p1 = x;
        for (int k = 2; k <= n; ++k) {
            const double p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
            p0 = p1;
            p1 = p2;
        }
        const double dp = n * (x * p1 - p0) / (x * x - 1.0);
        weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
}

// std::map keeps references to existing entries valid across insertions,
// which the thread-local cache relies on.
std::map<int, GLTable>& global_cache() {
    static std::map<int, GLTable> cache;
    return cache;
}

std::mutex& global_mutex() {
    static std::mutex mtx;
    return mtx;
}

}  // namespace

const GLTable& gauss_legendre(int n) {
    std::lock_guard<std::mutex> lock(global_mutex());

    auto& cache = global_cache();
    auto it = cache.find(n);
    if (it != cache.end()) return it->second;

    if (n < 1) throw std::invalid_argument("Gauss-Legendre requires n >= 1");

    GLTable table;
    compute_gl(n, table.nodes, table.weights);
    auto [inserted, _] = cache.emplace(n, std::move(table));
    return inserted->second;
}

const GLTable& gauss_legendre_local(int n) {
    thread_local std::unordered_map<int, const GLTable*> tl;
    auto it = tl.find(n);
    if (it != tl.end()) return *it->second;
    const GLTable& t = gauss_legendre(n);
    tl.emplace(n, &t);
    return t;
}

void warm_gl_cache(int n) {
    (void)gauss_legendre(n);
}

}  // namespace frgc::manifolds::quadrature