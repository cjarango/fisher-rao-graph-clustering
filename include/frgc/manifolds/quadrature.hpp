#pragma once

#include <vector>

namespace frgc::manifolds::quadrature {

// Gauss-Legendre nodes and weights on [-1, 1].
struct GLTable {
    std::vector<double> nodes;
    std::vector<double> weights;
};

// Returns a cached table for n nodes. Thread-safe.
const GLTable& gauss_legendre(int n);

// Thread-local cached lookup. The first call per (thread, n) acquires a
// lock; subsequent calls are lock-free. Use this inside parallel hot loops.
const GLTable& gauss_legendre_local(int n);

// Pre-populate the global cache. Call before a parallel region to avoid
// the first-call contention inside the loop.
void warm_gl_cache(int n);

// Integrates f on [a, b] using n-point Gauss-Legendre.
template <typename Func>
double integrate_gl(Func f, double a, double b, int n) {
    const GLTable& gl = gauss_legendre_local(n);
    const double half = 0.5 * (b - a);
    const double mid  = 0.5 * (a + b);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += gl.weights[i] * f(mid + half * gl.nodes[i]);
    }
    return half * sum;
}

}  // namespace frgc::manifolds::quadrature