#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const long N = 20000000;
    const int n_threads = (argc > 1) ? std::atoi(argv[1]) : omp_get_max_threads();

    omp_set_num_threads(n_threads);

    std::cout << "Threads requested: " << n_threads << "\n";
    std::cout << "Threads actual:    " << omp_get_max_threads() << "\n";

    double result = 0.0;
    const auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+ : result) schedule(static)
    for (long i = 0; i < N; ++i) {
        const double x = static_cast<double>(i) * 1e-7;
        result += std::sin(x) * std::cos(x) + std::exp(-x) * std::log1p(x);
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> dt = t1 - t0;

    std::cout << "Result: " << result << "\n";
    std::cout << "Time:   " << dt.count() << " s\n";
    return 0;
}