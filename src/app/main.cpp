#include <frgc/version.hpp>

#include <chrono>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
    std::cout << "frgc v" << frgc::version_string() << "\n";

#ifdef _OPENMP
    std::cout << "OpenMP enabled. Max threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP not enabled\n";
#endif

    const int N = 100000000;
    std::vector<double> x(N, 1.0);
    double sum = 0.0;

    const auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < N; ++i) {
        sum += x[i] * x[i];
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> dt = t1 - t0;

    std::cout << "Sum: " << sum << "\n";
    std::cout << "Time: " << dt.count() << " s\n";
    return 0;
}