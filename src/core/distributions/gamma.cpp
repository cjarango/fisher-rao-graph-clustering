#include "frgc/distributions/gamma.hpp"

#include <stdexcept>

namespace frgc {

Gamma::Gamma(double gamma, double k) : gamma_(gamma), k_(k) {
    if (!is_valid()) {
        throw std::invalid_argument(
            "Gamma: gamma and k must be positive, got gamma=" + std::to_string(gamma) +
            ", k=" + std::to_string(k));
    }
}

double Gamma::gamma() const { return gamma_; }
double Gamma::k() const { return k_; }

bool Gamma::is_valid() const { return gamma_ > 0.0 && k_ > 0.0; }

}  // namespace frgc