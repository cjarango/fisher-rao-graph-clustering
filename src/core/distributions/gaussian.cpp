#include "frgc/distributions/gaussian.hpp"

#include <stdexcept>

namespace frgc {

Gaussian::Gaussian(double mu, double sigma) : mu_(mu), sigma_(sigma) {
    if (!is_valid()) {
        throw std::invalid_argument(
            "Gaussian: sigma must be positive, got sigma=" + std::to_string(sigma));
    }
}

double Gaussian::mu() const { return mu_; }
double Gaussian::sigma() const { return sigma_; }
double Gaussian::variance() const { return sigma_ * sigma_; }

bool Gaussian::is_valid() const { return sigma_ > 0.0; }

}  // namespace frgc