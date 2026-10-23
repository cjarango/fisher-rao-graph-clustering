#include "frgc/distributions/beta.hpp"

#include <stdexcept>

namespace frgc {

Beta::Beta(double mu, double phi) : mu_(mu), phi_(phi) {
    if (!is_valid()) {
        throw std::invalid_argument(
            "Beta: mu must be in (0,1) and phi positive, got mu=" + std::to_string(mu) +
            ", phi=" + std::to_string(phi));
    }
}

double Beta::mu() const { return mu_; }
double Beta::phi() const { return phi_; }
double Beta::alpha() const { return mu_ * phi_; }
double Beta::beta() const { return (1.0 - mu_) * phi_; }

bool Beta::is_valid() const { return mu_ > 0.0 && mu_ < 1.0 && phi_ > 0.0; }

}  // namespace frgc