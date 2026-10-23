#pragma once

namespace frgc {

class Beta {
public:
    Beta(double mu, double phi);

    double mu() const;
    double phi() const;
    double alpha() const;  // derived: mu * phi
    double beta() const;   // derived: (1 - mu) * phi

    bool is_valid() const;

private:
    double mu_;
    double phi_;
};

}  // namespace frgc