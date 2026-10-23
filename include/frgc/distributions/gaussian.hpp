#pragma once

namespace frgc {

class Gaussian {
public:
    Gaussian(double mu, double sigma);

    double mu() const;
    double sigma() const;
    double variance() const;

    bool is_valid() const;

private:
    double mu_;
    double sigma_;
};

}  // namespace frgc