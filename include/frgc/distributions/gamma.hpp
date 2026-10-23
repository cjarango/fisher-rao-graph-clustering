#pragma once

namespace frgc {

class Gamma {
public:
    Gamma(double gamma, double k);

    double gamma() const;
    double k() const;

    bool is_valid() const;

private:
    double gamma_;
    double k_;
};

}  // namespace frgc