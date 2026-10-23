#pragma once

namespace frgc::experiment {

enum class Family {
    Gaussian,
    Gamma,
    Beta,
};

enum class Regime {
    U,
    M2,
    M3,
    L2,
    L3,
};

const char* to_string(Family);
const char* to_string(Regime);

struct FamilyBox {
    double a_min, a_max;
    double b_min, b_max;
    double margin_frac;   // fraction of each edge used as safety margin
};

FamilyBox get_family_box(Family);

}  // namespace frgc::experiment