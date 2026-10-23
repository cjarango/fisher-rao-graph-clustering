#include "frgc/experiment/config.hpp"

namespace frgc::experiment {

const char* to_string(Family f) {
    switch (f) {
    case Family::Gaussian: return "Gaussian";
    case Family::Gamma:    return "Gamma";
    case Family::Beta:     return "Beta";
    }
    return "Unknown";
}

const char* to_string(Regime r) {
    switch (r) {
    case Regime::U:  return "U";
    case Regime::M2: return "M2";
    case Regime::M3: return "M3";
    case Regime::L2: return "L2";
    case Regime::L3: return "L3";
    }
    return "Unknown";
}

FamilyBox get_family_box(Family f) {
    switch (f) {
    case Family::Gaussian: return {-8.0, 8.0,  0.5,  3.0, 0.10};
    case Family::Gamma:    return { 0.5, 8.0,  0.5,  3.0, 0.15};
    case Family::Beta:     return {0.05, 0.95, 0.5, 20.0, 0.15};
    }
    return {0.0, 0.0, 0.0, 0.0, 0.0};
}

}  // namespace frgc::experiment