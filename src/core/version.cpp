#include <frgc/version.hpp>

namespace frgc {

const char* build_version() noexcept {
    return FRGC_VERSION_STRING;
}

}  // namespace frgc