#pragma once

#define FRGC_VERSION_MAJOR 0
#define FRGC_VERSION_MINOR 1
#define FRGC_VERSION_PATCH 0

#define FRGC_VERSION_STRING "0.1.0"

namespace frgc {

constexpr int version_major() noexcept { return FRGC_VERSION_MAJOR; }
constexpr int version_minor() noexcept { return FRGC_VERSION_MINOR; }
constexpr int version_patch() noexcept { return FRGC_VERSION_PATCH; }

constexpr const char* version_string() noexcept {
    return FRGC_VERSION_STRING;
}

const char* build_version() noexcept;

}  // namespace frgc