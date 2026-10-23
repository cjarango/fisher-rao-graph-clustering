#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "frgc/stats/replicate_stats.hpp"

using namespace frgc::stats;

namespace {

bool is_nan(double x) { return std::isnan(x); }

CellStats make_cell(double rho, double rk0 = 0.0) {
    CellStats c;
    c.rho_mean = rho;
    c.rk_rel_err_mean.fill(std::numeric_limits<double>::quiet_NaN());
    c.rk_rel_err_mean[0] = rk0;
    return c;
}

}  // namespace

TEST_CASE("aggregate_replicates: known values R = 4") {
    // x = [1, 2, 3, 4]
    // mean = 2.5
    // ssd = 1.5^2 + 0.5^2 + 0.5^2 + 1.5^2 = 5
    // var_of_mean = 5 / (4*3) = 5/12
    std::vector<CellStats> cells = {
        make_cell(1.0), make_cell(2.0), make_cell(3.0), make_cell(4.0),
    };
    const auto agg = aggregate_replicates(cells);
    CHECK(agg.rho_expected == doctest::Approx(2.5).epsilon(1e-12));
    CHECK(agg.rho_var_hat  == doctest::Approx(5.0 / 12.0).epsilon(1e-12));
}

TEST_CASE("aggregate_replicates: R = 2") {
    // x = [1, 3], mean = 2, ssd = 2, var_of_mean = 2/(2*1) = 1
    std::vector<CellStats> cells = { make_cell(1.0), make_cell(3.0) };
    const auto agg = aggregate_replicates(cells);
    CHECK(agg.rho_expected == doctest::Approx(2.0).epsilon(1e-12));
    CHECK(agg.rho_var_hat  == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("aggregate_replicates: R = 1 gives NaN variance") {
    std::vector<CellStats> cells = { make_cell(5.0) };
    const auto agg = aggregate_replicates(cells);
    CHECK(agg.rho_expected == doctest::Approx(5.0).epsilon(1e-12));
    CHECK(is_nan(agg.rho_var_hat));
}

TEST_CASE("aggregate_replicates: constant values give zero variance") {
    std::vector<CellStats> cells = {
        make_cell(0.7), make_cell(0.7), make_cell(0.7), make_cell(0.7),
    };
    const auto agg = aggregate_replicates(cells);
    CHECK(agg.rho_expected == doctest::Approx(0.7).epsilon(1e-12));
    CHECK(agg.rho_var_hat  == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("aggregate_replicates: NaN entries are skipped") {
    // x = [NaN, 1, 3], finite = [1, 3], R_eff = 2
    // mean = 2, var_of_mean = 2/(2*1) = 1
    std::vector<CellStats> cells = {
        make_cell(std::numeric_limits<double>::quiet_NaN()),
        make_cell(1.0),
        make_cell(3.0),
    };
    const auto agg = aggregate_replicates(cells);
    CHECK(agg.rho_expected == doctest::Approx(2.0).epsilon(1e-12));
    CHECK(agg.rho_var_hat  == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("aggregate_replicates: empty input gives NaN") {
    const std::vector<CellStats> empty;
    const auto agg = aggregate_replicates(empty);
    CHECK(is_nan(agg.rho_expected));
    CHECK(is_nan(agg.rho_var_hat));
    for (std::size_t k = 0; k < K_NEIGHBORS; ++k) {
        CHECK(is_nan(agg.rk_rel_err_expected[k]));
        CHECK(is_nan(agg.rk_rel_err_var_hat[k]));
    }
}

TEST_CASE("aggregate_replicates: all NaN gives NaN") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<CellStats> cells = { make_cell(nan), make_cell(nan) };
    const auto agg = aggregate_replicates(cells);
    CHECK(is_nan(agg.rho_expected));
    CHECK(is_nan(agg.rho_var_hat));
}

TEST_CASE("aggregate_replicates: rk array uses same formula per k") {
    // rk[0] values across 4 replicates: 0.1, 0.2, 0.3, 0.4
    // mean = 0.25
    // ssd = 0.15^2 + 0.05^2 + 0.05^2 + 0.15^2 = 0.05
    // var = 0.05 / (4*3) = 0.05/12
    std::vector<CellStats> cells = {
        make_cell(0.0, 0.1),
        make_cell(0.0, 0.2),
        make_cell(0.0, 0.3),
        make_cell(0.0, 0.4),
    };
    const auto agg = aggregate_replicates(cells);
    CHECK(agg.rk_rel_err_expected[0] == doctest::Approx(0.25).epsilon(1e-12));
    CHECK(agg.rk_rel_err_var_hat[0]  == doctest::Approx(0.05 / 12.0).epsilon(1e-12));
    // Other k entries are NaN since the inputs are NaN.
    for (std::size_t k = 1; k < K_NEIGHBORS; ++k) {
        CHECK(is_nan(agg.rk_rel_err_expected[k]));
    }
}

TEST_CASE("aggregate_replicates: matches hand-computed mean over all fields") {
    std::vector<CellStats> cells(3);
    for (auto& c : cells) c.rk_rel_err_mean.fill(0.0);
    cells[0].rho_mean = 0.5; cells[0].rk_rel_err_mean[0] = 0.10;
    cells[1].rho_mean = 0.6; cells[1].rk_rel_err_mean[0] = 0.20;
    cells[2].rho_mean = 0.7; cells[2].rk_rel_err_mean[0] = 0.30;

    const auto agg = aggregate_replicates(cells);
    CHECK(agg.rho_expected == doctest::Approx(0.6).epsilon(1e-12));
    // ssd = 0.1^2 + 0 + 0.1^2 = 0.02, var = 0.02/(3*2) = 0.0033333...
    CHECK(agg.rho_var_hat == doctest::Approx(0.02 / 6.0).epsilon(1e-12));

    CHECK(agg.rk_rel_err_expected[0] == doctest::Approx(0.2).epsilon(1e-12));
    // ssd = 0.1^2 + 0 + 0.1^2 = 0.02, var = 0.02/6
    CHECK(agg.rk_rel_err_var_hat[0] == doctest::Approx(0.02 / 6.0).epsilon(1e-12));
}