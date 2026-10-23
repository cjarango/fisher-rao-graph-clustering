#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef FRGC_EXPERIMENT_BIN
#error "FRGC_EXPERIMENT_BIN must be defined by CMake"
#endif

namespace {

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, delim)) out.push_back(tok);
    return out;
}

std::size_t sum_n_failed(const fs::path& timing_csv) {
    std::ifstream f(timing_csv);
    std::string line;
    std::getline(f, line);
    auto hdr = split(line, ',');
    REQUIRE(hdr.size() >= 5);
    CHECK(hdr[4] == "n_failed");

    std::size_t total = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto toks = split(line, ',');
        REQUIRE(toks.size() >= 5);
        total += std::stoul(toks[4]);
    }
    return total;
}

}  // namespace

TEST_CASE("driver e2e: tiny config produces expected outputs") {
    const fs::path root = fs::temp_directory_path() / "frgc_e2e";
    fs::remove_all(root);
    fs::create_directories(root);

    const fs::path out_dir = root / "out";
    const fs::path cfg = root / "config.json";
    {
        std::ofstream f(cfg);
        f << R"({
            "output_dir": ")" << out_dir.string() << R"(",
            "replicates": 1,
            "N_max": 30,
            "N_values": [10, 30],
            "families": ["Gamma"],
            "regimes": ["U", "M2"],
            "bvp": { "max_iter": 100, "tol": 1e-8, "rk4_steps": 200 }
        })";
    }

    const std::string cmd = std::string(FRGC_EXPERIMENT_BIN)
                          + " " + cfg.string() + " > /dev/null 2>&1";
    const int rc = std::system(cmd.c_str());
    REQUIRE(rc == 0);

    CHECK(fs::exists(out_dir / "meta.json"));

    for (const char* stem : {"Gamma_U_0", "Gamma_M2_0"}) {
        const fs::path pf = out_dir / (std::string(stem) + ".csv");
        const fs::path tf = out_dir / (std::string(stem) + ".timing.csv");
        REQUIRE(fs::exists(pf));
        REQUIRE(fs::exists(tf));

        CHECK(sum_n_failed(tf) == 0);

        std::ifstream f(pf);
        std::string line;
        std::getline(f, line);
        auto hdr = split(line, ',');
        REQUIRE(hdr.size() >= 6);
        CHECK(hdr[0] == "N");
        CHECK(hdr[1] == "criterion");
        CHECK(hdr[2] == "i");
        CHECK(hdr[3] == "rho");
        CHECK(hdr[4] == "delta_h");
        CHECK(hdr[5] == "delta_f");
    }
}

int main(int argc, char** argv) {
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    return ctx.run();
}