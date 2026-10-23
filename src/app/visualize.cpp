#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "frgc/experiment/config.hpp"
#include "frgc/experiment/sampling.hpp"

using namespace frgc::experiment;

namespace {

Family parse_family(const std::string& s) {
    if (s == "gamma") return Family::Gamma;
    if (s == "beta")  return Family::Beta;
    return Family::Gaussian;
}

Regime parse_regime(const std::string& s) {
    if (s == "M2") return Regime::M2;
    if (s == "M3") return Regime::M3;
    if (s == "L2") return Regime::L2;
    if (s == "L3") return Regime::L3;
    return Regime::U;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: %s <family> <regime> <N> <replicate> <out.svg>\n"
            "  family: gaussian | gamma | beta\n"
            "  regime: U | M2 | M3 | L2 | L3\n",
            argv[0]);
        return 1;
    }

    const std::string fam_s = argv[1];
    const std::string reg_s = argv[2];
    const std::size_t N     = std::stoul(argv[3]);
    const int         rep   = std::stoi(argv[4]);
    const std::string out   = argv[5];

    const Family fam = parse_family(fam_s);
    const Regime reg = parse_regime(reg_s);

    SampleOptions opts;
    if (reg == Regime::M2 || reg == Regime::L2) {
        opts.proportions = {0.5, 0.5};
    } else if (reg == Regime::M3 || reg == Regime::L3) {
        opts.proportions = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    }
    opts.s_min = 0.05;
    opts.s_max = 0.20;

    const std::uint64_t seed = replicate_seed(0, fam, reg, rep);
    const auto result = generate_sample(fam, reg, N, opts, seed);

    const FamilyBox box = get_family_box(fam);

    const std::string data_file = "/tmp/frgc_plot.dat";
    {
        std::ofstream f(data_file);
        if (result.labels.empty()) {
            for (const auto& p : result.points) {
                f << p[0] << " " << p[1] << " -1\n";
            }
        } else {
            for (std::size_t i = 0; i < result.points.size(); ++i) {
                f << result.points[i][0] << " "
                  << result.points[i][1] << " "
                  << result.labels[i] << "\n";
            }
        }
    }

    const std::string gp_script = "/tmp/frgc_plot.gp";
    {
        std::ofstream f(gp_script);

        f << "set terminal svg size 900,700 dynamic\n";
        f << "set output '" << out << "'\n";
        f << "set title 'family=" << fam_s
          << "  regime=" << reg_s
          << "  N=" << N
          << "  rep=" << rep << "'\n";
        f << "set xlabel 'param 1'\n";
        f << "set ylabel 'param 2'\n";
        f << "unset grid\n";
        f << "set key outside right\n";
        f << "set xrange [" << box.a_min << ":" << box.a_max << "]\n";

        // For Beta, the range of phi (0.5 to 20) is so much larger than
        // the range of mu (0.05 to 0.95) that fixing the y-range hides
        // the clusters. Let gnuplot autoscale y for this family.
        if (fam != Family::Beta) {
            f << "set yrange [" << box.b_min << ":" << box.b_max << "]\n";
        }

        const double da = (box.a_max - box.a_min) * box.margin_frac;
        const double db = (box.b_max - box.b_min) * box.margin_frac;
        f << "set object rect from "
          << (box.a_min + da) << "," << (box.b_min + db) << " to "
          << (box.a_max - da) << "," << (box.b_max - db)
          << " behind fs empty border dt 2 lw 2 lc rgb '#888888'\n";

        if (!result.labels.empty()) {
            f << "unset colorbox\n";
            f << "set palette model RGB defined ("
              << "0 '#e41a1c', 1 '#377eb8', 2 '#4daf4a')\n";

            f << "plot '" << data_file
              << "' using 1:2:3 with points pt 7 ps 0.5 lc palette notitle\n";
        } else {
            f << "plot '" << data_file
              << "' using 1:2 with points pt 7 ps 0.5"
              << " lc rgb '#377eb8' title 'uniform'\n";
        }

        f << "unset output\n";
    }

    const std::string cmd = "gnuplot " + gp_script;
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "gnuplot exited with code %d\n", rc);
        return 1;
    }

    std::printf("wrote %s\n", out.c_str());
    return 0;
}