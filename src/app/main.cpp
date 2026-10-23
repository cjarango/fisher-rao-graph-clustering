#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "frgc/distributions/beta.hpp"
#include "frgc/distributions/gamma.hpp"
#include "frgc/distributions/gaussian.hpp"
#include "frgc/experiment/config.hpp"
#include "frgc/experiment/sampling.hpp"
#include "frgc/experiment/warm_fr.hpp"
#include "frgc/manifolds/beta_manifold.hpp"
#include "frgc/manifolds/distance_matrix.hpp"
#include "frgc/manifolds/gamma_manifold.hpp"
#include "frgc/manifolds/gaussian_manifold.hpp"
#include "frgc/manifolds/shooting_options.hpp"
#include "frgc/stats/point_stats.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace frgc;
using namespace frgc::experiment;
using namespace frgc::manifolds;
using namespace frgc::stats;
using Clock = std::chrono::steady_clock;

namespace {

// ---------------------------------------------------------------------------
// Progress bar

struct Progress {
    std::size_t total;
    std::size_t done = 0;
    double      ema_secs = 0.0;
    bool        tty;
    std::string current;

    explicit Progress(std::size_t total_)
        : total(total_), tty(::isatty(fileno(stderr))) {}

    void begin(const std::string& label) { current = label; draw(); }
    void end(double secs) {
        ++done;
        if (ema_secs == 0.0) ema_secs = secs;
        else                 ema_secs = 0.7 * ema_secs + 0.3 * secs;
        draw();
    }
    void skip(const std::string& label) {
        current = label;
        ++done;
        draw();
    }
    void finish() {
        if (tty) std::fprintf(stderr, "\n");
        std::fprintf(stderr, "all done: %zu cells\n", done);
    }

private:
    static const char* fmt_eta(double secs) {
        static thread_local char buf[24];
        if (secs < 1.0)         std::snprintf(buf, sizeof buf, "0s");
        else if (secs < 60.0)   std::snprintf(buf, sizeof buf, "%.0fs", secs);
        else if (secs < 3600.0) std::snprintf(buf, sizeof buf, "%dm%02ds",
                                              (int)(secs / 60.0),
                                              (int)secs % 60);
        else                    std::snprintf(buf, sizeof buf, "%dh%02dm",
                                              (int)(secs / 3600.0),
                                              ((int)secs % 3600) / 60);
        return buf;
    }

    void draw() {
        constexpr int width = 24;
        const int filled = (int)((double)done / (double)total * width);
        char bar[width + 2];
        for (int i = 0; i < width; ++i)
            bar[i] = (i < filled) ? '=' : ' ';
        if (filled > 0 && filled < width) bar[filled - 1] = '>';
        bar[width] = '\0';

        const double remaining = (double)(total - done) * ema_secs;
        const char* eta = fmt_eta(remaining);

        if (tty) {
            std::fprintf(stderr, "\r[%s] %zu/%zu (%.1f%%)  ETA %-7s  %-22s",
                         bar, done, total,
                         100.0 * (double)done / (double)total, eta,
                         current.c_str());
            std::fflush(stderr);
        } else {
            std::fprintf(stderr, "[%s] %zu/%zu (%.1f%%)  ETA %-7s  %s\n",
                         bar, done, total,
                         100.0 * (double)done / (double)total, eta,
                         current.c_str());
        }
    }
};

// ---------------------------------------------------------------------------

struct Config {
    fs::path                 output_dir = "results";
    std::uint64_t            base_seed  = 0;
    int                      replicates = 25;
    std::vector<std::size_t> N_values   = {10, 25, 50, 100, 200, 500, 1000};
    std::vector<Family>      families   = {Family::Gaussian, Family::Gamma, Family::Beta};
    std::vector<Regime>      regimes    = {Regime::U, Regime::M2, Regime::M3,
                                           Regime::L2, Regime::L3};

    ShootingOptions bvp;
    SampleOptions   sample;

    // Derived: the largest sample size is the maximum of N_values.
    std::size_t N_max() const {
        return *std::max_element(N_values.begin(), N_values.end());
    }
};

Family family_from_string(const std::string& s) {
    if (s == "Gaussian") return Family::Gaussian;
    if (s == "Gamma")    return Family::Gamma;
    if (s == "Beta")     return Family::Beta;
    throw std::runtime_error("unknown family: " + s);
}

Regime regime_from_string(const std::string& s) {
    if (s == "U")  return Regime::U;
    if (s == "M2") return Regime::M2;
    if (s == "M3") return Regime::M3;
    if (s == "L2") return Regime::L2;
    if (s == "L3") return Regime::L3;
    throw std::runtime_error("unknown regime: " + s);
}

Config load_config(const fs::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open config: " + path.string());
    json j = json::parse(f);

    Config c;
    c.output_dir = j.value("output_dir", "results");
    c.base_seed  = j.value("base_seed",  0);
    c.replicates = j.value("replicates", 25);

    if (j.contains("N_values"))
        c.N_values = j["N_values"].get<std::vector<std::size_t>>();

    if (c.N_values.empty())
        throw std::runtime_error("N_values must not be empty");

    if (j.contains("families")) {
        c.families.clear();
        for (const auto& s : j["families"]) c.families.push_back(family_from_string(s));
    }
    if (j.contains("regimes")) {
        c.regimes.clear();
        for (const auto& s : j["regimes"]) c.regimes.push_back(regime_from_string(s));
    }

    if (j.contains("bvp")) {
        const auto& b = j["bvp"];
        c.bvp.max_iter   = b.value("max_iter",   50);
        c.bvp.tol        = b.value("tol",        1e-6);
        c.bvp.rk4_steps  = b.value("rk4_steps",  100);
        c.bvp.fd_epsilon = b.value("fd_epsilon", 1e-7);
    }

    if (j.contains("sampling")) {
        const auto& s = j["sampling"];
        c.sample.s_min      = s.value("s_min",      0.05);
        c.sample.s_max      = s.value("s_max",      0.20);
        c.sample.halo_w_min = s.value("halo_w_min", 0.15);
        c.sample.halo_w_max = s.value("halo_w_max", 0.40);
        c.sample.halo_k_min = s.value("halo_k_min", 2.5);
        c.sample.halo_k_max = s.value("halo_k_max", 6.0);
    }

    return c;
}

void write_meta(const Config& cfg) {
    json j;
    j["base_seed"]   = cfg.base_seed;
    j["replicates"]  = cfg.replicates;
    j["N_max"]       = cfg.N_max();    // derived, kept for reference
    j["N_values"]    = cfg.N_values;
    j["K_NEIGHBORS"] = K_NEIGHBORS;

    json fams = json::array();
    for (auto f : cfg.families) fams.push_back(to_string(f));
    j["families"] = std::move(fams);

    json regs = json::array();
    for (auto r : cfg.regimes) regs.push_back(to_string(r));
    j["regimes"] = std::move(regs);

    j["bvp"] = {
        {"max_iter",   cfg.bvp.max_iter},
        {"tol",        cfg.bvp.tol},
        {"rk4_steps",  cfg.bvp.rk4_steps},
        {"fd_epsilon", cfg.bvp.fd_epsilon},
    };

    j["sampling"] = {
        {"s_min",      cfg.sample.s_min},
        {"s_max",      cfg.sample.s_max},
        {"halo_w_min", cfg.sample.halo_w_min},
        {"halo_w_max", cfg.sample.halo_w_max},
        {"halo_k_min", cfg.sample.halo_k_min},
        {"halo_k_max", cfg.sample.halo_k_max},
    };

    std::ofstream f(cfg.output_dir / "meta.json");
    f << j.dump(2) << "\n";
}

SampleOptions sample_opts_for(Regime reg, const SampleOptions& base) {
    SampleOptions o = base;
    if (reg == Regime::M2 || reg == Regime::L2)
        o.proportions = {0.5, 0.5};
    else if (reg == Regime::M3 || reg == Regime::L3)
        o.proportions = {1.0/3.0, 1.0/3.0, 1.0/3.0};
    return o;
}

// Stateless criteria per family. FR is handled separately because it
// needs warm-start state.
template <typename D> struct Criteria;

template <> struct Criteria<Gaussian> {
    auto h() const { return [](const Gaussian& p, const Gaussian& q) { return gaussian::hellinger(p, q); }; }
    auto t() const { return [](const Gaussian& p, const Gaussian& q) { return gaussian::tangent_norm(p, q); }; }
    auto s() const {
        return [](const Gaussian& p, const Gaussian& q) {
            gaussian::SegmentOptions opts;
            opts.use_closed_form = true;
            return gaussian::segment(p, q, opts);
        };
    }
    auto e() const { return [](const Gaussian& p, const Gaussian& q) { return gaussian::euclidean(p, q); }; }
};

template <> struct Criteria<Gamma> {
    auto h() const { return [](const Gamma& p, const Gamma& q) { return gamma::hellinger(p, q); }; }
    auto t() const { return [](const Gamma& p, const Gamma& q) { return gamma::tangent_norm(p, q); }; }
    auto s() const { return [](const Gamma& p, const Gamma& q) { return gamma::segment(p, q); }; }
    auto e() const { return [](const Gamma& p, const Gamma& q) { return gamma::euclidean(p, q); }; }
};

template <> struct Criteria<Beta> {
    auto h() const { return [](const Beta& p, const Beta& q) { return beta::hellinger(p, q); }; }
    auto t() const { return [](const Beta& p, const Beta& q) { return beta::tangent_norm(p, q); }; }
    auto s() const { return [](const Beta& p, const Beta& q) { return beta::segment(p, q); }; }
    auto e() const { return [](const Beta& p, const Beta& q) { return beta::euclidean(p, q); }; }
};

template <typename D>
std::vector<D> prefix_sorted(const std::vector<Point2D>& shuffled, std::size_t N) {
    std::vector<Point2D> p(shuffled.begin(), shuffled.begin() + N);
    sort_by_coordinates(p);
    std::vector<D> out;
    out.reserve(N);
    for (const auto& x : p) out.emplace_back(x[0], x[1]);
    return out;
}

void write_csv_headers(std::ofstream& pts, std::ofstream& tim) {
    pts << "N,criterion,i,rho,delta_h,delta_f";
    for (std::size_t k = 1; k <= K_NEIGHBORS; ++k)
        pts << ",r_h_" << k << ",r_f_" << k;
    pts << "\n";

    tim << "N,criterion,matrix_nanos,pairs,n_failed\n";
}

void write_criterion_rows(std::ofstream& out,
                          const std::vector<std::vector<double>>& D_crit,
                          const std::vector<std::vector<double>>& D_fr,
                          std::size_t N, const char* name, bool self_ref)
{
    for (std::size_t i = 0; i < N; ++i) {
        out << N << "," << name << "," << i;
        if (self_ref) {
            auto r = row_without_self(D_fr, i);
            std::sort(r.begin(), r.end());
            out << ",1.0," << r[0] << "," << r[0];
            for (std::size_t k = 0; k < K_NEIGHBORS; ++k)
                out << "," << r[k] << "," << r[k];
        } else {
            const auto s = compute_point_stats(D_crit, D_fr, i);
            out << "," << s.rho << "," << s.r_hat[0] << "," << s.r_fr[0];
            for (std::size_t k = 0; k < K_NEIGHBORS; ++k)
                out << "," << s.r_hat[k] << "," << s.r_fr[k];
        }
        out << "\n";
    }
}

template <typename D>
void run_cell(const std::vector<std::size_t>& N_values,
              const std::vector<Point2D>& shuffled,
              const Criteria<D>& cr,
              const ShootingOptions& bvp,
              std::ofstream& pts,
              std::ofstream& tim)
{
    for (std::size_t N : N_values) {
        const auto sample = prefix_sorted<D>(shuffled, N);

        // FR uses warm start. A fresh functor per N ensures state does
        // not leak between sample sizes.
        WarmFR<D> fr_fn;
        if constexpr (std::is_same_v<D, Gamma>) fr_fn.opts = bvp;
        if constexpr (std::is_same_v<D, Beta>)  fr_fn.opts = bvp;

        const auto t0 = Clock::now();
        const auto D_fr = distance_matrix_warm(sample, fr_fn);
        const auto t1 = Clock::now();
        const auto fr_ns = (std::uint64_t)
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        const std::size_t pairs = N * (N - 1) / 2;
        tim << N << ",fr," << fr_ns << "," << pairs
            << "," << D_fr.n_failed << "\n";

        write_criterion_rows(pts, D_fr.matrix, D_fr.matrix, N, "fr", true);

        auto emit = [&](const char* name, auto fn) {
            const auto u0 = Clock::now();
            const auto R = distance_matrix(sample, fn);
            const auto u1 = Clock::now();
            const auto ns = (std::uint64_t)
                std::chrono::duration_cast<std::chrono::nanoseconds>(u1 - u0).count();
            tim << N << "," << name << "," << ns << "," << pairs
                << "," << R.n_failed << "\n";
            write_criterion_rows(pts, R.matrix, D_fr.matrix, N, name, false);
        };
        emit("hellinger", cr.h());
        emit("tangent",   cr.t());
        emit("segment",   cr.s());
        emit("euclidean", cr.e());
    }
}

std::string cell_stem(Family f, Regime r, int rep) {
    return std::string(to_string(f)) + "_" + to_string(r)
         + "_" + std::to_string(rep);
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path cfg_path = (argc >= 2) ? argv[1] : "config.json";

    Config cfg;
    try {
        cfg = load_config(cfg_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "config error: %s\n", e.what());
        return 1;
    }

    fs::create_directories(cfg.output_dir);
    write_meta(cfg);

    std::vector<std::tuple<Family, Regime, int>> cells;
    for (Family f : cfg.families)
        for (Regime r : cfg.regimes)
            for (int i = 0; i < cfg.replicates; ++i)
                cells.emplace_back(f, r, i);

    std::fprintf(stderr, "total cells: %zu\n", cells.size());
    Progress progress(cells.size());

    for (const auto& [fam, reg, rep] : cells) {
        const std::string stem = cell_stem(fam, reg, rep);
        const fs::path pf = cfg.output_dir / (stem + ".csv");
        const fs::path tf = cfg.output_dir / (stem + ".timing.csv");

        if (fs::exists(pf) && fs::exists(tf)
            && fs::file_size(pf) > 0 && fs::file_size(tf) > 0) {
            progress.skip(stem);
            continue;
        }

        progress.begin(stem);
        const auto t0 = Clock::now();

        const auto seed = replicate_seed(cfg.base_seed, fam, reg, rep);
        const auto opts = sample_opts_for(reg, cfg.sample);
        auto result = generate_sample(fam, reg, cfg.N_max(), opts, seed);

        std::mt19937_64 rng(seed ^ 0xBEEFCAFEull);
        std::shuffle(result.points.begin(), result.points.end(), rng);

        const fs::path pts_tmp = pf.string() + ".tmp";
        const fs::path tim_tmp = tf.string() + ".tmp";

        {
            std::ofstream pts(pts_tmp);
            std::ofstream tim(tim_tmp);
            pts << std::setprecision(17);
            write_csv_headers(pts, tim);

            if (fam == Family::Gaussian) {
                run_cell<Gaussian>(cfg.N_values, result.points,
                                   Criteria<Gaussian>{}, cfg.bvp, pts, tim);
            } else if (fam == Family::Gamma) {
                run_cell<Gamma>(cfg.N_values, result.points,
                                Criteria<Gamma>{}, cfg.bvp, pts, tim);
            } else {
                run_cell<Beta>(cfg.N_values, result.points,
                               Criteria<Beta>{}, cfg.bvp, pts, tim);
            }
        }

        fs::rename(pts_tmp, pf);
        fs::rename(tim_tmp, tf);

        const auto t1 = Clock::now();
        progress.end(std::chrono::duration<double>(t1 - t0).count());
    }

    progress.finish();
    return 0;
}