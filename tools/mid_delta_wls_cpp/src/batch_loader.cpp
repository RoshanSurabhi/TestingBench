#include "batch_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr const char* kMarketPriceCandidates[] = {"call_mid", "call_price", "mid", "option_mid", "option_price"};
constexpr const char* kMarketVolCandidates[] = {"option_impl_vol", "impl_vol", "mid_impl_vol"};
constexpr const char* kBidVolCandidates[] = {"bid_impl_vol", "call_bid_impl_vol", "bid_vol", "call_bid_vol"};
constexpr const char* kAskVolCandidates[] = {"ask_impl_vol", "call_ask_impl_vol", "ask_vol", "call_ask_vol"};

struct SummaryRow {
    std::string subfolder;
    std::string timestamp;
    std::string date;
    int idx_in_bin = 0;
    double time_sec = std::numeric_limits<double>::quiet_NaN();
};

struct ExpRow {
    double fwd = std::numeric_limits<double>::quiet_NaN();
    double vol_time = std::numeric_limits<double>::quiet_NaN();
    double r = 0.0;
    double q = 0.0;
};

struct OptFitRow {
    double strike = std::numeric_limits<double>::quiet_NaN();
    double z = std::numeric_limits<double>::quiet_NaN();
    double fitted_vol = std::numeric_limits<double>::quiet_NaN();
    std::unordered_map<std::string, double> extras;
};

struct PriceRow {
    std::unordered_map<std::string, double> vals;
};

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> out;
    out.reserve(24);
    std::string field;
    bool in_quotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    field.push_back('"');
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                field.push_back(c);
            }
        } else if (c == ',') {
            out.push_back(field);
            field.clear();
        } else if (c == '"') {
            in_quotes = true;
        } else {
            field.push_back(c);
        }
    }
    out.push_back(field);
    return out;
}

std::vector<std::unordered_map<std::string, std::string>> read_csv_dict_rows(const fs::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open CSV: " + path.string());
    }
    std::string line;
    if (!std::getline(in, line)) {
        return {};
    }
    const auto headers = parse_csv_line(line);
    std::vector<std::unordered_map<std::string, std::string>> out;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = parse_csv_line(line);
        std::unordered_map<std::string, std::string> row;
        row.reserve(headers.size());
        for (std::size_t i = 0; i < headers.size(); ++i) {
            row[headers[i]] = (i < fields.size() ? fields[i] : "");
        }
        out.push_back(std::move(row));
    }
    return out;
}

std::optional<double> to_optional_double(const std::string& s) {
    if (s.empty()) {
        return std::nullopt;
    }
    try {
        const double v = std::stod(s);
        if (!std::isfinite(v)) {
            return std::nullopt;
        }
        return v;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<int> to_optional_int(const std::string& s) {
    if (s.empty()) {
        return std::nullopt;
    }
    try {
        std::size_t pos = 0;
        const int v = std::stoi(s, &pos, 10);
        if (pos != s.size()) {
            return std::nullopt;
        }
        return v;
    } catch (...) {
        return std::nullopt;
    }
}

std::string fmt_strike_key(double k) {
    const double r = std::round(k * 1e8) / 1e8;
    const double nr = std::round(r);
    if (std::abs(r - nr) < 1e-6) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(std::llround(nr)));
        return std::string(buf);
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.10g", k);
    return std::string(buf);
}

int days_from_civil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int>(doe) - 719468;
}

double parse_timestamp_to_seconds(const std::string& ts) {
    int y = 0;
    int mo = 0;
    int d = 0;
    int hh = 0;
    int mm = 0;
    int ss = 0;
    if (sscanf_s(ts.c_str(), "%d-%d-%d %d:%d:%d", &y, &mo, &d, &hh, &mm, &ss) != 6) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const int days = days_from_civil(y, static_cast<unsigned>(mo), static_cast<unsigned>(d));
    return static_cast<double>(days) * 86400.0 + static_cast<double>(hh * 3600 + mm * 60 + ss);
}

std::string extract_date(const std::string& ts) {
    const auto pos = ts.find(' ');
    return (pos == std::string::npos) ? ts : ts.substr(0, pos);
}

std::vector<std::string> split_csv_arg(const std::string& raw) {
    std::vector<std::string> out;
    std::stringstream ss(raw);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            out.push_back(tok);
        }
    }
    return out;
}

double pp_peizer_pratt(int n, double z) {
    const double dn = static_cast<double>(n);
    const double denom = dn + (1.0 / 3.0) + (0.1 / (dn + 1.0));
    const double exp_term = -((z / denom) * (z / denom)) * (dn + 1.0 / 6.0);
    const double inside = std::max(0.0, 1.0 - std::exp(exp_term));
    const double out = 0.5 + (z >= 0.0 ? 1.0 : -1.0) * 0.5 * std::sqrt(inside);
    return std::clamp(out, 1e-10, 1.0 - 1e-10);
}

double lr_d1(double fwd, double strike, double r, double vol_time, double t, double sigma, double q) {
    return (std::log(fwd / strike) + t * (r - q) + ((sigma * sigma) * vol_time / 2.0)) / (sigma * std::sqrt(vol_time));
}

std::optional<double> lr_call_price_from_fwd(
    double fwd,
    double strike,
    double sigma,
    double texp,
    double r,
    double q,
    int num_steps = 101
) {
    if (!(std::isfinite(fwd) && std::isfinite(strike) && std::isfinite(sigma) && std::isfinite(texp) &&
          fwd > 0.0 && strike > 0.0 && sigma > 0.0 && texp > 0.0)) {
        return std::nullopt;
    }
    int n = std::max(3, num_steps);
    if ((n % 2) == 0) {
        ++n;
    }
    const double dt = texp / static_cast<double>(n);
    if (!(dt > 0.0)) {
        return std::nullopt;
    }
    const double d1 = lr_d1(fwd, strike, r, texp, texp, sigma, q);
    const double d2 = d1 - sigma * std::sqrt(texp);
    const double p_prime = pp_peizer_pratt(n, d1);
    const double p = pp_peizer_pratt(n, d2);
    const double carry = std::exp((r - q) * dt);
    const double u = carry * (p_prime / p);
    const double d = carry * ((1.0 - p_prime) / (1.0 - p));
    if (!(std::isfinite(u) && std::isfinite(d) && u > 0.0 && d > 0.0)) {
        return std::nullopt;
    }
    const double disc = std::exp(-r * dt);
    std::vector<double> vals(static_cast<std::size_t>(n + 1), 0.0);
    double stock = fwd * std::pow(u, n);
    for (int j = 0; j <= n; ++j) {
        vals[static_cast<std::size_t>(j)] = std::max(stock - strike, 0.0);
        stock *= d / u;
    }
    for (int step = n - 1; step >= 0; --step) {
        stock = fwd * std::pow(u, step);
        for (int j = 0; j <= step; ++j) {
            const double cont = disc * (p * vals[static_cast<std::size_t>(j)] + (1.0 - p) * vals[static_cast<std::size_t>(j + 1)]);
            const double exer = std::max(stock - strike, 0.0);
            vals[static_cast<std::size_t>(j)] = std::max(exer, cont);
            stock *= d / u;
        }
    }
    return vals[0];
}

struct Greeks {
    double delta = std::numeric_limits<double>::quiet_NaN();
    double gamma = std::numeric_limits<double>::quiet_NaN();
    double vega = std::numeric_limits<double>::quiet_NaN();
};

std::optional<Greeks> lr_call_greeks_from_fwd(double fwd, double strike, double sigma, double texp, double r, double q) {
    const auto base_opt = lr_call_price_from_fwd(fwd, strike, sigma, texp, r, q);
    if (!base_opt.has_value()) {
        return std::nullopt;
    }
    const double base = *base_opt;
    const double h_f = std::max(0.01, std::abs(fwd) * 1e-4);
    const double h_s = std::max(1e-4, std::abs(sigma) * 1e-3);
    const double s_up = sigma + h_s;
    const double s_dn = std::max(1e-6, sigma - h_s);
    const double f_up = fwd + h_f;
    const double f_dn = std::max(1e-6, fwd - h_f);

    const auto c_fu = lr_call_price_from_fwd(f_up, strike, sigma, texp, r, q);
    const auto c_fd = lr_call_price_from_fwd(f_dn, strike, sigma, texp, r, q);
    const auto c_su = lr_call_price_from_fwd(fwd, strike, s_up, texp, r, q);
    const auto c_sd = lr_call_price_from_fwd(fwd, strike, s_dn, texp, r, q);
    if (!c_fu || !c_fd || !c_su || !c_sd) {
        return std::nullopt;
    }

    Greeks g;
    g.delta = (*c_fu - *c_fd) / (2.0 * h_f);
    g.gamma = (*c_fu - 2.0 * base + *c_fd) / (h_f * h_f);
    g.vega = (*c_su - *c_sd) / (s_up - s_dn);
    if (!(std::isfinite(g.delta) && std::isfinite(g.gamma) && std::isfinite(g.vega))) {
        return std::nullopt;
    }
    return g;
}

std::optional<double> lr_call_theta_from_fwd(double fwd, double strike, double sigma, double texp, double r, double q) {
    const auto c_now = lr_call_price_from_fwd(fwd, strike, sigma, texp, r, q);
    if (!c_now.has_value() || !(std::isfinite(texp) && texp > 0.0)) {
        return std::nullopt;
    }
    const double dt_year = 1.0 / (365.0 * 24.0 * 60.0);
    const double h = std::min(dt_year, std::max(texp * 0.5, 1e-8));
    if (!(h > 0.0)) {
        return std::nullopt;
    }
    const double t_next = std::max(texp - h, 1e-8);
    const auto c_next = lr_call_price_from_fwd(fwd, strike, sigma, t_next, r, q);
    if (!c_next.has_value()) {
        return std::nullopt;
    }
    return (*c_next - *c_now) / h;
}

std::optional<double> first_positive_from_row(
    const std::unordered_map<std::string, double>& extras,
    const std::vector<std::string>& cols
) {
    for (const auto& c : cols) {
        const auto it = extras.find(c);
        if (it != extras.end() && std::isfinite(it->second) && it->second > 0.0) {
            return it->second;
        }
    }
    return std::nullopt;
}

std::pair<std::optional<double>, std::string> market_price_from_row(
    const OptFitRow& row,
    double fwd,
    double texp
) {
    for (const auto* col : kMarketPriceCandidates) {
        const auto it = row.extras.find(col);
        if (it != row.extras.end()) {
            return {it->second, std::string(col)};
        }
    }
    for (const auto* col : kMarketVolCandidates) {
        const auto it = row.extras.find(col);
        if (it == row.extras.end() || !(it->second > 0.0)) {
            continue;
        }
        const auto p = lr_call_price_from_fwd(fwd, row.strike, it->second, texp, 0.0, 0.0);
        if (p.has_value()) {
            return {*p, std::string(col) + "_to_lr_price"};
        }
    }
    if (row.fitted_vol > 0.0) {
        const auto p = lr_call_price_from_fwd(fwd, row.strike, row.fitted_vol, texp, 0.0, 0.0);
        if (p.has_value()) {
            return {*p, "fitted_vol_to_lr_price"};
        }
    }
    return {std::nullopt, "unavailable"};
}

std::vector<SummaryRow> read_summary_rows(const fs::path& batch_dir, const std::set<std::string>& date_filter) {
    const auto raw_rows = read_csv_dict_rows(batch_dir / "batch_cvi_summary.csv");
    std::vector<SummaryRow> out;
    out.reserve(raw_rows.size());
    for (const auto& row : raw_rows) {
        const int ok = to_optional_int(row.count("ok") ? row.at("ok") : "0").value_or(0);
        if (ok != 1) {
            continue;
        }
        SummaryRow s;
        s.subfolder = row.at("subfolder");
        s.timestamp = row.at("timestamp");
        s.date = extract_date(s.timestamp);
        if (!date_filter.empty() && date_filter.find(s.date) == date_filter.end()) {
            continue;
        }
        s.idx_in_bin = to_optional_int(row.count("idx_in_bin") ? row.at("idx_in_bin") : "0").value_or(0);
        s.time_sec = parse_timestamp_to_seconds(s.timestamp);
        out.push_back(std::move(s));
    }
    std::sort(out.begin(), out.end(), [](const SummaryRow& a, const SummaryRow& b) {
        if (a.date != b.date) {
            return a.date < b.date;
        }
        return a.idx_in_bin < b.idx_in_bin;
    });
    return out;
}

std::map<int, ExpRow> read_expiry_rows(const fs::path& path) {
    const auto raw_rows = read_csv_dict_rows(path);
    std::map<int, ExpRow> out;
    for (const auto& row : raw_rows) {
        const auto idx = to_optional_int(row.count("expiry_idx") ? row.at("expiry_idx") : "");
        const auto fwd = to_optional_double(row.count("F") ? row.at("F") : "");
        const auto vol_time = to_optional_double(row.count("volTime") ? row.at("volTime") : "");
        if (!idx.has_value() || !fwd.has_value() || !vol_time.has_value()) {
            continue;
        }
        if (!(*fwd > 0.0 && *vol_time > 0.0)) {
            continue;
        }
        ExpRow e;
        e.fwd = *fwd;
        e.vol_time = *vol_time;
        e.r = to_optional_double(row.count("r") ? row.at("r") : "").value_or(0.0);
        e.q = to_optional_double(row.count("q") ? row.at("q") : "").value_or(0.0);
        out[*idx] = e;
    }
    return out;
}

std::map<int, std::vector<OptFitRow>> read_option_fit_by_expiry(const fs::path& path) {
    const auto raw_rows = read_csv_dict_rows(path);
    std::map<int, std::vector<OptFitRow>> out;
    for (const auto& row : raw_rows) {
        const auto ei = to_optional_int(row.count("expiry_index") ? row.at("expiry_index") : "");
        const auto strike = to_optional_double(row.count("strike") ? row.at("strike") : "");
        const auto z = to_optional_double(row.count("z") ? row.at("z") : "");
        const auto fv = to_optional_double(row.count("fitted_vol") ? row.at("fitted_vol") : "");
        if (!ei || !strike || !z || !fv) {
            continue;
        }
        if (!(std::isfinite(*strike) && std::isfinite(*z) && std::isfinite(*fv) && *fv > 0.0)) {
            continue;
        }
        OptFitRow rec;
        rec.strike = *strike;
        rec.z = *z;
        rec.fitted_vol = *fv;
        for (const auto& kv : row) {
            const auto v = to_optional_double(kv.second);
            if (v.has_value()) {
                rec.extras[kv.first] = *v;
            }
        }
        out[*ei].push_back(std::move(rec));
    }
    return out;
}

std::map<int, std::unordered_map<std::string, PriceRow>> read_price_cmp_by_expiry(const fs::path& path) {
    std::map<int, std::unordered_map<std::string, PriceRow>> out;
    if (!fs::is_regular_file(path)) {
        return out;
    }
    const auto raw_rows = read_csv_dict_rows(path);
    for (const auto& row : raw_rows) {
        const auto ei = to_optional_int(row.count("expiry_index") ? row.at("expiry_index") : "");
        const auto strike = to_optional_double(row.count("strike") ? row.at("strike") : "");
        if (!ei || !strike) {
            continue;
        }
        PriceRow rec;
        for (const auto& kv : row) {
            const auto v = to_optional_double(kv.second);
            if (v.has_value()) {
                rec.vals[kv.first] = *v;
            }
        }
        out[*ei][fmt_strike_key(*strike)] = std::move(rec);
    }
    return out;
}

template <std::size_t N>
std::vector<std::string> to_vec(const char* const (&arr)[N]) {
    std::vector<std::string> out;
    out.reserve(N);
    for (const auto* s : arr) {
        out.emplace_back(s);
    }
    return out;
}

}  // namespace

nlohmann::json build_engine_input_from_batch(const BuildInputConfig& cfg) {
    const fs::path batch_dir(cfg.batch_dir);
    if (!fs::is_directory(batch_dir)) {
        throw std::runtime_error("batch directory does not exist: " + cfg.batch_dir);
    }

    std::set<std::string> date_filter(cfg.dates.begin(), cfg.dates.end());
    auto summary = read_summary_rows(batch_dir, date_filter);
    if (summary.empty()) {
        throw std::runtime_error("no valid summary rows for requested dates");
    }

    std::set<int> requested_expiries;
    if (cfg.all_expiries) {
        const fs::path efq_path = batch_dir / summary.front().subfolder / "expiry_fwd_q.csv";
        const auto exp_rows = read_expiry_rows(efq_path);
        for (const auto& [idx, _row] : exp_rows) {
            requested_expiries.insert(idx);
        }
    } else {
        requested_expiries.insert(cfg.expiry_indices.begin(), cfg.expiry_indices.end());
    }
    if (requested_expiries.empty()) {
        throw std::runtime_error("no expiry indices selected");
    }

    std::map<std::pair<std::string, int>, std::vector<json>> grouped_snaps;
    const auto bid_cols = to_vec(kBidVolCandidates);
    const auto ask_cols = to_vec(kAskVolCandidates);
    const auto mid_cols = to_vec(kMarketVolCandidates);

    for (const auto& s : summary) {
        const fs::path sub = batch_dir / s.subfolder;
        const fs::path efq_path = sub / "expiry_fwd_q.csv";
        const fs::path opt_path = sub / "option_fit_comparison.csv";
        const fs::path px_path = sub / "price_comparison.csv";
        if (!fs::is_regular_file(efq_path) || !fs::is_regular_file(opt_path)) {
            continue;
        }

        const auto exp_rows = read_expiry_rows(efq_path);
        const auto by_exp = read_option_fit_by_expiry(opt_path);
        const auto price_by_exp = read_price_cmp_by_expiry(px_path);

        for (int ei : requested_expiries) {
            const auto it_exp = exp_rows.find(ei);
            if (it_exp == exp_rows.end()) {
                continue;
            }
            const auto it_opt = by_exp.find(ei);
            if (it_opt == by_exp.end() || it_opt->second.empty()) {
                continue;
            }
            const auto& exp = it_exp->second;
            const auto& opt_rows = it_opt->second;
            const auto it_price = price_by_exp.find(ei);
            const std::unordered_map<std::string, PriceRow>* px_map = (it_price == price_by_exp.end() ? nullptr : &it_price->second);

            json snap_rows = json::array();
            for (const auto& rr : opt_rows) {
                const std::string kkey = fmt_strike_key(rr.strike);
                const PriceRow* px_row = nullptr;
                if (px_map != nullptr) {
                    const auto itp = px_map->find(kkey);
                    if (itp != px_map->end()) {
                        px_row = &itp->second;
                    }
                }

                const auto [c_market, c_market_src] = market_price_from_row(rr, exp.fwd, exp.vol_time);
                std::optional<double> c_bid;
                std::optional<double> c_ask;
                std::optional<double> c_mid;
                if (px_row != nullptr) {
                    const auto itb = px_row->vals.find("market_call_bid");
                    if (itb != px_row->vals.end()) c_bid = itb->second;
                    const auto ita = px_row->vals.find("market_call_ask");
                    if (ita != px_row->vals.end()) c_ask = ita->second;
                    const auto itm = px_row->vals.find("market_call_mid");
                    if (itm != px_row->vals.end()) c_mid = itm->second;
                }
                if (!c_mid.has_value() && c_bid && c_ask) {
                    c_mid = 0.5 * (*c_bid + *c_ask);
                }

                const auto bid_vol = first_positive_from_row(rr.extras, bid_cols);
                const auto ask_vol = first_positive_from_row(rr.extras, ask_cols);
                const auto mid_vol = first_positive_from_row(rr.extras, mid_cols);

                // Match Python path: side LR repricing from implied vols uses zero carry inputs (r=0, q=0).
                auto c_bid_lr = bid_vol ? lr_call_price_from_fwd(exp.fwd, rr.strike, *bid_vol, exp.vol_time, 0.0, 0.0) : std::nullopt;
                auto c_ask_lr = ask_vol ? lr_call_price_from_fwd(exp.fwd, rr.strike, *ask_vol, exp.vol_time, 0.0, 0.0) : std::nullopt;
                auto c_mid_lr = mid_vol ? lr_call_price_from_fwd(exp.fwd, rr.strike, *mid_vol, exp.vol_time, 0.0, 0.0) : std::nullopt;

                std::string bid_src = "unavailable";
                std::string ask_src = "unavailable";
                std::string mid_src = "unavailable";

                if (!c_bid_lr && c_bid) {
                    c_bid_lr = c_bid;
                    bid_src = "market_call_bid";
                } else if (bid_vol) {
                    bid_src = "bid_impl_vol_to_lr_price";
                }
                if (!c_ask_lr && c_ask) {
                    c_ask_lr = c_ask;
                    ask_src = "market_call_ask";
                } else if (ask_vol) {
                    ask_src = "ask_impl_vol_to_lr_price";
                }
                if (!c_mid_lr && c_mid) {
                    c_mid_lr = c_mid;
                    mid_src = "market_call_mid";
                } else if (mid_vol) {
                    mid_src = "option_impl_vol_to_lr_price";
                }

                std::optional<double> c_fit;
                std::string c_fit_src = "unavailable";
                if (px_row != nullptr) {
                    const auto itf = px_row->vals.find("fitted_call");
                    if (itf != px_row->vals.end()) {
                        c_fit = itf->second;
                        c_fit_src = "price_comparison.fitted_call";
                    }
                }
                if (!c_fit) {
                    c_fit = lr_call_price_from_fwd(exp.fwd, rr.strike, rr.fitted_vol, exp.vol_time, exp.r, exp.q);
                    if (c_fit) {
                        c_fit_src = "fitted_vol_to_lr_price";
                    }
                }

                const auto greeks = lr_call_greeks_from_fwd(exp.fwd, rr.strike, rr.fitted_vol, exp.vol_time, exp.r, exp.q);
                const auto theta = lr_call_theta_from_fwd(exp.fwd, rr.strike, rr.fitted_vol, exp.vol_time, exp.r, exp.q);

                json out = {
                    {"key", kkey},
                    {"strike", rr.strike},
                    {"z", rr.z},
                    {"sigma", rr.fitted_vol},
                    {"c_market", c_market ? json(*c_market) : json(nullptr)},
                    {"c_market_source", c_market_src},
                    {"c_bid", c_bid ? json(*c_bid) : json(nullptr)},
                    {"c_ask", c_ask ? json(*c_ask) : json(nullptr)},
                    {"c_mid", c_mid ? json(*c_mid) : json(nullptr)},
                    {"c_bid_lr", c_bid_lr ? json(*c_bid_lr) : json(nullptr)},
                    {"c_ask_lr", c_ask_lr ? json(*c_ask_lr) : json(nullptr)},
                    {"c_mid_lr", c_mid_lr ? json(*c_mid_lr) : json(nullptr)},
                    {"bid_impl_vol", bid_vol ? json(*bid_vol) : json(nullptr)},
                    {"ask_impl_vol", ask_vol ? json(*ask_vol) : json(nullptr)},
                    {"mid_impl_vol", mid_vol ? json(*mid_vol) : json(nullptr)},
                    {"bid_price_source", bid_src},
                    {"ask_price_source", ask_src},
                    {"mid_price_source", mid_src},
                    {"c_fit", c_fit ? json(*c_fit) : json(nullptr)},
                    {"c_fit_source", c_fit_src},
                    {"delta_bs", (greeks ? json(greeks->delta) : json(nullptr))},
                    {"gamma_bs", (greeks ? json(greeks->gamma) : json(nullptr))},
                    {"vega_bs", (greeks ? json(greeks->vega) : json(nullptr))},
                    {"theta_bs", (theta ? json(*theta) : json(nullptr))},
                };
                snap_rows.push_back(std::move(out));
            }

            if (!snap_rows.empty()) {
                grouped_snaps[{s.date, ei}].push_back(
                    json{
                        {"timestamp", s.timestamp},
                        {"time_sec", s.time_sec},
                        {"fwd", exp.fwd},
                        {"rows", std::move(snap_rows)},
                    }
                );
            }
        }
    }

    json groups = json::array();
    for (auto& [key, snaps] : grouped_snaps) {
        std::sort(snaps.begin(), snaps.end(), [](const json& a, const json& b) {
            return a.value("time_sec", 0.0) < b.value("time_sec", 0.0);
        });
        groups.push_back(
            json{
                {"date", key.first},
                {"expiry_index", key.second},
                {"snaps", std::move(snaps)},
            }
        );
    }

    std::set<std::string> out_dates;
    std::set<int> out_expiries;
    for (const auto& g : groups) {
        out_dates.insert(g.value("date", ""));
        out_expiries.insert(g.value("expiry_index", -1));
    }

    return json{
        {"window_min", cfg.holding_min},
        {"snapshot_spacing_min", cfg.entry_spacing_min},
        {"min_abs_df_frac", cfg.min_abs_df_frac},
        {"eps_delta", cfg.eps_delta},
        {"n_days", static_cast<int>(out_dates.size())},
        {"n_expiries", static_cast<int>(out_expiries.size())},
        {"groups", std::move(groups)},
    };
}
