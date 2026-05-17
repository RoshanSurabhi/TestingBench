#include "decouplingEngine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "wls_shared.h"

namespace {

using json = nlohmann::json;

/** nlohmann `value(k, nullptr)` throws if k exists and holds a number; use this for optional fields. */
json JsonAtOrNull(const json& j, const char* key)
{
    const auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        return nullptr;
    }
    return *it;
}

int JsonToInt(const json& j, int fallback = -1)
{
    if (j.is_number_integer()) {
        return j.get<int>();
    }
    if (j.is_number_float()) {
        return static_cast<int>(std::lround(j.get<double>()));
    }
    if (j.is_number_unsigned()) {
        return static_cast<int>(j.get<unsigned long long>());
    }
    return fallback;
}

std::optional<double> GetOptionalDouble(const json& obj, const char* key)
{
    auto it = obj.find(key);
    if (it == obj.end() || it->is_null()) {
        return std::nullopt;
    }
    try {
        const double v = it->get<double>();
        if (!std::isfinite(v)) {
            return std::nullopt;
        }
        return v;
    }
    catch (...) {
        return std::nullopt;
    }
}

double GetDoubleOr(const json& obj, const char* key, double fallback)
{
    auto v = GetOptionalDouble(obj, key);
    return v.has_value() ? *v : fallback;
}

double PearsonCorr(const std::vector<double>& xs, const std::vector<double>& ys)
{
    const std::size_t n = xs.size();
    if (n < 2 || ys.size() != n) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double mx = std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(n);
    const double my = std::accumulate(ys.begin(), ys.end(), 0.0) / static_cast<double>(n);
    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = xs[i] - mx;
        const double dy = ys[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if (sxx <= 1e-18 || syy <= 1e-18) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double c = sxy / std::sqrt(sxx * syy);
    return std::isfinite(c) ? c : std::numeric_limits<double>::quiet_NaN();
}

double Median(std::vector<double> vals)
{
    if (vals.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const std::size_t n = vals.size();
    const std::size_t mid = n / 2;
    std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(mid), vals.end());
    if ((n % 2) == 1) {
        return vals[mid];
    }
    const double hi = vals[mid];
    std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(mid - 1), vals.begin() + static_cast<std::ptrdiff_t>(mid));
    const double lo = vals[mid - 1];
    return 0.5 * (lo + hi);
}

double Rms(const std::vector<double>& vals)
{
    if (vals.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double ss = 0.0;
    for (double v : vals) {
        ss += v * v;
    }
    return std::sqrt(ss / static_cast<double>(vals.size()));
}

json NullOrDouble(double v)
{
    if (!std::isfinite(v)) {
        return nullptr;
    }
    return v;
}

std::string FmtStrikeKey(double k)
{
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

struct MidWlsFitResult {
    double alpha = std::numeric_limits<double>::quiet_NaN();
    double beta = std::numeric_limits<double>::quiet_NaN();
    double r2 = std::numeric_limits<double>::quiet_NaN();
    bool ok = false;
};

MidWlsFitResult WlsMidDeltaAlphaBetaR2(const std::vector<double>& xs, const std::vector<double>& ys)
{
    MidWlsFitResult out;
    const auto fit = trdbclient::decoupling::ComputeInverseBivariateMoveWls(xs, ys);
    out.alpha = fit.alpha;
    out.beta = fit.beta;
    out.r2 = fit.r2;
    out.ok = fit.ok;
    return out;
}

struct StrikeMidPts {
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    double strikeFloat = std::numeric_limits<double>::quiet_NaN();
};

json BuildMidWlsPerStrike(const json& rowsOut)
{
    using GKey = std::tuple<std::string, int, std::string>;
    std::map<GKey, StrikeMidPts> groups;
    for (const auto& row : rowsOut) {
        if (!row.is_object()) {
            continue;
        }
        int valid = 0;
        if (row.contains("valid") && row["valid"].is_number()) {
            valid = JsonToInt(row["valid"], 0);
        }
        if (valid != 1) {
            continue;
        }
        const auto xopt = GetOptionalDouble(row, "delta_bs_entry");
        const auto yopt = GetOptionalDouble(row, "delta_realized_mid");
        if (!xopt.has_value() || !yopt.has_value()) {
            continue;
        }
        if (!row.contains("strike") || !row["strike"].is_number()) {
            continue;
        }
        const double sk = row["strike"].get<double>();
        if (!row.contains("date") || !row["date"].is_string()) {
            continue;
        }
        const std::string day = row["date"].get<std::string>();
        if (!row.contains("expiry_index") || !row["expiry_index"].is_number()) {
            continue;
        }
        const int ei = JsonToInt(row["expiry_index"], -1);
        if (ei < 0) {
            continue;
        }
        const std::string kkey = FmtStrikeKey(sk);
        const GKey gkey{day, ei, kkey};
        StrikeMidPts& g = groups[gkey];
        g.xs.push_back(*xopt);
        g.ys.push_back(*yopt);
        g.strikeFloat = sk;
        const auto zopt = GetOptionalDouble(row, "z_entry");
        if (zopt.has_value()) {
            g.zs.push_back(*zopt);
        }
    }

    std::vector<std::pair<GKey, StrikeMidPts>> items(groups.begin(), groups.end());
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (std::get<0>(a.first) != std::get<0>(b.first)) {
            return std::get<0>(a.first) < std::get<0>(b.first);
        }
        if (std::get<1>(a.first) != std::get<1>(b.first)) {
            return std::get<1>(a.first) < std::get<1>(b.first);
        }
        const double sa = a.second.strikeFloat;
        const double sb = b.second.strikeFloat;
        if (sa != sb) {
            return sa < sb;
        }
        return std::get<2>(a.first) < std::get<2>(b.first);
    });

    json arr = json::array();
    for (const auto& ent : items) {
        const GKey& key = ent.first;
        const StrikeMidPts& g = ent.second;
        const std::string& day = std::get<0>(key);
        const int ei = std::get<1>(key);
        MidWlsFitResult fit = WlsMidDeltaAlphaBetaR2(g.xs, g.ys);
        std::vector<double> zs_copy = g.zs;
        const double zmed = Median(std::move(zs_copy));
        json orow = {
            {"date", day},
            {"expiry_index", ei},
            {"strike", g.strikeFloat},
            {"z_entry_median", NullOrDouble(zmed)},
            {"n", static_cast<int>(g.xs.size())},
            {"alpha", nullptr},
            {"beta", nullptr},
            {"r2_wls", nullptr},
            {"weight_scheme", "w_i = 1/(eps + x_i^2 + y_i^2), eps=1e-8"},
            {"model", "delta_realized_mid ~ alpha + beta * delta_bs_entry (per strike)"},
        };
        if (fit.ok && g.xs.size() >= 2) {
            orow["alpha"] = fit.alpha;
            orow["beta"] = fit.beta;
            orow["r2_wls"] = NullOrDouble(fit.r2);
        }
        arr.push_back(std::move(orow));
    }
    return arr;
}

int FindForwardIdx(const std::vector<double>& timesSec, int i, double horizonMin, double maxSlipMin)
{
    const double ti = timesSec[static_cast<std::size_t>(i)];
    const double target = ti + horizonMin * 60.0;
    const double limit = target + maxSlipMin * 60.0;
    for (std::size_t j = static_cast<std::size_t>(i + 1); j < timesSec.size(); ++j) {
        const double tj = timesSec[j];
        if (!std::isfinite(tj)) {
            continue;
        }
        if (tj < target) {
            continue;
        }
        if (tj <= limit) {
            return static_cast<int>(j);
        }
        return -1;
    }
    return -1;
}

} // namespace

int RunDecouplingEngineFromStdinToStdout()
{
    try {
        json input = json::parse(std::cin);
        const double windowMin = GetDoubleOr(input, "window_min", 5.0);
        const double spacingMin = GetDoubleOr(input, "snapshot_spacing_min", 5.0);
        const double minAbsDfFrac = GetDoubleOr(input, "min_abs_df_frac", 0.0005);
        const double epsDelta = GetDoubleOr(input, "eps_delta", 1e-8);

        json rowsOut = json::array();
        json summaryOut = json::array();
        json diag = {
            {"n_days", input.value("n_days", 0)},
            {"n_expiries", input.value("n_expiries", 0)},
            {"n_rows_total", 0},
            {"n_rows_valid", 0},
            {"n_skipped_small_df", 0},
            {"n_skipped_missing_price", 0},
            {"n_skipped_missing_sigma", 0},
            {"n_skipped_missing_greeks", 0},
            {"n_skipped_missing_exit_strike", 0},
            {"n_skipped_other", 0},
        };

        const json groups = input.value("groups", json::array());
        for (const auto& group : groups) {
            const std::string day = group.value("date", "");
            const int expiryIndex =
                group.contains("expiry_index") && group["expiry_index"].is_number()
                    ? JsonToInt(group["expiry_index"], -1)
                    : -1;
            const json snaps = group.value("snaps", json::array());
            if (!snaps.is_array() || snaps.size() < 2) {
                continue;
            }

            std::vector<double> spotVals;
            spotVals.reserve(snaps.size());
            std::vector<double> timesSec;
            timesSec.reserve(snaps.size());
            for (const auto& s : snaps) {
                const double fwd = GetDoubleOr(s, "fwd", std::numeric_limits<double>::quiet_NaN());
                if (std::isfinite(fwd)) {
                    spotVals.push_back(fwd);
                }
                timesSec.push_back(GetDoubleOr(s, "time_sec", std::numeric_limits<double>::quiet_NaN()));
            }

            std::vector<int> entryIdx;
            entryIdx.reserve(snaps.size());
            const double spacingSec = spacingMin * 60.0;
            double lastT = std::numeric_limits<double>::quiet_NaN();
            for (int i = 0; i < static_cast<int>(timesSec.size()); ++i) {
                const double ti = timesSec[static_cast<std::size_t>(i)];
                if (!std::isfinite(ti)) {
                    continue;
                }
                if (!std::isfinite(lastT) || (ti - lastT) >= spacingSec) {
                    entryIdx.push_back(i);
                    lastT = ti;
                }
            }

            std::vector<double> valsDec;
            std::vector<double> valsDecBid;
            std::vector<double> valsDecAsk;
            std::vector<double> valsDecMid;
            std::vector<double> valsDecFit;
            std::vector<double> valsDr;
            std::vector<double> valsDb;
            std::vector<double> valsDrBid;
            std::vector<double> valsDbBid;
            std::vector<double> valsDrAsk;
            std::vector<double> valsDbAsk;
            std::vector<double> valsDrMid;
            std::vector<double> valsDbMid;
            std::vector<double> valsDrFit;
            std::vector<double> valsDbFit;
            int nTotalGroup = 0;
            int nValidGroup = 0;

            for (const int i : entryIdx) {
                const int j = FindForwardIdx(timesSec, i, windowMin, 2.0);
                if (j < 0) {
                    continue;
                }
                const auto& s0 = snaps[static_cast<std::size_t>(i)];
                const auto& s1 = snaps[static_cast<std::size_t>(j)];
                const double dtYear = (timesSec[static_cast<std::size_t>(j)] - timesSec[static_cast<std::size_t>(i)]) / (365.0 * 24.0 * 3600.0);
                const double f0 = GetDoubleOr(s0, "fwd", std::numeric_limits<double>::quiet_NaN());
                const double f1 = GetDoubleOr(s1, "fwd", std::numeric_limits<double>::quiet_NaN());
                if (!std::isfinite(f0) || !std::isfinite(f1)) {
                    continue;
                }
                const double dF = f1 - f0;
                const double dFFloor = minAbsDfFrac * f0;

                std::unordered_map<std::string, const json*> exitByKey;
                const json rows1 = s1.value("rows", json::array());
                exitByKey.reserve(rows1.size());
                for (const auto& r1 : rows1) {
                    const std::string key = r1.value("key", "");
                    if (!key.empty()) {
                        exitByKey[key] = &r1;
                    }
                }

                const json rows0 = s0.value("rows", json::array());
                for (const auto& r0 : rows0) {
                    ++nTotalGroup;
                    diag["n_rows_total"] = diag["n_rows_total"].get<int>() + 1;
                    json rowOut = {
                        {"date", day},
                        {"expiry_index", expiryIndex},
                        {"from_t", s0.value("timestamp", "")},
                        {"to_t", s1.value("timestamp", "")},
                        {"strike", JsonAtOrNull(r0, "strike")},
                        {"z_entry", JsonAtOrNull(r0, "z")},
                        {"F_entry", f0},
                        {"F_exit", f1},
                        {"dF", dF},
                        {"dF_floor", dFFloor},
                        {"dt_years", dtYear},
                        {"sigma_entry", JsonAtOrNull(r0, "sigma")},
                        {"sigma_exit", nullptr},
                        {"dSigma", nullptr},
                        {"c_bid_quote_entry", JsonAtOrNull(r0, "c_bid")},
                        {"c_bid_entry", JsonAtOrNull(r0, "c_bid_lr")},
                        {"c_bid_quote_exit", nullptr},
                        {"c_bid_exit", nullptr},
                        {"c_ask_quote_entry", JsonAtOrNull(r0, "c_ask")},
                        {"c_ask_entry", JsonAtOrNull(r0, "c_ask_lr")},
                        {"c_ask_quote_exit", nullptr},
                        {"c_ask_exit", nullptr},
                        {"c_mid_quote_entry", JsonAtOrNull(r0, "c_mid")},
                        {"c_mid_entry", JsonAtOrNull(r0, "c_mid_lr")},
                        {"c_mid_quote_exit", nullptr},
                        {"c_mid_exit", nullptr},
                        {"bid_impl_vol_entry", JsonAtOrNull(r0, "bid_impl_vol")},
                        {"bid_impl_vol_exit", nullptr},
                        {"ask_impl_vol_entry", JsonAtOrNull(r0, "ask_impl_vol")},
                        {"ask_impl_vol_exit", nullptr},
                        {"mid_impl_vol_entry", JsonAtOrNull(r0, "mid_impl_vol")},
                        {"mid_impl_vol_exit", nullptr},
                        {"c_entry", JsonAtOrNull(r0, "c_market")},
                        {"c_exit", nullptr},
                        {"c_fit_entry", JsonAtOrNull(r0, "c_fit")},
                        {"c_fit_exit", nullptr},
                        {"dC", nullptr},
                        {"dC_bid", nullptr},
                        {"dC_ask", nullptr},
                        {"dC_mid", nullptr},
                        {"dC_fit", nullptr},
                        {"delta_bs_entry", JsonAtOrNull(r0, "delta_bs")},
                        {"gamma_bs_entry", JsonAtOrNull(r0, "gamma_bs")},
                        {"vega_bs_entry", JsonAtOrNull(r0, "vega_bs")},
                        {"theta_bs_entry", JsonAtOrNull(r0, "theta_bs")},
                        {"dc_tilde", nullptr},
                        {"dc_tilde_bid", nullptr},
                        {"dc_tilde_ask", nullptr},
                        {"dc_tilde_mid", nullptr},
                        {"dc_tilde_fit", nullptr},
                        {"delta_realized", nullptr},
                        {"delta_realized_bid", nullptr},
                        {"delta_realized_ask", nullptr},
                        {"delta_realized_mid", nullptr},
                        {"delta_realized_fit", nullptr},
                        {"decoupling", nullptr},
                        {"decoupling_bid", nullptr},
                        {"decoupling_ask", nullptr},
                        {"decoupling_mid", nullptr},
                        {"decoupling_fit", nullptr},
                        {"decoupling_normalized", nullptr},
                        {"decoupling_normalized_bid", nullptr},
                        {"decoupling_normalized_ask", nullptr},
                        {"decoupling_normalized_mid", nullptr},
                        {"decoupling_normalized_fit", nullptr},
                        {"entry_price_source", JsonAtOrNull(r0, "c_market_source")},
                        {"exit_price_source", nullptr},
                        {"entry_bid_price_source", JsonAtOrNull(r0, "bid_price_source")},
                        {"exit_bid_price_source", nullptr},
                        {"entry_ask_price_source", JsonAtOrNull(r0, "ask_price_source")},
                        {"exit_ask_price_source", nullptr},
                        {"entry_mid_price_source", JsonAtOrNull(r0, "mid_price_source")},
                        {"exit_mid_price_source", nullptr},
                        {"entry_fit_price_source", JsonAtOrNull(r0, "c_fit_source")},
                        {"exit_fit_price_source", nullptr},
                        {"valid", 0},
                        {"skip_reason", nullptr},
                    };

                    const std::string key = r0.value("key", "");
                    const auto itExit = exitByKey.find(key);
                    if (itExit == exitByKey.end()) {
                        rowOut["skip_reason"] = "missing_exit_strike";
                        diag["n_skipped_missing_exit_strike"] = diag["n_skipped_missing_exit_strike"].get<int>() + 1;
                        rowsOut.push_back(std::move(rowOut));
                        continue;
                    }
                    const json& r1 = *(itExit->second);
                    rowOut["sigma_exit"] = JsonAtOrNull(r1, "sigma");
                    rowOut["exit_price_source"] = JsonAtOrNull(r1, "c_market_source");
                    rowOut["c_bid_quote_exit"] = JsonAtOrNull(r1, "c_bid");
                    rowOut["c_bid_exit"] = JsonAtOrNull(r1, "c_bid_lr");
                    rowOut["c_ask_quote_exit"] = JsonAtOrNull(r1, "c_ask");
                    rowOut["c_ask_exit"] = JsonAtOrNull(r1, "c_ask_lr");
                    rowOut["c_mid_quote_exit"] = JsonAtOrNull(r1, "c_mid");
                    rowOut["c_mid_exit"] = JsonAtOrNull(r1, "c_mid_lr");
                    rowOut["bid_impl_vol_exit"] = JsonAtOrNull(r1, "bid_impl_vol");
                    rowOut["ask_impl_vol_exit"] = JsonAtOrNull(r1, "ask_impl_vol");
                    rowOut["mid_impl_vol_exit"] = JsonAtOrNull(r1, "mid_impl_vol");
                    rowOut["exit_bid_price_source"] = JsonAtOrNull(r1, "bid_price_source");
                    rowOut["exit_ask_price_source"] = JsonAtOrNull(r1, "ask_price_source");
                    rowOut["exit_mid_price_source"] = JsonAtOrNull(r1, "mid_price_source");
                    rowOut["c_fit_exit"] = JsonAtOrNull(r1, "c_fit");
                    rowOut["exit_fit_price_source"] = JsonAtOrNull(r1, "c_fit_source");

                    const auto c0 = GetOptionalDouble(r0, "c_market");
                    const auto c1 = GetOptionalDouble(r1, "c_market");
                    if (!c0.has_value() || !c1.has_value()) {
                        rowOut["skip_reason"] = "missing_market_price";
                        diag["n_skipped_missing_price"] = diag["n_skipped_missing_price"].get<int>() + 1;
                        rowsOut.push_back(std::move(rowOut));
                        continue;
                    }
                    if (std::abs(dF) < dFFloor) {
                        rowOut["skip_reason"] = "small_dF";
                        diag["n_skipped_small_df"] = diag["n_skipped_small_df"].get<int>() + 1;
                        rowsOut.push_back(std::move(rowOut));
                        continue;
                    }

                    const auto sigma0 = GetOptionalDouble(r0, "sigma");
                    const auto sigma1 = GetOptionalDouble(r1, "sigma");
                    if (!sigma0.has_value() || !sigma1.has_value()) {
                        rowOut["skip_reason"] = "missing_sigma";
                        diag["n_skipped_missing_sigma"] = diag["n_skipped_missing_sigma"].get<int>() + 1;
                        rowsOut.push_back(std::move(rowOut));
                        continue;
                    }

                    const auto delta0 = GetOptionalDouble(r0, "delta_bs");
                    const auto gamma0 = GetOptionalDouble(r0, "gamma_bs");
                    const auto vega0 = GetOptionalDouble(r0, "vega_bs");
                    const auto theta0 = GetOptionalDouble(r0, "theta_bs");
                    if (!delta0.has_value() || !gamma0.has_value() || !vega0.has_value() || !theta0.has_value()) {
                        rowOut["skip_reason"] = "missing_greeks";
                        diag["n_skipped_missing_greeks"] = diag["n_skipped_missing_greeks"].get<int>() + 1;
                        rowsOut.push_back(std::move(rowOut));
                        continue;
                    }

                    const double dSigma = *sigma1 - *sigma0;
                    const double dC = *c1 - *c0;
                    const double dcTilde = dC - (*vega0) * dSigma - 0.5 * (*gamma0) * dF * dF - (*theta0) * dtYear;
                    const double deltaReal = dcTilde / dF;
                    const double dec = deltaReal - *delta0;
                    const double decNorm = dec / std::max(std::abs(*delta0), epsDelta);

                    rowOut["dSigma"] = dSigma;
                    rowOut["dC"] = dC;
                    rowOut["c_exit"] = *c1;
                    rowOut["dc_tilde"] = dcTilde;
                    rowOut["delta_realized"] = deltaReal;
                    rowOut["decoupling"] = dec;
                    rowOut["decoupling_normalized"] = decNorm;

                    for (const auto& side : std::vector<std::pair<std::string, std::string>>{
                             {"bid", "c_bid_lr"}, {"ask", "c_ask_lr"}, {"mid", "c_mid_lr"}}) {
                        const auto p0 = GetOptionalDouble(r0, side.second.c_str());
                        const auto p1 = GetOptionalDouble(r1, side.second.c_str());
                        if (!p0.has_value() || !p1.has_value()) {
                            continue;
                        }
                        const double dCSide = *p1 - *p0;
                        const double dcTildeSide = dCSide - (*vega0) * dSigma - 0.5 * (*gamma0) * dF * dF - (*theta0) * dtYear;
                        const double deltaSide = dcTildeSide / dF;
                        const double decSide = deltaSide - *delta0;
                        rowOut[std::string("dC_") + side.first] = dCSide;
                        rowOut[std::string("dc_tilde_") + side.first] = dcTildeSide;
                        rowOut[std::string("delta_realized_") + side.first] = deltaSide;
                        rowOut[std::string("decoupling_") + side.first] = decSide;
                        rowOut[std::string("decoupling_normalized_") + side.first] =
                            decSide / std::max(std::abs(*delta0), epsDelta);
                    }

                    const auto c0Fit = GetOptionalDouble(r0, "c_fit");
                    const auto c1Fit = GetOptionalDouble(r1, "c_fit");
                    if (c0Fit.has_value() && c1Fit.has_value()) {
                        const double dCFit = *c1Fit - *c0Fit;
                        const double dcTildeFit = dCFit - (*vega0) * dSigma - 0.5 * (*gamma0) * dF * dF - (*theta0) * dtYear;
                        const double deltaFit = dcTildeFit / dF;
                        const double decFit = deltaFit - *delta0;
                        rowOut["dC_fit"] = dCFit;
                        rowOut["dc_tilde_fit"] = dcTildeFit;
                        rowOut["delta_realized_fit"] = deltaFit;
                        rowOut["decoupling_fit"] = decFit;
                        rowOut["decoupling_normalized_fit"] = decFit / std::max(std::abs(*delta0), epsDelta);
                    }

                    rowOut["valid"] = 1;
                    ++nValidGroup;
                    diag["n_rows_valid"] = diag["n_rows_valid"].get<int>() + 1;

                    valsDec.push_back(dec);
                    valsDr.push_back(deltaReal);
                    valsDb.push_back(*delta0);

                    const auto decBid = GetOptionalDouble(rowOut, "decoupling_bid");
                    if (decBid.has_value()) {
                        valsDecBid.push_back(*decBid);
                    }
                    const auto decAsk = GetOptionalDouble(rowOut, "decoupling_ask");
                    if (decAsk.has_value()) {
                        valsDecAsk.push_back(*decAsk);
                    }
                    const auto decMid = GetOptionalDouble(rowOut, "decoupling_mid");
                    if (decMid.has_value()) {
                        valsDecMid.push_back(*decMid);
                    }
                    const auto decFit = GetOptionalDouble(rowOut, "decoupling_fit");
                    if (decFit.has_value()) {
                        valsDecFit.push_back(*decFit);
                    }

                    const auto drBid = GetOptionalDouble(rowOut, "delta_realized_bid");
                    if (drBid.has_value()) {
                        valsDrBid.push_back(*drBid);
                        valsDbBid.push_back(*delta0);
                    }
                    const auto drAsk = GetOptionalDouble(rowOut, "delta_realized_ask");
                    if (drAsk.has_value()) {
                        valsDrAsk.push_back(*drAsk);
                        valsDbAsk.push_back(*delta0);
                    }
                    const auto drMid = GetOptionalDouble(rowOut, "delta_realized_mid");
                    if (drMid.has_value()) {
                        valsDrMid.push_back(*drMid);
                        valsDbMid.push_back(*delta0);
                    }
                    const auto drFit = GetOptionalDouble(rowOut, "delta_realized_fit");
                    if (drFit.has_value()) {
                        valsDrFit.push_back(*drFit);
                        valsDbFit.push_back(*delta0);
                    }

                    rowsOut.push_back(std::move(rowOut));
                }
            }

            const double spotMean = spotVals.empty() ? std::numeric_limits<double>::quiet_NaN()
                                                     : (std::accumulate(spotVals.begin(), spotVals.end(), 0.0) /
                                                        static_cast<double>(spotVals.size()));
            double spotStd = std::numeric_limits<double>::quiet_NaN();
            if (spotVals.size() >= 2) {
                double ss = 0.0;
                for (double v : spotVals) {
                    const double d = v - spotMean;
                    ss += d * d;
                }
                spotStd = std::sqrt(ss / static_cast<double>(spotVals.size()));
            }
            const double spotMin = spotVals.empty() ? std::numeric_limits<double>::quiet_NaN()
                                                    : *std::min_element(spotVals.begin(), spotVals.end());
            const double spotMax = spotVals.empty() ? std::numeric_limits<double>::quiet_NaN()
                                                    : *std::max_element(spotVals.begin(), spotVals.end());

            std::vector<double> absDec;
            absDec.reserve(valsDec.size());
            for (double v : valsDec) {
                absDec.push_back(std::abs(v));
            }
            std::vector<double> absDecBid;
            absDecBid.reserve(valsDecBid.size());
            for (double v : valsDecBid) {
                absDecBid.push_back(std::abs(v));
            }
            std::vector<double> absDecAsk;
            absDecAsk.reserve(valsDecAsk.size());
            for (double v : valsDecAsk) {
                absDecAsk.push_back(std::abs(v));
            }
            std::vector<double> absDecMid;
            absDecMid.reserve(valsDecMid.size());
            for (double v : valsDecMid) {
                absDecMid.push_back(std::abs(v));
            }
            std::vector<double> absDecFit;
            absDecFit.reserve(valsDecFit.size());
            for (double v : valsDecFit) {
                absDecFit.push_back(std::abs(v));
            }

            const double corr = PearsonCorr(valsDr, valsDb);
            const double corrBid = PearsonCorr(valsDrBid, valsDbBid);
            const double corrAsk = PearsonCorr(valsDrAsk, valsDbAsk);
            const double corrMid = PearsonCorr(valsDrMid, valsDbMid);
            const double corrFit = PearsonCorr(valsDrFit, valsDbFit);

            summaryOut.push_back({
                {"date", day},
                {"expiry_index", expiryIndex},
                {"n_valid", nValidGroup},
                {"n_total", nTotalGroup},
                {"spot_n", static_cast<int>(spotVals.size())},
                {"spot_mean", NullOrDouble(spotMean)},
                {"spot_std_dev", NullOrDouble(spotStd)},
                {"spot_min", NullOrDouble(spotMin)},
                {"spot_max", NullOrDouble(spotMax)},
                {"rms_decoupling", NullOrDouble(Rms(valsDec))},
                {"rms_decoupling_bid", NullOrDouble(Rms(valsDecBid))},
                {"rms_decoupling_ask", NullOrDouble(Rms(valsDecAsk))},
                {"rms_decoupling_mid", NullOrDouble(Rms(valsDecMid))},
                {"rms_decoupling_fit", NullOrDouble(Rms(valsDecFit))},
                {"median_abs_decoupling", NullOrDouble(Median(absDec))},
                {"median_abs_decoupling_bid", NullOrDouble(Median(absDecBid))},
                {"median_abs_decoupling_ask", NullOrDouble(Median(absDecAsk))},
                {"median_abs_decoupling_mid", NullOrDouble(Median(absDecMid))},
                {"median_abs_decoupling_fit", NullOrDouble(Median(absDecFit))},
                {"corr_delta_realized_vs_bs", NullOrDouble(corr)},
                {"corr_delta_realized_bid_vs_bs", NullOrDouble(corrBid)},
                {"corr_delta_realized_ask_vs_bs", NullOrDouble(corrAsk)},
                {"corr_delta_realized_mid_vs_bs", NullOrDouble(corrMid)},
                {"corr_delta_realized_fit_vs_bs", NullOrDouble(corrFit)},
            });
        }

        const int nRowsTotal = diag["n_rows_total"].get<int>();
        const int nRowsValid = diag["n_rows_valid"].get<int>();
        const int nSkipSmallDf = diag["n_skipped_small_df"].get<int>();
        const int nSkipMissingPrice = diag["n_skipped_missing_price"].get<int>();
        const int nSkipMissingSigma = diag["n_skipped_missing_sigma"].get<int>();
        const int nSkipMissingGreeks = diag["n_skipped_missing_greeks"].get<int>();
        const int nSkipMissingExit = diag["n_skipped_missing_exit_strike"].get<int>();
        diag["n_skipped_other"] =
            std::max(nRowsTotal - nRowsValid - nSkipSmallDf - nSkipMissingPrice - nSkipMissingSigma - nSkipMissingGreeks - nSkipMissingExit, 0);

        json midWlsPerStrike = BuildMidWlsPerStrike(rowsOut);
        json out = {
            {"rows", std::move(rowsOut)},
            {"summary", std::move(summaryOut)},
            {"diagnostics", std::move(diag)},
            {"mid_wls_per_strike", std::move(midWlsPerStrike)},
        };

        std::cout << out.dump(-1, ' ', false, json::error_handler_t::replace);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "decoupling engine error: " << e.what() << '\n';
        return 1;
    }
}
