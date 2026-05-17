#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "batch_loader.h"
#include "csv_reader.h"
#include "grouping.h"
#include "wls_regression.h"
#include "decouplingEngine.h"

namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

struct CliArgs {
    std::string batch_dir;
    std::vector<std::string> dates;
    std::vector<int> expiry_indices;
    bool all_expiries = true;
    double entry_spacing_min = 5.0;
    double holding_min = 120.0;
    double min_abs_df_frac = 0.0005;
    std::string detail_csv;
    std::string fit_csv;
    std::string bid_csv;
    std::string ask_csv;
    std::string mid_csv;
    std::string html_dir;
};

std::vector<std::string> split_csv(const std::string& raw) {
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

void print_usage() {
    std::cerr
        << "Usage:\n"
        << "  mid_delta_wls_cpp --batch-dir <path> [--dates D1,D2] [--expiry-indices all|I1,I2]\n"
        << "                    [--entry-spacing-min 5] [--holding-min 120] [--min-abs-df-frac 0.0005]\n"
        << "                    [--detail-csv <path>] [--fit-csv <path>] [--bid-csv <path>] [--ask-csv <path>] [--mid-csv <path>]\n"
        << "                    [--html-dir <directory>]\n";
}

CliArgs parse_args(int argc, char** argv) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];
        if (token == "--batch-dir" && i + 1 < argc) {
            args.batch_dir = argv[++i];
        } else if (token == "--dates" && i + 1 < argc) {
            args.dates = split_csv(argv[++i]);
        } else if (token == "--expiry-indices" && i + 1 < argc) {
            const std::string raw = argv[++i];
            if (raw == "all") {
                args.all_expiries = true;
                args.expiry_indices.clear();
            } else {
                args.all_expiries = false;
                for (const auto& t : split_csv(raw)) {
                    args.expiry_indices.push_back(std::stoi(t));
                }
            }
        } else if (token == "--entry-spacing-min" && i + 1 < argc) {
            args.entry_spacing_min = std::stod(argv[++i]);
        } else if (token == "--holding-min" && i + 1 < argc) {
            args.holding_min = std::stod(argv[++i]);
        } else if (token == "--min-abs-df-frac" && i + 1 < argc) {
            args.min_abs_df_frac = std::stod(argv[++i]);
        } else if (token == "--detail-csv" && i + 1 < argc) {
            args.detail_csv = argv[++i];
        } else if (token == "--fit-csv" && i + 1 < argc) {
            args.fit_csv = argv[++i];
        } else if (token == "--bid-csv" && i + 1 < argc) {
            args.bid_csv = argv[++i];
        } else if (token == "--ask-csv" && i + 1 < argc) {
            args.ask_csv = argv[++i];
        } else if (token == "--mid-csv" && i + 1 < argc) {
            args.mid_csv = argv[++i];
        } else if (token == "--html-dir" && i + 1 < argc) {
            args.html_dir = argv[++i];
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + token);
        }
    }
    if (args.batch_dir.empty()) {
        throw std::runtime_error("--batch-dir is required");
    }
    const fs::path base(args.batch_dir);
    if (args.detail_csv.empty()) {
        args.detail_csv = (base / "decoupling_details_cpp_5m_120m.csv").string();
    }
    if (args.fit_csv.empty()) {
        args.fit_csv = (base / "decoupling_wls_per_strike_fit.csv").string();
    }
    if (args.bid_csv.empty()) {
        args.bid_csv = (base / "decoupling_wls_per_strike_bid.csv").string();
    }
    if (args.ask_csv.empty()) {
        args.ask_csv = (base / "decoupling_wls_per_strike_ask.csv").string();
    }
    if (args.mid_csv.empty()) {
        args.mid_csv = (base / "decoupling_wls_per_strike_mid.csv").string();
    }
    return args;
}

std::string safe_file_fragment(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (char c : raw) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    return out;
}

std::string escape_csv(const std::string& s) {
    bool quote = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            quote = true;
            break;
        }
    }
    if (!quote) {
        return s;
    }
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out.push_back(c);
    }
    out += "\"";
    return out;
}

std::string json_to_cell(const json& v) {
    if (v.is_null()) {
        return "";
    }
    if (v.is_number_float()) {
        std::ostringstream oss;
        oss.precision(17);
        oss << v.get<double>();
        return oss.str();
    }
    if (v.is_number_integer()) {
        return std::to_string(v.get<long long>());
    }
    if (v.is_number_unsigned()) {
        return std::to_string(v.get<unsigned long long>());
    }
    if (v.is_boolean()) {
        return v.get<bool>() ? "1" : "0";
    }
    if (v.is_string()) {
        return v.get<std::string>();
    }
    return v.dump();
}

void write_rows_json_csv(const std::string& path, const json& rows) {
    if (!rows.is_array() || rows.empty()) {
        throw std::runtime_error("no detail rows to write");
    }
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open detail CSV: " + path);
    }
    std::vector<std::string> headers;
    for (auto it = rows[0].begin(); it != rows[0].end(); ++it) {
        headers.push_back(it.key());
    }
    for (std::size_t i = 0; i < headers.size(); ++i) {
        if (i) out << ",";
        out << escape_csv(headers[i]);
    }
    out << "\n";
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < headers.size(); ++i) {
            if (i) out << ",";
            const auto it = row.find(headers[i]);
            const std::string cell = (it == row.end() ? "" : json_to_cell(*it));
            out << escape_csv(cell);
        }
        out << "\n";
    }
}

json run_engine_from_input_json(const json& input) {
    std::stringstream in_stream;
    in_stream << input.dump();
    std::stringstream out_stream;

    auto* old_in = std::cin.rdbuf(in_stream.rdbuf());
    auto* old_out = std::cout.rdbuf(out_stream.rdbuf());
    const int rc = RunDecouplingEngineFromStdinToStdout();
    std::cout.rdbuf(old_out);
    std::cin.rdbuf(old_in);
    if (rc != 0) {
        throw std::runtime_error("decoupling engine returned non-zero status");
    }
    return json::parse(out_stream.str());
}

std::vector<OutputRow> build_track_wls(const json& rows_in, const std::string& y_key, const std::string& model) {
    std::map<std::tuple<std::string, int, std::string>, GroupAccum> grouped;
    for (const auto& r : rows_in) {
        if (!r.is_object()) continue;
        const int valid = r.value("valid", 0);
        if (valid != 1) continue;
        if (!r.contains("date") || !r.contains("expiry_index") || !r.contains("strike")) continue;
        if (!r["strike"].is_number()) continue;
        const auto x_it = r.find("delta_bs_entry");
        const auto y_it = r.find(y_key);
        if (x_it == r.end() || y_it == r.end() || x_it->is_null() || y_it->is_null()) continue;
        if (!x_it->is_number() || !y_it->is_number()) continue;
        const double x = x_it->get<double>();
        const double y = y_it->get<double>();
        if (!(std::isfinite(x) && std::isfinite(y))) continue;
        const std::string day = r["date"].get<std::string>();
        const int ei = r["expiry_index"].get<int>();
        const double strike = r["strike"].get<double>();
        const auto key = std::make_tuple(day, ei, fmt_strike_key(strike));
        auto& g = grouped[key];
        g.date = day;
        g.expiry_index = ei;
        g.strike = strike;
        g.xs.push_back(x);
        g.ys.push_back(y);
        const auto z_it = r.find("z_entry");
        if (z_it != r.end() && z_it->is_number()) {
            const double z = z_it->get<double>();
            if (std::isfinite(z)) g.zs.push_back(z);
        }
    }

    std::vector<OutputRow> out;
    out.reserve(grouped.size());
    for (auto& [k, g] : grouped) {
        OutputRow row;
        row.date = g.date;
        row.expiry_index = g.expiry_index;
        row.strike = g.strike;
        row.z_entry_median = median_value(g.zs);
        row.n = static_cast<int>(g.xs.size());
        const auto fit = fit_mid_delta_wls(g.xs, g.ys);
        row.alpha = fit.alpha;
        row.beta = fit.beta;
        row.r2_wls = fit.r2;
        row.weight_scheme = "w_i = 1/(eps + x_i^2 + y_i^2), eps=1e-8";
        row.model = model;
        out.push_back(std::move(row));
    }
    std::sort(out.begin(), out.end(), [](const OutputRow& a, const OutputRow& b) {
        if (a.date != b.date) return a.date < b.date;
        if (a.expiry_index != b.expiry_index) return a.expiry_index < b.expiry_index;
        return a.strike < b.strike;
    });
    return out;
}

using GroupKey = std::tuple<std::string, int, std::string>;

std::unordered_map<std::string, OutputRow> index_wls_rows(const std::vector<OutputRow>& rows) {
    std::unordered_map<std::string, OutputRow> out;
    out.reserve(rows.size());
    for (const auto& r : rows) {
        const std::string key = r.date + "|" + std::to_string(r.expiry_index) + "|" + fmt_strike_key(r.strike);
        out[key] = r;
    }
    return out;
}

std::string js_num_or_null(double v) {
    if (!std::isfinite(v)) {
        return "null";
    }
    std::ostringstream oss;
    oss.precision(17);
    oss << v;
    return oss.str();
}

std::string js_opt_num_or_null(const std::optional<double>& v) {
    if (!v.has_value() || !std::isfinite(*v)) {
        return "null";
    }
    std::ostringstream oss;
    oss.precision(17);
    oss << *v;
    return oss.str();
}

std::string opt_to_text(const std::optional<double>& v) {
    if (!v.has_value() || !std::isfinite(*v)) {
        return "n/a";
    }
    std::ostringstream oss;
    oss.precision(17);
    oss << *v;
    return oss.str();
}

void write_per_strike_expiry_htmls(
    const std::string& html_dir,
    const json& rows_in,
    const std::vector<OutputRow>& fit_rows,
    const std::vector<OutputRow>& bid_rows,
    const std::vector<OutputRow>& ask_rows,
    const std::vector<OutputRow>& mid_rows
) {
    if (html_dir.empty()) {
        return;
    }
    fs::create_directories(html_dir);

    const auto fit_idx = index_wls_rows(fit_rows);
    const auto bid_idx = index_wls_rows(bid_rows);
    const auto ask_idx = index_wls_rows(ask_rows);
    const auto mid_idx = index_wls_rows(mid_rows);

    struct GroupPoints {
        std::string date;
        int expiry_index = -1;
        double strike = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> x_fit;
        std::vector<double> y_fit;
        std::vector<double> x_bid;
        std::vector<double> y_bid;
        std::vector<double> x_ask;
        std::vector<double> y_ask;
        std::vector<double> x_mid;
        std::vector<double> y_mid;
    };

    std::map<GroupKey, GroupPoints> groups;
    for (const auto& r : rows_in) {
        if (!r.is_object() || r.value("valid", 0) != 1) {
            continue;
        }
        if (!r.contains("date") || !r.contains("expiry_index") || !r.contains("strike")) {
            continue;
        }
        if (!r["strike"].is_number()) {
            continue;
        }
        const std::string day = r["date"].get<std::string>();
        const int ei = r["expiry_index"].get<int>();
        const double strike = r["strike"].get<double>();
        const auto gk = GroupKey{day, ei, fmt_strike_key(strike)};
        auto& g = groups[gk];
        g.date = day;
        g.expiry_index = ei;
        g.strike = strike;

        const auto xit = r.find("delta_bs_entry");
        if (xit == r.end() || !xit->is_number()) {
            continue;
        }
        const double x = xit->get<double>();
        if (!std::isfinite(x)) {
            continue;
        }
        const auto fit_it = r.find("delta_realized_fit");
        if (fit_it != r.end() && fit_it->is_number()) {
            const double y = fit_it->get<double>();
            if (std::isfinite(y)) {
                g.x_fit.push_back(x);
                g.y_fit.push_back(y);
            }
        }
        const auto bid_it = r.find("delta_realized_bid");
        if (bid_it != r.end() && bid_it->is_number()) {
            const double y = bid_it->get<double>();
            if (std::isfinite(y)) {
                g.x_bid.push_back(x);
                g.y_bid.push_back(y);
            }
        }
        const auto ask_it = r.find("delta_realized_ask");
        if (ask_it != r.end() && ask_it->is_number()) {
            const double y = ask_it->get<double>();
            if (std::isfinite(y)) {
                g.x_ask.push_back(x);
                g.y_ask.push_back(y);
            }
        }
        const auto mid_it = r.find("delta_realized_mid");
        if (mid_it != r.end() && mid_it->is_number()) {
            const double y = mid_it->get<double>();
            if (std::isfinite(y)) {
                g.x_mid.push_back(x);
                g.y_mid.push_back(y);
            }
        }
    }

    std::vector<std::string> index_rows;
    index_rows.reserve(groups.size());

    for (const auto& [key, g] : groups) {
        const std::string key_str = g.date + "|" + std::to_string(g.expiry_index) + "|" + fmt_strike_key(g.strike);
        const auto fit_it = fit_idx.find(key_str);
        const auto bid_it = bid_idx.find(key_str);
        const auto ask_it = ask_idx.find(key_str);
        const auto mid_it = mid_idx.find(key_str);

        std::vector<double> all_x;
        all_x.insert(all_x.end(), g.x_fit.begin(), g.x_fit.end());
        all_x.insert(all_x.end(), g.x_bid.begin(), g.x_bid.end());
        all_x.insert(all_x.end(), g.x_ask.begin(), g.x_ask.end());
        all_x.insert(all_x.end(), g.x_mid.begin(), g.x_mid.end());
        if (all_x.empty()) {
            continue;
        }
        const auto [mn_it, mx_it] = std::minmax_element(all_x.begin(), all_x.end());
        const double x_min = *mn_it;
        const double x_max = *mx_it;

        std::ostringstream x_fit_js, y_fit_js, x_bid_js, y_bid_js, x_ask_js, y_ask_js, x_mid_js, y_mid_js;
        auto append_vec = [](std::ostringstream& oss, const std::vector<double>& v) {
            oss << "[";
            for (std::size_t i = 0; i < v.size(); ++i) {
                if (i) oss << ",";
                oss << js_num_or_null(v[i]);
            }
            oss << "]";
        };
        append_vec(x_fit_js, g.x_fit);
        append_vec(y_fit_js, g.y_fit);
        append_vec(x_bid_js, g.x_bid);
        append_vec(y_bid_js, g.y_bid);
        append_vec(x_ask_js, g.x_ask);
        append_vec(y_ask_js, g.y_ask);
        append_vec(x_mid_js, g.x_mid);
        append_vec(y_mid_js, g.y_mid);

        const std::string fit_alpha = (fit_it == fit_idx.end() ? "null" : js_opt_num_or_null(fit_it->second.alpha));
        const std::string fit_beta = (fit_it == fit_idx.end() ? "null" : js_opt_num_or_null(fit_it->second.beta));
        const std::string fit_r2 = (fit_it == fit_idx.end() ? "null" : js_opt_num_or_null(fit_it->second.r2_wls));
        const std::string bid_alpha = (bid_it == bid_idx.end() ? "null" : js_opt_num_or_null(bid_it->second.alpha));
        const std::string bid_beta = (bid_it == bid_idx.end() ? "null" : js_opt_num_or_null(bid_it->second.beta));
        const std::string bid_r2 = (bid_it == bid_idx.end() ? "null" : js_opt_num_or_null(bid_it->second.r2_wls));
        const std::string ask_alpha = (ask_it == ask_idx.end() ? "null" : js_opt_num_or_null(ask_it->second.alpha));
        const std::string ask_beta = (ask_it == ask_idx.end() ? "null" : js_opt_num_or_null(ask_it->second.beta));
        const std::string ask_r2 = (ask_it == ask_idx.end() ? "null" : js_opt_num_or_null(ask_it->second.r2_wls));
        const std::string mid_alpha = (mid_it == mid_idx.end() ? "null" : js_opt_num_or_null(mid_it->second.alpha));
        const std::string mid_beta = (mid_it == mid_idx.end() ? "null" : js_opt_num_or_null(mid_it->second.beta));
        const std::string mid_r2 = (mid_it == mid_idx.end() ? "null" : js_opt_num_or_null(mid_it->second.r2_wls));

        std::ostringstream strike_oss;
        strike_oss.precision(12);
        strike_oss << g.strike;
        const std::string strike_txt = strike_oss.str();

        const std::string file_name =
            "decoupling_" + safe_file_fragment(g.date) + "_e" + std::to_string(g.expiry_index) +
            "_k" + safe_file_fragment(fmt_strike_key(g.strike)) + ".html";
        const fs::path file_path = fs::path(html_dir) / file_name;
        std::ofstream out(file_path);
        if (!out.is_open()) {
            throw std::runtime_error("failed to open html file: " + file_path.string());
        }

        out << "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/>"
            << "<title>Decoupling " << g.date << " e" << g.expiry_index << " K=" << strike_txt << "</title>"
            << "<script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js\"></script>"
            << "</head><body style=\"font-family:system-ui,Segoe UI,sans-serif;margin:16px;\">"
            << "<h2>Decoupling regression: " << g.date << " expiry " << g.expiry_index << " strike " << strike_txt << "</h2>"
            << "<div id=\"plot\" style=\"width:100%;height:640px;\"></div>"
            << "<script>"
            << "const xFit=" << x_fit_js.str() << ";const yFit=" << y_fit_js.str() << ";"
            << "const xBid=" << x_bid_js.str() << ";const yBid=" << y_bid_js.str() << ";"
            << "const xAsk=" << x_ask_js.str() << ";const yAsk=" << y_ask_js.str() << ";"
            << "const xMid=" << x_mid_js.str() << ";const yMid=" << y_mid_js.str() << ";"
            << "const xmin=" << js_num_or_null(x_min) << ";const xmax=" << js_num_or_null(x_max) << ";"
            << "const fitA=" << fit_alpha << ";const fitB=" << fit_beta << ";const fitR2=" << fit_r2 << ";"
            << "const bidA=" << bid_alpha << ";const bidB=" << bid_beta << ";const bidR2=" << bid_r2 << ";"
            << "const askA=" << ask_alpha << ";const askB=" << ask_beta << ";const askR2=" << ask_r2 << ";"
            << "const midA=" << mid_alpha << ";const midB=" << mid_beta << ";const midR2=" << mid_r2 << ";"
            << "const traces=["
            << "{type:'scatter',mode:'markers',name:'fit windows',x:xFit,y:yFit,marker:{size:6,opacity:0.7,color:'#1f77b4'}},"
            << "{type:'scatter',mode:'markers',name:'bid windows',x:xBid,y:yBid,marker:{size:6,opacity:0.7,color:'#2ca02c'}},"
            << "{type:'scatter',mode:'markers',name:'ask windows',x:xAsk,y:yAsk,marker:{size:6,opacity:0.7,color:'#d62728'}},"
            << "{type:'scatter',mode:'markers',name:'mid windows',x:xMid,y:yMid,marker:{size:6,opacity:0.7,color:'#9467bd'}},"
            << "{type:'scatter',mode:'lines',name:'y=x',x:[xmin,xmax],y:[xmin,xmax],line:{dash:'dot',color:'#666'}}"
            << "];"
            << "if(Number.isFinite(fitA)&&Number.isFinite(fitB)){traces.push({type:'scatter',mode:'lines',name:'fit WLS',x:[xmin,xmax],y:[fitA+fitB*xmin,fitA+fitB*xmax],line:{color:'#1f77b4',width:2}});}"
            << "if(Number.isFinite(bidA)&&Number.isFinite(bidB)){traces.push({type:'scatter',mode:'lines',name:'bid WLS',x:[xmin,xmax],y:[bidA+bidB*xmin,bidA+bidB*xmax],line:{color:'#2ca02c',width:2}});}"
            << "if(Number.isFinite(askA)&&Number.isFinite(askB)){traces.push({type:'scatter',mode:'lines',name:'ask WLS',x:[xmin,xmax],y:[askA+askB*xmin,askA+askB*xmax],line:{color:'#d62728',width:2}});}"
            << "if(Number.isFinite(midA)&&Number.isFinite(midB)){traces.push({type:'scatter',mode:'lines',name:'mid WLS',x:[xmin,xmax],y:[midA+midB*xmin,midA+midB*xmax],line:{color:'#9467bd',width:2}});}"
            << "Plotly.newPlot('plot',traces,{"
            << "title:'delta_realized vs delta_BS(entry)'"
            << ",xaxis:{title:'delta_BS(entry)'}"
            << ",yaxis:{title:'delta_realized'}"
            << ",legend:{orientation:'h'}"
            << "});"
            << "</script>"
            << "<p>fit r2: " << (fit_it == fit_idx.end() ? "n/a" : opt_to_text(fit_it->second.r2_wls))
            << " | bid r2: " << (bid_it == bid_idx.end() ? "n/a" : opt_to_text(bid_it->second.r2_wls))
            << " | ask r2: " << (ask_it == ask_idx.end() ? "n/a" : opt_to_text(ask_it->second.r2_wls))
            << " | mid r2: " << (mid_it == mid_idx.end() ? "n/a" : opt_to_text(mid_it->second.r2_wls)) << "</p>"
            << "</body></html>";

        index_rows.push_back("<li><a href=\"" + file_name + "\">" + g.date + " | e" + std::to_string(g.expiry_index) +
                             " | K=" + strike_txt + "</a></li>");
    }

    std::sort(index_rows.begin(), index_rows.end());
    std::ofstream idx(fs::path(html_dir) / "index.html");
    if (idx.is_open()) {
        idx << "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/><title>Decoupling per strike/expiry</title></head><body>"
            << "<h2>Per-strike, per-expiry decoupling plots</h2><ul>";
        for (const auto& row : index_rows) {
            idx << row;
        }
        idx << "</ul></body></html>";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliArgs args = parse_args(argc, argv);

        BuildInputConfig cfg;
        cfg.batch_dir = args.batch_dir;
        cfg.dates = args.dates;
        cfg.expiry_indices = args.expiry_indices;
        cfg.all_expiries = args.all_expiries;
        cfg.entry_spacing_min = args.entry_spacing_min;
        cfg.holding_min = args.holding_min;
        cfg.min_abs_df_frac = args.min_abs_df_frac;

        const json input = build_engine_input_from_batch(cfg);
        const json out = run_engine_from_input_json(input);

        const json rows = out.value("rows", json::array());
        write_rows_json_csv(args.detail_csv, rows);

        const auto fit_rows = build_track_wls(
            rows,
            "delta_realized_fit",
            "delta_realized_fit ~ alpha + beta * delta_bs_entry (per strike)"
        );
        const auto bid_rows = build_track_wls(
            rows,
            "delta_realized_bid",
            "delta_realized_bid ~ alpha + beta * delta_bs_entry (per strike)"
        );
        const auto ask_rows = build_track_wls(
            rows,
            "delta_realized_ask",
            "delta_realized_ask ~ alpha + beta * delta_bs_entry (per strike)"
        );
        const auto mid_rows = build_track_wls(
            rows,
            "delta_realized_mid",
            "delta_realized_mid ~ alpha + beta * delta_bs_entry (per strike)"
        );

        write_output_rows_csv(args.fit_csv, fit_rows);
        write_output_rows_csv(args.bid_csv, bid_rows);
        write_output_rows_csv(args.ask_csv, ask_rows);
        write_output_rows_csv(args.mid_csv, mid_rows);
        write_per_strike_expiry_htmls(args.html_dir, rows, fit_rows, bid_rows, ask_rows, mid_rows);

        std::cout << "Generated C++ decoupling outputs\n";
        std::cout << "  detail_rows: " << rows.size() << "\n";
        std::cout << "  fit_reg_rows: " << fit_rows.size() << "\n";
        std::cout << "  bid_reg_rows: " << bid_rows.size() << "\n";
        std::cout << "  ask_reg_rows: " << ask_rows.size() << "\n";
        std::cout << "  mid_reg_rows: " << mid_rows.size() << "\n";
        std::cout << "  detail_csv: " << args.detail_csv << "\n";
        std::cout << "  fit_csv: " << args.fit_csv << "\n";
        std::cout << "  bid_csv: " << args.bid_csv << "\n";
        std::cout << "  ask_csv: " << args.ask_csv << "\n";
        std::cout << "  mid_csv: " << args.mid_csv << "\n";
        if (!args.html_dir.empty()) {
            std::cout << "  html_dir: " << args.html_dir << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        print_usage();
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
