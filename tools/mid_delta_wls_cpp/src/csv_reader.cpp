#include "csv_reader.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> out;
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
        } else {
            if (c == ',') {
                out.push_back(field);
                field.clear();
            } else if (c == '"') {
                in_quotes = true;
            } else {
                field.push_back(c);
            }
        }
    }
    out.push_back(field);
    return out;
}

std::unordered_map<std::string, std::size_t> header_index(const std::vector<std::string>& headers) {
    std::unordered_map<std::string, std::size_t> idx;
    for (std::size_t i = 0; i < headers.size(); ++i) {
        idx[headers[i]] = i;
    }
    return idx;
}

std::string must_get(const std::vector<std::string>& row, const std::unordered_map<std::string, std::size_t>& idx, const std::string& name) {
    const auto it = idx.find(name);
    if (it == idx.end()) {
        throw std::runtime_error("missing required column: " + name);
    }
    if (it->second >= row.size()) {
        return "";
    }
    return row[it->second];
}

std::optional<double> parse_optional_double(const std::string& s) {
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

std::optional<int> parse_optional_int(const std::string& s) {
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

std::string escape_csv(const std::string& s) {
    bool needs_quote = false;
    for (const char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needs_quote = true;
            break;
        }
    }
    if (!needs_quote) {
        return s;
    }
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out.push_back(c);
        }
    }
    out += "\"";
    return out;
}

std::string format_optional_double(const std::optional<double>& v) {
    if (!v.has_value()) {
        return "";
    }
    std::ostringstream oss;
    oss << std::setprecision(17) << *v;
    return oss.str();
}

}  // namespace

std::vector<DetailRow> read_decoupling_details_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open input CSV: " + path);
    }
    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("input CSV is empty: " + path);
    }
    const auto headers = parse_csv_line(line);
    const auto idx = header_index(headers);

    std::vector<DetailRow> rows;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = parse_csv_line(line);
        DetailRow row;
        row.date = must_get(fields, idx, "date");
        row.expiry_index = parse_optional_int(must_get(fields, idx, "expiry_index")).value_or(0);
        row.strike = parse_optional_double(must_get(fields, idx, "strike")).value_or(std::numeric_limits<double>::quiet_NaN());
        row.z_entry = parse_optional_double(must_get(fields, idx, "z_entry"));
        row.valid = parse_optional_int(must_get(fields, idx, "valid")).value_or(0);
        row.delta_bs_entry = parse_optional_double(must_get(fields, idx, "delta_bs_entry"));
        row.delta_realized_mid = parse_optional_double(must_get(fields, idx, "delta_realized_mid"));
        rows.push_back(std::move(row));
    }
    return rows;
}

std::vector<OutputRow> read_output_rows_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open output CSV: " + path);
    }
    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("output CSV is empty: " + path);
    }
    const auto headers = parse_csv_line(line);
    const auto idx = header_index(headers);

    std::vector<OutputRow> rows;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = parse_csv_line(line);
        OutputRow row;
        row.date = must_get(fields, idx, "date");
        row.expiry_index = parse_optional_int(must_get(fields, idx, "expiry_index")).value_or(0);
        row.strike = parse_optional_double(must_get(fields, idx, "strike")).value_or(0.0);
        row.z_entry_median = parse_optional_double(must_get(fields, idx, "z_entry_median"));
        row.n = parse_optional_int(must_get(fields, idx, "n")).value_or(0);
        row.alpha = parse_optional_double(must_get(fields, idx, "alpha"));
        row.beta = parse_optional_double(must_get(fields, idx, "beta"));
        row.r2_wls = parse_optional_double(must_get(fields, idx, "r2_wls"));
        row.weight_scheme = must_get(fields, idx, "weight_scheme");
        row.model = must_get(fields, idx, "model");
        rows.push_back(std::move(row));
    }
    return rows;
}

void write_output_rows_csv(const std::string& path, const std::vector<OutputRow>& rows) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open output path: " + path);
    }
    out << "date,expiry_index,strike,z_entry_median,n,alpha,beta,r2_wls,weight_scheme,model\n";
    for (const auto& row : rows) {
        std::ostringstream strike_oss;
        strike_oss << std::setprecision(17) << row.strike;
        out << escape_csv(row.date) << ","
            << row.expiry_index << ","
            << strike_oss.str() << ","
            << format_optional_double(row.z_entry_median) << ","
            << row.n << ","
            << format_optional_double(row.alpha) << ","
            << format_optional_double(row.beta) << ","
            << format_optional_double(row.r2_wls) << ","
            << escape_csv(row.weight_scheme) << ","
            << escape_csv(row.model) << "\n";
    }
}
