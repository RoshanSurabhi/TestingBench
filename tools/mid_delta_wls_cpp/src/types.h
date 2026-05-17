#pragma once

#include <optional>
#include <string>
#include <vector>

struct DetailRow {
    std::string date;
    int expiry_index = 0;
    double strike = 0.0;
    std::optional<double> z_entry;
    int valid = 0;
    std::optional<double> delta_bs_entry;
    std::optional<double> delta_realized_mid;
};

struct OutputRow {
    std::string date;
    int expiry_index = 0;
    double strike = 0.0;
    std::optional<double> z_entry_median;
    int n = 0;
    std::optional<double> alpha;
    std::optional<double> beta;
    std::optional<double> r2_wls;
    std::string weight_scheme;
    std::string model;
};

struct CompareReport {
    std::size_t generated_rows = 0;
    std::size_t baseline_rows = 0;
    std::size_t missing_in_generated = 0;
    std::size_t missing_in_baseline = 0;
    std::size_t compared_rows = 0;
    double max_abs_diff_alpha = 0.0;
    double max_abs_diff_beta = 0.0;
    double max_abs_diff_r2 = 0.0;
    bool pass = false;
};

struct ParsedArgs {
    std::string input_csv;
    std::string output_csv;
    std::optional<std::string> compare_csv;
    double tolerance = 1e-8;
};

struct WlsFitResult {
    std::optional<double> alpha;
    std::optional<double> beta;
    std::optional<double> r2;
};

struct GroupAccum {
    std::string date;
    int expiry_index = 0;
    double strike = 0.0;
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
};
