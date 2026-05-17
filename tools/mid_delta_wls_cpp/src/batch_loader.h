#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

struct BuildInputConfig {
    std::string batch_dir;
    std::vector<std::string> dates;
    std::vector<int> expiry_indices;
    bool all_expiries = true;
    double entry_spacing_min = 5.0;
    double holding_min = 120.0;
    double min_abs_df_frac = 0.0005;
    double eps_delta = 1e-8;
};

nlohmann::json build_engine_input_from_batch(const BuildInputConfig& cfg);
