#include "grouping.h"

#include <cmath>
#include <iomanip>
#include <sstream>

std::string fmt_strike_key(double strike) {
    const double rounded8 = std::round(strike * 1e8) / 1e8;
    const double nearest_int = std::round(rounded8);
    if (std::abs(rounded8 - nearest_int) < 1e-6) {
        std::ostringstream oss;
        oss << static_cast<long long>(nearest_int);
        return oss.str();
    }
    std::ostringstream oss;
    oss << std::setprecision(10) << strike;
    return oss.str();
}

GroupMap group_rows_per_strike(const std::vector<DetailRow>& rows) {
    GroupMap grouped;
    for (const auto& row : rows) {
        if (row.valid != 1) {
            continue;
        }
        if (!row.delta_bs_entry.has_value() || !row.delta_realized_mid.has_value()) {
            continue;
        }
        if (!std::isfinite(row.strike)) {
            continue;
        }
        const auto key = GroupKey{row.date, row.expiry_index, fmt_strike_key(row.strike)};
        auto& g = grouped[key];
        g.date = row.date;
        g.expiry_index = row.expiry_index;
        g.strike = row.strike;
        g.xs.push_back(*row.delta_bs_entry);
        g.ys.push_back(*row.delta_realized_mid);
        if (row.z_entry.has_value()) {
            g.zs.push_back(*row.z_entry);
        }
    }
    return grouped;
}
