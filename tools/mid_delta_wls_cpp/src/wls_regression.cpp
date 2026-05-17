#include "wls_regression.h"

#include <algorithm>
#include <cmath>

#include "wls_shared.h"

WlsFitResult fit_mid_delta_wls(const std::vector<double>& xs, const std::vector<double>& ys) {
    WlsFitResult out;
    const auto fit = trdbclient::decoupling::ComputeInverseBivariateMoveWls(xs, ys);
    if (!fit.ok) {
        return out;
    }
    out.alpha = fit.alpha;
    out.beta = fit.beta;
    out.r2 = fit.r2;
    return out;
}

std::optional<double> median_value(std::vector<double> values) {
    if (values.empty()) {
        return std::nullopt;
    }
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return (values[n / 2 - 1] + values[n / 2]) * 0.5;
}
