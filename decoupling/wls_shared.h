#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace trdbclient::decoupling {

struct WlsFitComputation {
    double alpha = std::numeric_limits<double>::quiet_NaN();
    double beta = std::numeric_limits<double>::quiet_NaN();
    double r2 = std::numeric_limits<double>::quiet_NaN();
    bool ok = false;
};

inline WlsFitComputation ComputeInverseBivariateMoveWls(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    double eps = 1e-8)
{
    WlsFitComputation out;
    const std::size_t n = xs.size();
    if (n < 2 || ys.size() != n) {
        return out;
    }

    std::vector<double> ws(n);
    for (std::size_t i = 0; i < n; ++i) {
        ws[i] = 1.0 / (eps + xs[i] * xs[i] + ys[i] * ys[i]);
    }

    double sw = 0.0;
    for (double wi : ws) {
        sw += wi;
    }
    if (sw <= 0.0) {
        return out;
    }

    double mx = 0.0;
    double my = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        mx += ws[i] * xs[i];
        my += ws[i] * ys[i];
    }
    mx /= sw;
    my /= sw;

    double sxx = 0.0;
    double sxy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = xs[i] - mx;
        sxx += ws[i] * dx * dx;
        sxy += ws[i] * dx * (ys[i] - my);
    }
    if (sxx <= 1e-18) {
        return out;
    }

    const double beta = sxy / sxx;
    const double alpha = my - beta * mx;

    for (double& wi : ws) {
        wi = std::max(wi, 1e-15);
    }

    sw = 0.0;
    my = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sw += ws[i];
        my += ws[i] * ys[i];
    }
    if (sw <= 0.0) {
        return out;
    }
    my /= sw;

    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double y_hat = alpha + beta * xs[i];
        ss_res += ws[i] * (ys[i] - y_hat) * (ys[i] - y_hat);
        ss_tot += ws[i] * (ys[i] - my) * (ys[i] - my);
    }

    out.alpha = alpha;
    out.beta = beta;
    out.r2 = (ss_tot > 1e-18) ? (1.0 - ss_res / ss_tot) : std::numeric_limits<double>::quiet_NaN();
    out.ok = std::isfinite(alpha) && std::isfinite(beta);
    return out;
}

} // namespace trdbclient::decoupling
