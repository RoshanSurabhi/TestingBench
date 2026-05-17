#pragma once

#include <optional>
#include <vector>

#include "types.h"

WlsFitResult fit_mid_delta_wls(const std::vector<double>& xs, const std::vector<double>& ys);
std::optional<double> median_value(std::vector<double> values);
