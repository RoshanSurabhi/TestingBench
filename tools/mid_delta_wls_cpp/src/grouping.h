#pragma once

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "types.h"

using GroupKey = std::tuple<std::string, int, std::string>;
using GroupMap = std::map<GroupKey, GroupAccum>;

std::string fmt_strike_key(double strike);
GroupMap group_rows_per_strike(const std::vector<DetailRow>& rows);
