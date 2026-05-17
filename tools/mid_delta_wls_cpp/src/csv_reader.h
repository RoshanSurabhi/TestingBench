#pragma once

#include <string>
#include <vector>

#include "types.h"

std::vector<DetailRow> read_decoupling_details_csv(const std::string& path);
std::vector<OutputRow> read_output_rows_csv(const std::string& path);
void write_output_rows_csv(const std::string& path, const std::vector<OutputRow>& rows);
