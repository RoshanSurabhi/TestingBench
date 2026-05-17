#pragma once

#include <string>
#include <vector>

#include "ResearchTypes.h"

namespace ResearchBench::Exporters {

bool writeReturnCsv(const std::string& path, const std::vector<ReturnRecord>& rows);
bool writeFitDiagnosticsCsv(const std::string& path, const std::vector<FitDiagnosticRecord>& rows);
bool writeFitResultsCsv(const std::string& path, const std::vector<PerSnapshotFitResult>& rows);
bool writeSummaryJson(const std::string& path, const ResearchRunResult& run);

} // namespace ResearchBench::Exporters
