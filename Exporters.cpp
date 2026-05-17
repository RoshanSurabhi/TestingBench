#include "Exporters.h"

#include <fstream>
#include <iomanip>

#include <nlohmann/json.hpp>

namespace ResearchBench::Exporters {

bool writeReturnCsv(const std::string& path, const std::vector<ReturnRecord>& rows) {
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }
    out << "symbol,origin_timestamp,eval_timestamp,horizon_sec,expiry,forward_t,forward_t_plus_h,forward_return_log,spot_t,spot_t_plus_h,spot_return_log,valid\n";
    out << std::setprecision(12);
    for (const auto& r : rows) {
        out << r.symbol << ','
            << r.originTimestamp << ','
            << r.evalTimestamp << ','
            << r.horizonSec << ','
            << r.expiry << ','
            << r.forwardT << ','
            << r.forwardTPlusH << ','
            << r.forwardReturnLog << ','
            << r.spotT << ','
            << r.spotTPlusH << ','
            << r.spotReturnLog << ','
            << (r.valid ? 1 : 0) << '\n';
    }
    return true;
}

bool writeFitDiagnosticsCsv(const std::string& path, const std::vector<FitDiagnosticRecord>& rows) {
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }
    out << "timestamp,fit_success,total_expiries,total_strikes,fitted_expiries,nan_vol_count,mean_fitted_vol,objective,fit_error\n";
    out << std::setprecision(12);
    for (const auto& r : rows) {
        out << r.timestamp << ','
            << (r.fitSuccess ? 1 : 0) << ','
            << r.totalExpiries << ','
            << r.totalStrikes << ','
            << r.fittedExpiries << ','
            << r.nanVolCount << ','
            << r.meanFittedVol << ','
            << r.objective << ','
            << '"' << r.fitError << '"' << '\n';
    }
    return true;
}

bool writeFitResultsCsv(const std::string& path, const std::vector<PerSnapshotFitResult>& rows) {
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }
    out << "ticker,timestamp,success,clarabel_status,objective,error,num_expiries,num_basis\n";
    out << std::setprecision(12);
    for (const auto& r : rows) {
        out << r.ticker << ','
            << r.timestamp << ','
            << (r.success ? 1 : 0) << ','
            << r.clarabelStatus << ','
            << r.objective << ','
            << '"' << r.error << '"' << ','
            << r.numExpiries << ','
            << r.numBasis << '\n';
    }
    return true;
}

bool writeSummaryJson(const std::string& path, const ResearchRunResult& run) {
    nlohmann::json j;
    j["success"] = run.success;
    j["streamedSnapshots"] = run.streamedSnapshots;
    j["fitResults"] = run.fitResults;
    j["diagnostics"] = run.diagnostics;
    j["delayedOutputs"] = run.delayedOutputs;
    j["message"] = run.message;

    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }
    out << j.dump(2) << '\n';
    return true;
}

} // namespace ResearchBench::Exporters
