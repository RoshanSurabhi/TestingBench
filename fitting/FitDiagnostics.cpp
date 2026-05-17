#include "FitDiagnostics.h"

#include <cmath>

namespace ResearchBench {

FitDiagnosticRecord FitDiagnostics::evaluate(
    const VolSurfaceSnapshot& snapshot,
    const PerSnapshotFitResult& fitResult,
    const std::vector<ExpirSnapshot>& fittedSurface
) const {
    FitDiagnosticRecord d;
    d.timestamp = snapshot.timestamp;
    d.fitSuccess = fitResult.success;
    d.totalExpiries = static_cast<int>(snapshot.expirs.size());
    d.objective = fitResult.objective;
    d.fitError = fitResult.error;

    for (const auto& exp : snapshot.expirs) {
        d.totalStrikes += static_cast<int>(exp.strikes.size());
    }

    int volCount = 0;
    for (const auto& exp : fittedSurface) {
        if (exp.strikes.empty()) {
            continue;
        }
        d.fittedExpiries++;
        for (const auto& st : exp.strikes) {
            const double vol = (st.splineImplVol > 0.0) ? st.splineImplVol : st.weightedImplVol;
            if (!std::isfinite(vol) || vol <= 0.0) {
                d.nanVolCount++;
                continue;
            }
            d.meanFittedVol += vol;
            volCount++;
        }
    }
    if (volCount > 0) {
        d.meanFittedVol /= static_cast<double>(volCount);
    }
    return d;
}

} // namespace ResearchBench
