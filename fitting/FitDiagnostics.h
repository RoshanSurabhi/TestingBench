#pragma once

#include <vector>

#include "../ResearchTypes.h"

namespace ResearchBench {

class FitDiagnostics {
public:
    FitDiagnosticRecord evaluate(
        const VolSurfaceSnapshot& snapshot,
        const PerSnapshotFitResult& fitResult,
        const std::vector<ExpirSnapshot>& fittedSurface
    ) const;
};

} // namespace ResearchBench
