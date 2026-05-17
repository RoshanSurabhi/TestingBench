#pragma once

#include <string>
#include <vector>

#include "CviFitBridge.h"
#include "FitDiagnostics.h"
#include "../ResearchTypes.h"
#include "SnapshotStreamAdapter.h"

namespace ResearchBench {

class ResearchBenchEngine {
public:
    ResearchBenchEngine();

    /** Log each WebSocket response body to stderr (hex + ASCII). maxBytes 0 = no limit. */
    void setDumpPayload(bool enabled, size_t maxBytes);

    ResearchRunResult run(const StreamRequest& request);

    const std::vector<PerSnapshotFitResult>& fitResults() const { return fitResults_; }
    const std::vector<FitDiagnosticRecord>& diagnostics() const { return diagnostics_; }

private:
    SnapshotStreamAdapter streamAdapter_;
    CviFitBridge cviBridge_;
    FitDiagnostics diagnosticsEngine_;

    std::vector<PerSnapshotFitResult> fitResults_;
    std::vector<FitDiagnosticRecord> diagnostics_;
};

} // namespace ResearchBench
