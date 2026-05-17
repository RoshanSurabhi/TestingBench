#include "ResearchBenchEngine.h"

namespace ResearchBench {

ResearchBenchEngine::ResearchBenchEngine() = default;

void ResearchBenchEngine::setDumpPayload(bool enabled, size_t maxBytes) {
    streamAdapter_.setDumpPayload(enabled, maxBytes);
}

ResearchRunResult ResearchBenchEngine::run(const StreamRequest& request) {
    fitResults_.clear();
    diagnostics_.clear();

    ResearchRunResult run;
    run.success = true;
    run.message = "ok";

    streamAdapter_.stream(
        request,
        []() {},
        [this, &run](const VolSurfaceSnapshot& snapshot, const StreamContext&) {
            run.streamedSnapshots++;

            const auto fit = cviBridge_.fitSnapshot(snapshot);
            fitResults_.push_back(fit);
            run.fitResults++;

            const auto diag = diagnosticsEngine_.evaluate(snapshot, fit, cviBridge_.lastSurface());
            diagnostics_.push_back(diag);
            run.diagnostics++;
        },
        []() {},
        [&run](const std::string& err) {
            run.success = false;
            run.message = err;
        }
    );

    run.delayedOutputs = 0;
    return run;
}

} // namespace ResearchBench
