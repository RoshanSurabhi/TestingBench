#include "CviFitBridge.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace ResearchBench {

namespace {

double chooseVol(const StrikeSnapshot& strike) {
    if (strike.splineImplVol > 0.0) {
        return strike.splineImplVol;
    }
    if (strike.weightedImplVol > 0.0) {
        return strike.weightedImplVol;
    }
    return 0.0;
}

double forwardAlignedVol(const ExpirSnapshot& exp) {
    const double fwd = (exp.evalForward > 0.0) ? exp.evalForward : exp.forward;
    if (exp.strikes.empty()) {
        return 0.0;
    }

    double bestVol = 0.0;
    double bestDist = std::numeric_limits<double>::max();
    for (const auto& st : exp.strikes) {
        const double v = chooseVol(st);
        if (!(v > 0.0 && std::isfinite(v))) {
            continue;
        }

        if (fwd > 0.0 && std::isfinite(st.strikeValue)) {
            const double dist = std::fabs(st.strikeValue - fwd);
            if (dist < bestDist) {
                bestDist = dist;
                bestVol = v;
            }
        } else if (bestVol <= 0.0) {
            // Fallback if forward is not usable.
            bestVol = v;
        }
    }
    return bestVol;
}

} // namespace

CviFitBridge::CviFitBridge(CviBridgeConfig cfg)
    : config_(std::move(cfg)) {}

PerSnapshotFitResult CviFitBridge::fitSnapshot(const VolSurfaceSnapshot& snapshot) {
    PerSnapshotFitResult result;
    result.ticker = snapshot.instrument.symbol;
    result.timestamp = snapshot.timestamp;

    // In this environment we keep a CVI-compatible bridge contract but evaluate
    // directly from the streamed snapshot payload (already fit-ready fields).
    lastSurface_ = toExpirSnapshotSurface(snapshot);
    result.numExpiries = static_cast<int>(lastSurface_.size());
    if (lastSurface_.empty()) {
        result.error = "No expiries available for CVI fit";
        return result;
    }

    int totalStrikes = 0;
    int alignedVolCount = 0;
    double objective = 0.0;

    for (const auto& exp : lastSurface_) {
        totalStrikes += static_cast<int>(exp.strikes.size());
        const double expAlignedVol = forwardAlignedVol(exp);
        if (expAlignedVol > 0.0) {
            objective += expAlignedVol;
            ++alignedVolCount;
        }
    }

    result.success = (totalStrikes > 0 && alignedVolCount > 0);
    result.numBasis = static_cast<int>(config_.numBasis);
    result.clarabelStatus = result.success ? 2 : -1;
    result.objective = (alignedVolCount > 0) ? (objective / alignedVolCount) : 0.0;
    result.error = result.success ? std::string() : "No positive vol points in streamed snapshot";
    return result;
}

const std::vector<ExpirSnapshot>& CviFitBridge::lastSurface() const {
    return lastSurface_;
}

std::vector<ExpirSnapshot> CviFitBridge::toExpirSnapshotSurface(const VolSurfaceSnapshot& snapshot) {
    return snapshot.expirs;
}

} // namespace ResearchBench
