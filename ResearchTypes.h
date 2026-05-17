#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "volSurfaceUtils.h"
#include "../UtilLib/src/TrData/dbStructs.h"

namespace ResearchBench {

struct StreamRequest {
    std::string ticker;
    std::string startTimestamp;
    std::string endTimestamp;
    int splitMinutes = 30;
    SearchType searchType = SearchType::VOL_SURFACE;
    TransformationType transformationType = TransformationType::RAW_SERIES;
};

struct StreamContext {
    size_t sequenceIndex = 0;
    size_t totalSnapshots = 0;
    std::string timestamp;
    std::string symbol;
};

struct PerSnapshotFitResult {
    bool success = false;
    std::string ticker;
    std::string timestamp;
    int clarabelStatus = -1;
    double objective = 0.0;
    std::string error;
    int numExpiries = 0;
    int numBasis = 0;
};

struct FitDiagnosticRecord {
    std::string timestamp;
    bool fitSuccess = false;
    int totalExpiries = 0;
    int totalStrikes = 0;
    int fittedExpiries = 0;
    int nanVolCount = 0;
    double meanFittedVol = 0.0;
    double objective = 0.0;
    std::string fitError;
};

struct ReturnRecord {
    std::string symbol;
    std::string originTimestamp;
    std::string evalTimestamp;
    int horizonSec = 0;
    std::string expiry;
    double forwardT = 0.0;
    double forwardTPlusH = 0.0;
    double forwardReturnLog = 0.0;
    double spotT = 0.0;
    double spotTPlusH = 0.0;
    double spotReturnLog = 0.0;
    bool valid = false;
};

struct ResearchRunResult {
    bool success = false;
    size_t streamedSnapshots = 0;
    size_t fitResults = 0;
    size_t diagnostics = 0;
    size_t delayedOutputs = 0;
    std::string message;
};

struct DelayedEvaluationContext {
    VolSurfaceSnapshot originSnapshot;
    VolSurfaceSnapshot evalSnapshot;
    StreamContext originContext;
    StreamContext evalContext;
    int horizonSec = 0;
};

using SnapshotCallback = std::function<void(const VolSurfaceSnapshot&, const StreamContext&)>;
using DelayedCallback = std::function<void(const DelayedEvaluationContext&)>;

struct DelayedCallbackSpec {
    std::string name;
    int horizonSec = 5;
    DelayedCallback callback;
};

} // namespace ResearchBench
