#pragma once

#include <string>
#include <vector>

#include "../ResearchTypes.h"

namespace ResearchBench {

struct CviBridgeConfig {
    unsigned numBasis = 23;
    double lambda = 0.05;
    int deArbConstraintMask = -1;
};

class CviFitBridge {
public:
    explicit CviFitBridge(CviBridgeConfig cfg = {});

    PerSnapshotFitResult fitSnapshot(const VolSurfaceSnapshot& snapshot);
    const std::vector<ExpirSnapshot>& lastSurface() const;

private:
    static std::vector<ExpirSnapshot> toExpirSnapshotSurface(const VolSurfaceSnapshot& snapshot);

    CviBridgeConfig config_;
    std::vector<ExpirSnapshot> lastSurface_;
};

} // namespace ResearchBench
