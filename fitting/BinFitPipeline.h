#pragma once

#include <cstddef>
#include <string>

namespace ResearchBench {

struct BinFitConfig {
    std::string binPath;
    std::string outDir;
    std::size_t maxSnapshots = 0; // 0 => all snapshots
    unsigned int numBasis = 30u;
    double lambda = 0.005;
    int numConstraintStrikes = 20;
    bool writePriceComparison = false;
    bool writeDebugArtifacts = false;
};

int runFitFromBin(const BinFitConfig& config);

} // namespace ResearchBench

