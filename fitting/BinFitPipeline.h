#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

struct VolSurfaceSnapshot;

namespace ResearchBench {

struct BinFitConfig {
    std::string binPath;
    std::string outDir;
    std::size_t maxSnapshots = 0; // 0 => all snapshots
    /** If non-empty, only snapshots whose timestamp contains this substring (e.g. "2026-05-05 15:45:02"). */
    std::string fitTimestamp;
    /** If true, print idx + timestamp for each snapshot in the bin and exit without fitting. */
    bool listTimestampsOnly = false;
    unsigned int numBasis = 25u;
    double lambda = 1e-4;
    int numConstraintStrikes = 20;
    double z0 = -6.0;
    double zn1 = 6.0;
    bool writePriceComparison = false;
    bool writeDebugArtifacts = false;
    /**
     * Exact CVITestSurfaceFitterClampedMask artifact set (runs the test executable in-process via subprocess).
     * Replaces the CVISurfaceFitter-based artifact path.
     */
    bool writeFullArtifacts = false;
    /** If non-empty: "clarabel" or "qpoases" overrides CVIFitParams::dearb_solver_mode for pre-CVI de-arb. */
    std::string dearbSolver;
    /** If non-empty: "inverse_spread", "inverse_raw_spread", or "uniform" overrides CVIFitParams::dearb_loss_weight_mode. */
    std::string dearbLoss;
    /** Write solve_de_arb_qp_diagnostics.csv and de_arb_qp/ Clarabel matrix tree (also implied by writeFullArtifacts). */
    bool dearbQpDiagnostics = false;
};

int runFitFromBin(const BinFitConfig& config);

/** @param snapIdxInBin index inside the loaded .bin (for clamped replica --trdb-snap-index). */
bool fitOneSnapshot(
    const VolSurfaceSnapshot& snapshot,
    const std::filesystem::path& outDir,
    const BinFitConfig& cfg,
    int snapIdxInBin,
    int& clarabelStatus,
    double& clarabelObj,
    std::string& message);

} // namespace ResearchBench

