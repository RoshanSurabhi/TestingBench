#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "../../FinMath-Lib/CVI/CVISurfaceFitter.h"
#include "../../UtilLib/src/TrData/dbStructs.h"

namespace PricingTools {
struct SurfExpir;
}

namespace ResearchBench {

/**
 * CVITestSuite / testclampedmask-style artifact bundle (HTML + CSV diagnostics).
 * When allowPartial is true, writes pipeline + pre-solve artifacts even if FitSurface failed;
 * post-solve / price HTML only when a usable solution exists.
 */
bool writeClampedStyleCviArtifacts(
    const std::filesystem::path& outDir,
    const VolSurfaceSnapshot& snapshot,
    double spot,
    const std::vector<PricingTools::SurfExpir>& chainBeforePreCvi,
    const std::vector<PricingTools::SurfExpir>& chainAfterDearb,
    const std::vector<PricingTools::SurfExpir>& chainAfterSpline,
    const std::vector<std::string>& expirDates,
    PricingTools::CVI::CVISurfaceFitter& fitter,
    std::string& errOut,
    bool allowPartial = false);

} // namespace ResearchBench
