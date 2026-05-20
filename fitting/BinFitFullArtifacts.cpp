#include "BinFitFullArtifacts.h"

#include "../../FinMath-Lib/CVI/CVIOptionData.h"
#include "../../FinMath-Lib/CVI/CVITestSuite/CVIArtifactWriter.h"
#include "../../FinMath-Lib/CVI/VolSnapshotToSurfExpir.h"

#include <fstream>
#include <iomanip>

namespace fs = std::filesystem;
namespace CVI = PricingTools::CVI;

namespace {

bool writeExpiryFwdQCsv(
    const fs::path& path,
    const std::vector<PricingTools::SurfExpir>& chain,
    const std::vector<std::string>& expirDates,
    const std::vector<double>& sigmaStar,
    const std::vector<double>& vStar) {
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) {
        return false;
    }
    ofs << "expiry_idx,expiry_date,F,q,volTime,r,sigma_star,v_star\n";
    ofs << std::setprecision(17);
    for (std::size_t j = 0; j < chain.size(); ++j) {
        const auto& e = chain[j];
        const std::string date = (j < expirDates.size()) ? expirDates[j] : std::string();
        ofs << j << ",";
        if (!date.empty()) {
            ofs << "\"" << date << "\"";
        }
        ofs << "," << e.m_forward << "," << e.m_q << "," << e.m_volTime << "," << e.m_r << ",";
        if (j < sigmaStar.size()) {
            ofs << sigmaStar[j];
        }
        ofs << ",";
        if (j < vStar.size()) {
            ofs << vStar[j];
        }
        ofs << "\n";
    }
    return true;
}

const std::vector<PricingTools::SurfExpir>& chainForForwardQ(
    CVI::CVISurfaceFitter& fitter,
    const std::vector<PricingTools::SurfExpir>& chainAfterSpline) {
    if (!fitter.rSurface().empty()) {
        return fitter.rSurface();
    }
    return chainAfterSpline;
}

bool hasPreSolveArtifacts(const CVI::CVISurfaceFitter& fitter) {
    return fitter.NumBasis() > 0 && fitter.NumExpiries() > 0 && !fitter.Options().empty()
        && fitter.MatrixBuildResult().n_v > 0;
}

bool hasPostSolveArtifacts(const CVI::CVISurfaceFitter& fitter) {
    return fitter.IsFitted() && !fitter.FullSolution().empty();
}

} // namespace

namespace ResearchBench {

bool writeClampedStyleCviArtifacts(
    const fs::path& outDir,
    const VolSurfaceSnapshot& snapshot,
    double spot,
    const std::vector<PricingTools::SurfExpir>& chainBeforePreCvi,
    const std::vector<PricingTools::SurfExpir>& chainAfterDearb,
    const std::vector<PricingTools::SurfExpir>& chainAfterSpline,
    const std::vector<std::string>& expirDates,
    CVI::CVISurfaceFitter& fitter,
    std::string& errOut,
    bool allowPartial) {
    using CVITestSuiteArtifacts::WriteDeArbQpDiagnosticsHtml;
    using CVITestSuiteArtifacts::WriteFitSummary;
    using CVITestSuiteArtifacts::WriteForwardAndQHtml;
    using CVITestSuiteArtifacts::WritePipelineDiagnosticsHtml;
    using CVITestSuiteArtifacts::WritePipelineVolStepDiagnostics;
    using CVITestSuiteArtifacts::WritePostSolveArtifacts;
    using CVITestSuiteArtifacts::WritePreSolveArtifacts;
    using CVITestSuiteArtifacts::WritePriceArtifacts;
    using CVITestSuiteArtifacts::WriteTotalVariancePairsHtml;

    fs::create_directories(outDir);

    const int m = fitter.NumExpiries();
    const int nb = fitter.NumBasis();
    const int arb = static_cast<int>(fitter.m_cviParams.num_constraint_strikes);
    const bool preSolve = hasPreSolveArtifacts(fitter);
    const bool postSolve = hasPostSolveArtifacts(fitter);

    if (!preSolve && !postSolve && !allowPartial) {
        errOut = "no CVI problem built; cannot write clamped-style artifacts";
        return false;
    }

    std::vector<CVI::CVIOptionRecord> pipeOptions;
    std::vector<double> invNMid, invNAsk, invNBid, sigmaStarPipe;
    const std::vector<PricingTools::SurfExpir>& splineChain =
        chainAfterSpline.empty() ? chainBeforePreCvi : chainAfterSpline;
    CVI::BuildOptionListFromChain(
        splineChain,
        spot,
        pipeOptions,
        invNMid,
        invNAsk,
        invNBid,
        sigmaStarPipe,
        false,
        fitter.m_cviParams.weight_mode);

    const std::string runTitle =
        snapshot.instrument.symbol.empty() ? std::string("ResearchBench") : snapshot.instrument.symbol;

    bool ok = true;
    bool wroteAny = false;

    if (!chainBeforePreCvi.empty() && !splineChain.empty()) {
        ok = WritePipelineDiagnosticsHtml(
                  outDir,
                  chainBeforePreCvi,
                  splineChain,
                  spot,
                  expirDates,
                  static_cast<int>(CVI::kDeArbConstraintMask_AllSupported),
                  pipeOptions,
                  sigmaStarPipe,
                  runTitle.c_str())
            && ok;
        ok = WritePipelineVolStepDiagnostics(
                  outDir,
                  chainBeforePreCvi,
                  chainAfterDearb.empty() ? chainBeforePreCvi : chainAfterDearb,
                  splineChain,
                  spot,
                  expirDates,
                  pipeOptions,
                  sigmaStarPipe,
                  runTitle.c_str())
            && ok;
        ok = WriteForwardAndQHtml(outDir, chainForForwardQ(fitter, splineChain), expirDates) && ok;
        ok = WriteDeArbQpDiagnosticsHtml(outDir) && ok;
        wroteAny = true;
    }

    if (preSolve) {
        ok = WritePreSolveArtifacts(
                  outDir,
                  m,
                  nb,
                  arb,
                  fitter.BasisEvaluator(),
                  fitter.Options(),
                  fitter.MatrixBuildResult(),
                  fitter.ConstraintData())
            && ok;
        wroteAny = true;
    }

    if (postSolve) {
        const std::vector<double>& xOut = fitter.FullSolution();
        {
            ok = WritePostSolveArtifacts(
                      outDir,
                      m,
                      nb,
                      fitter.MatrixBuildResult().n_v_orig,
                      fitter.BasisEvaluator(),
                      fitter.Options(),
                      fitter.VStar(),
                      xOut,
                      fitter.LastClarabelStatus(),
                      fitter.LastClarabelObjective())
                && ok;
            ok = WriteFitSummary(
                      outDir,
                      m,
                      nb,
                      arb,
                      fitter.m_cviParams.lambda,
                      fitter.BasisEvaluator(),
                      fitter.Options(),
                      fitter.VStar(),
                      xOut,
                      fitter.LastClarabelStatus(),
                      fitter.LastClarabelObjective())
                && ok;
            ok = WritePriceArtifacts(
                      outDir,
                      m,
                      nb,
                      fitter.BasisEvaluator(),
                      fitter.Options(),
                      fitter.VStar(),
                      xOut,
                      fitter.rSurface(),
                      spot)
                && ok;
            ok = WriteTotalVariancePairsHtml(
                      outDir,
                      m,
                      nb,
                      fitter.BasisEvaluator(),
                      fitter.Options(),
                      fitter.VStar(),
                      fitter.SigmaStar(),
                      xOut,
                      fitter.rSurface())
                && ok;
            ok = writeExpiryFwdQCsv(
                      outDir / "expiry_fwd_q.csv",
                      fitter.rSurface(),
                      expirDates,
                      fitter.SigmaStar(),
                      fitter.VStar())
                && ok;
            wroteAny = true;
        }
    }

    if (!wroteAny) {
        errOut = "no artifacts could be written (pipeline chains empty and CVI problem not built)";
        return false;
    }

    if (!ok) {
        errOut = allowPartial
            ? "one or more artifact writes failed (partial bundle may still be on disk)"
            : "CVI fit solved but one or more clamped-style artifact writes failed.";
        return !allowPartial;
    }

    if (allowPartial && !postSolve) {
        errOut = "partial artifacts written (fit did not produce a usable solution)";
    } else {
        errOut.clear();
    }
    return true;
}

} // namespace ResearchBench
