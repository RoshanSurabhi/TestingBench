#include "BinFitPipeline.h"

#include "BinFitClampedReplica.h"
#include "fileutils.h"

#include "../../FinMath-Lib/CVI/CVISurfaceFitter.h"
#include "../../FinMath-Lib/CVI/VolSnapshotToSurfExpir.h"
#include "../../FinMath-Lib/OptionPricing/Black76.h"
#include "../../FinMath-Lib/OptionPricing/SolveDeArb.h"
#include "../../UtilLib/src/TrData/dbStructs.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace fs = std::filesystem;
namespace CVI = PricingTools::CVI;

namespace {

std::string sanitizeTimestampForPath(const std::string& timestamp) {
    std::string out = timestamp;
    for (char& c : out) {
        if (c == ' ' || c == ':' || c == '/' || c == '\\' || c == '.' || c == '+' || c == '|') {
            c = '_';
        }
    }
    return out.empty() ? std::string("snapshot") : out;
}

std::string csvEscape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }
    std::string out;
    out.reserve(value.size() + 4);
    out.push_back('"');
    for (char c : value) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
    return out;
}

bool writeExpiryFwdQCsv(
    const fs::path& path,
    const std::vector<PricingTools::SurfExpir>& chain,
    const std::vector<std::string>& expirDates,
    const std::vector<double>& sigmaStar,
    const std::vector<double>& vStar) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        return false;
    }
    out << "expiry_idx,expiry_date,F,q,volTime,r,sigma_star,v_star\n";
    out << std::setprecision(17);
    for (std::size_t j = 0; j < chain.size(); ++j) {
        const auto& e = chain[j];
        const std::string date = (j < expirDates.size()) ? expirDates[j] : std::string();
        out << j << "," << csvEscape(date) << "," << e.m_forward << "," << e.m_q << "," << e.m_volTime << "," << e.m_r
            << ",";
        if (j < sigmaStar.size()) {
            out << sigmaStar[j];
        }
        out << ",";
        if (j < vStar.size()) {
            out << vStar[j];
        }
        out << "\n";
    }
    return true;
}

bool writeOptionFitComparisonCsv(
    const fs::path& path,
    int numBasis,
    const CVI::CVIBasisEvaluator& evaluator,
    const std::vector<CVI::CVIOptionRecord>& options,
    const std::vector<double>& vStar,
    const std::vector<double>& fullSolution) {
    if (numBasis <= 0) {
        return false;
    }
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        return false;
    }

    out << "expiry_index,strike,z,fitted_var,fitted_vol,v_mid,v_bid,v_ask,w_mid,w_bid,w_ask\n";
    out << std::setprecision(17);
    std::vector<double> coeff(static_cast<std::size_t>(numBasis), 0.0);
    for (const auto& rec : options) {
        if (rec.expiry_index < 0) {
            continue;
        }
        const std::size_t expiry = static_cast<std::size_t>(rec.expiry_index);
        const std::size_t blockStart = expiry * static_cast<std::size_t>(numBasis);
        if (blockStart + static_cast<std::size_t>(numBasis) > fullSolution.size()) {
            continue;
        }

        for (int b = 0; b < numBasis; ++b) {
            coeff[static_cast<std::size_t>(b)] = fullSolution[blockStart + static_cast<std::size_t>(b)];
        }

        const double vScale = (expiry < vStar.size()) ? vStar[expiry] : 1.0;
        std::vector<double> basisBlock(static_cast<std::size_t>(numBasis), 0.0);
        evaluator.EvalVariance(rec.z, vScale, basisBlock.data());
        double fittedVar = 0.0;
        for (int b = 0; b < numBasis; ++b) {
            fittedVar += basisBlock[static_cast<std::size_t>(b)] * coeff[static_cast<std::size_t>(b)];
        }
        if (fittedVar < 1e-12) {
            fittedVar = 1e-12;
        }
        const double fittedVol = std::sqrt(std::max(0.0, fittedVar));

        out << rec.expiry_index << "," << rec.strike << "," << rec.z << "," << fittedVar << "," << fittedVol << ","
            << rec.v_mid << "," << rec.v_bid << "," << rec.v_ask << "," << rec.w_mid << "," << rec.w_bid << ","
            << rec.w_ask << "\n";
    }
    return true;
}

bool writePriceComparisonCsv(
    const fs::path& path,
    const std::vector<PricingTools::SurfExpir>& chain,
    int numBasis,
    const CVI::CVIBasisEvaluator& evaluator,
    const std::vector<double>& vStar,
    const std::vector<double>& fullSolution) {
    if (numBasis <= 0) {
        return false;
    }
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        return false;
    }

    out << "expiry_index,strike,fitted_call,fitted_put,market_call_mid,market_put_mid,fitted_vol,vol_time,forward,r\n";
    out << std::setprecision(17);

    std::vector<double> coeff(static_cast<std::size_t>(numBasis), 0.0);
    for (std::size_t expiry = 0; expiry < chain.size(); ++expiry) {
        const std::size_t blockStart = expiry * static_cast<std::size_t>(numBasis);
        if (blockStart + static_cast<std::size_t>(numBasis) > fullSolution.size()) {
            continue;
        }
        for (int b = 0; b < numBasis; ++b) {
            coeff[static_cast<std::size_t>(b)] = fullSolution[blockStart + static_cast<std::size_t>(b)];
        }
        const auto& e = chain[expiry];
        const double vScale = (expiry < vStar.size()) ? vStar[expiry] : 1.0;

        PricingTools::Black76 callModel;
        PricingTools::Black76 putModel;
        for (const auto& strike : e.m_strikes) {
            if (!(strike.Strike > 0.0) || !(e.m_forward > 0.0) || !(e.m_volTime > 0.0)) {
                continue;
            }
            const double zDen = std::sqrt(std::max(1e-12, vScale * std::max(e.m_volTime, 1e-12)));
            const double z = std::log(strike.Strike / e.m_forward) / zDen;
            std::vector<double> basisBlock(static_cast<std::size_t>(numBasis), 0.0);
            evaluator.EvalVariance(z, vScale, basisBlock.data());
            double fittedVar = 0.0;
            for (int b = 0; b < numBasis; ++b) {
                fittedVar += basisBlock[static_cast<std::size_t>(b)] * coeff[static_cast<std::size_t>(b)];
            }
            if (fittedVar < 1e-12) {
                fittedVar = 1e-12;
            }
            const double fittedVol = std::sqrt(std::max(0.0, fittedVar));

            callModel.TheoInput(
                e.m_forward, strike.Strike, fittedVol, e.m_r, PricingTools::OptType::CALL, e.m_carryTime, e.m_volTime);
            putModel.TheoInput(
                e.m_forward, strike.Strike, fittedVol, e.m_r, PricingTools::OptType::PUT, e.m_carryTime, e.m_volTime);

            const double fittedCall = callModel.TheoPrice();
            const double fittedPut = putModel.TheoPrice();
            const double marketCallMid = 0.5 * (std::max(0.0, strike.CallBid) + std::max(0.0, strike.CallAsk));
            const double marketPutMid = 0.5 * (std::max(0.0, strike.PutBid) + std::max(0.0, strike.PutAsk));

            out << expiry << "," << strike.Strike << "," << fittedCall << "," << fittedPut << "," << marketCallMid << ","
                << marketPutMid << "," << fittedVol << "," << e.m_volTime << "," << e.m_forward << "," << e.m_r << "\n";
        }
    }
    return true;
}

} // namespace

namespace {

void applyBinFitCviParams(CVI::CVISurfaceFitter& fitter, const ResearchBench::BinFitConfig& cfg) {
    fitter.m_cviParams = CVI::CVIFitParams{};
    fitter.m_cviParams.num_basis = cfg.numBasis;
    fitter.m_cviParams.lambda = cfg.lambda;
    fitter.m_cviParams.num_constraint_strikes = cfg.numConstraintStrikes;
    if (!cfg.dearbSolver.empty()) {
        if (cfg.dearbSolver == "qpoases") {
            fitter.m_cviParams.dearb_solver_mode = CVI::CVIDeArbSolverMode::QpOases;
        } else if (cfg.dearbSolver == "clarabel") {
            fitter.m_cviParams.dearb_solver_mode = CVI::CVIDeArbSolverMode::Clarabel;
        }
    }
    if (!cfg.dearbLoss.empty()) {
        if (cfg.dearbLoss == "uniform") {
            fitter.m_cviParams.dearb_loss_weight_mode = CVI::CVIDeArbLossWeightMode::Uniform;
        } else if (cfg.dearbLoss == "inverse_raw_spread" || cfg.dearbLoss == "inverse-raw-spread") {
            fitter.m_cviParams.dearb_loss_weight_mode = CVI::CVIDeArbLossWeightMode::InverseRawSpread;
        } else if (cfg.dearbLoss == "inverse_spread" || cfg.dearbLoss == "inverse-spread") {
            fitter.m_cviParams.dearb_loss_weight_mode = CVI::CVIDeArbLossWeightMode::InverseSpread;
        }
    }
}

} // namespace

namespace ResearchBench {

bool fitOneSnapshot(
    const VolSurfaceSnapshot& snapshot,
    const fs::path& outDir,
    const BinFitConfig& cfg,
    int snapIdxInBin,
    int& clarabelStatus,
    double& clarabelObj,
    std::string& message) {
    if (cfg.writeFullArtifacts) {
        fs::path exeDir = fs::current_path();
#ifdef _WIN32
        wchar_t modulePath[MAX_PATH]{};
        if (GetModuleFileNameW(nullptr, modulePath, MAX_PATH) != 0) {
            exeDir = fs::path(modulePath).parent_path();
        }
#endif
        const fs::path clampedExe = findClampedMaskTestExe(exeDir);
        return runClampedMaskReplica(
            clampedExe,
            cfg.binPath,
            snapIdxInBin,
            outDir,
            cfg.numBasis,
            cfg.lambda,
            cfg.numConstraintStrikes,
            cfg.z0,
            cfg.zn1,
            cfg.dearbSolver,
            cfg.dearbLoss,
            clarabelStatus,
            clarabelObj,
            message);
    }

    const double spot = CVI::TrdbUnderlierSpot(snapshot);
    std::vector<PricingTools::SurfExpir> chain = CVI::TrdbSnapshotToSurfExpirChain(snapshot, spot);
    const std::vector<std::string> expirDates = CVI::TrdbExpirDatesFromSnapshot(snapshot);
    if (chain.empty()) {
        message = "empty chain after TrDB -> SurfExpir conversion";
        return false;
    }

    CVI::CVISurfaceFitter fitter;
    applyBinFitCviParams(fitter, cfg);
    fitter.InputData(std::move(chain), spot, static_cast<int>(CVI::kDeArbConstraintMask_AllSupported));

    fs::create_directories(outDir);
    if (cfg.writeDebugArtifacts) {
        std::ofstream pre(outDir / "fit_debug.txt", std::ios::out | std::ios::trunc);
        if (pre) {
            pre << "symbol=" << snapshot.instrument.symbol << "\n";
            pre << "timestamp=" << snapshot.timestamp << "\n";
            pre << "spot=" << spot << "\n";
            pre << "num_expiries=" << snapshot.expirs.size() << "\n";
            size_t strikes = 0;
            for (const auto& e : snapshot.expirs) {
                strikes += e.strikes.size();
            }
            pre << "num_strikes=" << strikes << "\n";
            pre << "surfexpir_chain=" << fitter.NumExpiries() << "\n";
        }
    }

    const bool enableDearbQpDiag = cfg.writeFullArtifacts || cfg.dearbQpDiagnostics;
    if (enableDearbQpDiag) {
        std::vector<std::string> expirDatesMut = expirDates;
        DeArb::EnableDeArbQpDiagnosticsInDir(outDir, &expirDatesMut);
    }
    const bool fitOk = fitter.FitSurface(snapshot.instrument.symbol);
    if (enableDearbQpDiag) {
        DeArb::ClearSolveDeArbDiagnostics();
    }
    clarabelStatus = fitter.LastClarabelStatus();
    clarabelObj = fitter.LastClarabelObjective();

    if (!fitOk) {
        message = fitter.LastError().empty() ? std::string("FitSurface failed") : fitter.LastError();
    }

    if (!fs::exists(outDir)) {
        fs::create_directories(outDir);
    }

    if (!fitOk) {
        if (cfg.writeDebugArtifacts) {
            std::ofstream dbg(outDir / "fit_debug.txt", std::ios::out | std::ios::app);
            if (dbg) {
                dbg << "clarabel_status=" << clarabelStatus << "\n";
                dbg << "clarabel_objective=" << std::setprecision(17) << clarabelObj << "\n";
                dbg << "last_error=" << message << "\n";
            }
        }
        return false;
    } else {
        const bool wroteExpiry = writeExpiryFwdQCsv(
            outDir / "expiry_fwd_q.csv", fitter.rSurface(), expirDates, fitter.SigmaStar(), fitter.VStar());
        const bool wroteOption = writeOptionFitComparisonCsv(
            outDir / "option_fit_comparison.csv",
            fitter.NumBasis(),
            fitter.BasisEvaluator(),
            fitter.Options(),
            fitter.VStar(),
            fitter.FullSolution());
        if (cfg.writePriceComparison) {
            writePriceComparisonCsv(
                outDir / "price_comparison.csv",
                fitter.rSurface(),
                fitter.NumBasis(),
                fitter.BasisEvaluator(),
                fitter.VStar(),
                fitter.FullSolution());
        }
        if (!wroteExpiry || !wroteOption) {
            message = "fit solved, but artifact write failed (expiry=" + std::to_string(wroteExpiry)
                + ", option=" + std::to_string(wroteOption) + ")";
            return false;
        }
    }

    if (cfg.writeDebugArtifacts) {
        std::ofstream dbg(outDir / "fit_debug.txt", std::ios::out | std::ios::trunc);
        if (dbg) {
            dbg << "symbol=" << snapshot.instrument.symbol << "\n";
            dbg << "timestamp=" << snapshot.timestamp << "\n";
            dbg << "num_expiries=" << fitter.NumExpiries() << "\n";
            dbg << "num_basis=" << fitter.NumBasis() << "\n";
            dbg << "clarabel_status=" << fitter.LastClarabelStatus() << "\n";
            dbg << "clarabel_objective=" << std::setprecision(17) << fitter.LastClarabelObjective() << "\n";
        }
    }

    message = "ok";
    return true;
}

} // namespace ResearchBench

int ResearchBench::runFitFromBin(const BinFitConfig& config) {
    if (config.binPath.empty()) {
        std::cerr << "Missing --fit-from-bin=<path>\n";
        return 2;
    }
    if (config.outDir.empty()) {
        std::cerr << "Missing --fit-out-dir=<path>\n";
        return 2;
    }

    std::vector<VolSurfaceSnapshot> snapshots;
    try {
        snapshots = loadTrdbVolSnapshotsBinRaw(config.binPath);
    } catch (const std::exception& ex) {
        std::cerr << "Failed to load snapshots: " << ex.what() << "\n";
        return 2;
    }
    if (snapshots.empty()) {
        std::cerr << "No snapshots in " << config.binPath << "\n";
        return 2;
    }

    if (config.listTimestampsOnly) {
        for (std::size_t i = 0; i < snapshots.size(); ++i) {
            std::cout << i << "\t" << snapshots[i].timestamp << "\n";
        }
        std::cout << "Total: " << snapshots.size() << " snapshot(s)\n";
        return 0;
    }

    std::vector<std::size_t> indices;
    indices.reserve(snapshots.size());
    for (std::size_t i = 0; i < snapshots.size(); ++i) {
        if (config.fitTimestamp.empty()) {
            indices.push_back(i);
            continue;
        }
        if (snapshots[i].timestamp.find(config.fitTimestamp) != std::string::npos) {
            indices.push_back(i);
        }
    }
    if (!config.fitTimestamp.empty() && indices.empty()) {
        std::cerr << "No snapshot matched --fit-timestamp=\"" << config.fitTimestamp << "\"\n";
        std::cerr << "Use --fit-list-timestamps to see available values.\n";
        return 2;
    }

    fs::create_directories(config.outDir);
    const fs::path summaryPath = fs::path(config.outDir) / "batch_cvi_summary.csv";
    std::ofstream summary(summaryPath, std::ios::out | std::ios::trunc);
    if (!summary) {
        std::cerr << "Failed to open " << summaryPath.string() << "\n";
        return 2;
    }
    summary << "subfolder,idx_in_bin,timestamp,ok,clarabel_status,clarabel_objective,message\n";

    const std::size_t runCount = indices.empty() ? snapshots.size() : indices.size();
    const std::size_t limit = (config.maxSnapshots == 0)
        ? runCount
        : std::min(config.maxSnapshots, runCount);
    std::size_t okCount = 0;

    for (std::size_t run = 0; run < limit; ++run) {
        const std::size_t i = indices.empty() ? run : indices[run];
        const VolSurfaceSnapshot& snap = snapshots[i];
        const std::string subfolder =
            "cvi_" + std::to_string(i) + "_" + sanitizeTimestampForPath(snap.timestamp);
        const fs::path subDir = fs::path(config.outDir) / subfolder;

        int clarabelStatus = -1;
        double clarabelObj = 0.0;
        std::string message;
        const bool ok = fitOneSnapshot(snap, subDir, config, static_cast<int>(i), clarabelStatus, clarabelObj, message);
        if (ok) {
            ++okCount;
        }

        summary << subfolder << "," << i << "," << csvEscape(snap.timestamp) << "," << (ok ? 1 : 0) << ","
            << clarabelStatus << "," << std::setprecision(17) << clarabelObj << "," << csvEscape(message) << "\n";
        summary.flush();
    }

    std::cout << "FitFromBin: loaded " << snapshots.size() << " snapshots, processed " << limit;
    if (!config.fitTimestamp.empty()) {
        std::cout << " (filter=\"" << config.fitTimestamp << "\")";
    }
    std::cout << ".\n";
    std::cout << "Success: " << okCount << "/" << limit << "\n";
    std::cout << "Output: " << summaryPath.string() << "\n";
    std::cout << "Default artifacts per snapshot: expiry_fwd_q.csv + option_fit_comparison.csv\n";
    if (config.writePriceComparison) {
        std::cout << "Optional artifact enabled: price_comparison.csv\n";
    }
    if (config.writeDebugArtifacts) {
        std::cout << "Debug artifacts enabled.\n";
    }
    if (config.writeFullArtifacts) {
        std::cout << "Full artifacts: CVITestSurfaceFitterClampedMask replica (exact test_clamped set).\n";
    }
    return 0;
}

