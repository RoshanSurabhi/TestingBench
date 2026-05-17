#include "BinFitPipeline.h"

#include "fileutils.h"

#include "../../FinMath-Lib/CVI/CVISurfaceFitter.h"
#include "../../FinMath-Lib/CVI/VolSnapshotToSurfExpir.h"
#include "../../FinMath-Lib/OptionPricing/Black76.h"
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

bool fitOneSnapshot(
    const VolSurfaceSnapshot& snapshot,
    const fs::path& outDir,
    const ResearchBench::BinFitConfig& cfg,
    int& clarabelStatus,
    double& clarabelObj,
    std::string& message) {
    const double spot = CVI::TrdbUnderlierSpot(snapshot);
    std::vector<PricingTools::SurfExpir> chain = CVI::TrdbSnapshotToSurfExpirChain(snapshot, spot);
    const std::vector<std::string> expirDates = CVI::TrdbExpirDatesFromSnapshot(snapshot);
    if (chain.empty()) {
        message = "empty chain after TrDB -> SurfExpir conversion";
        return false;
    }

    CVI::CVISurfaceFitter fitter;
    auto& p = fitter.m_cviParams;
    p.num_basis = cfg.numBasis;
    p.lambda = cfg.lambda;
    p.num_constraint_strikes = cfg.numConstraintStrikes;
    p.z0 = -6.0;
    p.zn1 = 6.0;
    p.basis_placement_mode = CVI::CVIBasisPlacementMode::DataDrivenClampedToDomain;
    p.z_reference_mode = CVI::CVIZReferenceMode::Forward;
    p.weight_mode = CVI::CVIWeightMode::VarianceSpace;
    p.outside_support_mode = CVI::CVIOutsideSupportMode::KeepAll;
    p.q_treatment_mode = CVI::CVIQTreatmentMode::KeepPreludeQ;
    p.butterfly_refinement_mode = CVI::CVIButterflyRefinementMode::Enabled;
    p.dearb_solver_mode = CVI::CVIDeArbSolverMode::Clarabel;

    fitter.InputData(std::move(chain), spot, CVI::kDeArbConstraintMask_AllSupported);
    if (!fitter.FitSurface(snapshot.instrument.symbol)) {
        clarabelStatus = fitter.LastClarabelStatus();
        clarabelObj = fitter.LastClarabelObjective();
        message = fitter.LastError().empty() ? std::string("FitSurface failed") : fitter.LastError();
        return false;
    }

    clarabelStatus = fitter.LastClarabelStatus();
    clarabelObj = fitter.LastClarabelObjective();

    if (!fs::exists(outDir)) {
        fs::create_directories(outDir);
    }

    const bool wroteExpiry = writeExpiryFwdQCsv(
        outDir / "expiry_fwd_q.csv", fitter.rSurface(), expirDates, fitter.SigmaStar(), fitter.VStar());
    const bool wroteOption = writeOptionFitComparisonCsv(
        outDir / "option_fit_comparison.csv",
        fitter.NumBasis(),
        fitter.BasisEvaluator(),
        fitter.Options(),
        fitter.VStar(),
        fitter.FullSolution());
    const bool wrotePrice = (!cfg.writePriceComparison && !cfg.writeDebugArtifacts)
        || writePriceComparisonCsv(
            outDir / "price_comparison.csv",
            fitter.rSurface(),
            fitter.NumBasis(),
            fitter.BasisEvaluator(),
            fitter.VStar(),
            fitter.FullSolution());

    if (!wroteExpiry || !wroteOption || !wrotePrice) {
        message = "fit solved, but artifact write failed";
        return false;
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

} // namespace

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

    fs::create_directories(config.outDir);
    const fs::path summaryPath = fs::path(config.outDir) / "batch_cvi_summary.csv";
    std::ofstream summary(summaryPath, std::ios::out | std::ios::trunc);
    if (!summary) {
        std::cerr << "Failed to open " << summaryPath.string() << "\n";
        return 2;
    }
    summary << "subfolder,idx_in_bin,timestamp,ok,clarabel_status,clarabel_objective,message\n";

    const std::size_t limit = (config.maxSnapshots == 0)
        ? snapshots.size()
        : std::min(config.maxSnapshots, snapshots.size());
    std::size_t okCount = 0;

    for (std::size_t i = 0; i < limit; ++i) {
        const VolSurfaceSnapshot& snap = snapshots[i];
        const std::string subfolder =
            "cvi_" + std::to_string(i) + "_" + sanitizeTimestampForPath(snap.timestamp);
        const fs::path subDir = fs::path(config.outDir) / subfolder;

        int clarabelStatus = -1;
        double clarabelObj = 0.0;
        std::string message;
        const bool ok = fitOneSnapshot(snap, subDir, config, clarabelStatus, clarabelObj, message);
        if (ok) {
            ++okCount;
        }

        summary << subfolder << "," << i << "," << csvEscape(snap.timestamp) << "," << (ok ? 1 : 0) << ","
            << clarabelStatus << "," << std::setprecision(17) << clarabelObj << "," << csvEscape(message) << "\n";
        summary.flush();
    }

    std::cout << "FitFromBin: loaded " << snapshots.size() << " snapshots, processed " << limit << ".\n";
    std::cout << "Success: " << okCount << "/" << limit << "\n";
    std::cout << "Output: " << summaryPath.string() << "\n";
    std::cout << "Default artifacts per snapshot: expiry_fwd_q.csv + option_fit_comparison.csv\n";
    if (config.writePriceComparison) {
        std::cout << "Optional artifact enabled: price_comparison.csv\n";
    }
    if (config.writeDebugArtifacts) {
        std::cout << "Debug artifacts enabled.\n";
    }
    return 0;
}

