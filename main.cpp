#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

#include "fileutils.h"
#include "fitting/BinFitPipeline.h"
#include "fitting/CviFitBridge.h"
#include "Exporters.h"
#include "fitting/ResearchBenchEngine.h"
#include "Tests.h"

namespace {

std::string getArg(int argc, char* argv[], const std::string& key, const std::string& fallback = "") {
    const std::string prefix = key + "=";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind(prefix, 0) == 0) {
            return arg.substr(prefix.size());
        }
    }
    return fallback;
}

bool hasFlag(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == flag) {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char* argv[]) {
    if (hasFlag(argc, argv, "--run-tests")) {
        const bool ok = ResearchBench::runTests();
        return ok ? 0 : 1;
    }

    const std::string fitFromBin = getArg(argc, argv, "--fit-from-bin", "");
    if (!fitFromBin.empty()) {
        ResearchBench::BinFitConfig cfg;
        cfg.binPath = fitFromBin;
        cfg.outDir = getArg(argc, argv, "--fit-out-dir", "fit_from_bin_batch");
        cfg.maxSnapshots = static_cast<std::size_t>(std::stoull(getArg(argc, argv, "--fit-max-snaps", "0")));
        cfg.numBasis = static_cast<unsigned int>(std::stoul(getArg(argc, argv, "--fit-num-basis", "25")));
        cfg.lambda = std::stod(getArg(argc, argv, "--fit-lambda", "0.0001"));
        cfg.numConstraintStrikes = std::stoi(getArg(argc, argv, "--fit-arb-points", "20"));
        cfg.z0 = std::stod(getArg(argc, argv, "--fit-z0", "-6"));
        cfg.zn1 = std::stod(getArg(argc, argv, "--fit-zn1", "6"));
        cfg.writePriceComparison = hasFlag(argc, argv, "--fit-write-price-comparison");
        cfg.writeDebugArtifacts = hasFlag(argc, argv, "--fit-debug-artifacts");
        cfg.writeFullArtifacts = hasFlag(argc, argv, "--fit-full-artifacts");
        cfg.dearbQpDiagnostics = hasFlag(argc, argv, "--fit-dearb-qp-diagnostics");
        cfg.fitTimestamp = getArg(argc, argv, "--fit-timestamp", "");
        cfg.listTimestampsOnly = hasFlag(argc, argv, "--fit-list-timestamps");
        {
            std::string ds = getArg(argc, argv, "--fit-dearb-solver", "");
            std::transform(ds.begin(), ds.end(), ds.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (!ds.empty() && ds != "clarabel" && ds != "qpoases") {
                std::cerr << "Invalid --fit-dearb-solver=\"" << ds << "\" (expected clarabel|qpoases).\n";
                return 1;
            }
            cfg.dearbSolver = ds;
        }
        {
            std::string dl = getArg(argc, argv, "--fit-dearb-loss", "");
            std::transform(dl.begin(), dl.end(), dl.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (!dl.empty() && dl != "inverse_spread" && dl != "inverse-spread" && dl != "uniform"
                && dl != "inverse_raw_spread" && dl != "inverse-raw-spread") {
                std::cerr << "Invalid --fit-dearb-loss=\"" << dl << "\" (expected inverse_spread|inverse_raw_spread|uniform).\n";
                return 1;
            }
            cfg.dearbLoss = dl;
        }
        return ResearchBench::runFitFromBin(cfg);
    }

    // Offline: load TrDBClient-saved .bin and run CviFitBridge on the first N snapshots (no network).
    const std::string cviFromBin = getArg(argc, argv, "--cvi-from-bin", "");
    if (!cviFromBin.empty()) {
        const int firstN = std::max(1, std::stoi(getArg(argc, argv, "--first", "30")));
        const std::string outPrefix = getArg(argc, argv, "--out-prefix", "cvi_from_bin");

        std::vector<VolSurfaceSnapshot> all = loadTrdbVolSnapshotsBinRaw(cviFromBin);
        const size_t n = static_cast<size_t>(std::min<int>(firstN, static_cast<int>(all.size())));

        ResearchBench::CviFitBridge bridge;
        std::vector<ResearchBench::PerSnapshotFitResult> results;
        results.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            results.push_back(bridge.fitSnapshot(all[i]));
        }

        const std::string fitCsv = outPrefix + "_fit_results.csv";
        ResearchBench::Exporters::writeFitResultsCsv(fitCsv, results);

        std::cout << "CviFromBin: loaded " << all.size() << " snapshots, ran bridge on " << n << ".\n";
        std::cout << "  " << fitCsv << '\n';
        return 0;
    }

    const std::string ticker = getArg(argc, argv, "--ticker", "AMZN");
    const std::string startTs = getArg(argc, argv, "--start", "2026-04-09 09:30:00");
    const std::string endTs = getArg(argc, argv, "--end", "2026-04-09 10:30:00");
    const std::string outPrefix = getArg(argc, argv, "--out-prefix", "research_bench");
    const int splitMins = std::stoi(getArg(argc, argv, "--split", "60"));
    const bool dumpPayload = hasFlag(argc, argv, "--dump-payload");
    const size_t dumpPayloadMax = dumpPayload
        ? static_cast<size_t>(std::stoull(getArg(argc, argv, "--dump-payload-max", "4096")))
        : static_cast<size_t>(0);

    ResearchBench::StreamRequest request;
    request.ticker = ticker;
    request.startTimestamp = startTs;
    request.endTimestamp = endTs;
    request.splitMinutes = splitMins;
    request.searchType = SearchType::VOL_SURFACE;
    request.transformationType = TransformationType::RAW_SERIES;

    ResearchBench::ResearchBenchEngine engine;
    if (dumpPayload) {
        engine.setDumpPayload(true, dumpPayloadMax);
    }
    auto run = engine.run(request);

    const std::string diagCsv = outPrefix + "_fit_diagnostics.csv";
    const std::string fitCsv = outPrefix + "_fit_results.csv";
    const std::string summaryJson = outPrefix + "_summary.json";

    ResearchBench::Exporters::writeFitDiagnosticsCsv(diagCsv, engine.diagnostics());
    ResearchBench::Exporters::writeFitResultsCsv(fitCsv, engine.fitResults());
    ResearchBench::Exporters::writeSummaryJson(summaryJson, run);

    std::cout << "ResearchBench run: " << (run.success ? "SUCCESS" : "FAILED") << '\n';
    std::cout << "Message: " << run.message << '\n';
    std::cout << "Snapshots: " << run.streamedSnapshots << '\n';
    std::cout << "Fit results: " << run.fitResults << '\n';
    std::cout << "Diagnostics: " << run.diagnostics << '\n';
    std::cout << "Outputs:\n";
    std::cout << "  " << diagCsv << '\n';
    std::cout << "  " << fitCsv << '\n';
    std::cout << "  " << summaryJson << '\n';

    return run.success ? 0 : 1;
}
