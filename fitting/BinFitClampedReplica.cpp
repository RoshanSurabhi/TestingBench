#include "BinFitClampedReplica.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace fs = std::filesystem;

namespace {

std::wstring wideFromUtf8(const std::string& s) {
    if (s.empty()) {
        return {};
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) {
        return {};
    }
    std::wstring out(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

std::wstring quoteWinArg(const std::wstring& arg) {
    if (arg.empty()) {
        return L"\"\"";
    }
    if (arg.find_first_of(L" \t\"") == std::wstring::npos) {
        return arg;
    }
    std::wstring out;
    out.push_back(L'"');
    for (wchar_t c : arg) {
        if (c == L'"') {
            out += L"\\\"";
        } else {
            out.push_back(c);
        }
    }
    out.push_back(L'"');
    return out;
}

bool parseSummaryFile(
    const fs::path& path,
    int& clarabelStatus,
    double& clarabelObj,
    bool& fitOk,
    std::string& message) {
    std::ifstream in(path);
    if (!in) {
        message = "clamped replica finished but clamped_run_summary.txt is missing";
        return false;
    }
    fitOk = false;
    clarabelStatus = -1;
    clarabelObj = 0.0;
    std::string line;
    while (std::getline(in, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, eq);
        const std::string val = line.substr(eq + 1);
        if (key == "fit_ok") {
            fitOk = (std::stoi(val) != 0);
        } else if (key == "clarabel_status") {
            clarabelStatus = std::stoi(val);
        } else if (key == "clarabel_objective") {
            clarabelObj = std::stod(val);
        } else if (key == "pass1_status" && clarabelStatus < 0) {
            clarabelStatus = std::stoi(val);
        }
    }
    message = fitOk ? std::string("ok (clamped replica)") : std::string("clamped replica fit_ok=0");
    return true;
}

} // namespace

namespace ResearchBench {

fs::path findClampedMaskTestExe(const fs::path& researchBenchExeDir) {
    const wchar_t* kNames[] = {
        L"CVITestSurfaceFitterClampedMask.exe",
        L"testclamped_surface_fitter.exe",
    };
    for (const wchar_t* name : kNames) {
        const fs::path p = researchBenchExeDir / name;
        if (fs::is_regular_file(p)) {
            return p;
        }
    }
    const fs::path candidates[] = {
        researchBenchExeDir / ".." / ".." / "FinMath-Lib" / "CVI" / "CVITestSuite" / "x64" / "Release"
            / "CVITestSurfaceFitterClampedMask.exe",
        researchBenchExeDir / ".." / ".." / "FinMath-Lib" / "CVI" / "x64" / "Release" / "CVITestSurfaceFitterClampedMask.exe",
    };
    for (const fs::path& p : candidates) {
        if (fs::is_regular_file(p)) {
            return fs::weakly_canonical(p);
        }
    }
    return {};
}

bool runClampedMaskReplica(
    const fs::path& clampedExe,
    const std::string& binPath,
    int snapIndexInBin,
    const fs::path& artifactDir,
    unsigned int numBasis,
    double lambda,
    int arbPoints,
    double z0,
    double zn1,
    const std::string& dearbSolver,
    const std::string& dearbLoss,
    int& clarabelStatus,
    double& clarabelObj,
    std::string& message) {
#ifdef _WIN32
    if (clampedExe.empty() || !fs::is_regular_file(clampedExe)) {
        message = "CVITestSurfaceFitterClampedMask.exe not found (build FinMath-Lib CVI.sln Release|x64)";
        return false;
    }
    fs::create_directories(artifactDir);

    std::ostringstream oss;
    oss << numBasis << ' ' << lambda << ' ' << arbPoints << ' ' << z0 << ' ' << zn1 << " researchbench_run";
    std::wstring cmd = quoteWinArg(wideFromUtf8(clampedExe.string()))
        + L' ' + wideFromUtf8(oss.str()) + L' ' + quoteWinArg(wideFromUtf8(fs::absolute(binPath).string()))
        + L" --trdb-snap-index " + wideFromUtf8(std::to_string(snapIndexInBin)) + L" keep_all_outside";
    if (!dearbSolver.empty()) {
        cmd += L" --dearb-solver " + wideFromUtf8(dearbSolver);
    }
    if (!dearbLoss.empty()) {
        cmd += L" --dearb-loss " + wideFromUtf8(dearbLoss);
    }
    cmd += L" --artifact-dir " + quoteWinArg(wideFromUtf8(fs::absolute(artifactDir).string()));

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::wstring cmdMutable = cmd;
    if (!CreateProcessW(
            nullptr,
            cmdMutable.data(),
            nullptr,
            nullptr,
            FALSE,
            CREATE_NO_WINDOW,
            nullptr,
            nullptr,
            &si,
            &pi)) {
        message = "CreateProcess failed for clamped replica (err=" + std::to_string(GetLastError()) + ")";
        return false;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode = 1;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    bool fitOk = false;
    if (!parseSummaryFile(artifactDir / "clamped_run_summary.txt", clarabelStatus, clarabelObj, fitOk, message)) {
        message += " (exit=" + std::to_string(exitCode) + ")";
        return false;
    }
    if (!fitOk) {
        message += " (exit=" + std::to_string(exitCode) + ")";
    }
    return fitOk;
#else
    (void)clampedExe;
    (void)binPath;
    (void)snapIndexInBin;
    (void)artifactDir;
    (void)numBasis;
    (void)lambda;
    (void)arbPoints;
    (void)z0;
    (void)zn1;
    (void)clarabelStatus;
    (void)clarabelObj;
    message = "clamped replica runner is only implemented on Windows";
    return false;
#endif
}

} // namespace ResearchBench
