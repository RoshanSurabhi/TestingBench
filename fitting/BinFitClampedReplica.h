#pragma once

#include <filesystem>
#include <string>

namespace ResearchBench {

/** Run CVITestSurfaceFitterClampedMask.exe (exact test_clamped artifact pipeline). */
bool runClampedMaskReplica(
    const std::filesystem::path& clampedExe,
    const std::string& binPath,
    int snapIndexInBin,
    const std::filesystem::path& artifactDir,
    unsigned int numBasis,
    double lambda,
    int arbPoints,
    double z0,
    double zn1,
    const std::string& dearbSolver,
    const std::string& dearbLoss,
    int& clarabelStatus,
    double& clarabelObj,
    std::string& message);

std::filesystem::path findClampedMaskTestExe(const std::filesystem::path& researchBenchExeDir);

} // namespace ResearchBench
