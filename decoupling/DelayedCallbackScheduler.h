#pragma once

#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../ResearchTypes.h"

namespace ResearchBench {

class DelayedCallbackScheduler {
public:
    void registerCallback(const DelayedCallbackSpec& spec);
    void onSnapshot(const VolSurfaceSnapshot& snapshot, const StreamContext& ctx);
    void reset();

private:
    struct PendingSnapshotState {
        VolSurfaceSnapshot snapshot;
        StreamContext ctx;
        int64_t unixSeconds = 0;
    };

    struct CallbackState {
        DelayedCallbackSpec spec;
        std::deque<PendingSnapshotState> pending;
    };

    static std::optional<int64_t> parseTimestampSeconds(const std::string& ts);

    std::vector<CallbackState> states_;
};

} // namespace ResearchBench
