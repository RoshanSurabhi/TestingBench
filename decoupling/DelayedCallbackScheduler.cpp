#include "DelayedCallbackScheduler.h"

#include <cctype>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

namespace ResearchBench {

namespace {

int parseTwoDigits(const std::string& s, size_t pos) {
    if (pos + 1 >= s.size() || !std::isdigit(static_cast<unsigned char>(s[pos])) ||
        !std::isdigit(static_cast<unsigned char>(s[pos + 1]))) {
        throw std::runtime_error("Invalid 2-digit field");
    }
    return (s[pos] - '0') * 10 + (s[pos + 1] - '0');
}

int64_t toUtcUnixSeconds(std::tm tm, int tzOffsetSeconds) {
    tm.tm_isdst = -1;
#ifdef _WIN32
    const std::time_t utc = _mkgmtime(&tm);
#else
    const std::time_t utc = timegm(&tm);
#endif
    if (utc == static_cast<std::time_t>(-1)) {
        throw std::runtime_error("Failed to parse timestamp");
    }
    return static_cast<int64_t>(utc) - tzOffsetSeconds;
}

} // namespace

void DelayedCallbackScheduler::registerCallback(const DelayedCallbackSpec& spec) {
    if (!spec.callback || spec.horizonSec < 0) {
        return;
    }
    CallbackState state;
    state.spec = spec;
    states_.push_back(std::move(state));
}

void DelayedCallbackScheduler::onSnapshot(const VolSurfaceSnapshot& snapshot, const StreamContext& ctx) {
    const auto evalTs = parseTimestampSeconds(snapshot.timestamp);
    if (!evalTs.has_value()) {
        return;
    }

    for (auto& state : states_) {
        while (!state.pending.empty()) {
            const auto& origin = state.pending.front();
            const int64_t triggerTs = origin.unixSeconds + state.spec.horizonSec;
            if (*evalTs < triggerTs) {
                break;
            }

            DelayedEvaluationContext delayedCtx;
            delayedCtx.originSnapshot = origin.snapshot;
            delayedCtx.evalSnapshot = snapshot;
            delayedCtx.originContext = origin.ctx;
            delayedCtx.evalContext = ctx;
            delayedCtx.horizonSec = state.spec.horizonSec;
            state.spec.callback(delayedCtx);
            state.pending.pop_front();
        }

        PendingSnapshotState pending;
        pending.snapshot = snapshot;
        pending.ctx = ctx;
        pending.unixSeconds = *evalTs;
        state.pending.push_back(std::move(pending));
    }
}

void DelayedCallbackScheduler::reset() {
    for (auto& state : states_) {
        state.pending.clear();
    }
}

std::optional<int64_t> DelayedCallbackScheduler::parseTimestampSeconds(const std::string& ts) {
    // Expected: YYYY-MM-DD HH:MM:SS optionally followed by timezone offset, e.g. -04, -0400, -04:00.
    if (ts.size() < 19) {
        return std::nullopt;
    }
    try {
        std::tm tm{};
        tm.tm_year = std::stoi(ts.substr(0, 4)) - 1900;
        tm.tm_mon = std::stoi(ts.substr(5, 2)) - 1;
        tm.tm_mday = std::stoi(ts.substr(8, 2));
        tm.tm_hour = std::stoi(ts.substr(11, 2));
        tm.tm_min = std::stoi(ts.substr(14, 2));
        tm.tm_sec = std::stoi(ts.substr(17, 2));

        int tzOffsetSeconds = 0;
        if (ts.size() > 19) {
            const size_t tzPos = 19;
            const char sign = ts[tzPos];
            if (sign == '+' || sign == '-') {
                int hh = 0;
                int mm = 0;
                if (ts.size() >= tzPos + 3) {
                    hh = parseTwoDigits(ts, tzPos + 1);
                }
                if (ts.size() >= tzPos + 6 && ts[tzPos + 3] == ':') {
                    mm = parseTwoDigits(ts, tzPos + 4);
                } else if (ts.size() >= tzPos + 5) {
                    mm = parseTwoDigits(ts, tzPos + 3);
                }
                tzOffsetSeconds = hh * 3600 + mm * 60;
                if (sign == '-') {
                    tzOffsetSeconds = -tzOffsetSeconds;
                }
            }
        }
        return toUtcUnixSeconds(tm, tzOffsetSeconds);
    } catch (...) {
        return std::nullopt;
    }
}

} // namespace ResearchBench
