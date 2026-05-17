#include "Tests.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "decoupling/DelayedCallbackScheduler.h"
#include "decoupling/ReturnConsumers.h"

namespace ResearchBench {

namespace {

VolSurfaceSnapshot makeSnapshot(
    const std::string& ts,
    double spot,
    const std::string& expiry,
    double forward
) {
    VolSurfaceSnapshot s;
    s.timestamp = ts;
    s.instrument.symbol = "SPY";
    s.instrument.mark = spot;

    ExpirSnapshot e;
    e.expirDate = expiry;
    e.evalForward = forward;
    e.forward = forward;
    s.expirs.push_back(e);
    return s;
}

bool approx(double a, double b, double eps = 1e-10) {
    return std::fabs(a - b) <= eps;
}

bool testDelayedCallbacks() {
    DelayedCallbackScheduler sched;
    std::vector<ReturnRecord> out;

    DelayedCallbackSpec spec;
    spec.name = "ret";
    spec.horizonSec = 5;
    spec.callback = ReturnConsumers::makeLogReturnCallback(&out);
    sched.registerCallback(spec);

    std::vector<VolSurfaceSnapshot> snaps{
        makeSnapshot("2026-04-08 09:30:00-04", 100.0, "2026-06-20", 101.0),
        makeSnapshot("2026-04-08 09:30:03-04", 101.0, "2026-06-20", 102.0),
        makeSnapshot("2026-04-08 09:30:05-04", 102.0, "2026-06-20", 103.0),
    };

    for (size_t i = 0; i < snaps.size(); ++i) {
        StreamContext ctx;
        ctx.sequenceIndex = i;
        ctx.totalSnapshots = snaps.size();
        ctx.timestamp = snaps[i].timestamp;
        ctx.symbol = "SPY";
        sched.onSnapshot(snaps[i], ctx);
    }

    // t=09:30:00 should trigger at t=09:30:05. t=09:30:03 does not trigger yet.
    if (out.size() != 1) {
        return false;
    }
    const auto& row = out.front();
    return row.originTimestamp == "2026-04-08 09:30:00-04" &&
           row.evalTimestamp == "2026-04-08 09:30:05-04" &&
           approx(row.forwardReturnLog, std::log(103.0 / 101.0)) &&
           approx(row.spotReturnLog, std::log(102.0 / 100.0));
}

bool testLogReturnMath() {
    std::vector<ReturnRecord> out;
    auto cb = ReturnConsumers::makeLogReturnCallback(&out);

    DelayedEvaluationContext ctx;
    ctx.horizonSec = 5;
    ctx.originSnapshot = makeSnapshot("2026-04-08 09:30:00-04", 200.0, "2026-06-20", 210.0);
    ctx.evalSnapshot = makeSnapshot("2026-04-08 09:30:07-04", 198.0, "2026-06-20", 207.0);
    cb(ctx);

    if (out.size() != 1) {
        return false;
    }
    const auto& row = out.front();
    return approx(row.forwardReturnLog, std::log(207.0 / 210.0)) &&
           approx(row.spotReturnLog, std::log(198.0 / 200.0));
}

} // namespace

bool runTests() {
    const bool t1 = testDelayedCallbacks();
    const bool t2 = testLogReturnMath();
    std::cout << "[TEST] delayed callback trigger: " << (t1 ? "PASS" : "FAIL") << '\n';
    std::cout << "[TEST] log return math: " << (t2 ? "PASS" : "FAIL") << '\n';
    return t1 && t2;
}

} // namespace ResearchBench
