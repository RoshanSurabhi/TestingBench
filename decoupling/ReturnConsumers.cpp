#include "ReturnConsumers.h"

#include <cmath>
#include <limits>
#include <unordered_map>

namespace ResearchBench {

namespace {

double midpointSpot(const VolSurfaceSnapshot& snapshot) {
    if (snapshot.instrument.mark > 0.0) {
        return snapshot.instrument.mark;
    }
    if (snapshot.instrument.bid > 0.0 && snapshot.instrument.ask > 0.0) {
        return 0.5 * (snapshot.instrument.bid + snapshot.instrument.ask);
    }
    if (snapshot.instrument.last > 0.0) {
        return snapshot.instrument.last;
    }
    return 0.0;
}

} // namespace

DelayedCallback ReturnConsumers::makeLogReturnCallback(std::vector<ReturnRecord>* output) {
    return [output](const DelayedEvaluationContext& ctx) {
        if (!output) {
            return;
        }

        std::unordered_map<std::string, const ExpirSnapshot*> evalByExpiry;
        evalByExpiry.reserve(ctx.evalSnapshot.expirs.size());
        for (const auto& e : ctx.evalSnapshot.expirs) {
            evalByExpiry[e.expirDate] = &e;
        }

        const double spotT = midpointSpot(ctx.originSnapshot);
        const double spotTPlusH = midpointSpot(ctx.evalSnapshot);
        const bool spotValid = (spotT > 0.0 && spotTPlusH > 0.0);
        const double spotRet = spotValid ? std::log(spotTPlusH / spotT) : 0.0;

        for (const auto& originExp : ctx.originSnapshot.expirs) {
            auto it = evalByExpiry.find(originExp.expirDate);
            if (it == evalByExpiry.end()) {
                continue;
            }
            const ExpirSnapshot& evalExp = *it->second;

            ReturnRecord rec;
            rec.symbol = ctx.evalSnapshot.instrument.symbol;
            rec.originTimestamp = ctx.originSnapshot.timestamp;
            rec.evalTimestamp = ctx.evalSnapshot.timestamp;
            rec.horizonSec = ctx.horizonSec;
            rec.expiry = originExp.expirDate;
            rec.forwardT = originExp.evalForward;
            rec.forwardTPlusH = evalExp.evalForward;
            rec.spotT = spotT;
            rec.spotTPlusH = spotTPlusH;

            const bool fwdValid = (rec.forwardT > 0.0 && rec.forwardTPlusH > 0.0);
            rec.forwardReturnLog = fwdValid ? std::log(rec.forwardTPlusH / rec.forwardT) : 0.0;
            rec.spotReturnLog = spotRet;
            rec.valid = fwdValid && spotValid;

            output->push_back(std::move(rec));
        }
    };
}

} // namespace ResearchBench
