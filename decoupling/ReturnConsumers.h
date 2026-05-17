#pragma once

#include <vector>

#include "../ResearchTypes.h"

namespace ResearchBench {

class ReturnConsumers {
public:
    static DelayedCallback makeLogReturnCallback(std::vector<ReturnRecord>* output);
};

} // namespace ResearchBench
