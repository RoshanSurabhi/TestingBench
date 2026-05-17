#pragma once

#include <functional>
#include <string>
#include <vector>

#include "../ResearchTypes.h"

class VolSnapshotStream;

namespace ResearchBench {

class SnapshotStreamAdapter {
public:
    explicit SnapshotStreamAdapter(const std::string& configFile = "dbclient_start_config.xml");
    ~SnapshotStreamAdapter();

    void connect();
    void disconnect();
    bool isConnected() const;

    std::vector<VolSurfaceSnapshot> fetchSnapshots(const StreamRequest& request);
    void stream(
        const StreamRequest& request,
        const std::function<void()>& onStreamStart,
        const SnapshotCallback& onSnapshot,
        const std::function<void()>& onStreamEnd,
        const std::function<void(const std::string&)>& onError
    );

    void setDumpPayload(bool enabled, size_t maxBytes);

private:
    VolSnapshotStream* stream_ = nullptr;
};

} // namespace ResearchBench
