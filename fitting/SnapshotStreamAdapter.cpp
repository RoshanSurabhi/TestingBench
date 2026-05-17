#include "SnapshotStreamAdapter.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

#include "client.h"

namespace ResearchBench {

SnapshotStreamAdapter::SnapshotStreamAdapter(const std::string& configFile) {
    stream_ = new VolSnapshotStream(configFile);
}

SnapshotStreamAdapter::~SnapshotStreamAdapter() {
    delete stream_;
}

void SnapshotStreamAdapter::connect() {
    if (!stream_) {
        throw std::runtime_error("SnapshotStreamAdapter: stream is null");
    }
    stream_->connect();
}

void SnapshotStreamAdapter::disconnect() {
    if (stream_) {
        stream_->disconnect();
    }
}

void SnapshotStreamAdapter::setDumpPayload(bool enabled, size_t maxBytes) {
    if (stream_) {
        stream_->setDumpPayload(enabled, maxBytes);
    }
}

bool SnapshotStreamAdapter::isConnected() const {
    return stream_ && stream_->isConnected();
}

std::vector<VolSurfaceSnapshot> SnapshotStreamAdapter::fetchSnapshots(const StreamRequest& request) {
    if (!stream_) {
        throw std::runtime_error("SnapshotStreamAdapter: stream is null");
    }

    std::vector<VolSurfaceSnapshot> snapshots;
    if (!request.startTimestamp.empty() && !request.endTimestamp.empty()) {
        snapshots = stream_->fetchVolSnapshotsData(
            request.ticker,
            request.startTimestamp,
            request.endTimestamp,
            request.splitMinutes,
            false,
            request.searchType,
            request.transformationType
        );
    } else if (!request.startTimestamp.empty()) {
        snapshots = stream_->fetchVolSnapshotsData(
            request.ticker,
            request.startTimestamp,
            request.splitMinutes,
            false,
            request.searchType,
            request.transformationType
        );
    } else if (!request.endTimestamp.empty()) {
        snapshots = stream_->fetchVolSnapshotsDataUpTo(
            request.ticker,
            request.endTimestamp,
            request.splitMinutes,
            false,
            request.searchType,
            request.transformationType
        );
    } else {
        snapshots = stream_->fetchVolSnapshotsData(
            request.ticker,
            request.splitMinutes,
            false,
            request.searchType,
            request.transformationType
        );
    }

    std::sort(
        snapshots.begin(),
        snapshots.end(),
        [](const VolSurfaceSnapshot& a, const VolSurfaceSnapshot& b) {
            return a.timestamp < b.timestamp;
        }
    );
    return snapshots;
}

void SnapshotStreamAdapter::stream(
    const StreamRequest& request,
    const std::function<void()>& onStreamStart,
    const SnapshotCallback& onSnapshot,
    const std::function<void()>& onStreamEnd,
    const std::function<void(const std::string&)>& onError
) {
    try {
        connect();
        if (onStreamStart) {
            onStreamStart();
        }

        auto snapshots = fetchSnapshots(request);
        for (size_t i = 0; i < snapshots.size(); ++i) {
            StreamContext ctx;
            ctx.sequenceIndex = i;
            ctx.totalSnapshots = snapshots.size();
            ctx.timestamp = snapshots[i].timestamp;
            ctx.symbol = snapshots[i].instrument.symbol;

            if (onSnapshot) {
                onSnapshot(snapshots[i], ctx);
            }
        }

        if (onStreamEnd) {
            onStreamEnd();
        }
        disconnect();
    } catch (const std::exception& ex) {
        disconnect();
        if (onError) {
            onError(ex.what());
        }
    } catch (...) {
        disconnect();
        if (onError) {
            onError("Unknown streaming error");
        }
    }
}

} // namespace ResearchBench
