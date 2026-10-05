#include "intent_engine/telemetry.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <vector>
#include <numeric>

namespace intent_engine {

TelemetrySystem::TelemetrySystem() {
    configure(TelemetryConfig{});
}

TelemetrySystem::TelemetrySystem(const TelemetryConfig& config) {
    configure(config);
}

void TelemetrySystem::configure(const TelemetryConfig& config) {
    config_ = config;
    metrics_ = PerformanceMetrics();
    health_ = SystemHealth();
    health_.isHealthy = true;
    processingTimes_.clear();
    samplesProcessed_ = 0;
    using namespace std::chrono;
    startTime_ = static_cast<double>(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count()) / 1000.0;
    lastUpdateTime_ = startTime_;
    configured_ = true;
}

void TelemetrySystem::reset() {
    configure(config_);
}

void TelemetrySystem::recordSampleProcessed(double processingTimeMs) {
    samplesProcessed_ += 1;
    processingTimes_.push_back(processingTimeMs);
    if (processingTimes_.size() > config_.historySize) processingTimes_.erase(processingTimes_.begin());
    metrics_.processingTimeMs = processingTimeMs;
    metrics_.totalSamples = samplesProcessed_;
}

void TelemetrySystem::recordDroppedSample() {
    metrics_.droppedSamples += 1;
}

void TelemetrySystem::update() {
    using namespace std::chrono;
    double now = static_cast<double>(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count()) / 1000.0;
    metrics_.uptimeSeconds = now - startTime_;
    metrics_.latencyMs = metrics_.processingTimeMs;
    metrics_.throughputSamplesPerSec = samplesProcessed_ / (metrics_.uptimeSeconds + 1e-12);
    if (!processingTimes_.empty()) {
        double sum = std::accumulate(processingTimes_.begin(), processingTimes_.end(), 0.0);
        double mean = sum / static_cast<double>(processingTimes_.size());
        double var = 0.0;
        for (auto t : processingTimes_) var += (t - mean) * (t - mean);
        var /= static_cast<double>(processingTimes_.size());
        metrics_.jitterMs = std::sqrt(var);
    }
    health_.isHealthy = health_.errors.empty();
    health_.uptimeSeconds = metrics_.uptimeSeconds;
}

PerformanceMetrics TelemetrySystem::getPerformanceMetrics() const { return metrics_; }
SystemHealth TelemetrySystem::getSystemHealth() const { return health_; }
void TelemetrySystem::addWarning(const std::string& warning) { health_.warnings.push_back(warning); }
void TelemetrySystem::addError(const std::string& error) { health_.errors.push_back(error); }
void TelemetrySystem::clearWarnings() { health_.warnings.clear(); }
void TelemetrySystem::clearErrors() { health_.errors.clear(); }
bool TelemetrySystem::isConfigured() const { return configured_; }

}
