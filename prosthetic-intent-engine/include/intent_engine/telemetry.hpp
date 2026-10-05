#pragma once

#include <cstddef>
#include <vector>
#include <string>

namespace intent_engine {

struct PerformanceMetrics {
    double latencyMs = 0.0;
    double throughputSamplesPerSec = 0.0;
    double cpuUsagePercent = 0.0;
    double memoryUsageBytes = 0.0;
    double jitterMs = 0.0;
    double processingTimeMs = 0.0;
    double uptimeSeconds = 0.0;
    std::size_t totalSamples = 0;
    std::size_t droppedSamples = 0;
};

struct SystemHealth {
    bool isHealthy = true;
    double uptimeSeconds = 0.0;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};

struct TelemetryConfig {
    double samplingIntervalMs = 100.0;
    bool enableLatencyTracking = true;
    bool enableThroughputTracking = true;
    bool enableHealthMonitoring = true;
    bool enableProfiling = true;
    std::size_t historySize = 1000;
};

class TelemetrySystem {
public:
    TelemetrySystem();
    explicit TelemetrySystem(const TelemetryConfig& config);

    void configure(const TelemetryConfig& config);
    void reset();
    void recordSampleProcessed(double processingTimeMs);
    void recordDroppedSample();
    void update();
    PerformanceMetrics getPerformanceMetrics() const;
    SystemHealth getSystemHealth() const;
    void addWarning(const std::string& warning);
    void addError(const std::string& error);
    void clearWarnings();
    void clearErrors();
    bool isConfigured() const;

private:
    TelemetryConfig config_;
    PerformanceMetrics metrics_;
    SystemHealth health_;
    std::vector<double> processingTimes_;
    double startTime_ = 0.0;
    double lastUpdateTime_ = 0.0;
    std::size_t samplesProcessed_ = 0;
    bool configured_ = false;
};

}
