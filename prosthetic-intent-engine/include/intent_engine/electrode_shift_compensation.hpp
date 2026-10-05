#pragma once

#include <cstddef>
#include <vector>

namespace intent_engine {

struct ElectrodeShiftMetrics {
    double shiftMagnitude = 0.0;
    double driftRate = 0.0;
    bool shiftDetected = false;
    bool compensationActive = false;
    std::vector<double> channelShifts;
};

struct ElectrodeShiftConfig {
    double detectionThreshold = 0.15;
    double compensationFactor = 0.8;
    double adaptationRate = 0.01;
    double windowSeconds = 5.0;
    double sampleRateHz = 1000.0;
    bool enableAutoCompensation = true;
};

class ElectrodeShiftCompensator {
public:
    ElectrodeShiftCompensator() = default;
    ElectrodeShiftCompensator(std::size_t numChannels, const ElectrodeShiftConfig& config = ElectrodeShiftConfig{});

    void configure(std::size_t numChannels, const ElectrodeShiftConfig& config);
    void reset();
    void update(const std::vector<double>& channelValues);
    std::vector<double> compensate(const std::vector<double>& channelValues) const;
    ElectrodeShiftMetrics evaluate() const;
    bool isShiftDetected() const;
    bool isConfigured() const;

private:
    std::vector<double> baselineMeans_;
    std::vector<double> currentMeans_;
    std::vector<double> shiftEstimates_;
    ElectrodeShiftMetrics metrics_;
    ElectrodeShiftConfig config_;
    std::size_t numChannels_ = 0;
    std::size_t bufferSize_ = 0;
    std::vector<std::vector<double>> buffer_;
    bool configured_ = false;
    bool baselineEstablished_ = false;
};

}
