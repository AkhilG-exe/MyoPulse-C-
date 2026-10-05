#include "intent_engine/electrode_shift_compensation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace intent_engine {

namespace {
double meanValue(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / static_cast<double>(data.size());
}
}

ElectrodeShiftCompensator::ElectrodeShiftCompensator(std::size_t numChannels, const ElectrodeShiftConfig& config) {
    configure(numChannels, config);
}

void ElectrodeShiftCompensator::configure(std::size_t numChannels, const ElectrodeShiftConfig& config) {
    if (numChannels == 0) throw std::invalid_argument("requires channels");
    config_ = config;
    numChannels_ = numChannels;
    bufferSize_ = static_cast<std::size_t>(config_.sampleRateHz * config_.windowSeconds);
    baselineMeans_.assign(numChannels_, 0.0);
    currentMeans_.assign(numChannels_, 0.0);
    shiftEstimates_.assign(numChannels_, 0.0);
    buffer_.assign(numChannels_, std::vector<double>());
    metrics_.channelShifts.assign(numChannels_, 0.0);
    configured_ = true;
}

void ElectrodeShiftCompensator::reset() {
    for (auto& b : buffer_) b.clear();
    baselineEstablished_ = false;
    std::fill(shiftEstimates_.begin(), shiftEstimates_.end(), 0.0);
}

void ElectrodeShiftCompensator::update(const std::vector<double>& channelValues) {
    if (!configured_) throw std::runtime_error("not configured");
    if (channelValues.size() != numChannels_) throw std::invalid_argument("mismatch");
    for (std::size_t ch = 0; ch < numChannels_; ++ch) {
        buffer_[ch].push_back(channelValues[ch]);
        if (buffer_[ch].size() > bufferSize_) buffer_[ch].erase(buffer_[ch].begin());
        currentMeans_[ch] = meanValue(buffer_[ch]);
        if (!baselineEstablished_) baselineMeans_[ch] = currentMeans_[ch];
        shiftEstimates_[ch] = std::abs(currentMeans_[ch] - baselineMeans_[ch]) / (std::abs(baselineMeans_[ch]) + 1e-12);
        if (baselineEstablished_ && config_.enableAutoCompensation) {
            baselineMeans_[ch] = baselineMeans_[ch] * (1.0 - config_.adaptationRate) + currentMeans_[ch] * config_.adaptationRate;
        }
    }
    baselineEstablished_ = true;
    double maxShift = 0.0;
    for (auto s : shiftEstimates_) if (s > maxShift) maxShift = s;
    metrics_.shiftMagnitude = maxShift;
    metrics_.shiftDetected = maxShift > config_.detectionThreshold;
    metrics_.compensationActive = config_.enableAutoCompensation;
    metrics_.channelShifts = shiftEstimates_;
}

std::vector<double> ElectrodeShiftCompensator::compensate(const std::vector<double>& channelValues) const {
    std::vector<double> comp = channelValues;
    if (!config_.enableAutoCompensation) return comp;
    for (std::size_t ch = 0; ch < comp.size() && ch < numChannels_; ++ch) {
        double correction = (currentMeans_[ch] - baselineMeans_[ch]) * config_.compensationFactor;
        comp[ch] -= correction;
    }
    return comp;
}

ElectrodeShiftMetrics ElectrodeShiftCompensator::evaluate() const { return metrics_; }
bool ElectrodeShiftCompensator::isShiftDetected() const { return metrics_.shiftDetected; }
bool ElectrodeShiftCompensator::isConfigured() const { return configured_; }

}
