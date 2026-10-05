#include "intent_engine/signal_quality.hpp"

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

double varianceValue(const std::vector<double>& data, double mean) {
    if (data.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& val : data) {
        sum += (val - mean) * (val - mean);
    }
    return sum / static_cast<double>(data.size());
}

double rmsValue(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& val : data) {
        sum += val * val;
    }
    return std::sqrt(sum / static_cast<double>(data.size()));
}

}

SignalQualityAnalyzer::SignalQualityAnalyzer(std::size_t numChannels, const SignalQualityConfig& config) {
    configure(numChannels, config);
}

void SignalQualityAnalyzer::configure(std::size_t numChannels, const SignalQualityConfig& config) {
    if (numChannels == 0) {
        throw std::invalid_argument("SignalQualityAnalyzer requires at least one channel");
    }
    if (config.sampleRateHz <= 0.0) {
        throw std::invalid_argument("SignalQualityAnalyzer requires positive sample rate");
    }

    config_ = config;
    numChannels_ = numChannels;
    signalBufferSize_ = static_cast<std::size_t>(config_.sampleRateHz * config_.noiseFloorEstimateSeconds);
    noiseBufferSize_ = static_cast<std::size_t>(config_.sampleRateHz * config_.noiseFloorEstimateSeconds);
    baselineBufferSize_ = static_cast<std::size_t>(config_.sampleRateHz * config_.baselineWindowSeconds);

    signalBuffer_.assign(numChannels_, std::vector<double>());
    noiseBuffer_.assign(numChannels_, std::vector<double>());
    baselineBuffer_.assign(numChannels_, std::vector<double>());

    metrics_.channelQuality.assign(numChannels_, 1.0);
    configured_ = true;
}

void SignalQualityAnalyzer::reset() {
    for (auto& buf : signalBuffer_) buf.clear();
    for (auto& buf : noiseBuffer_) buf.clear();
    for (auto& buf : baselineBuffer_) buf.clear();
    metrics_ = SignalQualityMetrics();
    metrics_.channelQuality.assign(numChannels_, 1.0);
}

void SignalQualityAnalyzer::update(const std::vector<double>& channelValues) {
    if (!configured_) throw std::runtime_error("SignalQualityAnalyzer not configured");
    if (channelValues.size() != numChannels_) throw std::invalid_argument("channel count mismatch");

    for (std::size_t ch = 0; ch < numChannels_; ++ch) {
        signalBuffer_[ch].push_back(channelValues[ch]);
        noiseBuffer_[ch].push_back(channelValues[ch]);
        baselineBuffer_[ch].push_back(channelValues[ch]);
        if (signalBuffer_[ch].size() > signalBufferSize_) signalBuffer_[ch].erase(signalBuffer_[ch].begin());
        if (noiseBuffer_[ch].size() > noiseBufferSize_) noiseBuffer_[ch].erase(noiseBuffer_[ch].begin());
        if (baselineBuffer_[ch].size() > baselineBufferSize_) baselineBuffer_[ch].erase(baselineBuffer_[ch].begin());
    }
    computeMetrics();
}

void SignalQualityAnalyzer::update(const double* channelValues, std::size_t numChannels) {
    std::vector<double> vec(channelValues, channelValues + numChannels);
    update(vec);
}

void SignalQualityAnalyzer::computeMetrics() {
    metrics_.channelQuality.assign(numChannels_, 1.0);
    metrics_.isValid = true;
    metrics_.rootMeanSquare = 0.0;
    metrics_.variance = 0.0;
    metrics_.meanAmplitude = 0.0;

    for (std::size_t ch = 0; ch < numChannels_; ++ch) {
        if (signalBuffer_[ch].empty()) continue;
        double mean = meanValue(signalBuffer_[ch]);
        double var = varianceValue(signalBuffer_[ch], mean);
        double rms = rmsValue(signalBuffer_[ch]);
        metrics_.meanAmplitude += mean;
        metrics_.variance += var;
        metrics_.rootMeanSquare += rms;
        metrics_.channelQuality[ch] = 1.0 - std::min(1.0, std::sqrt(var) / (std::abs(mean) + 1e-10));
    }

    metrics_.meanAmplitude /= static_cast<double>(numChannels_);
    metrics_.variance /= static_cast<double>(numChannels_);
    metrics_.rootMeanSquare /= static_cast<double>(numChannels_);
    metrics_.dynamicRange = metrics_.rootMeanSquare;
    metrics_.crestFactor = computeCrestFactor(signalBuffer_.empty() ? std::vector<double>() : signalBuffer_[0]);
    metrics_.signalToNoiseRatio = computeSnr(signalBuffer_[0], noiseBuffer_[0]);
    if (metrics_.signalToNoiseRatio > 0.0) {
        metrics_.signalToNoiseRatioDb = 10.0 * std::log10(metrics_.signalToNoiseRatio);
    } else {
        metrics_.signalToNoiseRatioDb = -std::numeric_limits<double>::infinity();
    }
    metrics_.baselineStability = computeBaselineStability(baselineBuffer_[0]);
    metrics_.motionArtifactRatio = computeMotionArtifact(signalBuffer_[0]);
    metrics_.powerlineInterferenceRatio = computePowerlineInterference(signalBuffer_[0]);
    metrics_.electrodeContactQuality = 1.0 - metrics_.motionArtifactRatio;
    metrics_.isValid = metrics_.signalToNoiseRatioDb >= config_.snrMinimumDb;
}

const SignalQualityMetrics& SignalQualityAnalyzer::evaluate() const {
    return metrics_;
}

double SignalQualityAnalyzer::computeSnr(const std::vector<double>& signal, const std::vector<double>& noise) const {
    if (signal.empty() || noise.empty()) return 0.0;
    double sigPower = varianceValue(signal, meanValue(signal)) + 1e-12;
    double noisePower = varianceValue(noise, meanValue(noise)) + 1e-12;
    return sigPower / noisePower;
}

double SignalQualityAnalyzer::computePowerlineInterference(const std::vector<double>& data) const {
    if (!config_.enablePowerlineDetection || data.empty()) return 0.0;
    return std::min(1.0, std::abs(meanValue(data)) * 0.1);
}

double SignalQualityAnalyzer::computeMotionArtifact(const std::vector<double>& data) const {
    if (!config_.enableMotionArtifactDetection || data.empty()) return 0.0;
    double var = varianceValue(data, meanValue(data));
    return std::min(1.0, var * 10.0);
}

double SignalQualityAnalyzer::computeBaselineStability(const std::vector<double>& baselineData) const {
    if (baselineData.empty()) return 1.0;
    double var = varianceValue(baselineData, meanValue(baselineData));
    return std::max(0.0, 1.0 - var);
}

double SignalQualityAnalyzer::computeCrestFactor(const std::vector<double>& data) const {
    if (data.empty()) return 0.0;
    double mean = meanValue(data);
    double maxVal = *std::max_element(data.begin(), data.end());
    double minVal = *std::min_element(data.begin(), data.end());
    double peak = std::max(std::abs(maxVal - mean), std::abs(minVal - mean));
    double rms = rmsValue(data);
    if (rms < 1e-12) return 0.0;
    return peak / rms;
}

SignalQualityMetrics SignalQualityAnalyzer::evaluateWindow(const std::vector<std::vector<double>>& windowData) const {
    SignalQualityMetrics m;
    if (windowData.empty()) return m;
    std::size_t chCount = windowData.size();
    m.channelQuality.assign(chCount, 1.0);
    m.isValid = true;
    for (std::size_t ch = 0; ch < chCount; ++ch) {
        if (windowData[ch].empty()) continue;
        double mean = meanValue(windowData[ch]);
        double var = varianceValue(windowData[ch], mean);
        double rms = rmsValue(windowData[ch]);
        m.meanAmplitude += mean;
        m.variance += var;
        m.rootMeanSquare += rms;
        m.channelQuality[ch] = 1.0;
    }
    m.meanAmplitude /= static_cast<double>(chCount);
    m.variance /= static_cast<double>(chCount);
    m.rootMeanSquare /= static_cast<double>(chCount);
    return m;
}

bool SignalQualityAnalyzer::isSignalUsable(const SignalQualityMetrics& metrics) const {
    return metrics.isValid && metrics.electrodeContactQuality >= config_.contactQualityThreshold;
}

bool SignalQualityAnalyzer::isConfigured() const {
    return configured_;
}

}
