#include "intent_engine/fatigue_adaptation.hpp"

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

double rmsValue(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    double sum = 0.0;
    for (auto v : data) sum += v * v;
    return std::sqrt(sum / static_cast<double>(data.size()));
}

double medianFrequency(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    std::vector<double> tmp = data;
    std::sort(tmp.begin(), tmp.end());
    return tmp[tmp.size() / 2];
}

}

FatigueAdaptationEngine::FatigueAdaptationEngine(std::size_t numChannels, const FatigueAdaptationConfig& config) {
    configure(numChannels, config);
}

void FatigueAdaptationEngine::configure(std::size_t numChannels, const FatigueAdaptationConfig& config) {
    if (numChannels == 0) throw std::invalid_argument("requires channels");
    config_ = config;
    numChannels_ = numChannels;
    historySize_ = static_cast<std::size_t>(config_.sampleRateHz * 10.0);
    signalHistory_.assign(numChannels_, std::deque<double>());
    baselineMedianFreq_.assign(numChannels_, 100.0);
    baselineRms_.assign(numChannels_, 0.0);
    currentMedianFreq_.assign(numChannels_, 100.0);
    currentRms_.assign(numChannels_, 0.0);
    channelFatigue_.assign(numChannels_, 0.0);
    metrics_.channelFatigue.assign(numChannels_, 0.0);
    configured_ = true;
}

void FatigueAdaptationEngine::reset() {
    for (auto& d : signalHistory_) d.clear();
    std::fill(channelFatigue_.begin(), channelFatigue_.end(), 0.0);
    fatigueLevel_ = 0.0;
    baselineEstablished_ = false;
    recoveryMode_ = false;
}

void FatigueAdaptationEngine::update(const std::vector<double>& channelValues) {
    if (!configured_) throw std::runtime_error("not configured");
    if (channelValues.size() != numChannels_) throw std::invalid_argument("mismatch");
    for (std::size_t ch = 0; ch < numChannels_; ++ch) {
        signalHistory_[ch].push_back(channelValues[ch]);
        if (signalHistory_[ch].size() > historySize_) signalHistory_[ch].pop_front();
        std::vector<double> hist(signalHistory_[ch].begin(), signalHistory_[ch].end());
        currentRms_[ch] = rmsValue(hist);
        if (baselineEstablished_) {
            double rmsInc = currentRms_[ch] / (baselineRms_[ch] + 1e-12);
            double freqShift = (baselineMedianFreq_[ch] - medianFrequency(hist));
            double fatigue = std::min(1.0, std::max(0.0, rmsInc - 1.0 + freqShift * 0.001));
            channelFatigue_[ch] = channelFatigue_[ch] * config_.forgettingFactor + fatigue * (1.0 - config_.forgettingFactor);
        }
    }
    computeFatigueIndex();
}

void FatigueAdaptationEngine::computeFatigueIndex() {
    fatigueLevel_ = 0.0;
    if (!baselineEstablished_) fatigueLevel_ = 0.0;
    else {
        for (auto f : channelFatigue_) fatigueLevel_ += f;
        fatigueLevel_ /= static_cast<double>(numChannels_);
    }
    metrics_.fatigueIndex = fatigueLevel_;
    metrics_.isFatigued = fatigueLevel_ > config_.fatigueThreshold;
    metrics_.enduranceFactor = std::max(0.0, 1.0 - fatigueLevel_);
    metrics_.recoveryFactor = recoveryMode_ ? 0.5 : 0.0;
    metrics_.channelFatigue = channelFatigue_;
    metrics_.baselineRms = baselineRms_;
    metrics_.currentRms = currentRms_;
}

double FatigueAdaptationEngine::getAdaptedThreshold(double baseThreshold) const {
    if (!config_.adaptiveThresholding) return baseThreshold;
    return baseThreshold * (1.0 + fatigueLevel_ * 0.3);
}

double FatigueAdaptationEngine::getAdaptedGain(double baseGain) const {
    return baseGain * metrics_.enduranceFactor;
}

void FatigueAdaptationEngine::establishBaseline(const std::vector<std::vector<double>>& baselineData) {
    baselineRms_.assign(numChannels_, 0.0);
    baselineMedianFreq_.assign(numChannels_, 100.0);
    if (!baselineData.empty() && !baselineData[0].empty()) {
        for (std::size_t ch = 0; ch < numChannels_ && ch < baselineData.size(); ++ch) {
            baselineRms_[ch] = rmsValue(baselineData[ch]);
            baselineMedianFreq_[ch] = medianFrequency(baselineData[ch]);
        }
        baselineEstablished_ = true;
    }
}

void FatigueAdaptationEngine::setRecoveryMode(bool enabled) { recoveryMode_ = enabled; }
bool FatigueAdaptationEngine::isFatigued() const { return fatigueLevel_ > config_.fatigueThreshold; }
FatigueMetrics FatigueAdaptationEngine::evaluate() const { return metrics_; }
bool FatigueAdaptationEngine::isConfigured() const { return configured_; }

}
