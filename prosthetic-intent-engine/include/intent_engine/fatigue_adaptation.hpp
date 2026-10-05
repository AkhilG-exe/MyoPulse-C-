#pragma once

#include <cstddef>
#include <vector>
#include <deque>

namespace intent_engine {

struct FatigueMetrics {
    double fatigueIndex = 0.0;
    double enduranceFactor = 1.0;
    double recoveryFactor = 0.0;
    double spectralShift = 0.0;
    double medianFrequency = 0.0;
    double meanFrequency = 0.0;
    double rmsIncrease = 0.0;
    double muscleActivationDecrease = 0.0;
    bool isFatigued = false;
    std::vector<double> channelFatigue;
    std::vector<double> baselineRms;
    std::vector<double> currentRms;
};

struct FatigueAdaptationConfig {
    double sampleRateHz = 1000.0;
    double baselineDurationSeconds = 60.0;
    double fatigueThreshold = 0.3;
    double recoveryThreshold = 0.2;
    double adaptationRate = 0.01;
    double recoveryRate = 0.005;
    double forgettingFactor = 0.95;
    double frequencyShiftThreshold = 20.0;
    bool enableFrequencyShiftDetection = true;
    bool enableRmsAdaptation = true;
    bool adaptiveThresholding = true;
};

class FatigueAdaptationEngine {
public:
    FatigueAdaptationEngine() = default;
    FatigueAdaptationEngine(std::size_t numChannels, const FatigueAdaptationConfig& config = FatigueAdaptationConfig{});

    void configure(std::size_t numChannels, const FatigueAdaptationConfig& config);
    void reset();
    void establishBaseline(const std::vector<std::vector<double>>& baselineData);
    void update(const std::vector<double>& channelValues);
    void update(const double* channelValues, std::size_t numChannels);
    double computeMedianFrequency(const std::vector<double>& data) const;
    double computeMeanFrequency(const std::vector<double>& data) const;
    double computeSpectralMoment(const std::vector<double>& data, int order) const;
    FatigueMetrics evaluate() const;
    double getAdaptedThreshold(double baseThreshold) const;
    double getAdaptedGain(double baseGain) const;
    void setRecoveryMode(bool enabled);
    bool isFatigued() const;
    bool isConfigured() const;

private:
    void computeFatigueIndex();
    std::vector<std::deque<double>> signalHistory_;
    std::vector<double> baselineMedianFreq_;
    std::vector<double> baselineRms_;
    std::vector<double> currentMedianFreq_;
    std::vector<double> currentRms_;
    std::vector<double> channelFatigue_;
    FatigueMetrics metrics_;
    FatigueAdaptationConfig config_;
    std::size_t numChannels_ = 0;
    std::size_t historySize_ = 0;
    bool baselineEstablished_ = false;
    bool recoveryMode_ = false;
    double fatigueLevel_ = 0.0;
    bool configured_ = false;
};

}
