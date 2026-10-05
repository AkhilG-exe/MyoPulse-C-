#pragma once

#include <cstddef>
#include <limits>
#include <vector>

namespace intent_engine {

struct SignalQualityMetrics {
    double signalToNoiseRatio = 0.0;
    double signalToNoiseRatioDb = -std::numeric_limits<double>::infinity();
    double baselineStability = 1.0;
    double electrodeContactQuality = 1.0;
    double motionArtifactRatio = 0.0;
    double powerlineInterferenceRatio = 0.0;
    double dynamicRange = 0.0;
    double crestFactor = 0.0;
    double meanAmplitude = 0.0;
    double rootMeanSquare = 0.0;
    double variance = 0.0;
    bool isValid = false;
    std::vector<double> channelQuality;
};

struct SignalQualityConfig {
    double sampleRateHz = 1000.0;
    double baselineWindowSeconds = 1.0;
    double noiseFloorEstimateSeconds = 0.5;
    double powerlineFrequencyHz = 50.0;
    double powerlineBandwidthHz = 2.0;
    double motionArtifactThreshold = 0.3;
    double contactQualityThreshold = 0.7;
    double snrMinimumDb = -10.0;
    bool enablePowerlineDetection = true;
    bool enableMotionArtifactDetection = true;
};

class SignalQualityAnalyzer {
public:
    SignalQualityAnalyzer() = default;
    SignalQualityAnalyzer(std::size_t numChannels, const SignalQualityConfig& config = SignalQualityConfig{});

    void configure(std::size_t numChannels, const SignalQualityConfig& config);
    void reset();
    void update(const std::vector<double>& channelValues);
    void update(const double* channelValues, std::size_t numChannels);
    const SignalQualityMetrics& evaluate() const;
    SignalQualityMetrics evaluateWindow(const std::vector<std::vector<double>>& windowData) const;
    double computeSnr(const std::vector<double>& signal, const std::vector<double>& noise) const;
    double computePowerlineInterference(const std::vector<double>& data) const;
    double computeMotionArtifact(const std::vector<double>& data) const;
    double computeBaselineStability(const std::vector<double>& baselineData) const;
    double computeCrestFactor(const std::vector<double>& data) const;
    bool isSignalUsable(const SignalQualityMetrics& metrics) const;
    bool isConfigured() const;

private:
    void computeMetrics();
    std::vector<std::vector<double>> signalBuffer_;
    std::vector<std::vector<double>> noiseBuffer_;
    std::vector<std::vector<double>> baselineBuffer_;
    SignalQualityMetrics metrics_;
    SignalQualityConfig config_;
    std::size_t numChannels_ = 0;
    std::size_t signalBufferSize_ = 0;
    std::size_t noiseBufferSize_ = 0;
    std::size_t baselineBufferSize_ = 0;
    bool configured_ = false;
};

}
