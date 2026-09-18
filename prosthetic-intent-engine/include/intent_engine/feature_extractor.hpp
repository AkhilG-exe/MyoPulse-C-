#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace intent_engine {

enum class TimeDomainFeature : std::uint8_t {
    kMeanAbsoluteValue = 0,
    kZeroCrossings = 1,
    kSlopeSignChanges = 2,
    kWaveformLength = 3,
};

struct FeatureWindowConfig {
    double windowSeconds = 0.100;
    double stepSeconds = 0.010;
    double zeroCrossingThreshold = 0.0;
    double slopeThreshold = 0.0;
};

class FeatureExtractor {
public:
    FeatureExtractor() = default;

    FeatureExtractor(std::size_t numChannels, double sampleRateHz, const FeatureWindowConfig& config = FeatureWindowConfig{});

    void configure(std::size_t numChannels, double sampleRateHz, const FeatureWindowConfig& config);

    void setWindowConfig(const FeatureWindowConfig& config);

    void reset();

    bool processSample(const std::vector<double>& channelValues);

    bool processSample(const double* channelValues);

    const std::vector<double>& features() const;

    bool isConfigured() const;

    std::size_t numChannels() const;

    std::size_t numFeatures() const;

    std::size_t windowSamples() const;

    std::size_t stepSamples() const;

    double sampleRateHz() const;

    const FeatureWindowConfig& windowConfig() const;

    static double meanAbsoluteValue(const double* data, std::size_t count);

    static std::size_t zeroCrossings(const double* data, std::size_t count, double threshold);

    static std::size_t slopeSignChanges(const double* data, std::size_t count, double threshold);

    static double waveformLength(const double* data, std::size_t count);

private:
    void computeFeatures();

    std::vector<std::deque<double>> buffers_;
    std::vector<double> features_;
    FeatureWindowConfig config_{};
    double sampleRateHz_ = 0.0;
    std::size_t numChannels_ = 0;
    std::size_t windowSamples_ = 0;
    std::size_t stepSamples_ = 0;
    bool configured_ = false;
};

}