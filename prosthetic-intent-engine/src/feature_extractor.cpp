#include "intent_engine/feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace intent_engine {

namespace {

std::size_t windowLengthSamples(double sampleRateHz, double windowSeconds) {
    std::size_t count = static_cast<std::size_t>(std::llround(sampleRateHz * windowSeconds));
    if (count < 3) {
        count = 3;
    }
    return count;
}

std::size_t stepLengthSamples(double sampleRateHz, double stepSeconds, std::size_t windowSamples) {
    std::size_t count = static_cast<std::size_t>(std::llround(sampleRateHz * stepSeconds));
    if (count == 0) {
        count = 1;
    }
    if (count > windowSamples) {
        count = windowSamples;
    }
    return count;
}

}

FeatureExtractor::FeatureExtractor(std::size_t numChannels, double sampleRateHz, const FeatureWindowConfig& config) {
    configure(numChannels, sampleRateHz, config);
}

void FeatureExtractor::configure(std::size_t numChannels, double sampleRateHz, const FeatureWindowConfig& config) {
    if (numChannels == 0) {
        throw std::invalid_argument("FeatureExtractor requires at least one channel");
    }
    if (sampleRateHz <= 0.0) {
        throw std::invalid_argument("FeatureExtractor requires a positive sample rate");
    }
    if (config.windowSeconds <= 0.0) {
        throw std::invalid_argument("FeatureExtractor requires a positive window");
    }
    if (config.stepSeconds <= 0.0) {
        throw std::invalid_argument("FeatureExtractor requires a positive step");
    }

    config_ = config;
    sampleRateHz_ = sampleRateHz;
    numChannels_ = numChannels;
    windowSamples_ = windowLengthSamples(sampleRateHz, config.windowSeconds);
    stepSamples_ = stepLengthSamples(sampleRateHz, config.stepSeconds, windowSamples_);

    buffers_.assign(numChannels_, std::deque<double>{});
    features_.assign(numChannels_ * 4, 0.0);
    configured_ = true;
}

void FeatureExtractor::setWindowConfig(const FeatureWindowConfig& config) {
    configure(numChannels_, sampleRateHz_, config);
}

void FeatureExtractor::reset() {
    for (auto& buffer : buffers_) {
        buffer.clear();
    }
    std::fill(features_.begin(), features_.end(), 0.0);
}

bool FeatureExtractor::processSample(const std::vector<double>& channelValues) {
    if (!configured_) {
        throw std::runtime_error("FeatureExtractor is not configured");
    }
    if (channelValues.size() != numChannels_) {
        throw std::invalid_argument("FeatureExtractor channel count mismatch");
    }
    for (std::size_t channel = 0; channel < numChannels_; ++channel) {
        buffers_[channel].push_back(channelValues[channel]);
    }
    if (buffers_.front().size() < windowSamples_) {
        return false;
    }
    computeFeatures();
    for (std::size_t channel = 0; channel < numChannels_; ++channel) {
        for (std::size_t i = 0; i < stepSamples_; ++i) {
            buffers_[channel].pop_front();
        }
    }
    return true;
}

bool FeatureExtractor::processSample(const double* channelValues) {
    if (!configured_) {
        throw std::runtime_error("FeatureExtractor is not configured");
    }
    if (channelValues == nullptr) {
        throw std::invalid_argument("FeatureExtractor received a null channel pointer");
    }
    for (std::size_t channel = 0; channel < numChannels_; ++channel) {
        buffers_[channel].push_back(channelValues[channel]);
    }
    if (buffers_.front().size() < windowSamples_) {
        return false;
    }
    computeFeatures();
    for (std::size_t channel = 0; channel < numChannels_; ++channel) {
        for (std::size_t i = 0; i < stepSamples_; ++i) {
            buffers_[channel].pop_front();
        }
    }
    return true;
}

const std::vector<double>& FeatureExtractor::features() const {
    return features_;
}

bool FeatureExtractor::isConfigured() const {
    return configured_;
}

std::size_t FeatureExtractor::numChannels() const {
    return numChannels_;
}

std::size_t FeatureExtractor::numFeatures() const {
    return numChannels_ * 4;
}

std::size_t FeatureExtractor::windowSamples() const {
    return windowSamples_;
}

std::size_t FeatureExtractor::stepSamples() const {
    return stepSamples_;
}

double FeatureExtractor::sampleRateHz() const {
    return sampleRateHz_;
}

const FeatureWindowConfig& FeatureExtractor::windowConfig() const {
    return config_;
}

void FeatureExtractor::computeFeatures() {
    std::vector<double> frame;
    frame.reserve(windowSamples_);
    for (std::size_t channel = 0; channel < numChannels_; ++channel) {
        frame.clear();
        for (const double value : buffers_[channel]) {
            frame.push_back(value);
        }
        const std::size_t base = channel * 4;
        features_[base + 0] = meanAbsoluteValue(frame.data(), frame.size());
        features_[base + 1] = static_cast<double>(zeroCrossings(frame.data(), frame.size(), config_.zeroCrossingThreshold));
        features_[base + 2] = static_cast<double>(slopeSignChanges(frame.data(), frame.size(), config_.slopeThreshold));
        features_[base + 3] = waveformLength(frame.data(), frame.size());
    }
}

double FeatureExtractor::meanAbsoluteValue(const double* data, std::size_t count) {
    if (data == nullptr || count == 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += std::fabs(data[i]);
    }
    return sum / static_cast<double>(count);
}

std::size_t FeatureExtractor::zeroCrossings(const double* data, std::size_t count, double threshold) {
    if (data == nullptr || count < 2) {
        return 0;
    }
    std::size_t crossings = 0;
    const double absoluteThreshold = std::fabs(threshold);
    for (std::size_t i = 1; i < count; ++i) {
        const bool signChanged = (data[i] > 0.0 && data[i - 1] < 0.0) || (data[i] < 0.0 && data[i - 1] > 0.0);
        if (!signChanged) {
            continue;
        }
        if (std::fabs(data[i]) >= absoluteThreshold && std::fabs(data[i - 1]) >= absoluteThreshold) {
            ++crossings;
        }
    }
    return crossings;
}

std::size_t FeatureExtractor::slopeSignChanges(const double* data, std::size_t count, double threshold) {
    if (data == nullptr || count < 3) {
        return 0;
    }
    std::size_t changes = 0;
    const double absoluteThreshold = std::fabs(threshold);
    for (std::size_t i = 1; i + 1 < count; ++i) {
        const double firstSlope = data[i] - data[i - 1];
        const double secondSlope = data[i + 1] - data[i];
        if (firstSlope * secondSlope < 0.0 &&
            std::fabs(firstSlope) >= absoluteThreshold &&
            std::fabs(secondSlope) >= absoluteThreshold) {
            ++changes;
        }
    }
    return changes;
}

double FeatureExtractor::waveformLength(const double* data, std::size_t count) {
    if (data == nullptr || count < 2) {
        return 0.0;
    }
    double length = 0.0;
    for (std::size_t i = 1; i < count; ++i) {
        length += std::fabs(data[i] - data[i - 1]);
    }
    return length;
}

}