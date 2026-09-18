#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace intent_engine {

struct BiquadCoefficients {
    double b0 = 1.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
};

class BiquadFilter {
public:
    BiquadFilter() = default;

    explicit BiquadFilter(const BiquadCoefficients& coefficients);

    void setCoefficients(const BiquadCoefficients& coefficients);

    const BiquadCoefficients& coefficients() const;

    void reset();

    double process(double input);

    void processStream(double* data, std::size_t count);

    std::array<double, 2> state() const;

private:
    BiquadCoefficients coefficients_{};
    double z1_ = 0.0;
    double z2_ = 0.0;
};

enum class MainsFrequency : std::uint8_t {
    k50Hz = 50,
    k60Hz = 60,
};

BiquadCoefficients makeNotchCoefficients(double sampleRateHz, double centerFrequencyHz, double quality);

BiquadCoefficients makeButterworthLowpassSection(double sampleRateHz, double cutoffHz);

BiquadCoefficients makeButterworthHighpassSection(double sampleRateHz, double cutoffHz);

class NotchFilter {
public:
    NotchFilter() = default;

    void configure(double sampleRateHz, MainsFrequency mainsFrequency, double quality = 30.0);

    void configure(double sampleRateHz, double centerFrequencyHz, double quality = 30.0);

    void reset();

    double process(double input);

    void processStream(double* data, std::size_t count);

    bool isConfigured() const;

    double sampleRateHz() const;

    double centerFrequencyHz() const;

private:
    BiquadFilter stage_;
    double sampleRateHz_ = 0.0;
    double centerFrequencyHz_ = 0.0;
    bool configured_ = false;
};

class BandpassFilter {
public:
    BandpassFilter() = default;

    void configure(double sampleRateHz, double lowCutoffHz, double highCutoffHz, unsigned int order = 4);

    void reset();

    double process(double input);

    void processStream(double* data, std::size_t count);

    bool isConfigured() const;

    double sampleRateHz() const;

    double lowCutoffHz() const;

    double highCutoffHz() const;

    unsigned int order() const;

private:
    std::vector<BiquadFilter> stages_;
    double sampleRateHz_ = 0.0;
    double lowCutoffHz_ = 0.0;
    double highCutoffHz_ = 0.0;
    unsigned int order_ = 0;
    bool configured_ = false;
};

struct EmgFilterConfig {
    double sampleRateHz = 1000.0;
    MainsFrequency mainsFrequency = MainsFrequency::k50Hz;
    bool enableMainsNotch = true;
    double notchQuality = 30.0;
    double lowCutoffHz = 20.0;
    double highCutoffHz = 450.0;
    unsigned int bandpassOrder = 4;
};

class EmgFilterChain {
public:
    EmgFilterChain() = default;

    void configure(const EmgFilterConfig& config);

    void reset();

    double process(double input);

    void processStream(double* data, std::size_t count);

    bool isConfigured() const;

    double sampleRateHz() const;

private:
    BandpassFilter bandpass_;
    NotchFilter notch_;
    EmgFilterConfig config_{};
    bool configured_ = false;
};

}