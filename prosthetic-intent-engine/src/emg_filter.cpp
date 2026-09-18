#include "intent_engine/emg_filter.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace intent_engine {

namespace {

constexpr double kPi = 3.14159265358979323846;

double clampedCutoff(double sampleRateHz, double cutoffHz) {
    const double maximum = 0.49 * sampleRateHz;
    if (cutoffHz >= maximum) {
        return maximum;
    }
    return std::max(0.0, cutoffHz);
}

BiquadCoefficients butterworthSection(double sampleRateHz, double cutoffHz, std::size_t totalOrder, std::size_t sectionIndex, bool highpass) {
    if (sampleRateHz <= 0.0) {
        throw std::invalid_argument("butterworthSection requires a positive sample rate");
    }
    if (cutoffHz <= 0.0) {
        throw std::invalid_argument("butterworthSection requires a positive cutoff");
    }
    const double c = 2.0 * sampleRateHz;
    const double fc = clampedCutoff(sampleRateHz, cutoffHz);
    const double omegaC = c * std::tan(kPi * fc / sampleRateHz);
    const double angle = 0.5 * kPi + (2.0 * static_cast<double>(sectionIndex) + 1.0) * kPi / (2.0 * static_cast<double>(totalOrder));
    const double poleDamping = -2.0 * omegaC * std::cos(angle);
    const double poleFrequencySquared = omegaC * omegaC;
    const double denominator = c * c + poleDamping * c + poleFrequencySquared;
    BiquadCoefficients result;
    if (highpass) {
        result.b0 = c * c / denominator;
        result.b1 = -2.0 * c * c / denominator;
        result.b2 = c * c / denominator;
    } else {
        result.b0 = poleFrequencySquared / denominator;
        result.b1 = 2.0 * poleFrequencySquared / denominator;
        result.b2 = poleFrequencySquared / denominator;
    }
    result.a1 = (-2.0 * c * c + 2.0 * poleFrequencySquared) / denominator;
    result.a2 = (c * c - poleDamping * c + poleFrequencySquared) / denominator;
    return result;
}

void validateBiquadCoefficients(const BiquadCoefficients& coefficients) {
    if (!std::isfinite(coefficients.b0) ||
        !std::isfinite(coefficients.b1) ||
        !std::isfinite(coefficients.b2) ||
        !std::isfinite(coefficients.a1) ||
        !std::isfinite(coefficients.a2)) {
        throw std::invalid_argument("filter coefficients must be finite");
    }
}

}

BiquadFilter::BiquadFilter(const BiquadCoefficients& coefficients)
    : coefficients_(coefficients) {}

void BiquadFilter::setCoefficients(const BiquadCoefficients& coefficients) {
    validateBiquadCoefficients(coefficients);
    coefficients_ = coefficients;
    reset();
}

const BiquadCoefficients& BiquadFilter::coefficients() const {
    return coefficients_;
}

void BiquadFilter::reset() {
    z1_ = 0.0;
    z2_ = 0.0;
}

double BiquadFilter::process(double input) {
    const double output = coefficients_.b0 * input + z1_;
    z1_ = coefficients_.b1 * input - coefficients_.a1 * output + z2_;
    z2_ = coefficients_.b2 * input - coefficients_.a2 * output;
    return output;
}

void BiquadFilter::processStream(double* data, std::size_t count) {
    if (data == nullptr || count == 0) {
        return;
    }
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = process(data[i]);
    }
}

std::array<double, 2> BiquadFilter::state() const {
    return {z1_, z2_};
}

BiquadCoefficients makeNotchCoefficients(double sampleRateHz, double centerFrequencyHz, double quality) {
    if (sampleRateHz <= 0.0) {
        throw std::invalid_argument("makeNotchCoefficients requires a positive sample rate");
    }
    if (centerFrequencyHz <= 0.0 || centerFrequencyHz >= 0.5 * sampleRateHz) {
        throw std::invalid_argument("makeNotchCoefficients requires a center frequency below Nyquist");
    }
    if (quality <= 0.0) {
        throw std::invalid_argument("makeNotchCoefficients requires positive quality");
    }
    const double omega = 2.0 * kPi * centerFrequencyHz / sampleRateHz;
    const double alpha = std::sin(omega) / (2.0 * quality);
    const double a0 = 1.0 + alpha;
    const double b1 = -2.0 * std::cos(omega);
    BiquadCoefficients coefficients;
    coefficients.b0 = 1.0 / a0;
    coefficients.b1 = b1 / a0;
    coefficients.b2 = 1.0 / a0;
    coefficients.a1 = b1 / a0;
    coefficients.a2 = (1.0 - alpha) / a0;
    return coefficients;
}

BiquadCoefficients makeButterworthLowpassSection(double sampleRateHz, double cutoffHz) {
    return butterworthSection(sampleRateHz, cutoffHz, 2, 0, false);
}

BiquadCoefficients makeButterworthHighpassSection(double sampleRateHz, double cutoffHz) {
    return butterworthSection(sampleRateHz, cutoffHz, 2, 0, true);
}

void NotchFilter::configure(double sampleRateHz, MainsFrequency mainsFrequency, double quality) {
    configure(sampleRateHz, static_cast<double>(mainsFrequency), quality);
}

void NotchFilter::configure(double sampleRateHz, double centerFrequencyHz, double quality) {
    stage_.setCoefficients(makeNotchCoefficients(sampleRateHz, centerFrequencyHz, quality));
    sampleRateHz_ = sampleRateHz;
    centerFrequencyHz_ = centerFrequencyHz;
    configured_ = true;
}

void NotchFilter::reset() {
    stage_.reset();
}

double NotchFilter::process(double input) {
    if (!configured_) {
        return input;
    }
    return stage_.process(input);
}

void NotchFilter::processStream(double* data, std::size_t count) {
    if (data == nullptr || count == 0) {
        return;
    }
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = process(data[i]);
    }
}

bool NotchFilter::isConfigured() const {
    return configured_;
}

double NotchFilter::sampleRateHz() const {
    return sampleRateHz_;
}

double NotchFilter::centerFrequencyHz() const {
    return centerFrequencyHz_;
}

void BandpassFilter::configure(double sampleRateHz, double lowCutoffHz, double highCutoffHz, unsigned int order) {
    if (sampleRateHz <= 0.0) {
        throw std::invalid_argument("BandpassFilter requires a positive sample rate");
    }
    if (lowCutoffHz <= 0.0) {
        throw std::invalid_argument("BandpassFilter requires a positive low cutoff");
    }
    if (highCutoffHz <= lowCutoffHz) {
        throw std::invalid_argument("BandpassFilter high cutoff must exceed low cutoff");
    }
    if (highCutoffHz >= 0.5 * sampleRateHz) {
        throw std::invalid_argument("BandpassFilter high cutoff must stay below Nyquist");
    }

    unsigned int evenOrder = order < 2 ? 2u : order;
    if (evenOrder % 2u != 0u) {
        ++evenOrder;
    }

    const std::size_t sections = static_cast<std::size_t>(evenOrder / 2u);
    stages_.clear();
    for (std::size_t i = 0; i < sections; ++i) {
        BiquadFilter section;
        section.setCoefficients(butterworthSection(sampleRateHz, lowCutoffHz, evenOrder, i, true));
        stages_.push_back(section);
    }
    for (std::size_t i = 0; i < sections; ++i) {
        BiquadFilter section;
        section.setCoefficients(butterworthSection(sampleRateHz, highCutoffHz, evenOrder, i, false));
        stages_.push_back(section);
    }

    sampleRateHz_ = sampleRateHz;
    lowCutoffHz_ = lowCutoffHz;
    highCutoffHz_ = highCutoffHz;
    order_ = evenOrder;
    configured_ = true;
}

void BandpassFilter::reset() {
    for (auto& stage : stages_) {
        stage.reset();
    }
}

double BandpassFilter::process(double input) {
    if (!configured_) {
        return input;
    }
    double output = input;
    for (auto& stage : stages_) {
        output = stage.process(output);
    }
    return output;
}

void BandpassFilter::processStream(double* data, std::size_t count) {
    if (data == nullptr || count == 0) {
        return;
    }
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = process(data[i]);
    }
}

bool BandpassFilter::isConfigured() const {
    return configured_;
}

double BandpassFilter::sampleRateHz() const {
    return sampleRateHz_;
}

double BandpassFilter::lowCutoffHz() const {
    return lowCutoffHz_;
}

double BandpassFilter::highCutoffHz() const {
    return highCutoffHz_;
}

unsigned int BandpassFilter::order() const {
    return order_;
}

void EmgFilterChain::configure(const EmgFilterConfig& config) {
    if (config.sampleRateHz <= 0.0) {
        throw std::invalid_argument("EmgFilterChain requires a positive sample rate");
    }
    bandpass_.configure(config.sampleRateHz, config.lowCutoffHz, config.highCutoffHz, config.bandpassOrder);
    if (config.enableMainsNotch) {
        notch_.configure(config.sampleRateHz, config.mainsFrequency, config.notchQuality);
    }
    config_ = config;
    configured_ = true;
}

void EmgFilterChain::reset() {
    bandpass_.reset();
    notch_.reset();
}

double EmgFilterChain::process(double input) {
    if (!configured_) {
        return input;
    }
    double output = bandpass_.process(input);
    if (config_.enableMainsNotch) {
        output = notch_.process(output);
    }
    return output;
}

void EmgFilterChain::processStream(double* data, std::size_t count) {
    if (data == nullptr || count == 0) {
        return;
    }
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = process(data[i]);
    }
}

bool EmgFilterChain::isConfigured() const {
    return configured_;
}

double EmgFilterChain::sampleRateHz() const {
    return config_.sampleRateHz;
}

}