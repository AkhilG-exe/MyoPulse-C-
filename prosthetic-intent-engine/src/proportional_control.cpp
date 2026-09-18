#include "intent_engine/proportional_control.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace intent_engine {

void ProportionalControl::setParameters(const ProportionalControlParams& params) {
    if (params.saturationRms < params.deadbandRms) {
        throw std::invalid_argument("ProportionalControl saturation must not be below deadband");
    }
    if (params.maxDutyCycle < 0.0 || params.maxDutyCycle > 1.0) {
        throw std::invalid_argument("ProportionalControl maxDutyCycle must be within [0,1]");
    }
    if (params.outputScale < 0.0) {
        throw std::invalid_argument("ProportionalControl outputScale must be non-negative");
    }
    if (params.rampRatePerSecond < 0.0) {
        throw std::invalid_argument("ProportionalControl rampRatePerSecond must be non-negative");
    }
    if (params.currentSensorScale < 0.0 || params.forceSensorScale < 0.0) {
        throw std::invalid_argument("ProportionalControl sensor scales must be non-negative");
    }
    if (params.maxCurrentAmps < 0.0 || params.currentDropoutAmps < 0.0 || params.maxForceNewtons < 0.0) {
        throw std::invalid_argument("ProportionalControl safety limits must be non-negative");
    }
    if (params.initialDuty < 0.0 || params.initialDuty > params.maxDutyCycle) {
        throw std::invalid_argument("ProportionalControl initialDuty outside allowed duty range");
    }
    params_ = params;
    duty_ = params_.initialDuty;
    reset();
}

void ProportionalControl::reset() {
    duty_ = params_.initialDuty;
    emergency_ = false;
    currentLatch_ = false;
}

const ProportionalControlParams& ProportionalControl::parameters() const {
    return params_;
}

double ProportionalControl::mapRmsToDuty(double rmsValue) const {
    if (rmsValue <= params_.deadbandRms) {
        return 0.0;
    }
    const double span = params_.saturationRms - params_.deadbandRms;
    if (span <= 0.0) {
        return params_.maxDutyCycle * params_.outputScale;
    }
    double normalized = (rmsValue - params_.deadbandRms) / span;
    normalized = std::clamp(normalized, 0.0, 1.0);
    double duty = normalized * params_.maxDutyCycle * params_.outputScale;
    return std::clamp(duty, 0.0, params_.maxDutyCycle);
}

double ProportionalControl::update(double rmsValue, double dtSeconds) {
    return update(rmsValue, dtSeconds, 0.0, 0.0);
}

double ProportionalControl::update(double rmsValue, double dtSeconds, double measuredCurrent, double measuredForce) {
    if (emergency_) {
        duty_ = 0.0;
        return 0.0;
    }

    const double currentAmps = measuredCurrent * params_.currentSensorScale;
    const double forceNewtons = measuredForce * params_.forceSensorScale;

    if (params_.maxCurrentAmps > 0.0) {
        const double dropout = params_.currentDropoutAmps > 0.0 ? params_.currentDropoutAmps : 0.8 * params_.maxCurrentAmps;
        if (currentLatch_) {
            if (currentAmps < dropout) {
                currentLatch_ = false;
            }
        } else if (currentAmps > params_.maxCurrentAmps) {
            currentLatch_ = true;
        }
    }

    double target = mapRmsToDuty(rmsValue);
    if (currentLatch_ || (params_.maxForceNewtons > 0.0 && forceNewtons > params_.maxForceNewtons)) {
        target = 0.0;
    }

    double next = target;
    if (params_.rampRatePerSecond > 0.0 && dtSeconds > 0.0) {
        const double maximumDelta = params_.rampRatePerSecond * dtSeconds;
        if (next > duty_ + maximumDelta) {
            next = duty_ + maximumDelta;
        } else if (next < duty_ - maximumDelta) {
            next = duty_ - maximumDelta;
        }
    }
    duty_ = next;
    if (currentLatch_) {
        duty_ = 0.0;
    }
    return duty_;
}

double ProportionalControl::dutyCycle() const {
    return duty_;
}

void ProportionalControl::emergencyStop() {
    emergency_ = true;
    duty_ = 0.0;
}

void ProportionalControl::releaseEmergencyStop() {
    emergency_ = false;
}

bool ProportionalControl::isEmergencyStopped() const {
    return emergency_;
}

bool ProportionalControl::isCurrentLatched() const {
    return currentLatch_;
}

RmsTracker::RmsTracker(double sampleRateHz, double timeConstantSeconds) {
    configure(sampleRateHz, timeConstantSeconds);
}

void RmsTracker::configure(double sampleRateHz, double timeConstantSeconds) {
    if (sampleRateHz <= 0.0) {
        throw std::invalid_argument("RmsTracker requires a positive sample rate");
    }
    if (timeConstantSeconds <= 0.0) {
        throw std::invalid_argument("RmsTracker requires a positive time constant");
    }
    windowSamples_ = static_cast<std::size_t>(std::llround(sampleRateHz * timeConstantSeconds));
    if (windowSamples_ == 0) {
        windowSamples_ = 1;
    }
    squares_.clear();
    sumSquares_ = 0.0;
    configured_ = true;
}

void RmsTracker::reset() {
    squares_.clear();
    sumSquares_ = 0.0;
}

double RmsTracker::update(double sample) {
    if (!configured_) {
        throw std::runtime_error("RmsTracker is not configured");
    }
    const double square = sample * sample;
    squares_.push_back(square);
    sumSquares_ += square;
    if (squares_.size() > windowSamples_) {
        sumSquares_ -= squares_.front();
        squares_.pop_front();
    }
    const double count = static_cast<double>(squares_.size());
    return std::sqrt(sumSquares_ / count);
}

double RmsTracker::rms() const {
    if (!configured_ || squares_.empty()) {
        return 0.0;
    }
    return std::sqrt(sumSquares_ / static_cast<double>(squares_.size()));
}

std::size_t RmsTracker::windowSamples() const {
    return windowSamples_;
}

bool RmsTracker::isConfigured() const {
    return configured_;
}

}