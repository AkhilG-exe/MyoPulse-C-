#pragma once

#include <cstddef>
#include <deque>

namespace intent_engine {

struct ProportionalControlParams {
    double deadbandRms = 0.0;
    double saturationRms = 1.0;
    double maxDutyCycle = 1.0;
    double outputScale = 1.0;
    double rampRatePerSecond = 0.0;
    double maxCurrentAmps = 0.0;
    double currentDropoutAmps = 0.0;
    double currentSensorScale = 1.0;
    double maxForceNewtons = 0.0;
    double forceSensorScale = 1.0;
    double initialDuty = 0.0;
};

class ProportionalControl {
public:
    void setParameters(const ProportionalControlParams& params);

    const ProportionalControlParams& parameters() const;

    double update(double rmsValue, double dtSeconds);

    double update(double rmsValue, double dtSeconds, double measuredCurrent, double measuredForce);

    double mapRmsToDuty(double rmsValue) const;

    double dutyCycle() const;

    void reset();

    void emergencyStop();

    void releaseEmergencyStop();

    bool isEmergencyStopped() const;

    bool isCurrentLatched() const;

private:
    ProportionalControlParams params_{};
    double duty_ = 0.0;
    bool emergency_ = false;
    bool currentLatch_ = false;
};

class RmsTracker {
public:
    RmsTracker() = default;

    RmsTracker(double sampleRateHz, double timeConstantSeconds);

    void configure(double sampleRateHz, double timeConstantSeconds);

    void reset();

    double update(double sample);

    double rms() const;

    std::size_t windowSamples() const;

    bool isConfigured() const;

private:
    std::deque<double> squares_;
    double sumSquares_ = 0.0;
    std::size_t windowSamples_ = 0;
    bool configured_ = false;
};

}