#include "intent_engine/intent_engine.hpp"

#include <vector>

namespace intent_engine {

IntentEngine::IntentEngine(const EngineConfig& config) {
    configure(config);
}

void IntentEngine::configure(const EngineConfig& config) {
    config_ = config;

    EmgFilterChain filterChain;
    filterChain.configure(config_.filterConfig);
    filterChain_ = std::move(filterChain);

    featureExtractor_.configure(config_.numChannels, config_.sampleRateHz, config_.windowConfig);
    signalQuality_.configure(config_.numChannels, config_.qualityConfig);
    muscleSynergy_.configure(config_.numChannels, config_.synergyConfig);
    fatigueAdaptation_.configure(config_.numChannels, config_.fatigueConfig);
    uncertaintyEstimator_.configure(config_.uncertaintyConfig);
    adaptiveLearning_.configure(config_.learningConfig);
    electrodeShiftComp_.configure(config_.numChannels, config_.shiftConfig);
    domainAdapt_.configure(featureExtractor_.numFeatures(), config_.domainAdaptConfig);
    anomalyDetector_.configure(config_.anomalyConfig);
    realtimeGraph_.configure(config_.numChannels, config_.graphConfig);
    telemetry_.configure(config_.telemetryConfig);
    proportionalControl_.setParameters(config_.controlParams);

    configured_ = true;
    reset();
}

void IntentEngine::reset() {
    filterChain_.reset();
    featureExtractor_.reset();
    signalQuality_.reset();
    muscleSynergy_.reset();
    fatigueAdaptation_.reset();
    uncertaintyEstimator_.reset();
    adaptiveLearning_.reset();
    electrodeShiftComp_.reset();
    domainAdapt_.reset();
    anomalyDetector_.reset();
    realtimeGraph_.reset();
    telemetry_.reset();
    proportionalControl_.reset();
    predictedGesture_ = GestureClass::kRest;
    features_.clear();
}

bool IntentEngine::processSample(const std::vector<double>& channelValues, double dtSeconds) {
    if (!configured_) return false;

    std::vector<double> filtered(channelValues.size(), 0.0);
    for (std::size_t i = 0; i < channelValues.size(); ++i) {
        filtered[i] = filterChain_.process(channelValues[i]);
    }

    if (config_.enableSignalQuality) {
        signalQuality_.update(filtered);
    }

    if (config_.enableFatigueAdaptation) {
        fatigueAdaptation_.update(filtered);
    }

    if (config_.enableSynergyAnalysis) {
        muscleSynergy_.update(filtered);
    }

    bool windowReady = featureExtractor_.processSample(filtered);
    if (windowReady) {
        features_ = featureExtractor_.features();
        if (ldaClassifier_.isTrained()) {
            int pred = ldaClassifier_.predict(features_);
            predictedGesture_ = gestureClassFromInt(pred);
        } else if (svmClassifier_.isTrained()) {
            int pred = svmClassifier_.predict(features_);
            predictedGesture_ = gestureClassFromInt(pred);
        }

        if (config_.enableUncertaintyEstimation && ldaClassifier_.isTrained()) {
            auto scores = ldaClassifier_.decisionScores(features_);
            uncertaintyEstimator_.estimateFromScores(scores);
        } else if (config_.enableUncertaintyEstimation && svmClassifier_.isTrained()) {
            auto scores = svmClassifier_.decisionScores(features_);
            uncertaintyEstimator_.estimateFromScores(scores);
        }
    }

    proportionalControl_.update(0.5, dtSeconds);
    return windowReady;
}

bool IntentEngine::processSample(const double* channelValues, double dtSeconds) {
    std::vector<double> vec(channelValues, channelValues + config_.numChannels);
    return processSample(vec, dtSeconds);
}

}
