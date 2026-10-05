#pragma once

#include "intent_engine/emg_filter.hpp"
#include "intent_engine/feature_extractor.hpp"
#include "intent_engine/intent_classifier.hpp"
#include "intent_engine/proportional_control.hpp"
#include "intent_engine/signal_quality.hpp"
#include "intent_engine/muscle_synergy_graph.hpp"
#include "intent_engine/fatigue_adaptation.hpp"
#include "intent_engine/uncertainty_estimation.hpp"
#include "intent_engine/adaptive_learning.hpp"
#include "intent_engine/electrode_shift_compensation.hpp"
#include "intent_engine/domain_adaptation.hpp"
#include "intent_engine/anomaly_detection.hpp"
#include "intent_engine/realtime_graph.hpp"
#include "intent_engine/telemetry.hpp"

namespace intent_engine {

struct EngineConfig {
    EmgFilterConfig filterConfig;
    FeatureWindowConfig windowConfig;
    SvmParameters svmParams;
    ProportionalControlParams controlParams;
    SignalQualityConfig qualityConfig;
    MuscleSynergyConfig synergyConfig;
    FatigueAdaptationConfig fatigueConfig;
    UncertaintyConfig uncertaintyConfig;
    AdaptiveLearningConfig learningConfig;
    ElectrodeShiftConfig shiftConfig;
    DomainAdaptationConfig domainAdaptConfig;
    AnomalyDetectionConfig anomalyConfig;
    RealtimeGraphConfig graphConfig;
    TelemetryConfig telemetryConfig;
    double sampleRateHz = 1000.0;
    std::size_t numChannels = 4;
    bool enableSignalQuality = true;
    bool enableSynergyAnalysis = true;
    bool enableFatigueAdaptation = true;
    bool enableUncertaintyEstimation = true;
    bool enableAdaptiveLearning = true;
    bool enableElectrodeShiftCompensation = true;
    bool enableDomainAdaptation = true;
    bool enableAnomalyDetection = true;
    bool enableRealtimeGraph = true;
    bool enableTelemetry = true;
};

class IntentEngine {
public:
    IntentEngine() = default;
    IntentEngine(const EngineConfig& config);

    void configure(const EngineConfig& config);
    void reset();
    bool processSample(const std::vector<double>& channelValues, double dtSeconds = 0.001);
    bool processSample(const double* channelValues, double dtSeconds = 0.001);

    EmgFilterChain& filterChain() { return filterChain_; }
    FeatureExtractor& featureExtractor() { return featureExtractor_; }
    LdaClassifier& ldaClassifier() { return ldaClassifier_; }
    SvmClassifier& svmClassifier() { return svmClassifier_; }
    ProportionalControl& proportionalControl() { return proportionalControl_; }
    SignalQualityAnalyzer& signalQuality() { return signalQuality_; }
    MuscleSynergyGraph& muscleSynergy() { return muscleSynergy_; }
    FatigueAdaptationEngine& fatigueAdaptation() { return fatigueAdaptation_; }
    UncertaintyEstimator& uncertaintyEstimator() { return uncertaintyEstimator_; }
    AdaptiveLearningSystem& adaptiveLearning() { return adaptiveLearning_; }
    ElectrodeShiftCompensator& electrodeShiftCompensation() { return electrodeShiftComp_; }
    DomainAdaptator& domainAdaptation() { return domainAdapt_; }
    AnomalyDetector& anomalyDetector() { return anomalyDetector_; }
    RealtimeGraphNetwork& realtimeGraph() { return realtimeGraph_; }
    TelemetrySystem& telemetry() { return telemetry_; }

    const EngineConfig& config() const { return config_; }
    GestureClass predictedGesture() const { return predictedGesture_; }
    const std::vector<double>& features() const { return features_; }
    bool isConfigured() const { return configured_; }

private:
    EngineConfig config_;
    EmgFilterChain filterChain_;
    FeatureExtractor featureExtractor_;
    LdaClassifier ldaClassifier_;
    SvmClassifier svmClassifier_;
    ProportionalControl proportionalControl_;
    SignalQualityAnalyzer signalQuality_;
    MuscleSynergyGraph muscleSynergy_;
    FatigueAdaptationEngine fatigueAdaptation_;
    UncertaintyEstimator uncertaintyEstimator_;
    AdaptiveLearningSystem adaptiveLearning_;
    ElectrodeShiftCompensator electrodeShiftComp_;
    DomainAdaptator domainAdapt_;
    AnomalyDetector anomalyDetector_;
    RealtimeGraphNetwork realtimeGraph_;
    TelemetrySystem telemetry_;
    GestureClass predictedGesture_ = GestureClass::kRest;
    std::vector<double> features_;
    bool configured_ = false;
};

}
