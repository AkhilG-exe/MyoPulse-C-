#pragma once

#include <cstddef>
#include <vector>
#include <map>
#include <string>

namespace intent_engine {

struct LearningSample {
    std::vector<double> features;
    int label = 0;
    double confidence = 1.0;
    double timestamp = 0.0;
    bool isValidated = false;
};

struct AdaptationMetrics {
    std::size_t samplesAdded = 0;
    std::size_t samplesPruned = 0;
    double driftScore = 0.0;
    double conceptDriftDetected = 0.0;
    double performanceStability = 1.0;
    bool adaptationActive = false;
    std::vector<double> classDistribution;
};

struct AdaptiveLearningConfig {
    double driftThreshold = 0.1;
    double forgettingFactor = 0.95;
    double confidenceThreshold = 0.8;
    double validationIntervalSeconds = 60.0;
    std::size_t maxBufferSize = 10000;
    std::size_t minBufferSize = 100;
    std::size_t batchSize = 100;
    bool enableOnlineLearning = false;
    bool enableConceptDriftDetection = true;
    bool enableActiveLearning = true;
    bool enableCatastrophicForgettingMitigation = true;
};

class AdaptiveLearningSystem {
public:
    AdaptiveLearningSystem();
    explicit AdaptiveLearningSystem(const AdaptiveLearningConfig& config);

    void configure(const AdaptiveLearningConfig& config);
    void reset();
    void addSample(const std::vector<double>& features, int label, double confidence = 1.0);
    void addValidatedSample(const std::vector<double>& features, int label, double confidence = 1.0, bool validated = true);
    std::vector<LearningSample> selectActiveLearningSamples(const std::vector<LearningSample>& unlabeledSamples) const;
    double detectConceptDrift(const std::vector<double>& recentPredictions, const std::vector<int>& recentGroundTruth) const;
    void pruneOldSamples();
    void updateModelWeights();
    std::vector<LearningSample> getTrainingBuffer() const;
    AdaptationMetrics getMetrics() const;
    bool shouldAdapt() const;
    bool isConfigured() const;
    void setOnlineLearning(bool enabled);
    double computeDriftScore(const std::vector<double>& distribution1, const std::vector<double>& distribution2) const;
    std::vector<double> computeClassDistribution() const;

private:
    std::vector<LearningSample> trainingBuffer_;
    std::vector<double> errorHistory_;
    std::vector<std::vector<double>> referenceDistribution_;
    AdaptiveLearningConfig config_;
    AdaptationMetrics metrics_;
    double lastAdaptationTime_ = 0.0;
    bool configured_ = false;
};

}
