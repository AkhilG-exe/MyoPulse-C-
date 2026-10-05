#pragma once

#include <cstddef>
#include <vector>

namespace intent_engine {

enum class AnomalyType : unsigned char {
    kNone = 0,
    kOutlier = 1,
    kNovelGesture = 2,
    kSensorFault = 3
};

struct AnomalyResult {
    AnomalyType type = AnomalyType::kNone;
    double anomalyScore = 0.0;
    double threshold = 0.0;
    bool isAnomalous = false;
    int nearestClass = -1;
    double distanceToNearest = 0.0;
};

struct AnomalyDetectionConfig {
    double contaminationRate = 0.05;
    double threshold = 0.95;
    std::size_t minTrainingSamples = 50;
    bool enableIsolationForest = true;
    bool enableOneClassSvm = false;
    bool enableDistanceBased = true;
};

class AnomalyDetector {
public:
    AnomalyDetector();
    explicit AnomalyDetector(const AnomalyDetectionConfig& config);

    void configure(const AnomalyDetectionConfig& config);
    void reset();
    void train(const std::vector<std::vector<double>>& normalSamples);
    AnomalyResult detect(const std::vector<double>& sample) const;
    double computeMahalanobisDistance(const std::vector<double>& sample) const;
    double computeIsolationScore(const std::vector<double>& sample) const;
    bool isConfigured() const;
    bool isTrained() const;

private:
    std::vector<std::vector<double>> trainingData_;
    std::vector<double> mean_;
    std::vector<std::vector<double>> covariance_;
    std::vector<std::vector<double>> invCovariance_;
    AnomalyDetectionConfig config_;
    double maxDistance_ = 0.0;
    bool configured_ = false;
    bool trained_ = false;
};

}
