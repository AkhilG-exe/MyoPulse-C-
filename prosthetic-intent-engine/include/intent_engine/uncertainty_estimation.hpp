#pragma once

#include <cstddef>
#include <vector>

namespace intent_engine {

enum class UncertaintyType : unsigned char {
    kEpistemic = 0,
    kAleatoric = 1,
    kTotal = 2
};

struct UncertaintyMetrics {
    double confidence = 1.0;
    double entropy = 0.0;
    double margin = 0.0;
    double variationRatio = 0.0;
    double predictiveVariance = 0.0;
    double epistemicUncertainty = 0.0;
    double aleatoricUncertainty = 0.0;
    double totalUncertainty = 0.0;
    std::vector<double> classProbabilities;
    int predictedClass = -1;
    bool isHighUncertainty = false;
};

struct UncertaintyConfig {
    double uncertaintyThreshold = 0.5;
    double confidenceThreshold = 0.7;
    double temperatureScaling = 1.0;
    int monteCarloSamples = 10;
    bool enableMonteCarloDropout = false;
    bool enableTemperatureScaling = false;
    bool enablePlattScaling = false;
};

class UncertaintyEstimator {
public:
    UncertaintyEstimator();
    explicit UncertaintyEstimator(const UncertaintyConfig& config);

    void configure(const UncertaintyConfig& config);
    void reset();
    UncertaintyMetrics estimateFromScores(const std::vector<double>& decisionScores) const;
    UncertaintyMetrics estimateFromProbabilities(const std::vector<double>& probabilities) const;
    std::vector<double> softmax(const std::vector<double>& scores) const;
    std::vector<double> applyTemperatureScaling(const std::vector<double>& scores) const;
    double computeEntropy(const std::vector<double>& probabilities) const;
    double computeMargin(const std::vector<double>& probabilities) const;
    double computeVariationRatio(const std::vector<double>& probabilities) const;
    double computePredictiveVariance(const std::vector<std::vector<double>>& mcPredictions, std::size_t numClasses) const;
    bool rejectPrediction(const UncertaintyMetrics& metrics) const;
    bool isConfigured() const;

private:
    UncertaintyConfig config_;
    bool configured_ = false;
};

}
