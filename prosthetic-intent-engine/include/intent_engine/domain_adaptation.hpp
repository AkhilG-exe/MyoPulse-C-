#pragma once

#include <cstddef>
#include <vector>

namespace intent_engine {

struct DomainAdaptationMetrics {
    double domainDiscrepancy = 0.0;
    double adaptationProgress = 0.0;
    bool adaptationActive = false;
    bool domainShiftDetected = false;
};

struct DomainAdaptationConfig {
    double discrepancyThreshold = 0.2;
    double learningRate = 0.01;
    double regularization = 0.1;
    std::size_t adaptationEpochs = 10;
    bool enableAdversarialAdaptation = false;
    bool enableStatisticalAlignment = true;
};

class DomainAdaptator {
public:
    DomainAdaptator() = default;
    DomainAdaptator(std::size_t featureDim, const DomainAdaptationConfig& config = DomainAdaptationConfig{});

    void configure(std::size_t featureDim, const DomainAdaptationConfig& config);
    void reset();
    void setSourceDistribution(const std::vector<std::vector<double>>& sourceFeatures);
    void setTargetDistribution(const std::vector<std::vector<double>>& targetFeatures);
    std::vector<double> adaptFeatures(const std::vector<double>& features) const;
    DomainAdaptationMetrics evaluate() const;
    double computeMmd(const std::vector<std::vector<double>>& src, const std::vector<std::vector<double>>& tgt) const;
    bool isConfigured() const;

private:
    std::vector<std::vector<double>> sourceMeanCov_;
    std::vector<std::vector<double>> targetMeanCov_;
    DomainAdaptationMetrics metrics_;
    DomainAdaptationConfig config_;
    std::size_t featureDim_ = 0;
    bool configured_ = false;
    bool distributionsSet_ = false;
};

}
