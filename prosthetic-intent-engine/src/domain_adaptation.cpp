#include "intent_engine/domain_adaptation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace intent_engine {

namespace {
double meanValue(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / static_cast<double>(data.size());
}
}

DomainAdaptator::DomainAdaptator(std::size_t featureDim, const DomainAdaptationConfig& config) {
    configure(featureDim, config);
}

void DomainAdaptator::configure(std::size_t featureDim, const DomainAdaptationConfig& config) {
    config_ = config;
    featureDim_ = featureDim;
    configured_ = true;
}

void DomainAdaptator::reset() {
    distributionsSet_ = false;
}

void DomainAdaptator::setSourceDistribution(const std::vector<std::vector<double>>& sourceFeatures) {
    sourceMeanCov_.assign(1, std::vector<double>(featureDim_, 0.0));
    if (!sourceFeatures.empty()) {
        for (std::size_t j = 0; j < featureDim_ && j < sourceFeatures[0].size(); ++j) {
            std::vector<double> col(sourceFeatures.size(), 0.0);
            for (std::size_t i = 0; i < sourceFeatures.size(); ++i) col[i] = sourceFeatures[i][j];
            sourceMeanCov_[0][j] = meanValue(col);
        }
    }
    distributionsSet_ = true;
}

void DomainAdaptator::setTargetDistribution(const std::vector<std::vector<double>>& targetFeatures) {
    targetMeanCov_.assign(1, std::vector<double>(featureDim_, 0.0));
    if (!targetFeatures.empty()) {
        for (std::size_t j = 0; j < featureDim_ && j < targetFeatures[0].size(); ++j) {
            std::vector<double> col(targetFeatures.size(), 0.0);
            for (std::size_t i = 0; i < targetFeatures.size(); ++i) col[i] = targetFeatures[i][j];
            targetMeanCov_[0][j] = meanValue(col);
        }
    }
    distributionsSet_ = true;
}

std::vector<double> DomainAdaptator::adaptFeatures(const std::vector<double>& features) const {
    std::vector<double> adapted = features;
    if (!distributionsSet_ || adapted.empty()) return adapted;
    for (std::size_t j = 0; j < adapted.size() && j < featureDim_; ++j) {
        if (!sourceMeanCov_.empty() && !targetMeanCov_.empty()) {
            double diff = targetMeanCov_[0][j] - sourceMeanCov_[0][j];
            adapted[j] += diff * config_.learningRate;
        }
    }
    return adapted;
}

double DomainAdaptator::computeMmd(const std::vector<std::vector<double>>& src, const std::vector<std::vector<double>>& tgt) const {
    if (src.empty() || tgt.empty()) return 0.0;
    double sum = 0.0;
    std::size_t n = std::min(src.size(), tgt.size());
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < featureDim_ && j < src[i].size() && j < tgt[i].size(); ++j) {
            sum += std::abs(src[i][j] - tgt[i][j]);
        }
    }
    return n > 0 ? sum / (static_cast<double>(n) * static_cast<double>(featureDim_)) : 0.0;
}

DomainAdaptationMetrics DomainAdaptator::evaluate() const {
    DomainAdaptationMetrics m;
    m.domainDiscrepancy = computeMmd({}, {});
    m.adaptationActive = distributionsSet_;
    m.domainShiftDetected = m.domainDiscrepancy > config_.discrepancyThreshold;
    return m;
}

bool DomainAdaptator::isConfigured() const { return configured_; }

}
