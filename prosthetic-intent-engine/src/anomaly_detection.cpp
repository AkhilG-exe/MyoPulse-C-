#include "intent_engine/anomaly_detection.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <limits>

namespace intent_engine {

namespace {
double meanValue(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / static_cast<double>(data.size());
}

double distanceL2(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}
}

AnomalyDetector::AnomalyDetector() {
    configure(AnomalyDetectionConfig{});
}

AnomalyDetector::AnomalyDetector(const AnomalyDetectionConfig& config) {
    configure(config);
}

void AnomalyDetector::configure(const AnomalyDetectionConfig& config) {
    config_ = config;
    configured_ = true;
}

void AnomalyDetector::reset() {
    trainingData_.clear();
    trained_ = false;
}

void AnomalyDetector::train(const std::vector<std::vector<double>>& normalSamples) {
    trainingData_ = normalSamples;
    if (trainingData_.empty() || trainingData_[0].empty()) {
        trained_ = false;
        return;
    }
    std::size_t dim = trainingData_[0].size();
    mean_.assign(dim, 0.0);
    for (const auto& s : trainingData_) {
        for (std::size_t i = 0; i < dim && i < s.size(); ++i) mean_[i] += s[i];
    }
    for (auto& m : mean_) m /= static_cast<double>(trainingData_.size());
    maxDistance_ = 0.0;
    for (const auto& s : trainingData_) {
        double d = distanceL2(s, mean_);
        if (d > maxDistance_) maxDistance_ = d;
    }
    trained_ = trainingData_.size() >= config_.minTrainingSamples;
}

AnomalyResult AnomalyDetector::detect(const std::vector<double>& sample) const {
    AnomalyResult res;
    if (!trained_) return res;
    res.distanceToNearest = distanceL2(sample, mean_);
    res.anomalyScore = maxDistance_ > 0 ? res.distanceToNearest / maxDistance_ : 0.0;
    res.threshold = config_.threshold;
    res.isAnomalous = res.anomalyScore > config_.threshold;
    res.type = res.isAnomalous ? AnomalyType::kOutlier : AnomalyType::kNone;
    res.nearestClass = -1;
    return res;
}

bool AnomalyDetector::isConfigured() const { return configured_; }
bool AnomalyDetector::isTrained() const { return trained_; }

}
