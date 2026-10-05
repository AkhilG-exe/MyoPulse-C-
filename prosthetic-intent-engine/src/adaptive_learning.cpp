#include "intent_engine/adaptive_learning.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace intent_engine {

AdaptiveLearningSystem::AdaptiveLearningSystem() {
    configure(AdaptiveLearningConfig{});
}

AdaptiveLearningSystem::AdaptiveLearningSystem(const AdaptiveLearningConfig& config) {
    configure(config);
}

void AdaptiveLearningSystem::configure(const AdaptiveLearningConfig& config) {
    config_ = config;
    trainingBuffer_.clear();
    errorHistory_.clear();
    metrics_ = AdaptationMetrics();
    configured_ = true;
}

void AdaptiveLearningSystem::reset() {
    trainingBuffer_.clear();
    errorHistory_.clear();
    metrics_ = AdaptationMetrics();
}

void AdaptiveLearningSystem::addSample(const std::vector<double>& features, int label, double confidence) {
    LearningSample s;
    s.features = features;
    s.label = label;
    s.confidence = confidence;
    s.isValidated = false;
    trainingBuffer_.push_back(std::move(s));
    if (trainingBuffer_.size() > config_.maxBufferSize) {
        pruneOldSamples();
    }
}

void AdaptiveLearningSystem::addValidatedSample(const std::vector<double>& features, int label, double confidence, bool validated) {
    LearningSample s;
    s.features = features;
    s.label = label;
    s.confidence = confidence;
    s.isValidated = validated;
    trainingBuffer_.push_back(std::move(s));
}

std::vector<LearningSample> AdaptiveLearningSystem::selectActiveLearningSamples(const std::vector<LearningSample>& unlabeledSamples) const {
    std::vector<LearningSample> selected;
    if (!config_.enableActiveLearning) return selected;
    for (const auto& u : unlabeledSamples) {
        if (u.confidence < config_.confidenceThreshold) {
            selected.push_back(u);
            if (selected.size() >= config_.batchSize) break;
        }
    }
    return selected;
}

double AdaptiveLearningSystem::detectConceptDrift(const std::vector<double>& recentPredictions, const std::vector<int>& recentGroundTruth) const {
    if (!config_.enableConceptDriftDetection || recentPredictions.empty() || recentGroundTruth.empty()) return 0.0;
    double errors = 0.0;
    std::size_t n = std::min(recentPredictions.size(), recentGroundTruth.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (static_cast<int>(recentPredictions[i] + 0.5) != recentGroundTruth[i]) errors += 1.0;
    }
    double errorRate = errors / static_cast<double>(n);
    return errorRate > config_.driftThreshold ? errorRate : 0.0;
}

void AdaptiveLearningSystem::pruneOldSamples() {
    if (trainingBuffer_.size() <= config_.maxBufferSize) return;
    while (trainingBuffer_.size() > config_.minBufferSize) {
        trainingBuffer_.erase(trainingBuffer_.begin());
        metrics_.samplesPruned += 1;
        if (trainingBuffer_.size() <= config_.maxBufferSize) break;
    }
}

std::vector<LearningSample> AdaptiveLearningSystem::getTrainingBuffer() const {
    return trainingBuffer_;
}

AdaptationMetrics AdaptiveLearningSystem::getMetrics() const {
    return metrics_;
}

bool AdaptiveLearningSystem::shouldAdapt() const {
    return config_.enableOnlineLearning && !trainingBuffer_.empty();
}

bool AdaptiveLearningSystem::isConfigured() const { return configured_; }

double AdaptiveLearningSystem::computeDriftScore(const std::vector<double>& d1, const std::vector<double>& d2) const {
    if (d1.empty() || d2.empty()) return 0.0;
    std::size_t n = std::min(d1.size(), d2.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += std::abs(d1[i] - d2[i]);
    }
    return sum / static_cast<double>(n);
}

std::vector<double> AdaptiveLearningSystem::computeClassDistribution() const {
    std::vector<double> dist;
    for (const auto& s : trainingBuffer_) {
        if (static_cast<std::size_t>(s.label) >= dist.size()) dist.resize(s.label + 1, 0.0);
        dist[s.label] += 1.0;
    }
    if (std::accumulate(dist.begin(), dist.end(), 0.0) > 0) {
        double total = std::accumulate(dist.begin(), dist.end(), 0.0);
        for (auto& d : dist) d /= total;
    }
    return dist;
}

}
