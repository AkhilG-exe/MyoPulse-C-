#include "intent_engine/uncertainty_estimation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace intent_engine {

UncertaintyEstimator::UncertaintyEstimator() {
    configure(UncertaintyConfig{});
}

UncertaintyEstimator::UncertaintyEstimator(const UncertaintyConfig& config) {
    configure(config);
}

void UncertaintyEstimator::configure(const UncertaintyConfig& config) {
    config_ = config;
    configured_ = true;
}

void UncertaintyEstimator::reset() {}

std::vector<double> UncertaintyEstimator::softmax(const std::vector<double>& scores) const {
    if (scores.empty()) return scores;
    double maxv = *std::max_element(scores.begin(), scores.end());
    double sum = 0.0;
    std::vector<double> probs(scores.size(), 0.0);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        probs[i] = std::exp((scores[i] - maxv) / config_.temperatureScaling);
        sum += probs[i];
    }
    if (sum < 1e-12) return probs;
    for (auto& p : probs) p /= sum;
    return probs;
}

double UncertaintyEstimator::computeEntropy(const std::vector<double>& probabilities) const {
    double ent = 0.0;
    for (auto p : probabilities) {
        if (p > 1e-12) ent -= p * std::log2(p);
    }
    return ent;
}

double UncertaintyEstimator::computeMargin(const std::vector<double>& probabilities) const {
    if (probabilities.size() < 2) return 0.0;
    std::vector<double> tmp = probabilities;
    std::partial_sort(tmp.begin(), tmp.begin() + 2, tmp.end(), std::greater<double>());
    return tmp[0] - tmp[1];
}

double UncertaintyEstimator::computeVariationRatio(const std::vector<double>& probabilities) const {
    if (probabilities.empty()) return 0.0;
    auto it = std::max_element(probabilities.begin(), probabilities.end());
    double mode = *it;
    return 1.0 - mode;
}

UncertaintyMetrics UncertaintyEstimator::estimateFromProbabilities(const std::vector<double>& probabilities) const {
    UncertaintyMetrics m;
    m.classProbabilities = probabilities;
    if (probabilities.empty()) return m;
    auto it = std::max_element(probabilities.begin(), probabilities.end());
    m.predictedClass = static_cast<int>(std::distance(probabilities.begin(), it));
    m.confidence = *it;
    m.entropy = computeEntropy(probabilities);
    m.margin = computeMargin(probabilities);
    m.variationRatio = computeVariationRatio(probabilities);
    m.epistemicUncertainty = 0.5;
    m.aleatoricUncertainty = m.entropy;
    m.totalUncertainty = m.epistemicUncertainty + m.aleatoricUncertainty;
    m.isHighUncertainty = m.confidence < config_.confidenceThreshold || m.totalUncertainty > config_.uncertaintyThreshold;
    return m;
}

UncertaintyMetrics UncertaintyEstimator::estimateFromScores(const std::vector<double>& decisionScores) const {
    return estimateFromProbabilities(softmax(decisionScores));
}

bool UncertaintyEstimator::rejectPrediction(const UncertaintyMetrics& metrics) const {
    return metrics.isHighUncertainty;
}

bool UncertaintyEstimator::isConfigured() const { return configured_; }

}
