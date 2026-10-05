#include "intent_engine/realtime_graph.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace intent_engine {

RealtimeGraphNetwork::RealtimeGraphNetwork(std::size_t numNodes, const RealtimeGraphConfig& config) {
    configure(numNodes, config);
}

void RealtimeGraphNetwork::configure(std::size_t numNodes, const RealtimeGraphConfig& config) {
    config_ = config;
    nodeActivations_.assign(numNodes, 0.0);
    state_.nodes.assign(numNodes, GraphNode());
    for (std::size_t i = 0; i < numNodes; ++i) state_.nodes[i].id = i;
    state_.adjacency.assign(numNodes, std::vector<double>(numNodes, 0.0));
    for (std::size_t i = 0; i < numNodes; ++i) state_.adjacency[i][i] = 0.0;
    configured_ = true;
}

void RealtimeGraphNetwork::reset() {
    std::fill(nodeActivations_.begin(), nodeActivations_.end(), 0.0);
}

void RealtimeGraphNetwork::updateNode(std::size_t nodeId, const std::vector<double>& features) {
    if (nodeId >= state_.nodes.size()) return;
    state_.nodes[nodeId].features = features;
    if (!features.empty()) {
        double sum = 0.0;
        for (auto f : features) sum += std::abs(f);
        nodeActivations_[nodeId] = std::tanh(sum / static_cast<double>(features.size()));
    }
}

void RealtimeGraphNetwork::updateEdgeWeights() {
    std::size_t n = state_.adjacency.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (config_.enableDynamicTopology) {
                double weight = nodeActivations_[i] * nodeActivations_[j];
                state_.adjacency[i][j] = weight * config_.learningRate;
                state_.adjacency[j][i] = state_.adjacency[i][j];
            }
        }
    }
}

void RealtimeGraphNetwork::propagate() {
    std::vector<double> newActs = nodeActivations_;
    std::size_t n = state_.adjacency.size();
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (i != j) sum += state_.adjacency[i][j] * nodeActivations_[j];
        }
        newActs[i] = std::tanh(nodeActivations_[i] + sum);
    }
    nodeActivations_ = newActs;
}

RealtimeGraphState RealtimeGraphNetwork::getState() const { return state_; }
std::vector<double> RealtimeGraphNetwork::getNodeActivations() const { return nodeActivations_; }
bool RealtimeGraphNetwork::isConfigured() const { return configured_; }

}
