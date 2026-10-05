#pragma once

#include <cstddef>
#include <vector>

namespace intent_engine {

struct GraphNode {
    std::size_t id = 0;
    std::vector<double> features;
    double activation = 0.0;
};

struct GraphEdge {
    std::size_t sourceId = 0;
    std::size_t targetId = 0;
    double weight = 0.0;
    double delay = 0.0;
};

struct RealtimeGraphState {
    std::vector<GraphNode> nodes;
    std::vector<GraphEdge> edges;
    std::vector<std::vector<double>> adjacency;
    bool updated = false;
};

struct RealtimeGraphConfig {
    double learningRate = 0.01;
    double decayFactor = 0.95;
    double sparsity = 0.3;
    std::size_t maxNodes = 32;
    bool enableDynamicTopology = true;
    bool enableTemporalMemory = true;
};

class RealtimeGraphNetwork {
public:
    RealtimeGraphNetwork() = default;
    RealtimeGraphNetwork(std::size_t numNodes, const RealtimeGraphConfig& config = RealtimeGraphConfig{});

    void configure(std::size_t numNodes, const RealtimeGraphConfig& config);
    void reset();
    void updateNode(std::size_t nodeId, const std::vector<double>& features);
    void updateEdgeWeights();
    void propagate();
    RealtimeGraphState getState() const;
    std::vector<double> getNodeActivations() const;
    bool isConfigured() const;

private:
    RealtimeGraphState state_;
    RealtimeGraphConfig config_;
    std::vector<double> nodeActivations_;
    bool configured_ = false;
};

}
