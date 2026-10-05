#pragma once

#include <cstddef>
#include <vector>
#include <string>

namespace intent_engine {

struct AdjacencyEntry {
    std::size_t sourceIndex = 0;
    std::size_t targetIndex = 0;
    double weight = 0.0;
};

struct GraphMetrics {
    double density = 0.0;
    double averageDegree = 0.0;
    double modularity = 0.0;
    double clusteringCoefficient = 0.0;
    double pathLength = 0.0;
    double spectralRadius = 0.0;
    double smallWorldness = 0.0;
    double assortativity = 0.0;
    std::vector<double> eigenvalues;
    std::vector<std::vector<double>> adjacencyMatrix;
    std::vector<std::vector<double>> laplacianMatrix;
    bool isValid = false;
};

struct SynergyAnalysisResult {
    std::vector<std::vector<double>> synergyVectors;
    std::vector<double> synergyActivations;
    std::vector<double> varianceExplained;
    std::size_t numSynergies = 0;
    double totalVarianceExplained = 0.0;
    bool isValid = false;
};

struct MuscleSynergyConfig {
    double correlationThreshold = 0.6;
    double coherenceThreshold = 0.5;
    double graphThreshold = 0.3;
    std::size_t maxSynergies = 8;
    double varianceThreshold = 0.9;
    bool useCoherence = false;
    bool normalizeWeights = true;
};

class MuscleSynergyGraph {
public:
    MuscleSynergyGraph() = default;
    MuscleSynergyGraph(std::size_t numMuscles, const MuscleSynergyConfig& config = MuscleSynergyConfig{});

    void configure(std::size_t numMuscles, const MuscleSynergyConfig& config);
    void reset();
    void update(const std::vector<double>& muscleActivations);
    void buildConnectivityMatrix(const std::vector<std::vector<double>>& muscleSignals);
    std::vector<std::vector<double>> computeCorrelationMatrix(const std::vector<std::vector<double>>& data) const;
    std::vector<std::vector<double>> computeCoherenceMatrix(const std::vector<std::vector<double>>& data) const;
    std::vector<std::vector<double>> computeAdjacencyMatrix(const std::vector<std::vector<double>>& connectivity) const;
    std::vector<std::vector<double>> computeLaplacianMatrix(const std::vector<std::vector<double>>& adjacency) const;
    std::vector<double> eigenDecompose(const std::vector<std::vector<double>>& matrix) const;
    GraphMetrics computeGraphMetrics() const;
    SynergyAnalysisResult extractSynergies(const std::vector<std::vector<double>>& muscleSignals) const;
    std::vector<std::vector<double>> nonNegativeMatrixFactorization(const std::vector<std::vector<double>>& matrix, std::size_t rank) const;
    std::vector<std::vector<double>> principalComponentAnalysis(const std::vector<std::vector<double>>& matrix, std::size_t components) const;
    double computeClusteringCoefficient(std::size_t nodeIndex, const std::vector<std::vector<double>>& adjacency) const;
    double computeCharacteristicPathLength(const std::vector<std::vector<double>>& adjacency) const;
    bool isConfigured() const;
    std::size_t numMuscles() const;

private:
    std::vector<std::vector<double>> connectivityMatrix_;
    std::vector<std::vector<double>> adjacencyMatrix_;
    std::vector<std::vector<double>> laplacianMatrix_;
    MuscleSynergyConfig config_;
    std::size_t numMuscles_ = 0;
    bool configured_ = false;
};

}
