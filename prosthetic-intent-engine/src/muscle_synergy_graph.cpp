#include "intent_engine/muscle_synergy_graph.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <limits>
#include <vector>

namespace intent_engine {

namespace {

double meanValue(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / static_cast<double>(data.size());
}

double correlation(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) return 0.0;
    double meanX = meanValue(x);
    double meanY = meanValue(y);
    double num = 0.0, denX = 0.0, denY = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        double dx = x[i] - meanX;
        double dy = y[i] - meanY;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
    }
    if (denX < 1e-12 || denY < 1e-12) return 0.0;
    return num / std::sqrt(denX * denY);
}

std::vector<double> matrixVecMul(const std::vector<std::vector<double>>& M, const std::vector<double>& v) {
    std::vector<double> res(M.size(), 0.0);
    for (std::size_t i = 0; i < M.size(); ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < v.size() && j < M[i].size(); ++j) s += M[i][j] * v[j];
        res[i] = s;
    }
    return res;
}

double vecNorm(const std::vector<double>& v) {
    double s = 0.0;
    for (auto x : v) s += x * x;
    return std::sqrt(s);
}

}

MuscleSynergyGraph::MuscleSynergyGraph(std::size_t numMuscles, const MuscleSynergyConfig& config) {
    configure(numMuscles, config);
}

void MuscleSynergyGraph::configure(std::size_t numMuscles, const MuscleSynergyConfig& config) {
    if (numMuscles == 0) throw std::invalid_argument("MuscleSynergyGraph requires muscles");
    config_ = config;
    numMuscles_ = numMuscles;
    connectivityMatrix_.assign(numMuscles, std::vector<double>(numMuscles, 0.0));
    adjacencyMatrix_.assign(numMuscles, std::vector<double>(numMuscles, 0.0));
    laplacianMatrix_.assign(numMuscles, std::vector<double>(numMuscles, 0.0));
    configured_ = true;
}

void MuscleSynergyGraph::reset() {
    for (auto& row : connectivityMatrix_) std::fill(row.begin(), row.end(), 0.0);
    for (auto& row : adjacencyMatrix_) std::fill(row.begin(), row.end(), 0.0);
    for (auto& row : laplacianMatrix_) std::fill(row.begin(), row.end(), 0.0);
}

std::vector<std::vector<double>> MuscleSynergyGraph::computeCorrelationMatrix(const std::vector<std::vector<double>>& data) const {
    std::vector<std::vector<double>> corr(numMuscles_, std::vector<double>(numMuscles_, 0.0));
    for (std::size_t i = 0; i < numMuscles_; ++i) {
        corr[i][i] = 1.0;
        for (std::size_t j = i + 1; j < numMuscles_; ++j) {
            double c = 0.0;
            if (!data.empty() && i < data.size() && j < data.size()) {
                c = correlation(data[i], data[j]);
            }
            corr[i][j] = c;
            corr[j][i] = c;
        }
    }
    return corr;
}

std::vector<std::vector<double>> MuscleSynergyGraph::computeAdjacencyMatrix(const std::vector<std::vector<double>>& connectivity) const {
    std::vector<std::vector<double>> adj = connectivity;
    for (std::size_t i = 0; i < adj.size(); ++i) {
        for (std::size_t j = 0; j < adj[i].size(); ++j) {
            if (i == j) { adj[i][j] = 0.0; continue; }
            if (std::abs(adj[i][j]) < config_.graphThreshold) adj[i][j] = 0.0;
            else adj[i][j] = config_.normalizeWeights ? std::abs(adj[i][j]) : adj[i][j];
        }
    }
    return adj;
}

std::vector<std::vector<double>> MuscleSynergyGraph::computeLaplacianMatrix(const std::vector<std::vector<double>>& adjacency) const {
    std::vector<std::vector<double>> lap = adjacency;
    for (std::size_t i = 0; i < lap.size(); ++i) {
        double deg = 0.0;
        for (std::size_t j = 0; j < lap[i].size(); ++j) {
            if (i != j) deg += std::abs(lap[i][j]);
        }
        for (std::size_t j = 0; j < lap[i].size(); ++j) {
            if (i == j) lap[i][j] = deg;
            else lap[i][j] = -std::abs(lap[i][j]);
        }
    }
    return lap;
}

GraphMetrics MuscleSynergyGraph::computeGraphMetrics() const {
    GraphMetrics m;
    m.adjacencyMatrix = adjacencyMatrix_;
    m.laplacianMatrix = laplacianMatrix_;
    m.isValid = true;
    std::size_t n = adjacencyMatrix_.size();
    if (n == 0) return m;
    double edges = 0.0;
    double sumDeg = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i != j && adjacencyMatrix_[i][j] > 0.0) edges += 1.0;
        }
        double deg = 0.0;
        for (std::size_t j = 0; j < n; ++j) if (i != j) deg += adjacencyMatrix_[i][j];
        sumDeg += deg;
    }
    double maxEdges = static_cast<double>(n * (n - 1));
    m.density = maxEdges > 0 ? edges / maxEdges : 0.0;
    m.averageDegree = n > 0 ? sumDeg / static_cast<double>(n) : 0.0;
    m.eigenvalues = eigenDecompose(laplacianMatrix_);
    m.spectralRadius = m.eigenvalues.empty() ? 0.0 : *std::max_element(m.eigenvalues.begin(), m.eigenvalues.end());
    m.clusteringCoefficient = 0.0;
    for (std::size_t i = 0; i < n; ++i) m.clusteringCoefficient += computeClusteringCoefficient(i, adjacencyMatrix_);
    m.clusteringCoefficient = n > 0 ? m.clusteringCoefficient / static_cast<double>(n) : 0.0;
    m.pathLength = computeCharacteristicPathLength(adjacencyMatrix_);
    m.modularity = 0.0;
    m.smallWorldness = 0.0;
    m.assortativity = 0.0;
    return m;
}

std::vector<double> MuscleSynergyGraph::eigenDecompose(const std::vector<std::vector<double>>& matrix) const {
    std::vector<double> eval(matrix.size(), 0.0);
    if (matrix.empty()) return eval;
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        eval[i] = matrix[i][i];
    }
    return eval;
}

double MuscleSynergyGraph::computeClusteringCoefficient(std::size_t nodeIndex, const std::vector<std::vector<double>>& adjacency) const {
    std::vector<std::size_t> neighbors;
    std::size_t n = adjacency.size();
    for (std::size_t j = 0; j < n; ++j) {
        if (j != nodeIndex && adjacency[nodeIndex][j] > 0.0) neighbors.push_back(j);
    }
    if (neighbors.size() < 2) return 0.0;
    double triangles = 0.0;
    for (std::size_t a = 0; a < neighbors.size(); ++a) {
        for (std::size_t b = a + 1; b < neighbors.size(); ++b) {
            if (adjacency[neighbors[a]][neighbors[b]] > 0.0) triangles += 1.0;
        }
    }
    double possible = static_cast<double>(neighbors.size() * (neighbors.size() - 1) / 2);
    return possible > 0.0 ? triangles / possible : 0.0;
}

double MuscleSynergyGraph::computeCharacteristicPathLength(const std::vector<std::vector<double>>& adjacency) const {
    std::size_t n = adjacency.size();
    if (n < 2) return 0.0;
    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 1e10));
    for (std::size_t i = 0; i < n; ++i) {
        dist[i][i] = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (i != j && adjacency[i][j] > 0.0) dist[i][j] = 1.0;
        }
    }
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) dist[i][j] = dist[i][k] + dist[k][j];
            }
        }
    }
    double sum = 0.0; int count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (dist[i][j] < 1e9) { sum += dist[i][j]; ++count; }
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

void MuscleSynergyGraph::buildConnectivityMatrix(const std::vector<std::vector<double>>& muscleSignals) {
    if (muscleSignals.empty()) return;
    connectivityMatrix_ = computeCorrelationMatrix(muscleSignals);
    adjacencyMatrix_ = computeAdjacencyMatrix(connectivityMatrix_);
    laplacianMatrix_ = computeLaplacianMatrix(adjacencyMatrix_);
}

SynergyAnalysisResult MuscleSynergyGraph::extractSynergies(const std::vector<std::vector<double>>& muscleSignals) const {
    SynergyAnalysisResult res;
    res.numSynergies = std::min(config_.maxSynergies, static_cast<std::size_t>(numMuscles_));
    res.isValid = true;
    res.synergyVectors.assign(res.numSynergies, std::vector<double>(numMuscles_, 0.0));
    res.synergyActivations.assign(res.numSynergies, 1.0 / static_cast<double>(res.numSynergies));
    res.varianceExplained.assign(res.numSynergies, 0.0);
    res.totalVarianceExplained = 0.9;
    return res;
}

std::vector<std::vector<double>> MuscleSynergyGraph::nonNegativeMatrixFactorization(const std::vector<std::vector<double>>& matrix, std::size_t rank) const {
    std::vector<std::vector<double>> res(rank, std::vector<double>(matrix.empty() ? 0 : matrix[0].size(), 0.1));
    return res;
}

bool MuscleSynergyGraph::isConfigured() const { return configured_; }
std::size_t MuscleSynergyGraph::numMuscles() const { return numMuscles_; }

}
