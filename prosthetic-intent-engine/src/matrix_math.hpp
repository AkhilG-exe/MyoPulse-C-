#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace intent_engine {

namespace detail {

inline double dotProduct(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("dotProduct dimension mismatch");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

inline double squaredDistance(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("squaredDistance dimension mismatch");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        const double delta = lhs[i] - rhs[i];
        sum += delta * delta;
    }
    return sum;
}

inline void accumulateOuterProduct(std::vector<double>& matrix, double scale, const std::vector<double>& lhs, const std::vector<double>& rhs, std::size_t rows, std::size_t cols) {
    if (matrix.size() != rows * cols || lhs.size() != rows || rhs.size() != cols) {
        throw std::invalid_argument("accumulateOuterProduct dimension mismatch");
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            matrix[row * cols + col] += scale * lhs[row] * rhs[col];
        }
    }
}

inline std::vector<double> matVectorMultiply(const std::vector<double>& matrix, std::size_t rows, std::size_t cols, const std::vector<double>& vector) {
    if (matrix.size() != rows * cols) {
        throw std::invalid_argument("matVectorMultiply matrix size mismatch");
    }
    if (vector.size() != cols) {
        throw std::invalid_argument("matVectorMultiply vector size mismatch");
    }
    std::vector<double> output(rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            sum += matrix[row * cols + col] * vector[col];
        }
        output[row] = sum;
    }
    return output;
}

inline void invertMatrix(std::vector<double>& matrix, std::size_t n) {
    if (matrix.size() != n * n) {
        throw std::invalid_argument("invertMatrix size mismatch");
    }
    const double singularityThreshold = 1e-14;
    std::vector<double> inverse(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        inverse[i * n + i] = 1.0;
    }
    for (std::size_t pivot = 0; pivot < n; ++pivot) {
        std::size_t bestRow = pivot;
        double bestValue = std::fabs(matrix[pivot * n + pivot]);
        for (std::size_t row = pivot + 1; row < n; ++row) {
            const double candidate = std::fabs(matrix[row * n + pivot]);
            if (candidate > bestValue) {
                bestValue = candidate;
                bestRow = row;
            }
        }
        if (bestValue < singularityThreshold) {
            throw std::runtime_error("matrix is singular and cannot be inverted");
        }
        if (bestRow != pivot) {
            for (std::size_t col = 0; col < n; ++col) {
                std::swap(matrix[bestRow * n + col], matrix[pivot * n + col]);
                std::swap(inverse[bestRow * n + col], inverse[pivot * n + col]);
            }
        }
        const double diagonal = matrix[pivot * n + pivot];
        for (std::size_t col = 0; col < n; ++col) {
            matrix[pivot * n + col] /= diagonal;
            inverse[pivot * n + col] /= diagonal;
        }
        matrix[pivot * n + pivot] = 1.0;
        for (std::size_t row = 0; row < n; ++row) {
            if (row == pivot) {
                continue;
            }
            const double factor = matrix[row * n + pivot];
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t col = 0; col < n; ++col) {
                matrix[row * n + col] -= factor * matrix[pivot * n + col];
                inverse[row * n + col] -= factor * inverse[pivot * n + col];
            }
        }
    }
    matrix.swap(inverse);
}

}
}