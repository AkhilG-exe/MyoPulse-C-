#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace intent_engine {

enum class GestureClass : std::uint8_t {
    kRest = 0,
    kPowerGrip = 1,
    kPinch = 2,
    kPoint = 3,
    kUnknown = 255,
};

const char* gestureClassName(GestureClass gesture);

GestureClass gestureClassFromInt(int value);

struct TrainingSample {
    std::vector<double> features;
    int label = 0;
};

struct LdaModel {
    std::size_t numFeatures = 0;
    std::size_t numClasses = 0;
    std::vector<int> classLabels;
    std::vector<double> classPriors;
    std::vector<double> classMeans;
    std::vector<double> invWithinClassScatter;
};

class LdaClassifier {
public:
    void fit(const std::vector<TrainingSample>& samples);

    void load(const LdaModel& model);

    int predict(const std::vector<double>& features) const;

    std::vector<double> decisionScores(const std::vector<double>& features) const;

    const LdaModel& model() const;

    bool isTrained() const;

    std::size_t numClasses() const;

    std::size_t numFeatures() const;

private:
    void validateFeatures(const std::vector<double>& features) const;

    LdaModel model_{};
    bool trained_ = false;
};

enum class KernelType : std::uint8_t {
    kLinear = 0,
    kRbf = 1,
    kPolynomial = 2,
};

const char* kernelTypeName(KernelType kernel);

struct SvmParameters {
    KernelType kernel = KernelType::kLinear;
    double c = 1.0;
    double gamma = 0.0;
    double degree = 2.0;
    double coef0 = 1.0;
    double tolerance = 1e-3;
    int maxPasses = 60;
};

struct BinarySvmModel {
    int classA = 0;
    int classB = 0;
    double bias = 0.0;
    std::vector<double> coefficients;
    std::vector<std::vector<double>> supportVectors;
};

struct SvmModel {
    std::size_t numFeatures = 0;
    std::vector<int> classLabels;
    std::vector<BinarySvmModel> classifiers;
    KernelType kernel = KernelType::kLinear;
    double gamma = 0.0;
    double degree = 2.0;
    double coef0 = 1.0;
};

class SvmClassifier {
public:
    void fit(const std::vector<TrainingSample>& samples, const SvmParameters& params = SvmParameters{});

    void load(const SvmModel& model);

    int predict(const std::vector<double>& features) const;

    std::vector<double> decisionScores(const std::vector<double>& features) const;

    const SvmModel& model() const;

    bool isTrained() const;

    std::size_t numClasses() const;

    std::size_t numFeatures() const;

private:
    void validateFeatures(const std::vector<double>& features) const;

    double binaryMargin(const BinarySvmModel& classifier, const std::vector<double>& features) const;

    SvmModel model_{};
    bool trained_ = false;
};

}