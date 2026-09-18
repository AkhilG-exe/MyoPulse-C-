#include "intent_engine/intent_classifier.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "matrix_math.hpp"

namespace intent_engine {

namespace {

double kernelValue(const std::vector<double>& lhs, const std::vector<double>& rhs, KernelType kernel, double gamma, double degree, double coef0) {
    switch (kernel) {
        case KernelType::kLinear: {
            return detail::dotProduct(lhs, rhs);
        }
        case KernelType::kPolynomial: {
            return std::pow(gamma * detail::dotProduct(lhs, rhs) + coef0, degree);
        }
        case KernelType::kRbf: {
            return std::exp(-gamma * detail::squaredDistance(lhs, rhs));
        }
    }
    return 0.0;
}

std::vector<double> buildKernelMatrix(const std::vector<std::vector<double>>& samples, KernelType kernel, double gamma, double degree, double coef0) {
    const std::size_t size = samples.size();
    std::vector<double> kernelMatrix(size * size, 0.0);
    for (std::size_t i = 0; i < size; ++i) {
        for (std::size_t j = i; j < size; ++j) {
            const double value = kernelValue(samples[i], samples[j], kernel, gamma, degree, coef0);
            kernelMatrix[i * size + j] = value;
            kernelMatrix[j * size + i] = value;
        }
    }
    return kernelMatrix;
}

constexpr double kSolverTau = 1e-12;
constexpr double kInfinity = 1e30;
constexpr int kAlphaLower = 0;
constexpr int kAlphaFree = 1;
constexpr int kAlphaUpper = 2;

struct BinarySvmResult {
    std::vector<double> alpha;
    double rho = 0.0;
};

struct BinarySvmState {
    std::size_t size = 0;
    double c = 1.0;
    double eps = 1e-3;
    std::vector<int> y;
    std::vector<double> q;
    std::vector<double> qd;
    std::vector<double> gradient;
    std::vector<double> alpha;
    std::vector<int> status;
    std::size_t iterations = 0;

    BinarySvmState(const std::vector<double>& kernelMatrix, const std::vector<int>& labels, double cValue, double epsValue)
        : size(labels.size()), c(cValue), eps(epsValue), y(labels) {
        q.assign(size * size, 0.0);
        for (std::size_t i = 0; i < size; ++i) {
            for (std::size_t j = 0; j < size; ++j) {
                q[i * size + j] = static_cast<double>(y[i]) * static_cast<double>(y[j]) * kernelMatrix[i * size + j];
            }
        }
        qd.assign(size, 0.0);
        for (std::size_t i = 0; i < size; ++i) {
            qd[i] = q[i * size + i];
        }
        gradient.assign(size, -1.0);
        alpha.assign(size, 0.0);
        status.assign(size, kAlphaLower);
    }

    bool isUpper(std::size_t index) const {
        return status[index] == kAlphaUpper;
    }

    bool isLower(std::size_t index) const {
        return status[index] == kAlphaLower;
    }

    void updateStatus(std::size_t index) {
        if (alpha[index] >= c) {
            status[index] = kAlphaUpper;
        } else if (alpha[index] <= 0.0) {
            status[index] = kAlphaLower;
        } else {
            status[index] = kAlphaFree;
        }
    }

    bool selectWorkingSet(std::size_t& outI, std::size_t& outJ) {
        double gMax = -kInfinity;
        double gMax2 = -kInfinity;
        std::size_t gMaxIndex = 0;
        bool gMaxFound = false;
        bool gMinFound = false;
        std::size_t gMinIndex = 0;
        double objDiffMin = kInfinity;

        for (std::size_t t = 0; t < size; ++t) {
            if (y[t] == 1) {
                if (!isUpper(t)) {
                    const double candidate = -gradient[t];
                    if (candidate >= gMax) {
                        gMax = candidate;
                        gMaxIndex = t;
                        gMaxFound = true;
                    }
                }
            } else {
                if (!isLower(t)) {
                    const double candidate = gradient[t];
                    if (candidate >= gMax) {
                        gMax = candidate;
                        gMaxIndex = t;
                        gMaxFound = true;
                    }
                }
            }
        }

        if (!gMaxFound) {
            return true;
        }

        const std::size_t i = gMaxIndex;
        for (std::size_t j = 0; j < size; ++j) {
            if (y[j] == 1) {
                if (!isLower(j)) {
                    const double gradDiff = gMax + gradient[j];
                    if (gradient[j] >= gMax2) {
                        gMax2 = gradient[j];
                    }
                    if (gradDiff > 0.0) {
                        const double quadCoef = qd[i] + qd[j] - 2.0 * static_cast<double>(y[i]) * q[i * size + j];
                        const double denom = (quadCoef > 0.0) ? quadCoef : kSolverTau;
                        const double objDiff = -(gradDiff * gradDiff) / denom;
                        if (objDiff <= objDiffMin) {
                            objDiffMin = objDiff;
                            gMinIndex = j;
                            gMinFound = true;
                        }
                    }
                }
            } else {
                if (!isUpper(j)) {
                    const double gradDiff = gMax - gradient[j];
                    if (-gradient[j] >= gMax2) {
                        gMax2 = -gradient[j];
                    }
                    if (gradDiff > 0.0) {
                        const double quadCoef = qd[i] + qd[j] + 2.0 * static_cast<double>(y[i]) * q[i * size + j];
                        const double denom = (quadCoef > 0.0) ? quadCoef : kSolverTau;
                        const double objDiff = -(gradDiff * gradDiff) / denom;
                        if (objDiff <= objDiffMin) {
                            objDiffMin = objDiff;
                            gMinIndex = j;
                            gMinFound = true;
                        }
                    }
                }
            }
        }

        if (gMax + gMax2 < eps || !gMinFound) {
            return true;
        }
        outI = i;
        outJ = gMinIndex;
        return false;
    }

    void updateStep(std::size_t i, std::size_t j) {
        const double oldAlphaI = alpha[i];
        const double oldAlphaJ = alpha[j];
        const double* rowI = q.data() + i * size;
        const double* rowJ = q.data() + j * size;

        if (y[i] != y[j]) {
            double quadCoef = qd[i] + qd[j] + 2.0 * rowI[j];
            if (quadCoef <= 0.0) {
                quadCoef = kSolverTau;
            }
            const double delta = (-gradient[i] - gradient[j]) / quadCoef;
            const double diff = alpha[i] - alpha[j];
            alpha[i] += delta;
            alpha[j] += delta;

            if (diff > 0.0) {
                if (alpha[j] < 0.0) {
                    alpha[j] = 0.0;
                    alpha[i] = diff;
                }
            } else {
                if (alpha[i] < 0.0) {
                    alpha[i] = 0.0;
                    alpha[j] = -diff;
                }
            }
            if (diff > c - c) {
                if (alpha[i] > c) {
                    alpha[i] = c;
                    alpha[j] = c - diff;
                }
            } else {
                if (alpha[j] > c) {
                    alpha[j] = c;
                    alpha[i] = c + diff;
                }
            }
        } else {
            double quadCoef = qd[i] + qd[j] - 2.0 * rowI[j];
            if (quadCoef <= 0.0) {
                quadCoef = kSolverTau;
            }
            const double delta = (gradient[i] - gradient[j]) / quadCoef;
            const double total = alpha[i] + alpha[j];
            alpha[i] -= delta;
            alpha[j] += delta;

            if (total > c) {
                if (alpha[i] > c) {
                    alpha[i] = c;
                    alpha[j] = total - c;
                }
            } else {
                if (alpha[j] < 0.0) {
                    alpha[j] = 0.0;
                    alpha[i] = total;
                }
            }
            if (total > c) {
                if (alpha[j] > c) {
                    alpha[j] = c;
                    alpha[i] = total - c;
                }
            } else {
                if (alpha[i] < 0.0) {
                    alpha[i] = 0.0;
                    alpha[j] = total;
                }
            }
        }

        const double deltaAlphaI = alpha[i] - oldAlphaI;
        const double deltaAlphaJ = alpha[j] - oldAlphaJ;
        for (std::size_t k = 0; k < size; ++k) {
            gradient[k] += rowI[k] * deltaAlphaI + rowJ[k] * deltaAlphaJ;
        }

        updateStatus(i);
        updateStatus(j);
    }

    void solve() {
        std::size_t iter = 0;
        const std::size_t maxIterations = std::max<std::size_t>(10000000, 100 * size);
        while (iter < maxIterations) {
            std::size_t i = 0;
            std::size_t j = 0;
            if (selectWorkingSet(i, j)) {
                break;
            }
            ++iter;
            updateStep(i, j);
        }
        iterations = iter;
    }

    double rho() const {
        double ub = kInfinity;
        double lb = -kInfinity;
        double sumFree = 0.0;
        std::size_t nrFree = 0;
        for (std::size_t i = 0; i < size; ++i) {
            const double yG = static_cast<double>(y[i]) * gradient[i];
            if (isUpper(i)) {
                if (y[i] == -1) {
                    ub = std::min(ub, yG);
                } else {
                    lb = std::max(lb, yG);
                }
            } else if (isLower(i)) {
                if (y[i] == 1) {
                    ub = std::min(ub, yG);
                } else {
                    lb = std::max(lb, yG);
                }
            } else {
                ++nrFree;
                sumFree += yG;
            }
        }
        if (nrFree > 0) {
            return sumFree / static_cast<double>(nrFree);
        }
        return 0.5 * (ub + lb);
    }
};

BinarySvmResult trainBinarySvm(const std::vector<std::vector<double>>& samples, const std::vector<int>& labels, const SvmParameters& params) {
    const std::vector<double> kernelMatrix = buildKernelMatrix(samples, params.kernel, params.gamma, params.degree, params.coef0);

    BinarySvmState state(kernelMatrix, labels, params.c, params.tolerance);
    state.solve();

    BinarySvmResult result;
    result.alpha = std::move(state.alpha);
    result.rho = state.rho();
    return result;
}

std::vector<int> collectLabels(const std::vector<TrainingSample>& samples) {
    std::vector<int> labels;
    for (const auto& sample : samples) {
        if (std::find(labels.begin(), labels.end(), sample.label) == labels.end()) {
            labels.push_back(sample.label);
        }
    }
    std::sort(labels.begin(), labels.end());
    return labels;
}

std::size_t classIndexOf(const std::vector<int>& classLabels, int classLabel) {
    const auto found = std::find(classLabels.begin(), classLabels.end(), classLabel);
    if (found == classLabels.end()) {
        return classLabels.size();
    }
    return static_cast<std::size_t>(std::distance(classLabels.begin(), found));
}

BinarySvmModel trainPair(const std::vector<TrainingSample>& samples, int classA, int classB, const SvmParameters& params) {
    std::vector<std::vector<double>> pairSamples;
    std::vector<int> pairLabels;
    for (const auto& sample : samples) {
        if (sample.label == classA) {
            pairSamples.push_back(sample.features);
            pairLabels.push_back(1);
        } else if (sample.label == classB) {
            pairSamples.push_back(sample.features);
            pairLabels.push_back(-1);
        }
    }

    bool hasPositive = false;
    bool hasNegative = false;
    for (const int label : pairLabels) {
        hasPositive = hasPositive || label > 0;
        hasNegative = hasNegative || label < 0;
    }
    if (!hasPositive || !hasNegative) {
        throw std::invalid_argument("SVM class pair must contain both classes");
    }

    const BinarySvmResult trained = trainBinarySvm(pairSamples, pairLabels, params);

    BinarySvmModel model;
    model.classA = classA;
    model.classB = classB;
    model.bias = -trained.rho;
    for (std::size_t i = 0; i < pairLabels.size(); ++i) {
        if (trained.alpha[i] > 1e-6) {
            model.supportVectors.push_back(pairSamples[i]);
            model.coefficients.push_back(trained.alpha[i] * static_cast<double>(pairLabels[i]));
        }
    }
    return model;
}

}

const char* gestureClassName(GestureClass gesture) {
    switch (gesture) {
        case GestureClass::kRest:
            return "Rest";
        case GestureClass::kPowerGrip:
            return "Power Grip";
        case GestureClass::kPinch:
            return "Pinch";
        case GestureClass::kPoint:
            return "Point";
        default:
            return "Unknown";
    }
}

GestureClass gestureClassFromInt(int value) {
    switch (value) {
        case 0:
            return GestureClass::kRest;
        case 1:
            return GestureClass::kPowerGrip;
        case 2:
            return GestureClass::kPinch;
        case 3:
            return GestureClass::kPoint;
        default:
            return GestureClass::kUnknown;
    }
}

const char* kernelTypeName(KernelType kernel) {
    switch (kernel) {
        case KernelType::kLinear:
            return "linear";
        case KernelType::kRbf:
            return "rbf";
        case KernelType::kPolynomial:
            return "polynomial";
        default:
            return "unknown";
    }
}

void LdaClassifier::fit(const std::vector<TrainingSample>& samples) {
    if (samples.size() < 2) {
        throw std::invalid_argument("LDA fit requires at least two samples");
    }
    const std::size_t featureCount = samples.front().features.size();
    if (featureCount == 0) {
        throw std::invalid_argument("LDA fit features cannot be empty");
    }
    for (const auto& sample : samples) {
        if (sample.features.size() != featureCount) {
            throw std::invalid_argument("LDA fit feature dimension mismatch");
        }
    }

    model_.classLabels = collectLabels(samples);
    model_.numClasses = model_.classLabels.size();
    if (model_.numClasses < 2) {
        throw std::invalid_argument("LDA fit requires at least two unique class labels");
    }
    model_.numFeatures = featureCount;

    const std::size_t classCount = model_.numClasses;
    const double sampleCount = static_cast<double>(samples.size());
    std::vector<double> perClassCount(classCount, 0.0);
    std::vector<double> sums(classCount * featureCount, 0.0);

    for (const auto& sample : samples) {
        const std::size_t classIndex = classIndexOf(model_.classLabels, sample.label);
        if (classIndex == classCount) {
            throw std::invalid_argument("LDA fit encountered an unknown class label");
        }
        perClassCount[classIndex] += 1.0;
        for (std::size_t f = 0; f < featureCount; ++f) {
            sums[classIndex * featureCount + f] += sample.features[f];
        }
    }

    model_.classPriors.assign(classCount, 0.0);
    model_.classMeans.assign(classCount * featureCount, 0.0);
    for (std::size_t c = 0; c < classCount; ++c) {
        model_.classPriors[c] = perClassCount[c] / sampleCount;
        for (std::size_t f = 0; f < featureCount; ++f) {
            model_.classMeans[c * featureCount + f] = sums[c * featureCount + f] / perClassCount[c];
        }
    }

    std::vector<double> withinScatter(classCount * 0 + featureCount * featureCount, 0.0);
    for (const auto& sample : samples) {
        const std::size_t classIndex = classIndexOf(model_.classLabels, sample.label);
        std::vector<double> deviation(featureCount);
        for (std::size_t f = 0; f < featureCount; ++f) {
            deviation[f] = sample.features[f] - model_.classMeans[classIndex * featureCount + f];
        }
        detail::accumulateOuterProduct(withinScatter, 1.0, deviation, deviation, featureCount, featureCount);
    }

    double trace = 0.0;
    for (std::size_t f = 0; f < featureCount; ++f) {
        trace += withinScatter[f * featureCount + f];
    }
    const double ridge = std::max(1e-12, 1e-9 * trace / static_cast<double>(featureCount));
    for (std::size_t f = 0; f < featureCount; ++f) {
        withinScatter[f * featureCount + f] += ridge;
    }

    detail::invertMatrix(withinScatter, featureCount);
    model_.invWithinClassScatter = std::move(withinScatter);
    trained_ = true;
}

void LdaClassifier::load(const LdaModel& model) {
    if (model.numClasses == 0 || model.numFeatures == 0) {
        throw std::invalid_argument("LdaClassifier received an invalid model");
    }
    if (model.classLabels.size() != model.numClasses ||
        model.classPriors.size() != model.numClasses ||
        model.classMeans.size() != model.numClasses * model.numFeatures ||
        model.invWithinClassScatter.size() != model.numFeatures * model.numFeatures) {
        throw std::invalid_argument("LdaClassifier received mismatched model arrays");
    }
    model_ = model;
    trained_ = true;
}

int LdaClassifier::predict(const std::vector<double>& features) const {
    const std::vector<double>& scores = LdaClassifier::decisionScores(features);
    std::size_t bestClass = 0;
    for (std::size_t c = 1; c < scores.size(); ++c) {
        if (scores[c] > scores[bestClass]) {
            bestClass = c;
        }
    }
    return model_.classLabels[bestClass];
}

std::vector<double> LdaClassifier::decisionScores(const std::vector<double>& features) const {
    validateFeatures(features);
    const std::size_t featureCount = model_.numFeatures;
    const std::size_t classCount = model_.numClasses;
    std::vector<double> scores(classCount, 0.0);

    const std::vector<double> projectedFeatures = detail::matVectorMultiply(model_.invWithinClassScatter, featureCount, featureCount, features);
    for (std::size_t c = 0; c < classCount; ++c) {
        const double* mean = model_.classMeans.data() + c * featureCount;
        const std::vector<double> projectedMean = detail::matVectorMultiply(model_.invWithinClassScatter, featureCount, featureCount, std::vector<double>(mean, mean + featureCount));
        double quadratic = 0.0;
        for (std::size_t f = 0; f < featureCount; ++f) {
            quadratic += mean[f] * projectedMean[f];
        }
        scores[c] = std::log(model_.classPriors[c]) + detail::dotProduct(features, projectedMean) - 0.5 * quadratic;
    }
    return scores;
}

const LdaModel& LdaClassifier::model() const {
    return model_;
}

bool LdaClassifier::isTrained() const {
    return trained_;
}

std::size_t LdaClassifier::numClasses() const {
    return model_.numClasses;
}

std::size_t LdaClassifier::numFeatures() const {
    return model_.numFeatures;
}

void LdaClassifier::validateFeatures(const std::vector<double>& features) const {
    if (!trained_) {
        throw std::runtime_error("LdaClassifier is not trained");
    }
    if (features.size() != model_.numFeatures) {
        throw std::invalid_argument("LdaClassifier feature dimension mismatch");
    }
}

void SvmClassifier::fit(const std::vector<TrainingSample>& samples, const SvmParameters& params) {
    if (samples.size() < 2) {
        throw std::invalid_argument("SvmClassifier requires at least two training samples");
    }
    const std::size_t featureCount = samples.front().features.size();
    if (featureCount == 0) {
        throw std::invalid_argument("SvmClassifier features cannot be empty");
    }
    for (const auto& sample : samples) {
        if (sample.features.size() != featureCount) {
            throw std::invalid_argument("SvmClassifier feature dimension mismatch");
        }
    }

    model_.classLabels = collectLabels(samples);
    if (model_.classLabels.size() < 2) {
        throw std::invalid_argument("SvmClassifier requires at least two unique class labels");
    }

    SvmParameters effective = params;
    if (effective.c <= 0.0) {
        throw std::invalid_argument("SvmClassifier C must be positive");
    }
    if (effective.maxPasses <= 0) {
        throw std::invalid_argument("SvmClassifier maxPasses must be positive");
    }
    if (effective.gamma <= 0.0) {
        effective.gamma = 1.0 / static_cast<double>(featureCount);
    }

    model_.numFeatures = featureCount;
    model_.kernel = effective.kernel;
    model_.gamma = effective.gamma;
    model_.degree = effective.degree;
    model_.coef0 = effective.coef0;
    model_.classifiers.clear();

    for (std::size_t a = 0; a < model_.classLabels.size(); ++a) {
        for (std::size_t b = a + 1; b < model_.classLabels.size(); ++b) {
            model_.classifiers.push_back(trainPair(samples, model_.classLabels[a], model_.classLabels[b], effective));
        }
    }
    trained_ = true;
}

void SvmClassifier::load(const SvmModel& model) {
    if (model.numFeatures == 0 || model.classLabels.empty()) {
        throw std::invalid_argument("SvmClassifier received an invalid model");
    }
    for (const auto& classifier : model.classifiers) {
        if (classifier.coefficients.size() != classifier.supportVectors.size()) {
            throw std::invalid_argument("SvmClassifier received mismatched binary classifier arrays");
        }
        for (const auto& vector : classifier.supportVectors) {
            if (vector.size() != model.numFeatures) {
                throw std::invalid_argument("SvmClassifier received support vectors of wrong dimension");
            }
        }
    }
    if (model.kernel == KernelType::kRbf && model.gamma <= 0.0) {
        throw std::invalid_argument("SvmClassifier RBF model requires positive gamma");
    }
    model_ = model;
    trained_ = true;
}

int SvmClassifier::predict(const std::vector<double>& features) const {
    validateFeatures(features);
    const std::size_t classCount = model_.classLabels.size();
    std::vector<double> scores(classCount, 0.0);
    std::vector<int> votes(classCount, 0);

    for (const auto& classifier : model_.classifiers) {
        const std::size_t indexA = classIndexOf(model_.classLabels, classifier.classA);
        const std::size_t indexB = classIndexOf(model_.classLabels, classifier.classB);
        const double margin = binaryMargin(classifier, features);
        if (margin >= 0.0) {
            scores[indexA] += margin;
            ++votes[indexA];
            scores[indexB] -= margin;
        } else {
            scores[indexB] -= margin;
            ++votes[indexB];
            scores[indexA] += margin;
        }
    }

    std::size_t bestClass = 0;
    for (std::size_t c = 1; c < classCount; ++c) {
        if (votes[c] > votes[bestClass] || (votes[c] == votes[bestClass] && scores[c] > scores[bestClass])) {
            bestClass = c;
        }
    }
    return model_.classLabels[bestClass];
}

std::vector<double> SvmClassifier::decisionScores(const std::vector<double>& features) const {
    validateFeatures(features);
    const std::size_t classCount = model_.classLabels.size();
    std::vector<double> scores(classCount, 0.0);
    for (const auto& classifier : model_.classifiers) {
        const std::size_t indexA = classIndexOf(model_.classLabels, classifier.classA);
        const std::size_t indexB = classIndexOf(model_.classLabels, classifier.classB);
        const double margin = binaryMargin(classifier, features);
        scores[indexA] += margin;
        scores[indexB] -= margin;
    }
    return scores;
}

const SvmModel& SvmClassifier::model() const {
    return model_;
}

bool SvmClassifier::isTrained() const {
    return trained_;
}

std::size_t SvmClassifier::numClasses() const {
    return model_.classLabels.size();
}

std::size_t SvmClassifier::numFeatures() const {
    return model_.numFeatures;
}

void SvmClassifier::validateFeatures(const std::vector<double>& features) const {
    if (!trained_) {
        throw std::runtime_error("SvmClassifier is not trained");
    }
    if (features.size() != model_.numFeatures) {
        throw std::invalid_argument("SvmClassifier feature dimension mismatch");
    }
}

double SvmClassifier::binaryMargin(const BinarySvmModel& classifier, const std::vector<double>& features) const {
    double margin = classifier.bias;
    for (std::size_t i = 0; i < classifier.coefficients.size(); ++i) {
        margin += classifier.coefficients[i] * kernelValue(classifier.supportVectors[i], features, model_.kernel, model_.gamma, model_.degree, model_.coef0);
    }
    return margin;
}

}