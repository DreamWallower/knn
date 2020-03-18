//
// Lda.h
//
//      Author: Jachin Fang.
//
// The Data Reduction Library <Lda.h> header.
// Linear Discriminant Analysis (LDA) is most commonly used as dimensionality reduction technique in the pre-processing step for
// pattern-classification and machine learning applications. The goal is to project a dataset onto a lower-dimensional space
// with good class-separability in order avoid overfitting (“curse of dimensionality”) and also reduce computational costs.
//
#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <algorithm>

template <typename data_type, typename label_type>
class Lda {
    using eigMatrix = Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic>;
    using eigMatrixSet = std::vector<eigMatrix>;
    using eigVector = Eigen::Matrix<data_type, Eigen::Dynamic, 1>;
    using eigVectorSet = std::vector<eigVector>;
    using eigMap = Eigen::Map<const eigMatrix>;
    using eigEigenSolver = Eigen::EigenSolver<eigMatrix>;
    using stdTupleEigen = std::tuple<data_type, eigMatrix>;
    using stdVectorData = std::vector<data_type>;
    using stdVectorLabel = std::vector<label_type>;
    using stdVectorTuple = std::vector<stdTupleEigen>;
    using stdHashMap = std::unordered_map<label_type, stdVectorData>;

public:
    Lda() = default;                     /* Constructor. */
    ~Lda() = default;                    /* Destructor. */
    Lda(const Lda&) = delete;            /* Deleted the copy constructor. */
    Lda& operator=(const Lda&) = delete; /* Deleted the copy assignment operator. */

    /**
     * Overloaded the operator `[]`.
     * Used LDA to reduce the data dimension to `K`.
     */
    stdVectorData operator[](int K) {

        int classNum = static_cast<int>(m_labelSet.size());
        int dimNum = static_cast<int>(m_dataSet[0].rows());

        // calculate class means.
        eigVector totalMean = eigVector::Zero(dimNum);
        unsigned int totalNum = 0;
        eigVectorSet means;
        means.reserve(classNum);
        for (int i = 0; i < classNum; ++i) {
            eigVector sum = m_dataSet[i].rowwise().sum();
            means.push_back(sum / m_dataSet[i].cols());
            totalNum += static_cast<int>(m_dataSet[i].cols());
            totalMean += sum;
        }
        totalMean /= totalNum;

        // calculate within-classes scatter.
        eigMatrix Sw = eigMatrix::Zero(dimNum, dimNum);
        for (int i = 0; i < classNum; ++i) {
            eigMatrix temp = m_dataSet[i].colwise() - means[i];
            Sw += temp * temp.transpose() / static_cast<double>(temp.cols() - 1);
        }

        // calculate between-classes scatter.
        eigMatrix Sb = eigMatrix::Zero(dimNum, dimNum);
        for (int i = 0; i < classNum; ++i) {
            eigVector temp = means[i] - totalMean;
            Sb += temp * temp.transpose();
        }

        // W = inv(Sw) * Sb
        eigMatrix W = Sw.inverse() * Sb;

        // calculate eigenvalues and eigenvectors.
        eigEigenSolver solver(W);
        eigVector eigenValues = solver.pseudoEigenvalueMatrix().diagonal();
        eigMatrix eigenVectors = solver.pseudoEigenvectors();
        sortEigenVectorByValues(eigenValues, eigenVectors);

        // Make sure that K is valid.
        if (0 >= K && K >= dimNum) {
            K = dimNum - 1;
        }

        eigMatrix subVectors = eigenVectors.block(0, 0, eigenVectors.rows(), K);
        stdVectorData result;
        for (int i = 0; i < classNum; ++i) {
            eigMatrix projected = subVectors.transpose() * m_dataSet[i];
            result.insert(result.end(), projected.data(), projected.data() + projected.size());
        }
        return result;
    }

    /**
     * Loading data.
     */
    Lda& reduce(const data_type* data, unsigned int dim, const label_type* label, unsigned int size) {
        if (data && dim && label && size) {
            const data_type* pData;
            stdHashMap dataMapper;
            unsigned int i;

            for (i = 0, pData = data; i < size; ++i, pData += dim) {
                stdVectorData& subSet = dataMapper[label[i]];
                subSet.insert(subSet.end(), pData, pData + dim);
            }

            m_dataSet.clear();
            m_labelSet.clear();
            m_dataSet.reserve(dataMapper.size());
            m_labelSet.reserve(dataMapper.size());
            // c++17, Structured binding declaration.
            for (auto& [key, value] : dataMapper) {
                eigMatrix mat = eigMap(value.data(), dim, value.size() / dim);
                m_dataSet.push_back(eigMap(value.data(), dim, value.size() / dim));
                m_labelSet.push_back(key);
            }
        }
        return *this;
    }

    /**
     * Loading data.
     */
    Lda& reduce(const stdVectorData& data, unsigned int dim, const stdVectorLabel& label, unsigned int size) {
        return reduce(data.data(), dim, label.data(), size);
    }

private:
    /**
     * Sort eigen vector by eigen value.
     */
    void sortEigenVectorByValues(eigVector& eigenValues, eigMatrix& eigenVectors) {
        stdVectorTuple eigenValueAndVector;
        int size = static_cast<int>(eigenValues.size());

        eigenValueAndVector.reserve(size);
        for (int i = 0; i < size; ++i)
            eigenValueAndVector.push_back(stdTupleEigen(eigenValues[i], eigenVectors.col(i)));

        std::sort(eigenValueAndVector.begin(), eigenValueAndVector.end(),
                  [&](const stdTupleEigen& a, const stdTupleEigen& b) -> bool {
                      return std::get<0>(a) > std::get<0>(b);
                  });

        for (int i = 0; i < size; ++i) {
            //eigenValues[i] = std::get<0>(eigenValueAndVector[i]);
            eigenVectors.col(i).swap(std::get<1>(eigenValueAndVector[i]));
        }
    }

    eigMatrixSet m_dataSet;
    stdVectorLabel m_labelSet;
};