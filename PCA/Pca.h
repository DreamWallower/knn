//
// Pca.h
//
//      Author: Jachin Fang.
//
// The Data Reduction Library <Pca.h> header.
// Principal Component Analysis (PCA) is a statistical procedure that orthogonally transforms
// the original `N` coordinates of a data set into a new set of `K` coordinates called principal components. (K < N)
//
#pragma once

#include <Eigen/Core>
#include <Eigen/SVD>
#include <vector>

template <typename data_type>
class Pca {
    using stdVectorData = std::vector<data_type>;
    using eigMatrix = Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic>;
    using eigVector = Eigen::Matrix<data_type, Eigen::Dynamic, 1>;
    using eigMap = Eigen::Map<const eigMatrix>;
    using eigJacobiSVD = Eigen::JacobiSVD<eigMatrix>;

public:
    Pca() = default;                     /* Constructor. */
    ~Pca() = default;                    /* Destructor. */
    Pca(const Pca&) = delete;            /* Deleted the copy constructor. */
    Pca& operator=(const Pca&) = delete; /* Deleted the copy assignment operator. */

    /**
     * Overloaded the operator `[]`.
     * Used PCA to reduce the data dimension to `K` by using SVD.
     */
    stdVectorData operator[](int K) {
        // Singular Value Decomposition (SVD).
        eigJacobiSVD svd(m_dataSet, Eigen::ComputeThinU);
        eigMatrix U = svd.matrixU().transpose();

        // Adjusts the rows of U that are largest in absolute value are always positive.
        int cols, rows = U.rows();
        for (int i = rows; i--;) {
            U.row(i).cwiseAbs().maxCoeff(&cols);
            if (U(i, cols) < 0)
                U.row(i).array() *= -1;
        }

        // Make sure that K is valid.
        if (0 >= K && K >= U.cols()) {
            K = U.cols() - 1;
        }

        // Result.
        eigMatrix result = U.block(0, 0, K, U.cols()) * m_dataSet;
        return stdVectorData(result.data(), result.data() + result.size());
    }

    /**
     * Loading data, and then centralize.
     */
    Pca& reduce(const data_type* data, unsigned int dim, unsigned int size) {
        if (data && dim && size) {
            // Number of rows and columns of data.
            unsigned int& row = dim;
            unsigned int& col = size;

            // Copy.
            m_dataSet = eigMap(data, row, col);

            // Centralization.
            eigVector means = m_dataSet.rowwise().mean();
            m_dataSet.colwise() -= means;
        }
        return *this;
    }

    /**
     * Loading data, and then centralize.
     */
    Pca& reduce(const stdVectorData& data, unsigned int dim, unsigned int size) {
        return reduce(data.data(), dim, size);
    }

private:
    eigMatrix m_dataSet;
};