//
// Pca.h
//
//      Author: Jachin Fang.
//
// The Data Reduction Library <Pca.h> header.
// Principal Component Analysis (PCA) is a statistical procedure that orthogonally transforms
// the original `N` coordinates of a data set into a new set of `K` coordinates called principal components. (K < N)
//
//
// How to use:
//      For example, here is some data:
//      	vector<double> data{10.2352, 11.3220, 10.1223, 11.8110, 9.1902, 8.9049, 9.3064, 9.8474, 8.3301, 8.3404, 10.1528, 10.1235, 10.4085, 10.8220, 9.0036, 10.0392, 9.5349, 10.0970, 9.4982, 10.8254};
//          unsigned int dim = 2;
//          unsigned int size = 10;
//
//      then,
//          Pca<float> pca;
//          vector<double> result = pca.reduce(data, dim, size)[1]; // transforms `2` coordinates of a data into a new set of `1` coordinates.
//
//      Or
//          Pca<float> pca;
//          pca.reduce(data, dim, size);
//          vector<double> result = pca[1]; // transforms `2` coordinates of a data into a new set of `1` coordinates.
//
#pragma once

#include <Eigen/Core>
#include <Eigen/SVD>
#include <vector>

template <typename data_type>
class Pca {
    using stdVectorData = std::vector<data_type>;
    using eigenMatrix = Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic>;
    using eigenVector = Eigen::Matrix<data_type, Eigen::Dynamic, 1>;
    using eigenMap = Eigen::Map<const eigenMatrix>;
    using eigenJacobiSVD = Eigen::JacobiSVD<eigenMatrix>;

public:
    Pca() = default;                     /* Constructor. */
    ~Pca() = default;                    /* Destructor. */
    Pca(const Pca&) = delete;            /* Deleted the copy constructor. */
    Pca& operator=(const Pca&) = delete; /* Deleted the copy assignment operator. */

    /**
     * Overloaded the operator `[]`.
     * Used PCA to reduce the data dimension to `K` by using SVD.
     */
    stdVectorData operator[](const unsigned int K) {
        // Singular Value Decomposition (SVD).
        eigenJacobiSVD svd(m_dataSet, Eigen::ComputeThinU);
        eigenMatrix U = svd.matrixU().transpose();

        // Adjusts the rows of U that are largest in absolute value are always positive.
        int cols, rows = U.rows();
        for (int i = rows; i--;) {
            U.row(i).cwiseAbs().maxCoeff(&cols);
            if (U(i, cols) < 0)
                U.row(i).array() *= -1;
        }

        // Result.
        eigenMatrix result = U.block(0, 0, K, U.cols()) * m_dataSet;
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
            m_dataSet = eigenMap(data, row, col);

            // Centralization.
            eigenVector means = m_dataSet.rowwise().mean();
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
    eigenMatrix m_dataSet;
};