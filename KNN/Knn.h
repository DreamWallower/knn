//
// Knn.h
//
//      Author: Jachin Fang.
//
// The Pattern Recognition Library <Knn.h> header.
// K-nearest-neighbor(kNN) classification is one of the most fundamental and simple classification methods
// and should be one of the first choices for a classification study  when there is little or no prior knowledge about the distribution of the data.
//
//
// How to use:
//		For example, here is some data:
//			vector<double> data{ 1, 101, 5, 89, 108, 5, 115, 8 };
//			vector<string> label{ "A", "A", "B", "B" };
//          unsigned int dim = 2;
//          unsigned int size = 4;
//
//		And here is test data:
//			vector<double> test{ 10, 202 };
//
//		Then,
//      	Knn<double, string> knn;
//			knn.init(data, dim, label, size);
//			string result = knn.classify(test)[1]; /* 1-NN */
//			string result2 = knn.classify(test)[3]; /* 3-NN */
//
//		Or,
//      	Knn<double, string> knn;
//			knn.init(data, dim, label, size);
//			knn.classify(test);
//			string result = knn[1]; /* 1-NN */
//			string result2 = knn[3]; /* 3-NN */
//
#pragma once

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

////////////////////////////////////////////////////////////////
// The namespace `KNN`
// In general, it should not be used or modified externally.
namespace KNN {

////////////////////////////////////////////////////////////////
// Data structure for KNN classifier.
template <typename data_type, typename label_type>
struct Data {
    using stdVectorData = std::vector<data_type>;
    stdVectorData m_data;
    label_type m_label;

    /**
	 * Constructor.
	 */
    Data(const data_type* data, const unsigned int dim, label_type label) {
        m_data = stdVectorData(data, data + dim);
        m_label = label;
    }

    /**
	 * Calculated Euclidean distance.
	 */
    double EuclideanDistance(const data_type* test) {
        double result = 0;
        for (int i = m_data.size(); i--;)
            result += (m_data[i] - test[i]) * (m_data[i] - test[i]);
        return std::sqrt(result);
    }
};

////////////////////////////////////////////////////////////////
// STRUCT TEMPLATE greater.
template <typename label_type = int>
struct KnnLabelGreater {
    using stdPair = std::pair<double, label_type>;
    constexpr bool operator()(const stdPair& left, const stdPair& right) const {
        return left.first > right.first;
    }
};
} // namespace KNN

////////////////////////////////////////////////////////////////
// KNN classifier.
template <typename data_type, typename label_type = int>
class Knn {
    using KnnComparator = KNN::KnnLabelGreater<label_type>;
    using KnnData = KNN::Data<data_type, label_type>;
    using stdVectorKnnData = std::vector<KnnData>;
    using stdVectorData = std::vector<data_type>;
    using stdVectorLabel = std::vector<label_type>;
    using stdPair = std::pair<double, label_type>;
    using stdVectorPair = std::vector<stdPair>;
    using stdHashMap = std::unordered_map<label_type, int>;

public:
    Knn() : m_testData(nullptr){};       /* Constructor. */
    ~Knn() = default;                    /* Destructor. */
    Knn(const Knn&) = delete;            /* Deleted the copy constructor. */
    Knn& operator=(const Knn&) = delete; /* Deleted the copy assignment operator. */

    /**
	 * Overloaded the operator `[]`.
	 * Finded K-nearest-neighbor and decided the label of `m_testData`.
	 */
    label_type operator[](const unsigned int K) {
        if (K <= m_dataSet.size()) {
            stdVectorPair distance;
            for (KnnData& data : m_dataSet)
                distance.push_back(std::make_pair(data.EuclideanDistance(m_testData), data.m_label));

            std::make_heap(distance.begin(), distance.end(), m_greater);
            if (K == 1) { // 1nn
                return distance.front().second;
            } else { // knn
                stdVectorLabel result;
                result.reserve(K);
                for (int i = K; i--;) {
                    std::pop_heap(distance.begin(), distance.end(), m_greater);
                    result.push_back(distance.back().second);
                    distance.pop_back();
                }
                return neighborVote(result);
            }
        }
        return label_type();
    }

    /**
	 * Input the data to be classified.
	 */
    Knn<data_type, label_type>& classify(const data_type* data) {
        this->m_testData = data;
        return *this;
    }

    Knn<data_type, label_type>& classify(const stdVectorData& data) {
        return classify(data.data());
    }

    /**
	 * Loading all data.
	 *
	 *      /▔▔▔▔▔▔▔▔▔▔   size   ▔▔▔▔▔▔▔▔▔▔\
	 *      +--------+--------+--------+--------
	 *      | data A | data B | data C | ......
	 *      +--------+--------+--------+--------
	 *      \_ dim _/
	 *
	 * @note  `sizeof(data) / sizeof(data_type) == dim * size`
	 *        `sizeof(label) / sizeof(label_type) == size`
	 */
    void init(const data_type* data, unsigned int dim, const label_type* label, unsigned int size) {
        if (data && dim && label && size) {
            m_dataSet.clear();
            m_dataSet.reserve(size);
            for (unsigned int i = 0, idx = 0; i < size; ++i, idx += dim)
                m_dataSet.push_back(KnnData(data + idx, dim, label[i]));
        }
    }

    void init(const stdVectorData& data, unsigned int dim, const stdVectorLabel& label, unsigned int size) {
        init(data.data(), dim, label.data(), size);
    }

private:
    /**
	 * Voted for result by finding the majority labels.
	 */
    label_type neighborVote(stdVectorLabel& labels) {
        stdHashMap voter;
        label_type label;
        int count = 0;
        for (auto& res : labels) {
            auto got = voter.find(res);
            if (got == voter.end())
                voter.insert(std::make_pair(res, 1));
            else
                (got->second)++;
        }
        for (auto& v : voter) {
            if (v.second > count) {
                count = v.second;
                label = v.first;
            }
        }
        return label;
    }

private:
    const data_type* m_testData;
    stdVectorKnnData m_dataSet;
    KnnComparator m_greater;
};
