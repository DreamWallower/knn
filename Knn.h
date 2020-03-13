//
// Knn.h
//
//      Author: Jachin Fang.
//
// The Pattern Recognition Library <Knn.h> header.  
// K-nearest-neighbor(kNN) classification is one of the most fundamental and simple classification methods 
// and should be one of the first choices for a classification study  when there is little or no prior knowledge about the distribution of the data.
//
// REFENCE:
//      [1] Cover T, Hart P. Nearest neighbor pattern classification[M]. IEEE Press, 1967.
//
//
//
// How to use:
//		For example, here is some data:
//			double data[]{ 1, 101, 1, 5, 89, 1, 108, 5, 1, 115, 8, 1 };
//			string label[]{ "A", "A", "B", "B" };
//
//		And here is test data:
//			double test[]{ 10, 202, 1 };
//
//		Then,
//      	auto& knn = Knn<double, string>::getInstance();
//			knn.init(data, 3, label, 4);
//			string result = knn.classify(test)[1]; /* 1-NN */
//			string result2 = knn.classify(test)[3]; /* 3-NN */
//
//		Or
//      	Knn<double, string>& knn = Knn<double, string>::getInstance();
//			knn.init(data, 3, label, 4);
//			knn.classify(test);
//			string result = knn[1]; /* 1-NN */
//			string result2 = knn[3]; /* 3-NN */
//
#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <utility>

using std::sqrt;
using std::vector;
using std::pair;
using std::unordered_map;
using std::make_heap;
using std::sort_heap;
using std::pop_heap;


////////////////////////////////////////////////////////////////
// The namespace `KNN`
//
// In general, it should not be used or modified externally.
namespace KNN {
	////////////////////////////////////////////////////////////////
	// Data structure for KNN classifier.
	template<typename type_data, typename type_label>
	struct KnnData {
		vector<type_data> m_data;
		type_label m_label;

		/**
		 * Constructor.
		 *
		 * @params { data }     array of data.
		 * @params { dim }      the range of array.
		 * @params { label }    the label of this data.
		 */
		KnnData(const type_data* data, const unsigned int dim, type_label& label) {
			m_data = vector<type_data>(data, data + dim);
			m_label = label;
		}

		/**
		 * Calculated Euclidean distance.
		 *
		 * @returns    the distance between `test` and `m_data`.
		 */
		double EuclideanDistance(type_data* test) {
			double result = 0;
			for (int i = m_data.size(); i--; )
				result += (m_data[i] - test[i]) * (m_data[i] - test[i]);
			return sqrt(result);
		}
	};

	////////////////////////////////////////////////////////////////
	// STRUCT TEMPLATE greater.
	template<typename type_label = int>
	struct KnnLabelGreater {
		using KnnLabelPair = const pair<double, type_label>;
		constexpr bool operator()(KnnLabelPair& left, KnnLabelPair& right) const {
			return left.first > right.first;
		}
	};


	////////////////////////////////////////////////////////////////
	// STRUCT TEMPLATE less.
	template<typename type_label = int>
	struct KnnLabelLess {
		using KnnLabelPair = const pair<double, type_label>;
		constexpr bool operator()(KnnLabelPair& left, KnnLabelPair& right) const {
			return left.first < right.first;
		}
	};
}

////////////////////////////////////////////////////////////////
// KNN classifier.
template<typename type_data, typename type_label = int>
class Knn
{
public:
	/**
	 * Destructor.
	 */
	~Knn() {};


	/**
	 * Deleted the copy constructor.
	 */
	Knn(const Knn&) = delete;


	/**
	 * Deleted the copy assignment operator.
	 */
	Knn& operator=(const Knn&) = delete;


	/**
	 * Overloaded the operator `[]`.
	 * Finded K-nearest-neighbor and decided the label of `m_testData`.
	 */
	type_label operator[](const unsigned int K) {
		if (K <= m_dataSet.size()) {
			vector<pair<double, type_label>> distance;
			for (KNN::KnnData<type_data, type_label>& data : m_dataSet) {
				distance.push_back(std::make_pair(data.EuclideanDistance(m_testData), data.m_label));
			}

			if (K == 1) { // 1nn
				make_heap(distance.begin(), distance.end(), m_greater);
				return distance.front().second;
			}
			else { // knn
				vector<type_label> result;
				result.reserve(K);
				for (int i = K; i--; ) {
					make_heap(distance.begin(), distance.end(), m_greater);
					pop_heap(distance.begin(), distance.end(), m_greater);
					result.push_back(distance.back().second);
					distance.pop_back();
				}
				return vote(result);
			}
		}

		return type_label();
	}


	/**
	 * Input the data to be classified.
	 *
	 * @params { data }     data with unknown label.
	 */
	Knn<type_data, type_label>& classify(type_data* data) {
		this->m_testData = data;
		return *this;
	}


	/**
	 * C++ Singleton Pattern.
	 * Meyers' Singleton (magic static).
	 */
	static Knn<type_data, type_label>& getInstance() {
		static Knn<type_data, type_label> knn;
		return knn;
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
	 * @params { data }     a cluster of data.
	 * @params { dim }      dimensions of the data.
	 * @params { label }    the label of the data.
	 * @params { size }     the size of data.
	 *
	 * @note: `sizeof(data) / sizeof(type_data) == dim * size`
	 *        `sizeof(label) / sizeof(type_label) == size`
	 */
	void init(type_data* data, int dim, type_label* label, int size) {
		m_dataSet.clear();
		m_dataSet.reserve(size);
		for (int i = 0, idx = 0; i < size; ++i, idx += dim)
			m_dataSet.push_back(KNN::KnnData<type_data, type_label>(data + idx, dim, label[i]));
	}

private:
	/**
	 * Constructor.
	 */
	Knn() :m_testData(nullptr) {};


	/**
	 * Voted for result by finding the majority labels.
	 *
	 * @params { labels }
	 */
	type_label vote(vector<type_label>& labels) {
		unordered_map<type_label, int> voter;
		type_label label;
		int count = 0;
		for (auto& res : labels) {
			auto got = voter.find(res);
			if (got == voter.end()) voter.insert(make_pair(res, 1));
			else (got->second)++;
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
	vector<KNN::KnnData<type_data, type_label>> m_dataSet;
	type_data* m_testData;
	KNN::KnnLabelGreater<type_label> m_greater;
	KNN::KnnLabelLess<type_label> m_less;
};

