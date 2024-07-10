//
// Created by juelin on 6/30/24.
//

#ifndef PICKLE_DATALOADER_HPP
#define PICKLE_DATALOADER_HPP
#include <cassert>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <memory>
#include <chrono>
#include <algorithm>
#include "util.hpp"
#include "timer.hpp"
#include "common.hpp"

namespace pickle {

    struct Array2D {
        void *_data{nullptr};
        size_t _shape[2]{0, 0};
        size_t _word_size{0};
        DataType _data_type{DataType::Float32};

        Array2D() = default;

        ~Array2D() {
            _shape[0] = 0;
            _shape[1] = 0;
            _word_size = 0;
            if (_data)
                free(_data);
        }

        void *get_raw(size_t row_id) {
            return static_cast<char *>(_data) + 1ull * row_id * _shape[1] * _word_size;
        };

        template <typename T> T *get_ptr(size_t row_id) {
            return static_cast<T *>(get_raw(row_id));
        }

        template <typename T> std::span<T> get_span(size_t row_id) {
            return std::span{get_ptr<T>(row_id), _shape[1]};
        }

        template <typename T, size_t Dim> std::span<T, Dim> get_span(size_t row_id) {
            assert(Dim == _shape[1]);
            return std::span<T, Dim>{get_ptr<T>(row_id), _shape[1]};
        }

        template <typename T> T *data() { return static_cast<T *>(_data); }

        template <typename T> std::span<T> span() {
            return {data<T>(), _shape[0] * _shape[1]};
        };
    };
    using Array2DPtr = std::shared_ptr<Array2D>;

struct GroundTruth {
  std::vector<uint32_t> _label;
  std::vector<float> _distance;
  size_t _shape[2]{0, 0};

  std::span<uint32_t> get_labels(size_t top_k, size_t q_id) {
    assert(top_k <= _shape[1]);
    assert(q_id < _shape[0]);
    return {_label.begin() + q_id * _shape[1], top_k};
  }

  std::span<float> get_distances(size_t top_k, size_t q_id) {
    assert(top_k <= _shape[1]);
    assert(q_id < _shape[0]);
    return {_distance.begin() + q_id * _shape[1], top_k};
  }
};
using GroundTruthPtr = std::shared_ptr<GroundTruth>;

Array2DPtr LoadArray2D(const std::string &filename,
                       size_t max_elements = std::dynamic_extent,
                       bool to_float = false);
GroundTruthPtr LoadGroundTruth(const std::string &filename);
} // namespace pickle
#endif // PICKLE_DATALOADER_HPP

