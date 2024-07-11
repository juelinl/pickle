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
#include "array2d.hpp"

namespace pickle {

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

