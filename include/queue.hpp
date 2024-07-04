//
// Created by juelin on 7/1/24.
//

#ifndef PICKLE_QUEUE_HPP
#define PICKLE_QUEUE_HPP

#include "common.hpp"

#include <queue>
#include <numeric>

namespace pickle {
    struct Entry {
        distance_t _distance{std::numeric_limits<distance_t>::max()};
        internal_id_t _vid{empty_internal_id};

        Entry(distance_t distance, internal_id_t vid) : _distance{distance}, _vid{vid} {};

        Entry() = default;

        bool inline operator==(const Entry &other) const {
            return _distance == other._distance && _vid == other._vid;
        }
    };

    struct MaxFirst {
        inline bool operator()(const Entry &lhs, const Entry &rhs) const {
            return lhs._distance < rhs._distance;
        };
    };

    struct MinFirst {
        inline bool operator()(const Entry &lhs, const Entry &rhs) const {
            return lhs._distance > rhs._distance;
        };
    };

    // TODO: accelerate pop operation using SIMD instructions
    // TODO: customized container to reduce memory allocation overhead
    typedef std::priority_queue<Entry, std::vector<Entry>, MaxFirst> MaxQueue;
    typedef std::priority_queue<Entry, std::vector<Entry>, MinFirst> MinQueue;

}
#endif //PICKLE_QUEUE_HPP
