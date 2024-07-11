//
// Created by juelin on 7/10/24.
//

#ifndef PICKLE_ADJLIST_HPP
#define PICKLE_ADJLIST_HPP
#include "common.hpp"
#include "mempool.hpp"
#include "queue.hpp"
#include "lock.hpp"

namespace pickle
{

    class AdjList {
    private:
        uint16_t _num_level{0};
        uint16_t _max_degree{0};
        external_id_t _ext_id{empty_external_id};
        char *_data{nullptr};
    public:
        AdjList() = default;

        AdjList(external_id_t ext_id, internal_id_t max_degree, internal_id_t max_level) {
            assert(max_level <= std::numeric_limits<uint8_t>::max());
            assert(max_degree <= std::numeric_limits<uint16_t>::max());
            _ext_id = ext_id;
            _max_degree = max_degree;
            _num_level = max_level + 1;
            size_t degree_size = sizeof(internal_id_t) * _num_level;
            size_t adj_list_size = sizeof(internal_id_t) * (max_degree + max_degree * _num_level);
            size_t distance_size = sizeof(distance_t) * (max_degree + max_degree * _num_level);
            size_t data_size = degree_size + adj_list_size + distance_size;
            _data = DataMemoryPool::Global().Alloc<char>(data_size);
            for (int i = 0; i < _num_level; i++) {
                GetDegreeData()[i] = 0;
            }
        }

        [[nodiscard]] external_id_t GetExtID() const { return _ext_id; };

        [[nodiscard]] internal_id_t *GetDegreeData() const {
            return reinterpret_cast<internal_id_t *>(_data);
        }

        [[nodiscard]] internal_id_t *GetAdjData() const {
            return reinterpret_cast<internal_id_t *>(_data + _num_level * sizeof(internal_id_t));
        }

        [[nodiscard]] distance_t *GetDistanceData() const {
            return reinterpret_cast<distance_t *>(_data + (_num_level + _num_level * _max_degree + _max_degree) *
                                                          sizeof(internal_id_t));
        }

        [[nodiscard]] size_t GetDegree(size_t level) const {
            return GetDegreeData()[level];
        }

        [[nodiscard]] std::span<distance_t> GetDistance(size_t level) const {
            return {GetDistanceData() + _max_degree * level + (level > 0) * _max_degree, GetDegree(level)};
        }

        [[nodiscard]] std::span<internal_id_t> GetAdj(size_t level) const {
            return {GetAdjData() + _max_degree * level + (level > 0) * _max_degree, GetDegree(level)};
        }

        internal_id_t GetMaxLevel() const {
            return _num_level - 1;
        }

        void AddSingle(size_t level, internal_id_t vid, distance_t distance) {
            auto adj = GetAdj(level);
            auto dist = GetDistance(level);
            auto max_deg = (level == 0) ? 2 * _max_degree : _max_degree;
            auto offset = std::lower_bound(dist.begin(), dist.end(), distance) - dist.begin();
            assert(offset < adj.size());
            if (offset < max_deg) {
                for (int i = adj.size(); i > offset; i--) {
                    dist[i] = dist[i - 1];
                    adj[i] = adj[i - 1];
                }
                adj[offset] = vid;
                dist[offset] = distance;
                GetDegreeData()[level] += adj.size() < max_deg;
            }
        }

        void Add(size_t level, std::span<Entry> entries) {
            auto adj = GetAdj(level);
            auto dist = GetDistance(level);
            for (size_t i = 0; i < entries.size(); i++) {
                adj[i] = entries[i]._vid;
                dist[i] = entries[i]._distance;
            }
            GetDegreeData()[level] = entries.size();
        }
    };


//    struct Neighbors {
//        // the adjacency lists is organized as following:
//        // the first num_level elements store the current length of each adjacency list
//        // the neighbors in level 0 (base layer) to layer _level_max are store contiguously
//        // the base level holds: 2 * max_degree elements
//        // other levels hold: max_degree elements
//
//        // the distance to neighbors in level 0 (base layer) to layer _level_max are store contiguously
//        internal_id_t *data{nullptr};
//        internal_id_t *_adj_list{nullptr};
//        distance_t *_distance{nullptr};
//
//        Neighbors() = default;
//
//        Neighbors(size_t max_level, size_t max_degree) {
//            // assume base level has 2 * max_degree neighbors
//            size_t num_level = max_level + 1;
//            size_t degree_size = sizeof(internal_id_t) * num_level;
//            size_t adj_list_size = sizeof(internal_id_t) * (max_degree + max_degree * num_level);
//            size_t distance_size = sizeof(distance_t) * (max_degree + max_degree * num_level);
//            size_t size = degree_size + adj_list_size + distance_size;
//            data = DataMemoryPool::Global().Alloc<internal_id_t>(size);
//            _adj_list = data + num_level;
//            _distance = reinterpret_cast<distance_t *>(_adj_list + max_degree * num_level + max_degree);
//        }
//
//        [[nodiscard]] size_t GetDegree(size_t level) const {
//            return data[level];
//        }
//
//        [[nodiscard]] std::span<distance_t> GetDistance(size_t level, size_t max_degree) {
//            return {_distance + max_degree * level + (level > 0) * max_degree, GetDegree(level)};
//        }
//
//        [[nodiscard]] std::span<const distance_t> GetDistance(size_t level, size_t max_degree) const {
//            return {_distance + max_degree * level + (level > 0) * max_degree, GetDegree(level)};
//        }
//
//        [[nodiscard]] std::span<internal_id_t> GetAdj(size_t level, size_t max_degree) {
//            return {_adj_list + max_degree * level + (level > 0) * max_degree, GetDegree(level)};
//        }
//
//        [[nodiscard]] std::span<const internal_id_t> GetAdj(size_t level, size_t max_degree) const {
//            return {_adj_list + max_degree * level + (level > 0) * max_degree, GetDegree(level)};
//        }
//    };
//
//    struct Node {
//        external_id_t _ext_id{empty_external_id}; // or label in other system
//        internal_id_t _level_max{0}; // highest level this vertex has been inserted to
//        Neighbors _neighbors{};
//
//        Node() = default;
//
//        ~Node() = default;
//
//        Node(external_id_t ext_id, internal_id_t max_level, size_t max_degree) : _ext_id{ext_id},
//                                                                                 _level_max{max_level} {
//            _neighbors = Neighbors(max_level, max_degree);
//        }
//
//        [[nodiscard]] size_t GetDegree(size_t level) const {
//            return _neighbors.GetDegree(level);
//        }
//
//        [[nodiscard]] std::span<distance_t> GetDistance(size_t level, size_t max_degree) {
//            return _neighbors.GetDistance(level, max_degree);
//        }
//
//        [[nodiscard]] std::span<const distance_t> GetDistance(size_t level, size_t max_degree) const {
//            return _neighbors.GetDistance(level, max_degree);
//        }
//
//        [[nodiscard]] std::span<internal_id_t> GetAdj(size_t level, size_t max_degree) {
//            return _neighbors.GetAdj(level, max_degree);
//        }
//
//        [[nodiscard]] std::span<const internal_id_t> GetAdj(size_t level, size_t max_degree) const {
//            return _neighbors.GetAdj(level, max_degree);
//        }
//    };

}
#endif //PICKLE_ADJLIST_HPP
