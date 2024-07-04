#pragma once

//#include <cstdint>
//#include <span>
//#include <atomic>
//#include <shared_mutex>
//#include <thread>
#include <mutex>
#include <vector>
#include <cassert>
#include <memory>
#include <iostream>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include "common.hpp"
#include "queue.hpp"

namespace pickle {

    // graph interface
//    class GraphIF {
//    public:
//        virtual ~GraphIF() = default;
//        virtual external_id_t GetExternalId(internal_id_t vid) = 0;
//        virtual internal_id_t GetEntryInternalId(external_id_t external_id) = 0;
//        virtual internal_id_t GetInternalId(external_id_t external_id) = 0;
//        virtual std::span<internal_id_t> GetNeighborsID(internal_id_t vid) = 0;
//        virtual void Init(size_t max_node_degree, size_t max_num_node) = 0;
//    };
//    using GraphIFPtr = std::shared_ptr<GraphIF>;

    class Graph {
    private:
        constexpr const static size_t _num_mutex{8192};

        size_t _max_degree{0};
        size_t _node_capacity{0};
        internal_id_t _entry_internal_id{0};
        internal_id_t _next_internal_id{0};
        std::mutex _id_increment_mutex;

        std::map<external_id_t, internal_id_t> _external_to_internal;
//        std::unordered_map<external_id_t, internal_id_t> _external_to_internal;
        std::vector<internal_id_t> _adjacent_lists;
        std::vector<distance_t> _distance_lists;
        std::vector<internal_id_t> _internal_degrees;
        std::vector<external_id_t> _external_ids;

        std::vector<std::mutex> _rw_mutex{_num_mutex}; // guard write to vector

        std::mutex &GetMutex(internal_id_t vid) {
            return _rw_mutex.at(vid % _num_mutex);
        };

        void Clear() {
            _max_degree = 0;
            _node_capacity = 0;
            _external_ids.clear();
            _internal_degrees.clear();
            _adjacent_lists.clear();
            _distance_lists.clear();
        };

    public:
        Graph() = default;

        void Init(size_t max_node_degree, size_t node_capacity) {
            Clear();
            _max_degree = max_node_degree;
            _node_capacity = node_capacity;
            _adjacent_lists.resize(_node_capacity * _max_degree, empty_internal_id);
            _distance_lists.resize(_node_capacity * _max_degree, std::numeric_limits<distance_t>::max());
            _external_ids.resize(_node_capacity, empty_external_id);
            _internal_degrees.resize(_node_capacity, 0);
        };

        external_id_t GetExternalId(internal_id_t vid) const {
            return _external_ids.at(vid);
        };


        internal_id_t GetNewInternalId(external_id_t external_id) {
            internal_id_t new_internal_id{empty_internal_id};
            {
                std::unique_lock<std::mutex> guard{_id_increment_mutex};
                assert(!_external_to_internal.contains(external_id));
                new_internal_id = _next_internal_id++;
                _external_to_internal.insert({external_id, new_internal_id});
                _external_ids.at(new_internal_id) = external_id;
                assert(_external_to_internal.contains(external_id));
            }
            return new_internal_id;
        };

        internal_id_t GetInternalId(external_id_t external_id) const {
            assert(_external_to_internal.contains(external_id));
            return _external_to_internal.at(external_id);
        };

        internal_id_t GetEntryInternalId(external_id_t external_entry_id = empty_external_id) const {
            //TODO: better strategy for updating entry id
            if (external_entry_id == empty_external_id) return _entry_internal_id;
            return GetInternalId(external_entry_id);
        };

        std::span<const internal_id_t> GetNeighborsID(internal_id_t vid) const {
            auto start = &_adjacent_lists.at(_max_degree * vid);
            auto end = start + _internal_degrees.at(vid);
            return {start, end};
        };

        std::span<const distance_t> GetNeighborsDistance(internal_id_t vid) const {
            auto start = &_distance_lists.at(_max_degree * vid);
            auto end = start + _internal_degrees.at(vid);
            return {start, end};
        };

        // it should be called only once for each vertex
        void AddNode(internal_id_t vid, std::span<Entry> neighbors) {
            std::unique_lock<std::mutex> writeLock{GetMutex(vid)};
            assert(vid < _node_capacity);
            size_t idx{0};
            for (const auto edge: neighbors) {
                assert(edge._vid < _node_capacity);
                if (edge._vid != vid) {
                    _adjacent_lists.at(_max_degree * vid + idx) = edge._vid;
                    _distance_lists.at(_max_degree * vid + idx) = edge._distance;
                    idx++;
                }
            }
            _internal_degrees.at(vid) = idx;
        };

        // it can be called multiple times for each vertex (critical path?)
        void AddReverseEdge(internal_id_t nid, const Entry &edge) {
            // TODO: one mutex per node?
            auto vid = edge._vid;
            assert(vid < _node_capacity);
            assert(nid < _node_capacity);
            if (nid == edge._vid) return;
            // use mutex to prevent race condition on update
            std::unique_lock<std::mutex> writeLock{GetMutex(vid)};
            auto v_deg = _internal_degrees.at(vid);
            assert(v_deg <= _max_degree);

            if (v_deg == 0) {
                _distance_lists.at(vid * _max_degree) = edge._distance;
                _adjacent_lists.at(vid * _max_degree) = nid;
                _internal_degrees.at(vid) = 1;
                return;
            }

            // Always use greedy approach for updating edges
            // The greedy approach always keeps top max_degree closest edges
            // keep adjacency list and distance sorted, small distance edges will be stored in the front
            auto dist_start = &_distance_lists.at(vid * _max_degree);
            auto edge_start = &_adjacent_lists.at(vid * _max_degree);
            auto offset = std::lower_bound(dist_start, dist_start + v_deg, edge._distance) - dist_start;
            assert(offset <= v_deg);
            if (offset == _max_degree) {
                return;
            } else if (offset == v_deg) {
                // add to the end if within capacity
                *(edge_start + offset) = nid;
                *(dist_start + offset) = edge._distance;
                _internal_degrees.at(vid) += v_deg < _max_degree;
            } else {
                for (int i = v_deg; i > offset; i--) {
                    dist_start[i] = dist_start[i - 1];
                    edge_start[i] = edge_start[i - 1];
                }
                edge_start[offset] = nid;
                dist_start[offset] = edge._distance;
                _internal_degrees.at(vid) += v_deg < _max_degree;
            }


//            for (int i = 0; i < _internal_degrees.at(vid); i++) {
//                assert(edge_start[i] >= 0);
//            }
        };

        internal_id_t GetNumNodes() const {
            return _next_internal_id;
        }

        internal_id_t GetNodeCapacity() const {
            return _node_capacity;
        }
    };

    using DynamicNSWGraphPtr = std::shared_ptr<Graph>;

    template<class T, std::size_t Dim>
    inline std::span<T, Dim> GetDataForExternalID(external_id_t q_id, size_t dim, std::span<T> all_data) {
        auto start = all_data.data() + q_id * dim;
        std::span<T, Dim> data{start, dim};
        return data;
    };

    template<class T, std::size_t Dim>
    inline std::span<T, Dim> GetDataForInternalID(internal_id_t vid, size_t dim, std::span<T> all_data,
                                                  const DynamicNSWGraphPtr &graph) {
        auto start = all_data.data() + graph->GetExternalId(vid) * dim;
        std::span<T, Dim> data{start, dim};
        return data;
    };
}