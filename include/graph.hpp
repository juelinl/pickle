#pragma once
#include <cstdint>
#include <span>
#include <atomic>
#include <shared_mutex>
#include <mutex>
#include <vector>
#include <thread>
#include <cassert>
#include <memory>

namespace pickle
{

    typedef int32_t external_id_t;
    typedef int32_t internal_id_t;
    typedef float distance_t;

    class DynamicNSWGraph
    {
    private:
        constexpr const static size_t _num_mutex{4096};

        size_t _max_degree{0};
        size_t _node_capacity{0};

        internal_id_t _entry_internal_id{0};
        std::atomic<internal_id_t> _next_internal_id{0};


        std::shared_mutex _resize_mutex; // guard resize and expand vector
        std::vector<std::mutex> _rw_mutex{_num_mutex}; // guard write to vector

        std::vector<internal_id_t> _adjacent_lists;
        std::vector<internal_id_t> _internal_degrees;
        std::vector<external_id_t> _external_ids;
        std::vector<distance_t > _distance_lists;

        std::mutex& GetMutex(internal_id_t vid) {
            return _rw_mutex.at(vid % _num_mutex);
        };

        void Resize(internal_id_t new_minimum_capacity) {
            if (_node_capacity < new_minimum_capacity) {
                std::unique_lock<std::shared_mutex> writeLock{_resize_mutex};
                _node_capacity = new_minimum_capacity * 1.2 + 4096;
                _adjacent_lists.resize(_node_capacity * _max_degree);
                _external_ids.resize(_node_capacity);
                _internal_degrees.resize(_node_capacity);
            }
        };

        void Clear() {
            _max_degree = 0;
            _node_capacity = 0;
            _adjacent_lists.clear();
            _external_ids.clear();
            _internal_degrees.clear();
        };

    public:
        DynamicNSWGraph() = default;

        void Init(size_t max_node_degree, size_t node_capacity){
            Clear();
            _max_degree = max_node_degree;
            _node_capacity = node_capacity;
            _adjacent_lists.resize(_node_capacity * _max_degree);
            _external_ids.resize(_node_capacity);
            _internal_degrees.resize(_node_capacity);
        };

        external_id_t GetExternalId(internal_id_t vid) const {
            return _external_ids.at(vid);
        };

        internal_id_t GetNewInternalId(external_id_t external_id){
            internal_id_t new_internal_id = _next_internal_id++;
            if (new_internal_id >= _node_capacity) {
                Resize(new_internal_id);
            }
            _external_ids.at(new_internal_id) = external_id;
            return new_internal_id++;
        };

        internal_id_t GetInternalId(external_id_t external_id){
            for(internal_id_t internal_id = 0; internal_id < _external_ids.size(); internal_id++) {
                if (_external_ids.at(internal_id) == external_id) return internal_id;
            }
            return -1;
        };

        internal_id_t GetEntryInternalId() {
            //TODO: better strategy for updating entry id
            return _entry_internal_id;
        };

        std::span<internal_id_t> GetAdjlist(internal_id_t vid){
            //TODO: should it handle race condition against write?
            auto start = _adjacent_lists.begin() + _max_degree * vid;
            auto end = start + _internal_degrees.at(vid);
            return {start, end};
        };

        // it should be called only once for each vertex
        void AddNode(internal_id_t vid, std::span<internal_id_t> adjlist, std::span<distance_t > distances){
            // expand should be handled when requesting for new internal ids
            assert(vid < _node_capacity);
            assert(adjlist.size() == distances.size());

            // use shared_mutex to prevent update while resize
            std::shared_lock<std::shared_mutex> updateLock{_resize_mutex};
            _internal_degrees.at(vid) = adjlist.size();
            auto adjlist_start = _adjacent_lists.begin() + _max_degree * vid;
            auto dislist_start = _distance_lists.begin() + _max_degree * vid;
            for (size_t i = 0; i < adjlist.size(); i++) {
                adjlist_start[i] = adjlist[i];
                dislist_start[i] = distances[i];
            }
        };

        // it can be called multiple times for each vertex (critical path)
        bool UpdateNode(internal_id_t vid, internal_id_t nid, distance_t distance) {
            // TODO: handle cases where v_degree >= max_degree

            // use mutex to prevent race condition on update
            // TODO: one mutex per node?
            std::unique_lock<std::mutex> writeLock{GetMutex(vid)};

            // use share_mutex to prevent resize while update
            std::shared_lock<std::shared_mutex> updateLock{_resize_mutex};

            auto cur_v_degree = _internal_degrees.at(vid) ;
            if (cur_v_degree == _max_degree) return false;
            _adjacent_lists.at(vid * _max_degree + cur_v_degree) = nid;
            _distance_lists.at(vid * _max_degree + cur_v_degree) = distance;
            _internal_degrees.at(vid)++;
            return true;
        };
    };

    using DynamicNSWGraphPtr = std::shared_ptr<DynamicNSWGraph>;

}