//
// Created by juelin on 7/10/24.
//

#ifndef PICKLE_ADJLIST_HPP
#define PICKLE_ADJLIST_HPP

#include "common.hpp"
#include "mempool.hpp"
#include "queue.hpp"
#include "lock.hpp"

namespace pickle {
    class AdjList {
    private:
        internal_id_t _degree{0};
        internal_id_t _capacity{0};
        char *_data{nullptr};


    public:
        AdjList() = default;
        explicit AdjList(internal_id_t level, internal_id_t capacity){
            _capacity = capacity;
            _data = DataMemoryPool::Global().Alloc<char>(capacity * (sizeof(internal_id_t ) + sizeof(distance_t)));
        }

        ~AdjList() = default;

        [[nodiscard]] internal_id_t GetDegree() const {
            return _degree;
        }

        [[nodiscard]] internal_id_t GetCapacity() const {
            return _capacity;
        }

        [[nodiscard]] std::span<internal_id_t> GetAdj() {
            return {reinterpret_cast<internal_id_t *>(_data), static_cast<size_t>(_degree)};
        }

        [[nodiscard]] std::span<const internal_id_t> GetAdj() const {
            return {reinterpret_cast<const internal_id_t *>(_data), static_cast<size_t>(_degree)};
        }

        [[nodiscard]] std::span<distance_t> GetDist() {
            return {reinterpret_cast<distance_t *>(_data + _capacity * sizeof(internal_id_t)),
                    static_cast<size_t>(_degree)};
        }

        [[nodiscard]] std::span<const distance_t> GetDist() const {
            return {reinterpret_cast<distance_t *>(_data + _capacity * sizeof(internal_id_t)),
                    static_cast<size_t>(_degree)};
        }

        void CheckValid() const {
            auto adj = GetAdj();
            assert(adj.size() <= _capacity);
            for (int i = 1; i < adj.size(); i++) {
                assert(adj[i] != adj[i - 1]);
            }
        }

        void AddSingle(internal_id_t vid, distance_t distance) {
            assert(_capacity > 0);
            auto adj = GetAdj();
            auto dist = GetDist();
            auto offset = std::lower_bound(dist.begin(), dist.end(), distance) - dist.begin();
            assert(offset <= adj.size());
            if (offset < _capacity) {
                int end = std::min(_capacity - 1, (int) adj.size());
                for (int i = end; i > offset; i--) {
                    assert(i < _capacity);
                    dist[i] = dist[i - 1];
                    adj[i] = adj[i - 1];
                }
                adj[offset] = vid;
                dist[offset] = distance;
                _degree += _degree < _capacity;
            }
        }

        void Add(std::span<Entry> entries) {
            assert(_capacity > 0);
            auto adj = GetAdj();
            auto dist = GetDist();
            for (size_t i = 0; i < entries.size(); i++) {
                adj[i] = entries[i].m_vid;
                dist[i] = entries[i].m_dist;
            }
            _degree = entries.size();
        }
    };

    class AdjLists {
    private:
        external_id_t _ext_id{empty_external_id};
        std::vector<AdjList> _adj_lists;

    public:
        AdjLists() = default;

        AdjLists(external_id_t ext_id, internal_id_t max_degree, internal_id_t max_level) {
            _ext_id = ext_id;
//            _adj_lists.reserve(max_level + 1);
            for (int i = 0; i <= max_level; i++) {
                _adj_lists.emplace_back(i, max_degree + (i == 0) * max_degree);
            }
        }

        [[nodiscard]] external_id_t GetExtID() const { return _ext_id; };

        [[nodiscard]] internal_id_t GetMaxLevel() const { return _adj_lists.size() - 1; }

        [[nodiscard]] internal_id_t GetDegree(int level) {
            return _adj_lists.at(level).GetDegree();
        }

        [[nodiscard]] std::span<distance_t> GetDist(int level) {
            return _adj_lists.at(level).GetDist();
        }

        [[nodiscard]] std::span<const distance_t> GetDist(int level) const {
            return _adj_lists.at(level).GetDist();
        }

        [[nodiscard]] std::span<internal_id_t> GetAdj(int level) {
            return _adj_lists.at(level).GetAdj();
        }

        [[nodiscard]] std::span<const internal_id_t> GetAdj(int level) const {
            return _adj_lists.at(level).GetAdj();
        }

        void CheckValid() const {
            assert(_ext_id != empty_external_id);
            for (auto& adj: _adj_lists) {
                adj.CheckValid();
            }
        }

        void AddSingle(internal_id_t level, internal_id_t vid, distance_t distance) {
            _adj_lists.at(level).AddSingle(vid, distance);
        }

        void Add(internal_id_t level, std::span<Entry> entries) {
            _adj_lists.at(level).Add(entries);
        }
    };

//    class AdjLists {
//    private:
//        uint16_t _num_level{0};
//        uint16_t m_max_deg{0};
//        external_id_t _ext_id{empty_external_id};
//        char *m_data{nullptr};
//
////        std::span<internal_id_t> _deg;
////        std::span<internal_id_t> _adj;
////        std::span<distance_t> _dist;
//    public:
//        AdjLists() = default;
//
//        AdjLists(external_id_t ext_id, internal_id_t max_degree, internal_id_t max_level) {
//            assert(max_level <= std::numeric_limits<uint8_t>::max());
//            assert(max_degree <= std::numeric_limits<uint16_t>::max());
//            _ext_id = ext_id;
//            m_max_deg = max_degree;
//            _num_level = max_level + 1;
//            size_t degree_size = sizeof(internal_id_t) * _num_level;
//            size_t adj_list_size = sizeof(internal_id_t) * (max_degree + max_degree * _num_level);
//            size_t distance_size = sizeof(distance_t) * (max_degree + max_degree * _num_level);
//            size_t data_size = degree_size + adj_list_size + distance_size;
//            m_data = DataMemoryPool::Global().Alloc<char>(data_size);
//            for (int i = 0; i < _num_level; i++) {
//                GetDegreeData()[i] = 0;
//            }
////            _deg = {GetDegreeData(), static_cast<size_t>(_num_level)};
////            _adj = {GetAdjData(), static_cast<size_t>(max_degree + max_degree * _num_level)};
////            _dist = {GetDistData(), static_cast<size_t>(max_degree + max_degree * _num_level)};
////            CheckValid();
//        }
//
//        [[nodiscard]] external_id_t GetExtID() const { return _ext_id; };
//
//        [[nodiscard]] internal_id_t GetMaxLevel() const { return _num_level - 1; }
//
//        [[nodiscard]] internal_id_t *GetDegreeData() {
//            return reinterpret_cast<internal_id_t *>(m_data);
//        }
//
//        [[nodiscard]] const internal_id_t *GetDegreeData() const {
//            return reinterpret_cast<const internal_id_t *>(m_data);
//        }
//
//        [[nodiscard]] internal_id_t &GetDegree(int level) {
//            return GetDegreeData()[level];
//        }
//
//        [[nodiscard]] const internal_id_t &GetDegree(int level) const {
//            return GetDegreeData()[level];
//        }
//
//        [[nodiscard]] internal_id_t *GetAdjData() {
//            return reinterpret_cast<internal_id_t *>(m_data + _num_level * sizeof(internal_id_t));
//        }
//
//        [[nodiscard]] const internal_id_t *GetAdjData() const {
//            return reinterpret_cast<const internal_id_t *>(m_data + _num_level * sizeof(internal_id_t));
//        }
//
//        [[nodiscard]] internal_id_t *GetAdjData(int level) {
//            return GetAdjData() + (level + level > 0) * m_max_deg;
//        }
//
//        [[nodiscard]] const internal_id_t *GetAdjData(int level) const {
//            return GetAdjData() + (level + level > 0) * m_max_deg;
//        }
//
//        [[nodiscard]] distance_t *GetDistData() {
//            return reinterpret_cast<distance_t *>(m_data + (_num_level + _num_level * m_max_deg + m_max_deg) *
//                                                          sizeof(internal_id_t));
//        }
//
//        [[nodiscard]] const distance_t *GetDistData() const {
//            return reinterpret_cast<const distance_t *>(m_data + (_num_level + _num_level * m_max_deg + m_max_deg) *
//                                                                sizeof(internal_id_t));
//        }
//
//        [[nodiscard]] distance_t *GetDistData(int level) {
//            return GetDistData() + (level + level > 0) * m_max_deg;
//        }
//
//        [[nodiscard]] const distance_t *GetDistData(int level) const {
//            return GetDistData() + (level + level > 0) * m_max_deg;
//        }
//
//        [[nodiscard]] std::span<distance_t> GetDist(int level) {
//            return {GetDistData(level), static_cast<size_t>(GetDegree(level))};
//        }
//
//        [[nodiscard]] std::span<const distance_t> GetDist(int level) const {
//            return {GetDistData(level), static_cast<size_t>(GetDegree(level))};
//        }
//
//        [[nodiscard]] std::span<internal_id_t> GetAdj(int level) {
//            return {GetAdjData(level), static_cast<size_t>(GetDegree(level))};
//        }
//
//        [[nodiscard]] std::span<const internal_id_t> GetAdj(int level) const {
//            return {GetAdjData(level), static_cast<size_t>(GetDegree(level))};
//        }
//
//        void CheckValid() const {
//            assert(_ext_id != empty_external_id);
//            for (int level = 0; level < _num_level; level++) {
//                auto adj = GetAdj(level);
//                assert(adj.size() <= m_max_deg + m_max_deg * (level == 0));
//                for (int i = 1; i < adj.size(); i++) {
//                    assert(adj[i] != adj[i - 1]);
//                    assert(adj[i] < 10000000);
//                }
//            }
//        }
//
//        void AddSingle(internal_id_t level, internal_id_t vid, distance_t distance) {
//            CheckValid();
//            auto adj = GetAdj(level);
//            auto dist = GetDist(level);
//            auto max_deg = (level == 0) ? 2 * m_max_deg : m_max_deg;
//            auto offset = std::lower_bound(dist.begin(), dist.end(), distance) - dist.begin();
//            assert(offset <= adj.size());
//            if (offset < max_deg) {
//                int end = std::min(max_deg - 1, (int) adj.size());
//                for (int i = end; i > offset; i--) {
//                    assert(i < max_deg);
//                    dist[i] = dist[i - 1];
//                    adj[i] = adj[i - 1];
//                }
//                adj[offset] = vid;
//                dist[offset] = distance;
//                GetDegreeData()[level] += adj.size() < max_deg;
//            }
//            CheckValid();
//        }
//
//        void Add(internal_id_t level, std::span<Entry> entries) {
//            CheckValid();
//            auto adj = GetAdj(level);
//            auto dist = GetDist(level);
//            for (size_t i = 0; i < entries.size(); i++) {
//                adj[i] = entries[i].m_vid;
//                dist[i] = entries[i].m_dist;
//            }
//            GetDegree(level) = static_cast<internal_id_t>(entries.size());
//            CheckValid();
//        }
//    };


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
//        distance_t *m_dist{nullptr};
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
//            m_dist = reinterpret_cast<distance_t *>(_adj_list + max_degree * num_level + max_degree);
//        }
//
//        [[nodiscard]] size_t GetDegree(size_t level) const {
//            return data[level];
//        }
//
//        [[nodiscard]] std::span<distance_t> GetDist(size_t level, size_t max_degree) {
//            return {m_dist + max_degree * level + (level > 0) * max_degree, GetDegree(level)};
//        }
//
//        [[nodiscard]] std::span<const distance_t> GetDist(size_t level, size_t max_degree) const {
//            return {m_dist + max_degree * level + (level > 0) * max_degree, GetDegree(level)};
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
//        [[nodiscard]] std::span<distance_t> GetDist(size_t level, size_t max_degree) {
//            return _neighbors.GetDist(level, max_degree);
//        }
//
//        [[nodiscard]] std::span<const distance_t> GetDist(size_t level, size_t max_degree) const {
//            return _neighbors.GetDist(level, max_degree);
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
