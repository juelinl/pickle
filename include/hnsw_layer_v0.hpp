#pragma once

#include <mutex>
#include <vector>
#include <cassert>
#include <memory>
#include <iostream>
#include <map>
#include <algorithm>
#include <cstring>
#include <immintrin.h>

#include "common.hpp"
#include "queue.hpp"

namespace pickle::v0 {
    
    class Serializer;

    class HNSWLayer {
    private:
        friend Serializer;
        bool m_is_base{false};
        bool m_is_empty{true};
        internal_id_t m_ent_id{0};
        internal_id_t m_next_id{0};
        size_t m_max_deg{0};
        size_t m_max_node{0};
        std::mutex m_id_mutex{};
        std::map<external_id_t, internal_id_t> m_ext2in; // mapping external id to internal id (only use if is_base)
        std::vector<internal_id_t > m_ext2in_base; // mapping external id to internal id (only use if !is_base)
        std::vector<internal_id_t> m_adj_list; // adjacency list
        std::vector<distance_t> m_dist_list; // distance between node to its neighbors
        std::vector<internal_id_t> m_deg_list; // degree of adjacency list
        std::vector<external_id_t> m_ext_list; // external ids received
        std::vector<std::mutex> m_update_mutex; // guard write to vector

        std::mutex &GetMutex(internal_id_t vid) {
            return m_update_mutex.at(vid % m_update_mutex.size());
        };

        void Clear() {
            m_is_empty = true;
            m_max_deg = 0;
            m_max_node = 0;
            m_ext_list.clear();
            m_deg_list.clear();
            m_adj_list.clear();
            m_dist_list.clear();
            m_ext2in.clear();
        };

    public:
        HNSWLayer() = default;
        [[nodiscard]] bool empty() const {
            return m_is_empty;
        }
        [[nodiscard]] internal_id_t GetSize() const {
            return m_next_id;
        }

        [[nodiscard]] size_t GetCapacity() const {
            return m_max_node;
        }


        [[nodiscard]] external_id_t GetExtID(internal_id_t vid) const {
            assert(!m_is_empty);
            assert(vid < m_max_node);
            return m_ext_list.at(vid);
        };


        [[nodiscard]] internal_id_t NewInID(external_id_t ext_id) {
            const std::lock_guard<std::mutex> guard{m_id_mutex};
            internal_id_t new_in_id = m_next_id++;
            assert(new_in_id < m_max_node);

            m_ext_list.at(new_in_id) = ext_id;
            if (m_is_base) {
                assert(m_ext2in_base.at(ext_id) == empty_internal_id);
                m_ext2in_base.at(ext_id) = new_in_id;
            } else {
                assert(!m_ext2in.contains(ext_id));
                m_ext2in.insert({ext_id, new_in_id});
            }
            m_is_empty = false;
            return new_in_id;
        };

        [[nodiscard]] internal_id_t GetInID(external_id_t external_id) const {
            assert(!m_is_empty);
            if (m_is_base) {
                auto id = m_ext2in_base.at(external_id);
                assert(id != empty_internal_id);
                return id;
            } else {
                assert(m_ext2in.contains(external_id));
                auto id = m_ext2in.at(external_id);
                return id;
            }
        };

        [[nodiscard]] internal_id_t GetEntInID(external_id_t external_entry_id = empty_external_id) const {
            //TODO: better strategy for updating entry id
            if (external_entry_id == empty_external_id) return m_ent_id;
            return GetInID(external_entry_id);
        };

        [[nodiscard]] std::span<internal_id_t> GetAdj(internal_id_t vid) {
            assert(vid < m_max_node);
            return {m_adj_list.data() + m_max_deg * vid, (size_t) m_deg_list.at(vid)};
        };

        [[nodiscard]] std::span<const internal_id_t> GetAdj(internal_id_t vid) const {
            assert(vid < m_max_node);
            return {m_adj_list.data() + m_max_deg * vid, (size_t) m_deg_list.at(vid)};
        };

        [[nodiscard]] std::span<distance_t> GetDist(internal_id_t vid) {
            assert(vid < m_max_node);
            return {m_dist_list.data() + m_max_deg * vid, (size_t) m_deg_list.at(vid)};
        };

        [[nodiscard]] std::span<const distance_t> GetDist(internal_id_t vid) const {
            return {m_dist_list.data() + m_max_deg * vid, (size_t) m_deg_list.at(vid)};
        };

        [[nodiscard]] bool IsBase() const {return m_is_base;};

        void Init(size_t max_node_degree, size_t node_capacity, bool is_base = false) {
            Clear();
            m_is_base = is_base;
            m_max_deg = max_node_degree;
            m_max_node = node_capacity;
            m_adj_list.resize(m_max_node * m_max_deg, empty_internal_id);
            m_dist_list.resize(m_max_node * m_max_deg, std::numeric_limits<distance_t>::max());
            m_ext_list.resize(m_max_node, empty_external_id);
            m_deg_list.resize(m_max_node, 0);
            if (m_is_base) {
                m_ext2in_base.resize(m_max_node, empty_internal_id);
                m_update_mutex = std::vector<std::mutex>(8192);
            } else {
                m_update_mutex = std::vector<std::mutex>(m_max_node);
            }
        };

        void CreateMap() {
            if (m_is_base) {
                m_ext2in_base.clear();
                m_ext2in_base.resize(m_max_node, empty_internal_id);
                for (internal_id_t i = 0; i < m_next_id; i++) {
                    m_ext2in_base.at(m_ext_list[i]) = i;
                }
            } else {
                m_ext2in.clear();
                for (internal_id_t i = 0; i < m_next_id; i++) {
                    m_ext2in.emplace(m_ext_list[i], i);
                }
            }
        };

        bool IsValid(internal_id_t vid) const {
            auto adj = GetAdj(vid);
            auto dist = GetDist(vid);
            assert(adj.size() == dist.size());
            for (int i = 1; i < adj.size(); i++) {
                assert(adj[i] != adj[i - 1]);
                assert(dist[i] >= dist[i - 1]);
            }
            if (dist.size() > 0) assert(dist[0] > 0);
            return true;
        };

        void Add(internal_id_t vid, std::span<const Entry> neighbors) {
            std::lock_guard<std::mutex> writeLock{GetMutex(vid)};
            assert(vid < m_max_node);
            auto adj = GetAdj(vid);
            auto dist = GetDist(vid);
            for (size_t i = 0; i < neighbors.size(); i++) {
                assert(vid != neighbors[i].m_vid);
                assert(neighbors[i].m_vid < m_max_node);
                assert(i+1 == neighbors.size() || neighbors[i].m_vid != neighbors[i+1].m_vid);
                adj[i] = neighbors[i].m_vid;
                dist[i] = neighbors[i].m_dist;
            }
            m_deg_list.at(vid) = (int ) neighbors.size();
            assert(IsValid(vid));
        }

        void AddEdge(internal_id_t vid, internal_id_t nid, distance_t distance) {
            std::lock_guard<std::mutex> writeLock{GetMutex(vid)};
            auto v_deg = m_deg_list.at(vid);
            assert(vid != nid);
            assert(vid < m_max_node);
            assert(nid < m_max_node);
            assert(v_deg <= m_max_deg);
            assert(IsValid(vid));

            // Greedy approach for updating edges
            // The greedy approach always keeps top max_degree closest edges
            // keep adjacency list and distance sorted, small distance edges will be stored in the front
            auto dist_ptr = &m_dist_list.at(vid * m_max_deg);
            auto adj_ptr = &m_adj_list.at(vid * m_max_deg);
            auto offset = std::lower_bound(dist_ptr, dist_ptr + v_deg, distance) - dist_ptr;
            if (offset < m_max_deg) {
                int end = std::min(v_deg, (int) m_max_deg - 1);
                for (int i = end; i > offset; i--) {
                    dist_ptr[i] = dist_ptr[i - 1];
                    adj_ptr[i] = adj_ptr[i - 1];
                }
                adj_ptr[offset] = nid;
                dist_ptr[offset] = distance;
                m_deg_list.at(vid) += v_deg < m_max_deg;
            }
            assert(IsValid(vid));
        };
    };

    using HNSWLayerPtr = std::shared_ptr<HNSWLayer>;
}