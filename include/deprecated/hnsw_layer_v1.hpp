//
// Created by juelin on 7/25/24.
//

#ifndef PICKLE_HNSW_LAYER_V1_HPP
#define PICKLE_HNSW_LAYER_V1_HPP
#include <mutex>
#include <vector>
#include <cassert>
#include <memory>
#include <iostream>
#include <map>
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <shared_mutex>

#include "common.hpp"
#include "queue.hpp"
namespace pickle::v1 {

    class Serializer;

    inline std::vector<Entry> merge(const std::vector<Entry>& vec1, const std::vector<Entry>& vec2) {
        std::vector<Entry> merged;
        merged.reserve(vec1.size() + vec2.size()); // Reserve space to avoid multiple allocations
        size_t i = 0, j = 0;

        // Merge vectors while there are elements in both
        while (i < vec1.size() && j < vec2.size()) {
            if (vec1[i].m_dist < vec2[j].m_dist) {
                merged.push_back(vec1[i]);
                ++i;
            } else {
                merged.push_back(vec2[j]);
                ++j;
            }
        }

        // Copy remaining elements from vec1
        while (i < vec1.size()) {
            merged.push_back(vec1[i]);
            ++i;
        }

        // Copy remaining elements from vec2
        while (j < vec2.size()) {
            merged.push_back(vec2[j]);
            ++j;
        }

        return merged;
    }

    class HNSWLayer {

    private:
        typedef uint16_t degree_t;

        struct NodeMeta
        {
            degree_t m_degree{0};
            degree_t m_heuristic{0};
            external_id_t m_ext_id{empty_external_id};
        };

    private:
        friend Serializer;
        bool m_use_table{false};
        bool m_is_empty{true};
        internal_id_t m_ent_id{0};
        internal_id_t m_next_id{0};
        size_t m_max_degree{0};
        size_t m_capacity{0};
        std::mutex m_id_mutex{};
        std::map<external_id_t, internal_id_t> m_ext2in_map; // mapping external id to internal id (only use if not base layer)
        mutable std::shared_mutex m_map_mutex;
        std::vector<internal_id_t > m_ext2in_table; // mapping external id to internal id (only use if is base layer)
        std::vector<internal_id_t> m_adj_list; // adjacency list
        std::vector<distance_t> m_dist_list; // distance between node to its neighbors
        std::vector<NodeMeta> m_node_list; // degree of adjacency list
        std::vector<std::mutex> m_update_mutex; // guard write to vector

        DistFunc m_df;
        NDArray m_data;

        std::mutex &GetMutex(internal_id_t vid) {
            return m_update_mutex.at(vid % m_update_mutex.size());
        };

        void Clear() {
            m_is_empty = true;
            m_max_degree = 0;
            m_capacity = 0;
            m_node_list.clear();
            m_adj_list.clear();
            m_dist_list.clear();
            m_ext2in_map.clear();
            m_data = NDArray();
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
            return m_capacity;
        }

        [[nodiscard]] size_t GetDegree(internal_id_t vid) const {
            return m_node_list.at(vid).m_degree;
        }

        void SetDegree(internal_id_t vid, degree_t degree) {
            m_node_list.at(vid).m_degree = degree;
        }

        [[nodiscard]] external_id_t GetExtID(internal_id_t vid) const {
            assert(!m_is_empty);
            assert(vid < m_capacity);
            return m_node_list.at(vid).m_ext_id;
        };

        void SetExtID(internal_id_t vid, external_id_t ext_id) {
            assert(vid < m_capacity);
            m_node_list.at(vid).m_ext_id = ext_id;
        };

        template<class T, size_t Dim>
        [[nodiscard]] std::span<const T, Dim> GetData(internal_id_t vid) {
            return m_data.get_span<T, Dim>(vid);
        }

        template<class T, size_t Dim>
        [[nodiscard]] std::span<T, Dim> GetMutableData(internal_id_t vid) {
            return m_data.get_span<T, Dim>(vid);
        }

        [[nodiscard]] internal_id_t NewInID(external_id_t ext_id) {
            const std::lock_guard<std::mutex> guard{m_id_mutex};
            internal_id_t new_in_id = m_next_id++;
            assert(new_in_id < m_capacity);
            SetExtID(new_in_id, ext_id);
            if (m_use_table) {
                assert(m_ext2in_table.at(ext_id) == empty_internal_id);
                m_ext2in_table.at(ext_id) = new_in_id;
            } else {
                std::unique_lock<std::shared_mutex> lock(m_map_mutex);
                assert(!m_ext2in_map.contains(ext_id));
                m_ext2in_map[ext_id] = new_in_id;
            }
            m_is_empty = false;
            return new_in_id;
        };

        [[nodiscard]] internal_id_t GetInID(external_id_t external_id) const {
            assert(!m_is_empty);
            if (m_use_table) {
                auto id = m_ext2in_table.at(external_id);
                assert(id != empty_internal_id);
                return id;
            } else {
                std::shared_lock<std::shared_mutex> lock(m_map_mutex);
                assert(m_ext2in_map.contains(external_id));
                auto id = m_ext2in_map.at(external_id);
                return id;
            }
        };

        [[nodiscard]] internal_id_t GetEntInID(external_id_t external_entry_id = empty_external_id) const {
            //TODO: better strategy for updating entry id
            if (external_entry_id == empty_external_id) return m_ent_id;
            return GetInID(external_entry_id);
        };

        [[nodiscard]] std::span<internal_id_t> GetAdj(internal_id_t vid) {
            assert(vid < m_capacity);
            return {m_adj_list.data() + m_max_degree * vid, GetDegree(vid)};
        };

        [[nodiscard]] std::span<const internal_id_t> GetAdj(internal_id_t vid) const {
            assert(vid < m_capacity);
            return {m_adj_list.data() + m_max_degree * vid, GetDegree(vid)};
        };

        [[nodiscard]] std::span<distance_t> GetDist(internal_id_t vid) {
            assert(vid < m_capacity);
            return {m_dist_list.data() + m_max_degree * vid, GetDegree(vid)};
        };

        [[nodiscard]] std::span<const distance_t> GetDist(internal_id_t vid) const {
            return {m_dist_list.data() + m_max_degree * vid, GetDegree(vid)};
        };

        [[nodiscard]] bool IsBase() const {return m_use_table;};

        [[nodiscard]] bool IsFirstTimeAddEdge(internal_id_t vid) const {
            return m_node_list.at(vid).m_heuristic == 0;
        }

        void Init(size_t max_degree, size_t node_capacity, external_id_t max_node_id, DataType dtype, const std::vector<size_t>& shape, DistFunc df) {
            Clear();
            m_use_table = true;
            m_max_degree = max_degree;
            m_capacity = node_capacity;
            m_adj_list.resize(m_capacity * m_max_degree, empty_internal_id);
            m_dist_list.resize(m_capacity * m_max_degree, std::numeric_limits<distance_t>::max());
            m_node_list.resize(m_capacity);
            m_ext2in_table.resize(max_node_id, empty_internal_id);
            m_update_mutex = std::vector<std::mutex>(std::max(8192ul, node_capacity));
            m_data = NDArray(dtype, shape);
            m_df = df;
        };

        void CreateMap() {
            if (m_use_table) {
                m_ext2in_table.clear();
                m_ext2in_table.resize(m_capacity, empty_internal_id);
                for (internal_id_t i = 0; i < m_next_id; i++) {
                    m_ext2in_table.at(GetExtID(i)) = i;
                }
            } else {
                m_ext2in_map.clear();
                for (internal_id_t i = 0; i < m_next_id; i++) {
                    m_ext2in_map.insert({GetExtID(i), i});
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

        template<class T, size_t Dim>
        void AddNode(internal_id_t vid, std::span<const Entry> neighbors, std::span<const T, Dim> vdata) {
            std::lock_guard<std::mutex> writeLock{GetMutex(vid)};
            assert(vid < m_capacity);
            auto adj = GetAdj(vid);
            auto dist = GetDist(vid);
            for (size_t i = 0; i < neighbors.size(); i++) {
                assert(vid != neighbors[i].m_vid);
                assert(neighbors[i].m_vid < m_capacity);
                assert(i+1 == neighbors.size() || neighbors[i].m_vid != neighbors[i+1].m_vid);
                adj[i] = neighbors[i].m_vid;
                dist[i] = neighbors[i].m_dist;
            }
            SetDegree(vid, neighbors.size());
            assert(IsValid(vid));
            auto buf = GetMutableData<T, Dim>(vid);
            for (size_t i = 0; i < vdata.size(); i++) {
                buf[i] = vdata[i];
            }
        }

        void AddEdgeSimple(internal_id_t vid, internal_id_t nid, distance_t distance) {
//            auto v_deg = m_deg_list.at(vid);
            auto v_deg = GetDegree(vid);
            assert(v_deg <= m_max_degree);
            assert(vid != nid);
            assert(vid < m_capacity);
            assert(nid < m_capacity);
            // Greedy approach for updating edges
            // The greedy approach always keeps top max_degree closest edges
            // keep adjacency list and distance sorted, small distance edges will be stored in the front
            auto dist_ptr = &m_dist_list.at(vid * m_max_degree);
            auto adj_ptr = &m_adj_list.at(vid * m_max_degree);
            auto offset = std::lower_bound(dist_ptr, dist_ptr + v_deg, distance) - dist_ptr;
            if (offset < m_max_degree) {
                int end = std::min(v_deg, m_max_degree - 1);
                for (int i = end; i > offset; i--) {
                    dist_ptr[i] = dist_ptr[i - 1];
                    adj_ptr[i] = adj_ptr[i - 1];
                }
                adj_ptr[offset] = nid;
                dist_ptr[offset] = distance;
                degree_t new_deg = v_deg + (v_deg < m_max_degree);
                SetDegree(vid, new_deg);
            }
            assert(IsValid(vid));
        }

        template<class T, size_t Dim>
        void AddEdgeHeuristicFirstTimeV1(internal_id_t vid, internal_id_t nid, distance_t distance, bool keep_pruned) {
            auto v_deg = GetDegree(vid);
            assert(v_deg == m_max_degree);
            assert(vid != nid);
            assert(vid < m_capacity);
            assert(nid < m_capacity);
//            MinHeap W;
            MinHeap Wd;
            std::vector<Entry> R;

            auto adj = GetAdj(vid);
            auto dist = GetDist(vid);
            auto offset = std::lower_bound(dist.begin(), dist.end(), distance) - dist.begin();
            {
                for (size_t v1_idx = 0; v1_idx < offset; v1_idx++){
                    auto v1_q_dist = dist[v1_idx];
                    auto v1_id = adj[v1_idx];
                    Entry v1{v1_q_dist, v1_id};

                    std::span<const T, Dim> v1_data = GetData<T, Dim>(v1_id);
                    bool insert_v1{true};

                    for (const auto& v2: R) {
                        auto v2_id = v2.m_vid; // v2 is the nearest neighbors to return
                        std::span<const T, Dim> v2_data = GetData<T, Dim>(v2_id);
                        auto v1_v2_dist = Distance(v1_data, v2_data, m_df);
                        // v1 is only inserted if it is closer to the query than any v2
                        // this step avoids adding only nearby nodes to the query's adj list
                        // it allows remote edges to be added as well
                        if (v1_q_dist > v1_v2_dist) {
                            insert_v1 = false;
                            break;
                        }
                    }
                    if (insert_v1) {
                        R.push_back(v1);
                    } else if (keep_pruned){
                        Wd.insert(v1);
                    }
                }

            }

            {
                auto v1_id = nid;
                auto v1_q_dist = distance;
                Entry v1{distance, nid};
                std::span<const T, Dim> v1_data = GetData<T, Dim>(v1_id);

                bool insert_v1{true};
                for (const auto& v2: R) {
                    auto v2_id = v2.m_vid; // v2 is the nearest neighbors to return
                    std::span<const T, Dim> v2_data = GetData<T, Dim>(v2_id);
                    auto v1_v2_dist = Distance(v1_data, v2_data, m_df);
                    // v1 is only inserted if it is closer to the query than any v2
                    // this step avoids adding only nearby nodes to the query's adj list
                    // it allows remote edges to be added as well
                    if (v1_q_dist > v1_v2_dist) {
                        insert_v1 = false;
                        break;
                    }
                }
                if (insert_v1) {
                    R.push_back(v1);
                } else if (keep_pruned){
                    Wd.insert(v1);
                }
            }

            {
                for (size_t v1_idx = offset; v1_idx < adj.size(); v1_idx++){
                    auto v1_q_dist = dist[v1_idx];
                    auto v1_id = adj[v1_idx];
                    Entry v1{v1_q_dist, v1_id};

                    std::span<const T, Dim> v1_data = GetData<T, Dim>(v1_id);
                    bool insert_v1{true};

                    for (const auto& v2: R) {
                        auto v2_id = v2.m_vid; // v2 is the nearest neighbors to return
                        std::span<const T, Dim> v2_data = GetData<T, Dim>(v2_id);
                        auto v1_v2_dist = Distance(v1_data, v2_data, m_df);
                        // v1 is only inserted if it is closer to the query than any v2
                        // this step avoids adding only nearby nodes to the query's adj list
                        // it allows remote edges to be added as well
                        if (v1_q_dist > v1_v2_dist) {
                            insert_v1 = false;
                            break;
                        }
                    }
                    if (insert_v1) {
                        R.push_back(v1);
                    } else if (keep_pruned){
                        Wd.insert(v1);
                    }
                }
            }


            if (keep_pruned) {
                std::vector<Entry> other;
                while(other.size() + R.size() < m_max_degree && !Wd.empty()) {
                    other.push_back(Wd.top());
                    Wd.pop();
                }

                R = merge(R, other);
            }

            assert(R.size() <= m_max_degree);
            degree_t new_deg = std::min(m_max_degree, R.size());

            for (degree_t i = 0; i < new_deg; i++) {
                adj[i] = R[i].m_vid;
                dist[i] = R[i].m_dist;
            }
            m_node_list.at(vid).m_heuristic = 1;
            SetDegree(vid, new_deg);
            assert(IsValid(vid));
        }

        template<class T, size_t Dim>
        void AddEdgeHeuristicFirstTimeV0(internal_id_t vid, internal_id_t nid, distance_t distance, bool keep_pruned) {
            auto v_deg = GetDegree(vid);
            assert(v_deg == m_max_degree);
            assert(vid != nid);
            assert(vid < m_capacity);
            assert(nid < m_capacity);
            MinHeap W;
            MinHeap Wd;
            std::vector<Entry> R;

            auto adj = GetAdj(vid);
            auto dist = GetDist(vid);

            W.insert(distance, nid);
            for (size_t i = 0; i < adj.size(); i++) {
                W.insert(dist[i], adj[i]);
            }

            while(!W.empty()) {
                auto v1 = W.top(); // v1 is the closest candidate to the query in the queue
                auto v1_q_dist = v1.m_dist;
                auto v1_id = v1.m_vid;
                std::span<const T, Dim> v1_data = GetData<T, Dim>(v1_id);

                bool insert_v1{true};
                for (const auto& v2: R) {
                    auto v2_id = v2.m_vid; // v2 is the nearest neighbors to return
                    std::span<const T, Dim> v2_data = GetData<T, Dim>(v2_id);
                    auto v1_v2_dist = Distance(v1_data, v2_data, m_df);
                    // v1 is only inserted if it is closer to the query than any v2
                    // this step avoids adding only nearby nodes to the query's adj list
                    // it allows remote edges to be added as well
                    if (v1_q_dist > v1_v2_dist) {
                        insert_v1 = false;
                        break;
                    }
                }
                if (insert_v1) {
                    R.push_back(v1);
                } else if (keep_pruned){
                    Wd.insert(v1);
                }
                W.pop();
            };

            if (keep_pruned) {
                std::vector<Entry> other;
                while(other.size() + R.size() < m_max_degree && !Wd.empty()) {
                    other.push_back(Wd.top());
                    Wd.pop();
                }

                R = merge(R, other);
            }

            assert(R.size() <= m_max_degree);
            degree_t new_deg = std::min(m_max_degree, R.size());

            for (degree_t i = 0; i < new_deg; i++) {
                adj[i] = R[i].m_vid;
                dist[i] = R[i].m_dist;
            }
            m_node_list.at(vid).m_heuristic = 1;
            SetDegree(vid, new_deg);
            assert(IsValid(vid));
        }

        // TODO: this method is buggy, fix it
        template<class T, size_t Dim>
        void AddEdgeHeuristic(const internal_id_t vid, const internal_id_t nid, const distance_t distance, bool keep_pruned) {
            auto v_deg = GetDegree(vid);
            assert(v_deg == m_max_degree);
            assert(vid != nid);
            assert(vid < m_capacity);
            assert(nid < m_capacity);

            auto dist = GetDist(vid);
            auto adj = GetAdj(vid);
            auto offset = std::lower_bound(dist.begin(), dist.end(), distance) - dist.begin();
            if (offset == v_deg) return;

            auto n_data = GetData<T, Dim>(nid);

            for (size_t v2_idx = 0; v2_idx < offset; v2_idx++) {
                auto v2_id = adj[v2_idx];
                auto v2_data = GetData<T, Dim>(v2_id);
                auto v1_v2_dist = Distance(n_data, v2_data, m_df);

                if (distance > v1_v2_dist) {
                    // no need to insert v1 hence return
                    return;
                }
            }

            if (keep_pruned) {
                int v2_idx = v_deg - 1; // remove last one

                while(v2_idx > offset) {
                    auto v2_id = adj[v2_idx];
                    auto v2_q_dist = dist[v2_idx];
                    auto v2_data = GetData<T, Dim>(v2_id);
                    auto n_v2_dist = Distance(n_data, v2_data, m_df);
                    if (v2_q_dist > n_v2_dist) {
                        break; // no need to keep v2 hence break
                    } else {
                        v2_idx--;
                    }
                }

                for (int i = v2_idx; i > offset; i--) {
                    adj[i] = adj[i - 1];
                    dist[i] = dist[i - 1];
                }
                adj[offset] = nid;
                dist[offset] = distance;
            } else {
                std::vector<Entry> R;
                for (size_t v2_idx = offset; v2_idx < m_max_degree; v2_idx++){
                    auto v2_id = adj[v2_idx];
                    auto v2_q_dist = dist[v2_idx];
                    auto v2_data = GetData<T, Dim>(v2_id);
                    auto v1_v2_dist = Distance(n_data, v2_data, m_df);

                    if (v2_q_dist < v1_v2_dist)  {
                        // no need to insert v1 hence return
                        R.emplace_back(v2_q_dist, v2_id);
                    }
                }

                adj[offset] = nid;
                dist[offset] = distance;
                for (size_t i = 0; i < R.size(); i++) {
                    if (i + offset + 1 < m_max_degree) {
                        adj[i + offset + 1] = R[i].m_vid;
                        dist[i + offset + 1] = R[i].m_dist;
                    }
                }
            }
            assert(IsValid(vid));
        }

        template<class T, size_t Dim>
        void AddEdge(internal_id_t vid, internal_id_t nid, distance_t distance) {

            std::lock_guard<std::mutex> writeLock{GetMutex(vid)};
            auto v_deg = GetDegree(vid);
            assert(IsValid(vid));
            if (v_deg < m_max_degree) {
                AddEdgeSimple(vid, nid, distance);
            } else {
                AddEdgeHeuristicFirstTimeV1<T, Dim>(vid, nid, distance, true);
            }
//            if (v_deg < m_max_deg) {
//                AddEdgeSimple(vid, nid, distance);
//            } else if (IsFirstTimeAddEdge(vid)) {
//                AddEdgeHeuristicFirstTimeV1<T, Dim>(vid, nid, distance, true);
//            } else {
//                AddEdgeHeuristic<T, Dim>(vid, nid, distance, true);
//            }
            assert(IsValid(vid));
        };
    };

    using HNSWLayerPtr = std::shared_ptr<HNSWLayer>;
}
#endif //PICKLE_HNSW_LAYER_V1_HPP
