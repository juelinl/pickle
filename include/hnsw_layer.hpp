//
// Created by juelin on 7/25/24.
//

#ifndef PICKLE_HNSW_LAYER_HPP
#define PICKLE_HNSW_LAYER_HPP

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

namespace pickle::v2 {

    class Serializer;

    inline std::vector<Entry> merge(std::span<Entry> vec1, std::span<Entry> vec2) {
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
        typedef int16_t degree_t;
        struct NodeMeta {
            degree_t m_degree{0};
            degree_t m_heuristic{0};
            external_id_t m_ext_id{empty_external_id};
        };

    private:
        friend Serializer;
        bool m_is_empty{true};
        internal_id_t m_ent_id{0};
        internal_id_t m_next_id{0};
        size_t m_max_degree{0};
        size_t m_capacity{0};
        std::mutex m_id_mutex{};
        std::vector<internal_id_t> m_ext2in_table; // mapping external id to internal id (only use if is base layer)
        std::vector<internal_id_t> m_adj_list; // adjacency list
        std::vector<distance_t> m_dist_list; // distance between node to its neighbors
        std::vector<NodeMeta> m_node_list; // degree of adjacency list
//        std::vector<std::mutex> m_update_mutex; // guard write to vector
        std::vector<omp_lock_t> m_update_lock; // guard write to vector

        DistFunc m_df{DistFunc::RUNTIME};
        NDArray m_data;

//        std::mutex &GetMutex(internal_id_t vid) {
//            return m_update_mutex.at(vid % m_update_mutex.size());
//        };

        void Clear() {
            m_is_empty = true;
            m_max_degree = 0;
            m_capacity = 0;
            m_node_list.clear();
            m_adj_list.clear();
            m_dist_list.clear();
            m_update_lock.clear();
            m_data = NDArray();
        };

    public:
        HNSWLayer() = default;

        [[nodiscard]] bool empty() const {
            return m_is_empty;
        }

//        [[nodiscard]] internal_id_t GetSize() const {
//            return m_next_id;
//        }
//
//        [[nodiscard]] size_t GetCapacity() const {
//            return m_capacity;
//        }

        [[nodiscard]] size_t GetDegree(internal_id_t vid) const {
            return m_node_list.at(vid).m_degree;
        }

        void SetDegree(internal_id_t vid, degree_t degree) {
            m_node_list.at(vid).m_degree = degree;
        }

        [[nodiscard]] size_t GetHeuristic(internal_id_t vid) const {
            return m_node_list.at(vid).m_heuristic;
        }

        void SetHeuristic(internal_id_t vid, degree_t heuristic) {
            m_node_list.at(vid).m_heuristic = heuristic;
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

        void LockID(internal_id_t vid) {
            omp_set_lock(&m_update_lock.at(vid));
        }

        void UnlockID(internal_id_t vid) {
            omp_unset_lock(&m_update_lock.at(vid));
        }

        void PrefetchData(internal_id_t vid) const {
            m_data.Prefetch(vid);
        }

        void PrefetchAdj(internal_id_t vid) const {
            _mm_prefetch(m_adj_list.data() + m_max_degree * vid, _MM_HINT_T0);
        }

        template<class T, size_t Dim>
        [[nodiscard]] std::span<const T, Dim> GetData(internal_id_t vid) {
            return m_data.get_span<T, Dim>(vid);
        }

        template<class T, size_t Dim>
        [[nodiscard]] std::span<T, Dim> GetMutableData(internal_id_t vid) {
            return m_data.get_span<T, Dim>(vid);
        }

        [[nodiscard]] internal_id_t NewInID(external_id_t ext_id) {
            std::lock_guard<std::mutex> guard{m_id_mutex};
            internal_id_t new_in_id = m_next_id++;
            SetExtID(new_in_id, ext_id);
            assert(new_in_id < m_capacity);
            assert(m_ext2in_table.at(ext_id) == empty_internal_id);
            m_ext2in_table.at(ext_id) = new_in_id;
            m_is_empty = false;
            return new_in_id;
        };

        [[nodiscard]] internal_id_t GetInID(external_id_t external_id) const {
            assert(!m_is_empty);
            auto id = m_ext2in_table.at(external_id);
            assert(id != empty_internal_id);
            return id;
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

        [[nodiscard]] std::span<distance_t> GetDist(internal_id_t vid) {
            assert(vid < m_capacity);
            return {m_dist_list.data() + m_max_degree * vid, GetDegree(vid)};
        };


        [[nodiscard]] std::span<const distance_t> GetDist(internal_id_t vid) const {
            assert(vid < m_capacity);
            return {m_dist_list.data() + m_max_degree * vid, GetDegree(vid)};
        };

        [[nodiscard]] std::span<distance_t> GetDistHeuristic(internal_id_t vid) {
            return {m_dist_list.data() + m_max_degree * vid, GetHeuristic(vid)};
        };

        [[nodiscard]] std::span<const distance_t> GetDistHeuristic(internal_id_t vid) const {
            return {m_dist_list.data() + m_max_degree * vid, GetHeuristic(vid)};
        };

        [[nodiscard]] std::span<internal_id_t> GetAdjHeuristic(internal_id_t vid) {
            return {m_adj_list.data() + m_max_degree * vid, GetHeuristic(vid)};
        };

        [[nodiscard]] std::span<const internal_id_t> GetAdjHeuristic(internal_id_t vid) const {
            return {m_adj_list.data() + m_max_degree * vid, GetHeuristic(vid)};
        };

        [[nodiscard]] std::span<distance_t> GetDistPruned(internal_id_t vid) {
            auto offset = GetHeuristic(vid);
            auto deg = GetDegree(vid);
            return {m_dist_list.data() + m_max_degree * vid + offset, deg - offset};
        };

        [[nodiscard]] std::span<const distance_t> GetDistPruned(internal_id_t vid) const {
            auto offset = GetHeuristic(vid);
            auto deg = GetDegree(vid);
            return {m_dist_list.data() + m_max_degree * vid + offset, deg - offset};
        };

        [[nodiscard]] std::span<internal_id_t> GetAdjPruned(internal_id_t vid) {
            auto offset = GetHeuristic(vid);
            auto deg = GetDegree(vid);
            return {m_adj_list.data() + m_max_degree * vid + offset, deg - offset};
        };

        [[nodiscard]] std::span<const internal_id_t> GetAdjPruned(internal_id_t vid) const {
            auto offset = GetHeuristic(vid);
            auto deg = GetDegree(vid);
            return {m_adj_list.data() + m_max_degree * vid + offset, deg - offset};
        };

        [[nodiscard]] bool IsFirstTimeAddEdge(internal_id_t vid) const {
            return m_node_list.at(vid).m_heuristic == 0;
        }

        void Init(size_t max_degree, size_t node_capacity, external_id_t max_node_id, DataType dtype,
                  const std::vector<size_t> &shape, DistFunc df) {
            Clear();
            m_max_degree = max_degree;
            m_capacity = node_capacity;
            m_adj_list.resize(m_capacity * m_max_degree, empty_internal_id);
            m_dist_list.resize(m_capacity * m_max_degree, std::numeric_limits<distance_t>::max());
            m_node_list.resize(m_capacity);
            m_ext2in_table.resize(max_node_id, empty_internal_id);
//            m_update_mutex = std::vector<std::mutex>(std::min(8192ul, m_capacity));
            m_update_lock.resize(m_capacity);
            for (size_t i = 0; i < m_capacity; i++) {
                omp_init_lock(&m_update_lock.at(i));
            }
            m_data = NDArray(dtype, shape);
            m_df = df;
        };

        void CreateMap() {
            m_ext2in_table.clear();
            m_ext2in_table.resize(m_capacity, empty_internal_id);
            for (internal_id_t i = 0; i < m_next_id; i++) {
                m_ext2in_table.at(GetExtID(i)) = i;
            }
        };

        [[nodiscard]] bool IsValid(internal_id_t vid) {
            auto adj_h = GetAdjHeuristic(vid);
            auto dist_h = GetDistHeuristic(vid);
            for (int i = 1; i < adj_h.size(); i++) {
                assert(adj_h[i] != empty_internal_id);
                assert(adj_h[i] != adj_h[i - 1]);
                assert(dist_h[i] >= dist_h[i - 1]);
            }

            auto adj_p = GetAdjPruned(vid);
            auto dist_p = GetDistPruned(vid);
            for (int i = 1; i < adj_p.size(); i++) {
                assert(adj_p[i] != empty_internal_id);
                assert(adj_p[i] != adj_p[i - 1]);
                assert(dist_p[i] >= dist_p[i - 1]);
            }
            return true;
        };

        template<class T, size_t Dim>
        void AddNode(internal_id_t vid,
                     std::span<const Entry> R,
                     std::span<const Entry> Pruned,
                     std::span<const T, Dim> vdata) {
            LockID(vid);

            auto buf = GetMutableData<T, Dim>(vid);
            for (size_t i = 0; i < vdata.size(); i++) {
                buf[i] = vdata[i];
            }

//            std::unique_lock<std::mutex> lock{GetMutex(vid)};
            assert(IsValid(vid));
            assert(vid < m_capacity);
            auto adj = GetAdj(vid);
            auto dist = GetDist(vid);
            degree_t heuristic = std::min(m_max_degree, R.size());

            for (size_t i = 0; i < heuristic; i++) {
                adj[i] = R[i].m_vid;
                dist[i] = R[i].m_dist;
            }

            degree_t degree = std::min(heuristic + Pruned.size(), m_max_degree);
            for (size_t i = 0; i < degree - heuristic; i++) {
                adj[i + heuristic] = Pruned[i].m_vid;
                dist[i + heuristic] = Pruned[i].m_dist;
            }

            SetDegree(vid, degree);
            SetHeuristic(vid, heuristic);
            assert(IsValid(vid));

            UnlockID(vid);
        }

//        void AddEdgePruned(internal_id_t vid, internal_id_t nid, distance_t distance) {
//            auto dist_p = GetDistPruned(vid);
//            auto adj_p = GetAdjPruned(vid);
//            degree_t heuristic = GetHeuristic(vid);
//            degree_t degree = GetDegree(vid);
//            degree_t pruned = degree - heuristic;
//            assert(degree == heuristic + adj_p.size());
//
//            auto offset = std::lower_bound(dist_p.begin(), dist_p.end(), distance) - dist_p.begin();
//            if (offset == pruned && degree < m_max_degree) {
//                adj_p[offset] = nid;
//                dist_p[offset] = distance;
//                SetDegree(vid, degree + 1);
//            } else if (offset < pruned) {
//                int end = (degree < m_max_degree) ? pruned : pruned - 1;
//                for (int i = end; i >= offset; i--){
//                    adj_p[i] = adj_p[i - 1];
//                    dist_p[i] = dist_p[i - 1];
//                }
//                adj_p[offset] = nid;
//                dist_p[offset] = distance;
//                degree += (degree < m_max_degree);
//                SetDegree(vid, degree);
//            }
//        }

        template<class T, size_t Dim>
        void AddEdge(internal_id_t vid, internal_id_t nid, distance_t distance) {
//            std::unique_lock<std::mutex> lock{GetMutex(vid)};
            LockID(vid);
            assert(IsValid(vid));
            auto adj = GetAdj(vid);
            auto dist = GetDist(vid);
            degree_t heuristic = GetHeuristic(vid);
            auto offset = std::lower_bound(dist.begin(), dist.begin() + heuristic, distance) - dist.begin();
            auto v2span = adj.subspan(0, offset);

            std::span<const T, Dim> n_data = GetData<T, Dim>(nid);

            bool insert_n{true};
            for (const auto &v2_id: v2span) {
                std::span<const T, Dim> v2_data = GetData<T, Dim>(v2_id);
                auto n_v2_dist = Distance<T, Dim>(n_data, v2_data, m_df);
                // v1 is only inserted if it is closer to the query than any v2
                // this step avoids adding only nearby nodes to the query's adj list
                // it allows remote edges to be added as well
                if (distance > n_v2_dist) {
                    insert_n = false;
                    break;
                }
            }

            if (!insert_n) {
//                AddEdgePruned(vid, nid, distance);
                auto dist_p = GetDistPruned(vid);
                auto adj_p = GetAdjPruned(vid);
                degree_t degree = adj.size();
                degree_t pruned = degree - heuristic;
                auto offset = std::lower_bound(dist_p.begin(), dist_p.end(), distance) - dist_p.begin();
                if (offset == pruned && degree < m_max_degree) {
                    adj_p[offset] = nid;
                    dist_p[offset] = distance;
                    SetDegree(vid, degree + 1);
                } else if (offset < pruned) {
                    int end = (degree < m_max_degree) ? pruned : pruned - 1;
                    for (int i = end; i >= offset; i--){
                        adj_p[i] = adj_p[i - 1];
                        dist_p[i] = dist_p[i - 1];
                    }
                    adj_p[offset] = nid;
                    dist_p[offset] = distance;
                    degree += (degree < m_max_degree);
                    SetDegree(vid, degree);
                }
                assert(IsValid(vid));
            } else {
                // need to update both R and Pruned
                MinHeap W;
                MinHeap Wd;
                std::vector<Entry> R;

                // populate the heap before inserting
                for (size_t i = offset; i < adj.size(); i++) {
                    W.insert(dist[i], adj[i]);
                }

                adj[offset] = nid;
                dist[offset] = distance;
                degree_t cur_h_size = offset + 1;
                while (!W.empty()) {
                    auto v1 = W.extractTop();
                    auto v1_id = v1.m_vid;
                    auto v1_q_dist = v1.m_dist;
                    auto v1_data = GetData<T, Dim>(v1_id);
                    bool insert_v1{true};

                    for (size_t i = 0; i < cur_h_size; i++) {
                        auto v2_id = adj[i];
                        auto v2_q_dist = dist[i];
                        auto v2_data = GetData<T, Dim>(v2_id);
                        auto v2_v1_dist = Distance<T, Dim>(v1_data, v2_data, m_df);
                        if (v2_v1_dist < v1_q_dist) {
                            insert_v1 = false;
                            break;
                        }
                    }

                    if (insert_v1) {
                        // R contains all elements in adj upto cur_h_size
                        // need to update inplace here so
                        // the remaining vertices in W can compare against all vertices in R
                        adj[cur_h_size] = v1_id;
                        dist[cur_h_size] = v1_q_dist;
                        cur_h_size++;
                    } else {
                        Wd.insert(v1);
                    }
                }
                degree_t cur_deg = cur_h_size;
                while (cur_deg < m_max_degree && !Wd.empty()) {
                    auto e = Wd.extractTop();
                    adj[cur_deg] = e.m_vid;
                    dist[cur_deg] = e.m_dist;
                    cur_deg++;
                }

                SetHeuristic(vid, cur_h_size);
                SetDegree(vid, cur_deg);
                assert(IsValid(vid));
            }
            UnlockID(vid);
        };
    };

    using HNSWLayerPtr = std::shared_ptr<HNSWLayer>;
}
#endif //PICKLE_HNSW_LAYER_HPP
