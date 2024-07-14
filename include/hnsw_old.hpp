//
// Created by juelin on 7/9/24.
//

#ifndef PICKLE_HNSW_OLD_HPP
#define PICKLE_HNSW_OLD_HPP

#include "adjlist.hpp"
#include "array2d.hpp"
#include "distance.hpp"
#include "mempool.hpp"
#include "util.hpp"
#include "visited.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <immintrin.h>
#include <iostream>

namespace pickle {
    class HNSWGraph {
    public:
        HNSWGraph() = default;

        HNSWGraph(size_t max_degree, size_t build_ef, size_t search_ef, DistFunc df) {
            Init(max_degree, build_ef, search_ef, df);
        };

        void Init(size_t max_degree, size_t build_ef, size_t search_ef, DistFunc df) {
            this->max_degree = max_degree;
            this->build_ef = build_ef;
            this->search_ef = search_ef;
            this->df = df;
            this->cum_prob = get_cumulative_probability(max_degree);
        }

        template<class T, size_t Dim>
        void Build(const std::vector<external_id_t> &ext_ids, const Array2DPtr &data);

        template<class T, size_t Dim>
        MaxHeap Search(internal_id_t enter_id, int level, int ef,
                       std::span<const T, Dim> query) const;

        template<class T, size_t Dim>
        internal_id_t Slide(internal_id_t enter_id, int level,
                            std::span<const T, Dim> query) const;

        static inline EntryVector SelectSimple(int top_k,
                                               MaxHeap &nearest_neighbors) {
            return nearest_neighbors.extractMinTopK(top_k);
        };

    private:
        size_t max_degree{0};
        size_t build_ef{0};
        size_t node_capacity{0};
        size_t search_ef{0};
        DistFunc df{DistFunc::L2};

        internal_id_t entrance_id{0};
        internal_id_t next_id{0};
        std::mutex id_mutex;

        std::vector<AdjLists> adj_lists;
        //        std::vector<DynamicNSWGraph> graphs;
        std::vector<double> cum_prob;
        std::vector<std::vector<std::mutex>> update_mutex;
        std::vector<size_t> num_nodes;
        Array2DPtr data{nullptr};

        //        template<class T, size_t Dim>
        //        [[nodiscard]] std::span<const T, Dim> GetData(internal_id_t vid)
        //        const {
        //            const external_id_t ext_id = adj_lists.at(vid).GetExtID();
        //            return data->get_span<T, Dim>(ext_id);
        //        }

        template<class T, size_t Dim>
        [[nodiscard]] std::span<const T, Dim> GetData(internal_id_t vid,
                                                      internal_id_t level) const {
            //            const external_id_t ext_id = graphs.at(vid).GetExtID(vid);
            const external_id_t ext_id = adj_lists.at(vid).GetExtID();
            return data->get_span<T, Dim>(ext_id);
        }

        template<class T, size_t Dim>
        [[nodiscard]] std::span<const T, Dim> GetDataExt(external_id_t ext_id) const {
            return data->get_span<T, Dim>(ext_id);
        }

        [[nodiscard]] std::span<const internal_id_t>
        GetAdj(internal_id_t vid, internal_id_t level) const {
            //            return graphs.at(level).GetAdj(vid);
            return adj_lists.at(vid).GetAdj(level);
        }

        [[nodiscard]] std::span<const distance_t> GetDist(internal_id_t vid,
                                                          internal_id_t level) const {
            //            return graphs.at(level).GetDist(vid);
            return adj_lists.at(vid).GetDist(level);
        }

        void AddEdge(internal_id_t vid, internal_id_t nid, distance_t distance,
                     internal_id_t level) {
            //            graphs.at(level).AddEdge(vid, nid, distance);
            std::unique_lock<std::mutex> lock(GetMutex(vid, level));
            return adj_lists.at(vid).AddSingle(level, nid, distance);
        }

        void AddNode(internal_id_t vid, std::span<Entry> neighbors,
                     internal_id_t level) {
            //            graphs.at(level).Add(vid, neighbors);
            std::unique_lock<std::mutex> lock(GetMutex(vid, level));
            return adj_lists.at(vid).Add(level, neighbors);
        }

        //        void Prefetch(internal_id_t vid) const {
        //            const external_id_t ext_id = adj_lists.at(vid).GetExtID();
        //            _mm_prefetch(data->get_raw(ext_id), _MM_HINT_T1);
        //        }

        std::mutex &GetMutex(internal_id_t vid, internal_id_t level) {
            auto &mutex_vec = update_mutex.at(level);
            return mutex_vec.at(vid % mutex_vec.size());
        }
    };

    template<class T, size_t Dim>
    void HNSWGraph::Build(const std::vector<external_id_t> &ext_ids,
                          const Array2DPtr &all_data) {
        assert(ext_ids.size() == all_data->_shape[0]);
        assert(max_degree > 0);
        data = all_data;
        node_capacity = ext_ids.size();
        std::vector<int> max_levels = get_random_levels(ext_ids.size(), cum_prob);
        int global_max_level =
                *std::max_element(max_levels.begin(), max_levels.end());
        num_nodes = std::vector<size_t>(global_max_level + 1, 0);
        for (int l: max_levels) {
            for (size_t j = 0; j <= l; j++) {
                num_nodes.at(j)++;
            }
        }

        for (int i = 0; i <= global_max_level; i++) {
            size_t num_mutex = (i == 0) ? 8192 : num_nodes[i];
            update_mutex.emplace_back(num_mutex);
        }

        adj_lists.clear();
        adj_lists.resize(node_capacity);
//        std::vector<external_id_t> sorted_ext_id(ext_ids);
//        std::sort(sorted_ext_id.begin(), sorted_ext_id.end(),
//                  [&max_levels](const external_id_t a, const external_id_t b) {
//                      return max_levels[a] > max_levels[b];
//                  }); // sort ext_ids by max level so ext_ids with higher levels have smaller id

        for (size_t i = 0; i < ext_ids.size(); i++) {
            auto vid = next_id++;
            auto ext_id = ext_ids.at(i);
            auto l = max_levels.at(ext_id);
            adj_lists.at(vid) = AdjLists(ext_id, max_degree, l);
        }

#pragma omp parallel for schedule(static, 128)
        for (internal_id_t vid = 1; vid < adj_lists.size(); vid++) {
            auto &new_adj = adj_lists.at(vid);
            auto ext_id = new_adj.GetExtID();
            auto enter_id = entrance_id;
            auto entry_node = adj_lists.at(enter_id);
            auto query = GetDataExt<T, Dim>(ext_id);
            int l = new_adj.GetMaxLevel();
            int L = entry_node.GetMaxLevel();

            for (int level = L; level >= l + 1; level--) {
                enter_id = Slide<T, Dim>(enter_id, level, query);
            }

            std::stack<EntryVector> e_stack;
            for (int lc = std::min(l, L); lc >= 0; lc--) {
                auto top_k = (lc == 0) ? 2 * max_degree : max_degree;
                auto ret = Search<T, Dim>(enter_id, lc, build_ef, query);
                auto entries = SelectSimple(top_k, ret);
                enter_id = entries[0].m_vid;
                new_adj.CheckValid();
                e_stack.push(std::move(entries));
            }
            adj_lists.at(vid) = new_adj;
            int lc = 0;
            while (!e_stack.empty()) {
                auto &neighbors = e_stack.top();
                AddNode(vid, neighbors, lc);
                //                new_adj.Add(lc, neighbors);
                for (const auto &e: neighbors) {
                    AddEdge(e.m_vid, vid, e.m_dist, lc);
                    //                    std::unique_lock<std::mutex>
                    //                    update_lock(GetMutex(e.m_vid, lc)); auto
                    //                    &e_adj_list = adj_lists.at(e.m_vid);
                    //                    e_adj_list.AddSingle(lc, vid, e.m_dist);
                }
                e_stack.pop();
                lc++;
            }

            {
                std::unique_lock<std::mutex> id_lock{id_mutex};
                if (L > adj_lists.at(entrance_id).GetMaxLevel()) {
                    entrance_id = vid;
                }
            }
        }
    }

    template<class T, size_t Dim>
    MaxHeap HNSWGraph::Search(internal_id_t enter_id, int level, int ef,
                              std::span<const T, Dim> query) const {
        MinHeap top_candidates;
        MaxHeap nearest_neighbors;
        auto visited = VisitedTable::Global(node_capacity);
        auto ent_data = GetData<T, Dim>(enter_id, level);
        auto ent_dist = Distance(query, ent_data, df);

        nearest_neighbors.insert(ent_dist, enter_id);
        top_candidates.insert(ent_dist, enter_id);
        visited.Mark(enter_id);
        while (!top_candidates.empty()) {
            auto [c_dist, c_id] = top_candidates.extractTop();
            auto f_dist = nearest_neighbors.top().m_dist;
            if (c_dist > f_dist)
                break;
            //            auto c_adj = adj_lists.at(c_id).GetAdj(level);
            //            auto c_adj = graphs.at(level).GetAdj(c_id);
            auto c_adj = GetAdj(c_id, level);
            for (int i = 0; i < c_adj.size(); i++) {
                const internal_id_t vid = c_adj[i];
                //                if (i < c_adj.size() - 1) Prefetch(c_adj[i + 1]);
                if (!visited.IsVisited(vid)) {
                    visited.Mark(vid);
                    auto v_data = GetData<T, Dim>(vid, level);
                    auto v_dist = Distance(query, v_data, df);
                    if (nearest_neighbors.size() < ef ||
                        nearest_neighbors.top().m_dist > v_dist) {
                        nearest_neighbors.insert(v_dist, vid);
                        top_candidates.insert(v_dist, vid);
                        if (nearest_neighbors.size() > ef)
                            nearest_neighbors.pop();
                    }
                }
            }
        }
        visited.Advance();
        return nearest_neighbors;
    };

    template<class T, size_t Dim>
    internal_id_t HNSWGraph::Slide(internal_id_t enter_id, int level,
                                   std::span<const T, Dim> query) const {
        auto visited = VisitedTable::Global(node_capacity);
        auto c_id = enter_id;
        auto c_data = GetData<T, Dim>(enter_id, level);
        auto c_dist = Distance(query, c_data, df);
        visited.Mark(enter_id);

        bool updated;
        do {
            updated = false;
            //            auto c_adj = adj_lists.at(c_id).GetAdj(level);
            //            auto c_adj = graphs.at(level).GetAdj(c_id);
            auto c_adj = GetAdj(c_id, level);
            for (int i = 0; i < c_adj.size(); i++) {
                const internal_id_t vid = c_adj[i];
                //                if (i < c_adj.size() - 1) Prefetch(c_adj[i + 1]);
                if (!visited.IsVisited(vid)) {
                    auto v_data = GetData<T, Dim>(vid, level);
                    auto v_dist = Distance(query, v_data, df);
                    if (v_dist < c_dist) {
                        c_id = vid;
                        c_dist = v_dist;
                        updated = true;
                    }
                    visited.Mark(vid);
                }
            }
        } while (updated);

        visited.Advance();
        return c_id;
    };
} // namespace pickle
#endif // PICKLE_HNSW_OLD_HPP
