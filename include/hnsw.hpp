//
// Created by juelin on 7/13/24.
//

#ifndef PICKLE_HNSW_HPP
#define PICKLE_HNSW_HPP

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

        HNSWGraph(size_t max_degree, size_t build_ef, DistFunc df) {
            Init(max_degree, build_ef, df);
        };

        void Init(size_t max_degree_, size_t build_ef_, DistFunc df_) {
            max_degree = max_degree_;
            build_ef = build_ef_;
            df = df_;
            cum_prob = get_cumulative_probability(max_degree);
        }

        template<class T, size_t Dim>
        void Build(const std::vector<external_id_t> &ext_ids, const Array2DPtr &data);

        template<class T, size_t Dim>
        MaxHeap Search(internal_id_t enter_id, int level, int ef,
                       std::span<const T, Dim> query) const;

        template<class T, size_t Dim>
        EntryVector Search(int top_k, int search_ef, std::span<const T, Dim> query) const;

        template<class T, size_t Dim>
        internal_id_t Slide(internal_id_t enter_id, int level,
                            std::span<const T, Dim> query) const;

        static inline EntryVector SelectSimple(int top_k,
                                               MaxHeap &nearest_neighbors) {
            return nearest_neighbors.extractMinTopK(top_k);
        };

        inline EntryVector SelectSimpleExt(int top_k,
                                               MaxHeap &nearest_neighbors) const {
            auto res = nearest_neighbors.extractMinTopK(top_k);
            for (auto& e: res) {
                e.m_vid = GetExtID(e.m_vid, 0);
            }
            return res;
        };
    private:
        size_t max_degree{0};
        size_t build_ef{0};
        size_t node_capacity{0};
        DistFunc df{DistFunc::L2};

        internal_id_t ent_level{0};
        internal_id_t next_id{0};
        std::mutex id_mutex;
        std::mutex level_mutex;

        std::vector<DynamicNSWGraphPtr> graphs;
        std::vector<double> cum_prob;
        std::vector<size_t> num_nodes;
        Array2DPtr data{nullptr};

        [[nodiscard]] internal_id_t GetEntLevel() const {
            return ent_level;
        }

        void SetEntLevel(internal_id_t new_level) {
            std::lock_guard<std::mutex> guard{level_mutex};
            if (new_level > ent_level) {
                ent_level = new_level;
            }
        }

        [[nodiscard]] bool empty(internal_id_t level) const {
            return graphs.at(level)->empty();
        }

        [[nodiscard]] size_t GetNumNode(internal_id_t level) const {
            return num_nodes.at(level);
        }

        [[nodiscard]] external_id_t GetExtID(internal_id_t vid, internal_id_t level) const {
            return graphs.at(level)->GetExtID(vid);
        }

        [[nodiscard]] internal_id_t GetInID(external_id_t ext_id, internal_id_t level) const {
            return graphs.at(level)->GetInID(ext_id);
        }

        [[nodiscard]] internal_id_t NewInID(external_id_t ext_id, internal_id_t level) {
            return graphs.at(level)->NewInID(ext_id);
        }

        [[nodiscard]] internal_id_t GetEntInID(external_id_t ext_id, internal_id_t level) const {
            return graphs.at(level)->GetEntInID(ext_id);
        }

        template<class T, size_t Dim>
        [[nodiscard]] std::span<const T, Dim> GetData(internal_id_t vid,
                                                      internal_id_t level) const {
            const external_id_t ext_id = graphs.at(level)->GetExtID(vid);
            return data->get_span<T, Dim>(ext_id);
        }

        template<class T, size_t Dim>
        [[nodiscard]] std::span<const T, Dim> GetDataExt(external_id_t ext_id) const {
            return data->get_span<T, Dim>(ext_id);
        }

        [[nodiscard]] std::span<const internal_id_t>
        GetAdj(internal_id_t vid, internal_id_t level) const {
            return graphs.at(level)->GetAdj(vid);
        }

        [[nodiscard]] std::span<const distance_t> GetDist(internal_id_t vid,
                                                          internal_id_t level) const {
            return graphs.at(level)->GetDist(vid);
        }

        void AddEdge(internal_id_t vid, internal_id_t nid, distance_t distance,
                     internal_id_t level) {
            return graphs.at(level)->AddEdge(vid, nid, distance);
        }

        void AddNode(internal_id_t vid, std::span<const Entry> neighbors,
                     internal_id_t level) {
            return graphs.at(level)->Add(vid, neighbors);
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

        graphs.clear();
        for (int i = 0; i<=global_max_level; i++) {
            size_t max_node_degree = (i == 0) ? 2 * max_degree : max_degree;
            auto g = std::make_shared<DynamicNSWGraph>();
            bool is_base = (i == 0);
            g->Init(max_node_degree, num_nodes[i], is_base);
            graphs.push_back(g);
        }

#pragma omp parallel for schedule(dynamic, 64)
        for (auto ext_id: ext_ids) {
            auto query = GetDataExt<T, Dim>(ext_id);
            int l = max_levels.at(ext_id);
            int L = GetEntLevel();
            auto enter_ext_id{empty_external_id};
            auto enter_in_id{empty_internal_id};

            for (int level = L; level >= l + 1; level--) {
                if (!empty(level)) {
                    enter_in_id = GetEntInID(enter_ext_id, level);
                    enter_in_id = Slide<T, Dim>(enter_in_id, level, query);
                    enter_ext_id = GetExtID(enter_in_id, level);
                }
            }

            int top_search_level = std::min(l, L);
            int top_inserted_level = 0;
            std::vector<EntryVector> entry_list(top_search_level + 1);
            for (int level = top_search_level; level >= 0; level--) {
                if (!empty(level)) {
                    enter_in_id = GetEntInID(enter_ext_id, level);
                    auto ret = Search<T, Dim>(enter_in_id, level, build_ef, query);
                    auto top_k = (level == 0) ? 2 * max_degree : max_degree;
                    auto entries = SelectSimple(top_k, ret);
                    enter_in_id = entries[0].m_vid;
                    enter_ext_id = GetExtID(enter_in_id, level);
                    entry_list.at(level) = std::move(entries);
                    top_inserted_level = std::max(top_inserted_level, level);
                }
            }

            // insert from base layer
            for (int level = 0; level <= l; level++){
                auto vid = NewInID(ext_id, level);
                if (level <= top_inserted_level) {
                    const auto &neighbors = entry_list.at(level);
                    AddNode(vid, neighbors, level);
                    for (const auto &e: neighbors) {
                        AddEdge(e.m_vid, vid, e.m_dist, level);
                    }
                }
            }

            // initialize empty adjacency list for empty level


            if (l > ent_level) SetEntLevel(l);
        }
    }

    template<class T, size_t Dim>
    MaxHeap HNSWGraph::Search(internal_id_t enter_id, int level, int ef,
                              std::span<const T, Dim> query) const {
        assert(!empty(level));
        MinHeap top_candidates{};
        MaxHeap nearest_neighbors{};

        assert(top_candidates.begin() != nearest_neighbors.begin());
        auto& visited = VisitedTable::Global(node_capacity, GetNumNode(level));
        auto ent_data = GetData<T, Dim>(enter_id, level);
        auto ent_dist = Distance(query, ent_data, df);
        Entry entry(ent_dist, enter_id);
        nearest_neighbors.insert(entry);
        top_candidates.insert(entry);
        visited.Mark(enter_id);
        while (!top_candidates.empty()) {
            auto top_e = top_candidates.extractTop();
            auto c_dist = top_e.m_dist;
            auto c_id = top_e.m_vid;
            auto f_dist = nearest_neighbors.top().m_dist;
            if (c_dist > f_dist)
                break;
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
    EntryVector HNSWGraph::Search(int top_k, int search_ef, std::span<const T, Dim> query) const {
        auto ent_l = GetEntLevel();
        auto enter_ext_id{empty_external_id};
        auto enter_in_id{empty_internal_id};

        for (int level = ent_l; level >= 1; level--) {
            enter_in_id = GetEntInID(enter_ext_id, level);
            enter_in_id = Slide<T, Dim>(enter_in_id, level, query);
            enter_ext_id = GetExtID(enter_in_id, level);
        }

        enter_in_id = GetEntInID(enter_ext_id, 0);
        auto res = Search<T, Dim>(enter_in_id, 0, search_ef, query);
        return SelectSimpleExt(top_k, res);
    };

    template<class T, size_t Dim>
    internal_id_t HNSWGraph::Slide(internal_id_t enter_id, int level,
                                   std::span<const T, Dim> query) const {
        assert(!empty(level));
        auto& visited = VisitedTable::Global(node_capacity, GetNumNode(level));
        auto c_id = enter_id;
        auto c_data = GetData<T, Dim>(enter_id, level);
        auto c_dist = Distance(query, c_data, df);
        visited.Mark(enter_id);

        bool updated;
        do {
            updated = false;
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

#endif //PICKLE_HNSW_HPP
