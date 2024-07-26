//
// Created by juelin on 7/25/24.
//

#ifndef PICKLE_HNSW_V1_HPP
#define PICKLE_HNSW_V1_HPP

#include "ndarray.hpp"
#include "distance.hpp"
#include "mempool.hpp"
#include "hnsw_layer_v0_backup.hpp"
#include "util.hpp"
#include "visited.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <immintrin.h>
#include <iostream>

namespace pickle::v0 {

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

        void InitBuffer(size_t max_nodes);

        template<class T, size_t Dim>
        void Build(const std::vector<external_id_t> &ext_ids, const NDArray &data);

        template<class T, size_t Dim>
        void Insert(external_id_t ext_id, std::span<const T, Dim> query);

        template<class T, size_t Dim>
        MaxHeap SearchLayer(internal_id_t enter_id, int level, int ef,
                            std::span<const T, Dim> query) const;

        template<class T, size_t Dim>
        MaxHeap SearchBaseLayer(internal_id_t enter_id, int level, int ef,
                                std::span<const T, Dim> query) const;

        template<class T, size_t Dim>
        EntryVector KnnSearch(int top_k, int search_ef, std::span<const T, Dim> query) const;

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
            for (auto &e: res) {
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

        std::vector<HNSWLayerPtr> graphs;
        std::vector<double> cum_prob;
        std::vector<uint8_t> insert_level;
        std::vector<size_t> num_nodes;
        NDArray data;

        [[nodiscard]] internal_id_t GetEnterLevel() const {
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
            return data.get_span<T, Dim>(ext_id);
        }

        void PrefetchData(internal_id_t vid, internal_id_t level) const {
            const external_id_t ext_id = graphs.at(level)->GetExtID(vid);
            _mm_prefetch(data.get_raw(ext_id), _MM_HINT_T0);
        }

        void PrefetchAdj(internal_id_t vid, internal_id_t level) const {
            graphs.at(level)->PrefetchAdj(vid);
        }

        template<class T, size_t Dim>
        [[nodiscard]] std::span<const T, Dim> GetDataExt(external_id_t ext_id) const {
            return data.get_span<T, Dim>(ext_id);
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

    void HNSWGraph::InitBuffer(size_t max_nodes) {
        node_capacity = max_nodes;
        insert_level = get_random_levels(node_capacity, cum_prob);
        int max_level = *std::max_element(insert_level.begin(), insert_level.end());
        num_nodes = std::vector<size_t>(max_level + 1, 0);
        for (int l: insert_level) {
            for (size_t j = 0; j <= l; j++) {
                num_nodes.at(j)++;
            }
        }

        graphs.clear();
        for (int i = 0; i <= max_level; i++) {
            size_t max_node_degree = (i == 0) ? 2 * max_degree : max_degree;
            auto g = std::make_shared<HNSWLayer>();
            bool is_base = (i == 0);
            g->Init(max_node_degree, num_nodes[i], is_base);
            graphs.push_back(g);
        }
    };

    template<class T, size_t Dim>
    void HNSWGraph::Insert(external_id_t ext_id, std::span<const T, Dim> query) {
        int l = insert_level.at(ext_id);
        int L = GetEnterLevel();
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
                auto top_k = (level == 0) ? 2 * max_degree : max_degree;
                auto ret = SearchLayer<T, Dim>(enter_in_id, level, build_ef, query);
                auto entries = SelectSimple(top_k, ret);
                enter_in_id = entries[0].m_vid;
                enter_ext_id = GetExtID(enter_in_id, level);
                entry_list.at(level) = std::move(entries);
                top_inserted_level = std::max(top_inserted_level, level);
            }
        }

        // insert from base layer
        for (int level = 0; level <= l; level++) {
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
    };

    template<class T, size_t Dim>
    void HNSWGraph::Build(const std::vector<external_id_t> &ext_ids,
                          const NDArray &all_data) {
        data = all_data;
        InitBuffer(ext_ids.size());

#pragma omp parallel for schedule(static, 128)
        for (auto ext_id : ext_ids) {
            auto query = GetDataExt<T, Dim>(ext_id);
            Insert<T, Dim>(ext_id, query);
        }
    };

//    template<class T, size_t Dim>
//    void HNSWGraph::Build(const std::vector<external_id_t> &ext_ids,
//                          const NDArray &all_data) {
//
//    }
//    template<class T, size_t Dim>
//    void HNSWGraph::Build(const std::vector<external_id_t> &ext_ids,
//                          const NDArray &all_data) {
//
//        assert(ext_ids.size() == all_data.m_shape[0]);
//        assert(max_degree > 0);
//        data = all_data;
//        node_capacity = ext_ids.size();
//        insert_level = get_random_levels(ext_ids.size(), cum_prob);
//        int max_level = *std::max_element(insert_level.begin(), insert_level.end());
//        num_nodes = std::vector<size_t>(max_level + 1, 0);
//        for (int l: insert_level) {
//            for (size_t j = 0; j <= l; j++) {
//                num_nodes.at(j)++;
//            }
//        }
//
//        graphs.clear();
//        for (int i = 0; i <= max_level; i++) {
//            size_t max_node_degree = (i == 0) ? 2 * max_degree : max_degree;
//            auto g = std::make_shared<HNSWLayer>();
//            bool is_base = (i == 0);
//            g->Init(max_node_degree, num_nodes[i], is_base);
//            graphs.push_back(g);
//        }
//
//#pragma omp parallel for schedule(static, 128)
//        for (auto ext_id: ext_ids) {
//            auto query = GetDataExt<T, Dim>(ext_id);
//            int l = insert_level.at(ext_id);
//            int L = GetEnterLevel();
//            auto enter_ext_id{empty_external_id};
//            auto enter_in_id{empty_internal_id};
//
//            for (int level = L; level >= l + 1; level--) {
//                if (!empty(level)) {
//                    enter_in_id = GetEntInID(enter_ext_id, level);
//                    enter_in_id = Slide<T, Dim>(enter_in_id, level, query);
//                    enter_ext_id = GetExtID(enter_in_id, level);
//                }
//            }
//
//            int top_search_level = std::min(l, L);
//            int top_inserted_level = 0;
//            std::vector<EntryVector> entry_list(top_search_level + 1);
//            for (int level = top_search_level; level >= 0; level--) {
//                if (!empty(level)) {
//                    enter_in_id = GetEntInID(enter_ext_id, level);
//                    auto top_k = (level == 0) ? 2 * max_degree : max_degree;
//
////                    if(level==0) {
////                        auto ret = SearchBaseLayer<T, Dim>(enter_in_id, level, build_ef, query);
////                        auto entries = SelectSimple(top_k, ret);
//////                        enter_in_id = entries[0].m_vid;
//////                        enter_ext_id = GetExtID(enter_in_id, level);
////                        entry_list.at(level) = std::move(entries);
////                        top_inserted_level = std::max(top_inserted_level, level);
////                    } else {
////                        auto ret = SearchLayer<T, Dim>(enter_in_id, level, build_ef, query);
////                        auto entries = SelectSimple(top_k, ret);
////                        enter_in_id = entries[0].m_vid;
////                        enter_ext_id = GetExtID(enter_in_id, level);
////                        entry_list.at(level) = std::move(entries);
////                        top_inserted_level = std::max(top_inserted_level, level);
////                    }
//
//
//                    auto ret = SearchLayer<T, Dim>(enter_in_id, level, build_ef, query);
//                    auto entries = SelectSimple(top_k, ret);
//                    enter_in_id = entries[0].m_vid;
//                    enter_ext_id = GetExtID(enter_in_id, level);
//                    entry_list.at(level) = std::move(entries);
//                    top_inserted_level = std::max(top_inserted_level, level);
//
//                }
//            }
//
//            // insert from base layer
//            for (int level = 0; level <= l; level++){
//                auto vid = NewInID(ext_id, level);
//                if (level <= top_inserted_level) {
//                    const auto &neighbors = entry_list.at(level);
//                    AddNode(vid, neighbors, level);
//                    for (const auto &e: neighbors) {
//                        AddEdge(e.m_vid, vid, e.m_dist, level);
//                    }
//                }
//            }
//
//            // initialize empty adjacency list for empty level
//
//
//            if (l > ent_level) SetEntLevel(l);
//        }
//    }

    template<class T, size_t Dim>
    MaxHeap HNSWGraph::SearchLayer(internal_id_t enter_id, int level, int ef,
                                   std::span<const T, Dim> query) const {
        assert(!empty(level));
        MinHeap top_candidates{};
        MaxHeap nearest_neighbors{};

        assert(top_candidates.begin() != nearest_neighbors.begin());
        auto &visited = VisitedTable::Global(node_capacity, GetNumNode(level));
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
            for (int vid: c_adj) {
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
    MaxHeap HNSWGraph::SearchBaseLayer(internal_id_t enter_id, int level, int ef,
                                       std::span<const T, Dim> query) const {
        assert(!empty(level));
        MinHeap top_candidates{};
        MaxHeap nearest_neighbors{};

        assert(top_candidates.begin() != nearest_neighbors.begin());
        auto &visited = VisitedTable::Global(node_capacity, GetNumNode(level));
//        HashBitmap visited(GetNumNode(level));
        auto ent_data = GetData<T, Dim>(enter_id, level);
        auto ent_dist = Distance(query, ent_data, df);
        Entry entry(ent_dist, enter_id);
        nearest_neighbors.insert(entry);
        top_candidates.insert(entry);
        visited.Mark(enter_id);

        while (!top_candidates.empty()) {
            PrefetchData(top_candidates.top().m_vid, level);

            auto top_e = top_candidates.extractTop();
            auto c_dist = top_e.m_dist;
            auto c_id = top_e.m_vid;
            auto f_dist = nearest_neighbors.top().m_dist;
            if (c_dist > f_dist)
                break;

            auto c_adj = GetAdj(c_id, level);
            for (int i = 0; i < c_adj.size(); i++) {
                const internal_id_t vid = c_adj[i];

                if (i + 1 < c_adj.size()) {
                    const internal_id_t next_vid = c_adj[i + 1];
                    PrefetchData(next_vid, level);
                    visited.Prefetch(next_vid);
                }

                if (!visited.IsVisited(vid)) {
                    visited.Mark(vid);
                    auto v_data = GetData<T, Dim>(vid, level);
                    auto v_dist = Distance(query, v_data, df);
                    if (nearest_neighbors.size() < ef || f_dist > v_dist) {
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
    EntryVector HNSWGraph::KnnSearch(int top_k, int search_ef, std::span<const T, Dim> query) const {
        auto ent_l = GetEnterLevel();
        auto enter_ext_id{empty_external_id};
        auto enter_in_id{empty_internal_id};

        for (int level = ent_l; level >= 1; level--) {
            enter_in_id = GetEntInID(enter_ext_id, level);
            enter_in_id = Slide<T, Dim>(enter_in_id, level, query);
            enter_ext_id = GetExtID(enter_in_id, level);
        }

        enter_in_id = GetEntInID(enter_ext_id, 0);
        auto res = SearchBaseLayer<T, Dim>(enter_in_id, 0, search_ef, query);
        return SelectSimpleExt(top_k, res);
    };

    template<class T, size_t Dim>
    internal_id_t HNSWGraph::Slide(internal_id_t enter_id, int level,
                                   std::span<const T, Dim> query) const {
        assert(!empty(level));
        auto &visited = VisitedTable::Global(node_capacity, GetNumNode(level));
        auto c_id = enter_id;
        auto c_data = GetData<T, Dim>(enter_id, level);
        auto c_dist = Distance(query, c_data, df);
        visited.Mark(enter_id);

        bool updated;
        do {
            updated = false;
            auto c_adj = GetAdj(c_id, level);
            for (const auto vid: c_adj) {
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

#endif //PICKLE_HNSW_V1_HPP
