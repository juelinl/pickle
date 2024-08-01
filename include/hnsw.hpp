//
// Created by juelin on 7/13/24.
//

#ifndef PICKLE_HNSW_HPP
#define PICKLE_HNSW_HPP
#include "ndarray.hpp"
#include "distance.hpp"
#include "mempool.hpp"
#include "hnsw_layer.hpp"
#include "util.hpp"
#include "visited.hpp"
#include "profiler.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <immintrin.h>
#include <iostream>

namespace pickle::v2 {

    class HNSWGraph {
    public:
        HNSWGraph() = default;

        HNSWGraph(size_t max_degree, size_t build_ef, DistFunc df) {
            Init(max_degree, build_ef, df);
        };

        void Init(size_t max_degree_, size_t build_ef_, DistFunc df_) {
            m_max_degree = max_degree_;
            m_build_ef = build_ef_;
            m_df = df_;
            m_cumulative_probability = get_cumulative_probability(m_max_degree);
        }

        void InitBuffer(size_t max_nodes);

        template<class T, size_t Dim>
        void Build(const std::vector<external_id_t> &ext_ids, const NDArray &data);

        template<class T, size_t Dim>
        void Insert(external_id_t ext_id, std::span<const T, Dim> query);

        template<class T, size_t Dim, bool collect_metric=false>
        MaxHeap SearchLayer(internal_id_t enter_id, int level, int ef,
                            std::span<const T, Dim> query) const;

        template<class T, size_t Dim, bool collect_metric=false>
        MaxHeap SearchLayerFast(internal_id_t enter_id, int level, int ef,
                                           std::span<const T, Dim> query) const;

        template<class T, size_t Dim, bool collect_metric=false>
        std::vector<Entry> AnnSearch(int top_k, int search_ef, std::span<const T, Dim> query) const;

        template<class T, size_t Dim, bool collect_metric=false>
        [[nodiscard]] std::vector<std::vector<Entry>> AnnSearch(int top_k, int search_ef, NDArray query) const;

        template<class T, size_t Dim, bool collect_metric=false>
        internal_id_t SlideLayer(internal_id_t enter_id, int level,
                                 std::span<const T, Dim> query) const;

        inline std::vector<Entry> SelectSimple(int top_k,
                                               MaxHeap &nearest_neighbors) {
            auto ret = nearest_neighbors.extractMinTopK(top_k);
            return {ret.begin(), ret.end()};
        };

        template<class T, size_t Dim>
        inline std::pair<std::vector<Entry>, std::vector<Entry>> SelectHeuristic(int top_k, int level, MaxHeap& candidates, bool keep_pruned) {
            MinHeap W;
            MinHeap Wd;
            std::vector<Entry> R;
            std::vector<Entry> Pruned;
            for (const auto& e: candidates) {
                W.insert(e);
            }

            while(!W.empty() && R.size() < top_k) {
                auto v1 = W.top(); // v1 is the closest candidate to the query in the queue
                auto v1_q_dist = v1.m_dist;
                auto v1_id = v1.m_vid;
                std::span<const T, Dim> v1_data = GetData<T, Dim>(v1_id, level);

                bool insert_v1{true};
                for (const auto& v2: R) {
                    auto v2_id = v2.m_vid; // v2 is the nearest neighbors to return
                    std::span<const T, Dim> v2_data = GetData<T, Dim>(v2_id, level);
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
                while (!Wd.empty() && R.size() + Pruned.size() < top_k) {
                    Pruned.push_back(Wd.extractTop());
                }
            }
            assert(!R.empty());
            return std::make_pair(R, Pruned);
        };

        inline std::vector<Entry> SelectSimpleExt(int top_k,
                                           MaxHeap &nearest_neighbors) const {
            auto res = nearest_neighbors.extractMinTopK(top_k);
            for (auto &e: res) {
                e.m_vid = GetExtID(e.m_vid, 0);
            }
            return {res.begin(), res.end()};
        };

    private:
        size_t m_max_degree{0};
        size_t m_build_ef{0};
        size_t m_capacity{0};
        DistFunc m_df{DistFunc::L2};

        internal_id_t m_enter_level{0};
        std::mutex m_mutex_level;

        std::vector<HNSWLayerPtr> m_hnsw_layers;
        std::vector<double> m_cumulative_probability;
        std::vector<uint8_t> m_insert_level;
        std::vector<size_t> m_layer_capacity;
        NDArray m_data;

        [[nodiscard]] internal_id_t GetEnterLevel() const {
            return m_enter_level;
        }

        void SetEntLevel(internal_id_t new_level) {
            std::lock_guard<std::mutex> guard{m_mutex_level};
            if (new_level > m_enter_level) {
                m_enter_level = new_level;
            }
        }

        [[nodiscard]] bool empty(internal_id_t level) const {
            return m_hnsw_layers.at(level)->empty();
        }

        [[nodiscard]] size_t GetNumNode(internal_id_t level) const {
            return m_layer_capacity.at(level);
        }

        [[nodiscard]] external_id_t GetExtID(internal_id_t vid, internal_id_t level) const {
            return m_hnsw_layers.at(level)->GetExtID(vid);
        }

        [[nodiscard]] internal_id_t GetInID(external_id_t ext_id, internal_id_t level) const {
            return m_hnsw_layers.at(level)->GetInID(ext_id);
        }

        [[nodiscard]] internal_id_t NewInID(external_id_t ext_id, internal_id_t level) {
            return m_hnsw_layers.at(level)->NewInID(ext_id);
        }

        [[nodiscard]] internal_id_t GetEntInID(external_id_t ext_id, internal_id_t level) const {
            return m_hnsw_layers.at(level)->GetEntInID(ext_id);
        }

        template<class T, size_t Dim>
        [[nodiscard]] std::span<const T, Dim> GetData(internal_id_t vid,
                                                      internal_id_t level) const {
            return m_hnsw_layers.at(level)->GetData<T, Dim>(vid);
        }

        template<class T, size_t Dim>
        [[nodiscard]] std::span<const T, Dim> GetDataExt(external_id_t ext_id) const {
            return m_data.get_span<T, Dim>(ext_id);
        }

        [[nodiscard]] std::span<const internal_id_t>
        GetAdj(internal_id_t vid, internal_id_t level) const {
            return m_hnsw_layers.at(level)->GetAdj(vid);
        }

        [[nodiscard]] std::span<const distance_t> GetDist(internal_id_t vid,
                                                          internal_id_t level) const {
            return m_hnsw_layers.at(level)->GetDist(vid);
        }

        template<class T, size_t Dim>
        void AddEdge(internal_id_t vid, internal_id_t nid, distance_t distance,
                     internal_id_t level) {
            return m_hnsw_layers.at(level)->AddEdge<T, Dim>(vid, nid, distance);
        }

        template<class T, size_t Dim>
        void AddNode(internal_id_t vid,
                     std::span<const Entry> R,
                     std::span<const Entry> Pruned,
                     std::span<const T, Dim> vdata,
                     internal_id_t level) {
            return m_hnsw_layers.at(level)->AddNode<T, Dim>(vid, R, Pruned, vdata);
        }

        void PrefetchAdj(internal_id_t vid, int level) const {
            m_hnsw_layers.at(level)->PrefetchAdj(vid);
        }

        void PrefetchData(internal_id_t vid, int level) const {
            m_hnsw_layers.at(level)->PrefetchData(vid);
        }
    };

    void HNSWGraph::InitBuffer(size_t max_nodes) {
        m_capacity = max_nodes;
        m_insert_level = get_random_levels(m_capacity, m_cumulative_probability);
        int max_level = *std::max_element(m_insert_level.begin(), m_insert_level.end());
        m_layer_capacity = std::vector<size_t>(max_level + 1, 0);
        for (int l: m_insert_level) {
            for (size_t j = 0; j <= l; j++) {
                m_layer_capacity.at(j)++;
            }
        }

        m_hnsw_layers.clear();
        for (int i = 0; i <= max_level; i++) {
            size_t max_degree = (i == 0) ? 2 * m_max_degree : m_max_degree;
            size_t node_capacity = m_layer_capacity[i];
            size_t num_col = m_data.m_shape[1];
            std::vector<size_t> shape = {node_capacity, num_col};
            DataType dtype = m_data.m_dtype;
            auto max_node_id = static_cast<external_id_t>(m_capacity);
            auto g = std::make_shared<HNSWLayer>();
            g->Init(max_degree, node_capacity, max_node_id, dtype, shape, m_df);
            m_hnsw_layers.push_back(g);
        }
    }

    template<class T, size_t Dim>
    void HNSWGraph::Insert(external_id_t ext_id, std::span<const T, Dim> query) {
        int l = m_insert_level.at(ext_id);
        int L = GetEnterLevel();
        auto enter_ext_id{empty_external_id};
        auto enter_in_id{empty_internal_id};

        for (int level = L; level >= l + 1; level--) {
            if (!empty(level)) {
                enter_in_id = GetEntInID(enter_ext_id, level);
                enter_in_id = SlideLayer<T, Dim>(enter_in_id, level, query);
                enter_ext_id = GetExtID(enter_in_id, level);
            }
        }

        int top_search_level = std::min(l, L);
        int top_inserted_level = 0;

        typedef std::pair<std::vector<Entry>, std::vector<Entry>> ResultType;

        std::vector<ResultType> entry_list(top_search_level + 1);
        for (int level = top_search_level; level >= 0; level--) {
            if (!empty(level)) {
                enter_in_id = GetEntInID(enter_ext_id, level);
                auto top_k = (level == 0) ? 2 * m_max_degree : m_max_degree;
                auto ret = SearchLayer<T, Dim>(enter_in_id, level, m_build_ef, query);
                auto neighbors = SelectHeuristic<T, Dim>(top_k, level, ret, true);

                enter_in_id = neighbors.first[0].m_vid;
                enter_ext_id = GetExtID(enter_in_id, level);
                entry_list.at(level) = neighbors;
                top_inserted_level = std::max(top_inserted_level, level);
            }
        }

        // insert from base layer
        for (int level = 0; level <= l; level++) {
            auto vid = NewInID(ext_id, level);
            if (level <= top_inserted_level) {
                const auto &neighbors = entry_list.at(level);
                AddNode<T, Dim>(vid, neighbors.first, neighbors.second, query, level);

                for (const auto &e: neighbors.first) {
                    AddEdge<T, Dim>(e.m_vid, vid, e.m_dist, level);
                }

                for (const auto &e: neighbors.second) {
                    AddEdge<T, Dim>(e.m_vid, vid, e.m_dist, level);
                }
            }
        }

        // initialize empty adjacency list for empty level

        if (l > m_enter_level) SetEntLevel(l);
    };

    template<class T, size_t Dim, bool collect_metric>
    MaxHeap HNSWGraph::SearchLayer(internal_id_t enter_id, int level, int ef,
                                   std::span<const T, Dim> query) const {

        if (level==0) {
            return SearchLayerFast<T, Dim, collect_metric>(enter_id, level, ef, query);
        }
        assert(!empty(level));
        MinHeap top_candidates{};
        MaxHeap nearest_neighbors{};

        // local variables for profiling
        int num_hop{0};
        int num_dist{0};
        int num_neighbor{0};

        assert(top_candidates.begin() != nearest_neighbors.begin());
//        auto &visited = BitMap::ThreadLocal(m_capacity, GetNumNode(level));
        auto &visited = ByteMap::ThreadLocal(level, int(m_hnsw_layers.size()), GetNumNode(level));
        auto ent_data = GetData<T, Dim>(enter_id, level);
        auto ent_dist = Distance(query, ent_data, m_df);
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

            if (collect_metric) {
                num_hop++;
                num_neighbor += int(c_adj.size());
            }

            for (int vid: c_adj) {
                if (!visited.IsVisited(vid)) {
                    if (collect_metric) {
                        num_dist++;
                    }
                    visited.Mark(vid);
                    auto v_data = GetData<T, Dim>(vid, level);
                    auto v_dist = Distance(query, v_data, m_df);
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

        if (collect_metric) {
            Profiler::Global()->AddDist(level, num_dist);
            Profiler::Global()->AddNeighbor(level, num_neighbor);
            Profiler::Global()->AddHop(level, num_hop);
        }

        visited.Advance();
        return nearest_neighbors;
    };

    // prefetch is not helpful in current data layout
    template<class T, size_t Dim, bool collect_metric>
    MaxHeap HNSWGraph::SearchLayerFast(internal_id_t enter_id, int level, int ef,
                                   std::span<const T, Dim> query) const {
        assert(!empty(level));
        auto &visited = ByteMap::ThreadLocal(level, int(m_hnsw_layers.size()), GetNumNode(level));
        PrefetchAdj(enter_id, level);
        PrefetchData(enter_id, level);
        visited.Prefetch(enter_id);

        // local variables for profiling
        int num_hop{0};
        int num_dist{0};
        int num_neighbor{0};

        MinHeap top_candidates{};
        MaxHeap nearest_neighbors{};
        auto ent_data = GetData<T, Dim>(enter_id, level);
        auto ent_dist = Distance(query, ent_data, m_df);
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

            if constexpr (collect_metric) {
                num_hop++;
                num_neighbor += int(c_adj.size());
            }

            int idx{0};
            for (internal_id_t vid: c_adj) {
                const auto next_id = c_adj[++idx];
                PrefetchData(next_id, level);
                visited.Prefetch(next_id);

                if (!visited.IsVisited(vid)) {
                    if constexpr (collect_metric) {
                        num_dist++;
                    }
                    visited.Mark(vid);
                    auto v_data = GetData<T, Dim>(vid, level);
                    auto v_dist = Distance(query, v_data, m_df);
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

        if constexpr (collect_metric) {
            Profiler::Global()->AddHop(level, num_hop);
            Profiler::Global()->AddDist(level, num_dist);
            Profiler::Global()->AddNeighbor(level, num_neighbor);
        }
        return nearest_neighbors;
    };

    template<class T, size_t Dim, bool collect_metric>
    internal_id_t HNSWGraph::SlideLayer(internal_id_t enter_id, int level,
                                        std::span<const T, Dim> query) const {
        assert(!empty(level));
//        auto &visited = BitMap::ThreadLocal(m_capacity, GetNumNode(level));
        auto &visited = ByteMap::ThreadLocal(level, int(m_hnsw_layers.size()), GetNumNode(level));
        auto c_id = enter_id;
        auto c_data = GetData<T, Dim>(enter_id, level);
        auto c_dist = Distance(query, c_data, m_df);
        visited.Mark(enter_id);

        int num_hop{0};
        int num_dist{0};
        int num_neighbor{0};

        bool updated;
        do {
            updated = false;
            auto c_adj = GetAdj(c_id, level);
            if (collect_metric) {
                num_hop++;
                num_neighbor += int(c_adj.size());
            }
            for (const auto vid: c_adj) {
                if (!visited.IsVisited(vid)) {
                    if (collect_metric) {
                        num_dist++;
                    }
                    auto v_data = GetData<T, Dim>(vid, level);
                    auto v_dist = Distance(query, v_data, m_df);
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

        if constexpr (collect_metric) {
            Profiler::Global()->AddHop(level, num_hop);
            Profiler::Global()->AddDist(level, num_dist);
            Profiler::Global()->AddNeighbor(level, num_neighbor);
        }
        return c_id;
    };


    template<class T, size_t Dim>
    void HNSWGraph::Build(const std::vector<external_id_t> &ext_ids,
                          const NDArray &all_data) {
        m_data = all_data;
        InitBuffer(ext_ids.size());
#pragma omp parallel for schedule(static, 100)
        for (auto ext_id : ext_ids) {
            auto query = GetDataExt<T, Dim>(ext_id);
            Insert<T, Dim>(ext_id, query);
        }
    };

    template<class T, size_t Dim, bool collect_metric>
    std::vector<Entry> HNSWGraph::AnnSearch(int top_k, int search_ef, std::span<const T, Dim> query) const {
        auto ent_l = GetEnterLevel();
        auto enter_ext_id{empty_external_id};
        auto enter_in_id{empty_internal_id};

        for (int level = ent_l; level >= 1; level--) {
            enter_in_id = GetEntInID(enter_ext_id, level);
            enter_in_id = SlideLayer<T, Dim, collect_metric>(enter_in_id, level, query);
            enter_ext_id = GetExtID(enter_in_id, level);
        }

        enter_in_id = GetEntInID(enter_ext_id, 0);
//        auto res = SearchLayer<T, Dim, collect_metric>(enter_in_id, 0, search_ef, query);
        auto res = SearchLayerFast<T, Dim, collect_metric>(enter_in_id, 0, search_ef, query);
        return SelectSimpleExt(top_k, res);
    };


    template<class T, size_t Dim, bool collect_metric>
    std::vector<std::vector<Entry>> HNSWGraph::AnnSearch(int top_k, int search_ef, NDArray all_query) const {
        size_t num_row = all_query.m_shape[0];
        size_t num_col = all_query.m_shape[1];
        assert(num_col == Dim || Dim == std::dynamic_extent);
        std::vector<std::vector<Entry>> ret(num_row);

#pragma omp parallel for schedule(dynamic, 20)
        for (size_t i = 0; i < num_row; i++) {
            auto query = all_query.get_span<T, Dim>(i);
            ret[i] = AnnSearch<T, Dim, collect_metric>(top_k, search_ef, query);
        }
        return ret;
    };
} // namespace pickle

#endif //PICKLE_HNSW_HPP
