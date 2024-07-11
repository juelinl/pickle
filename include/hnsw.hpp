//
// Created by juelin on 7/9/24.
//

#ifndef PICKLE_HNSW_HPP
#define PICKLE_HNSW_HPP

#include "mempool.hpp"
#include "array2d.hpp"
#include "adjlist.hpp"
#include "util.hpp"
#include "visited.hpp"
#include "distance.hpp"

#include <cassert>
#include <atomic>
#include <iostream>
#include <immintrin.h>

namespace pickle {
    using T = uint8_t;
    static constexpr size_t Dim = 128;

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

        void Build(const std::vector<external_id_t> &ext_ids, const Array2DPtr &data);

        MaxHeap Search(internal_id_t enter_id, int level, int ef, std::span<const T, Dim> query) const ;

        // ef = 1
        internal_id_t Slide(internal_id_t enter_id, int level, std::span<const T, Dim> query) const;

        EntryVector SelectSimple(int top_k, MaxHeap& nearest_neighbors) {
            return nearest_neighbors.extractLastK(top_k);
        };



    private:
        size_t max_degree{0};
        size_t build_ef{0};
        size_t search_ef{0};
        size_t node_capacity{0};

        DistFunc df{DistFunc::L2};
        internal_id_t entrance_id{0};
        std::atomic<internal_id_t> next_id{0};
        std::mutex entrance_id_lock;

        std::vector<AdjList> adj_lists;
        std::vector<double> cum_prob;
        std::vector<spinlock> locks;

        Array2DPtr data{nullptr};

        template<class T, size_t Dim>
        std::span<const T, Dim> GetData(internal_id_t vid) const {
            const external_id_t ext_id = adj_lists.at(vid).GetExtID();
            return data->get_span<T, Dim>(ext_id);
        }

        void Prefetch(internal_id_t vid) const {
            const external_id_t ext_id = adj_lists.at(vid).GetExtID();
            _mm_prefetch(data->get_raw(ext_id), _MM_HINT_T1);
        }
    };


    void HNSWGraph::Build(const std::vector<external_id_t> &ext_ids, const Array2DPtr &all_data) {
        assert(ext_ids.size() == data->_shape[0]);
        assert(max_degree > 0);
        data = all_data;
        adj_lists.resize(ext_ids.size());
        node_capacity = ext_ids.size();
        locks = std::vector<spinlock>(ext_ids.size());

        for (auto ext_id: ext_ids) {
            auto vid = next_id++;
            auto enter_id = entrance_id;
            auto entry_node = adj_lists.at(enter_id);
            auto query = data->get_span<T, Dim>(ext_id);

            int l = get_random_level(cum_prob);
            int L = entry_node.GetMaxLevel();
            adj_lists.at(vid) = AdjList(ext_id, max_degree, l);
            auto& new_adj = adj_lists.at(vid);

            for (int level = L; level >= l + 1; level--) {
                enter_id = Slide(enter_id, level, query);
            }

            for (int lc = l; lc >= 0; lc--) {
                auto top_k = (lc == 0) ? 2 * max_degree : max_degree;
                auto ret = Search(enter_id, lc, build_ef, query);
                auto entries = SelectSimple(top_k, ret);
                enter_id = entries[0]._vid;
                new_adj.Add(lc, entries);
                for (auto e: entries) {
                    while(!locks.at(e._vid).try_lock());
                    adj_lists.at(e._vid).AddSingle(lc, vid, e._distance);
                    locks.at(e._vid).unlock();
                }
            }

            {
                std::lock_guard<std::mutex> lock(entrance_id_lock);
                if (L > adj_lists.at(entrance_id).GetMaxLevel()) {
                    entrance_id = vid;
                }
            }
        }
    }

    MaxHeap HNSWGraph::Search(internal_id_t enter_id, int level, int ef, std::span<const T, Dim> query) const {
        MinHeap top_candidates;
        MaxHeap nearest_neighbors;
        auto visited = VisitedTable::Global(node_capacity);
        auto c_id = enter_id;
        auto c_data = GetData<T, Dim>(enter_id);
        auto c_dist = Distance(query, c_data, df);
        auto f_dist = c_dist;

        nearest_neighbors.insert(c_dist, c_id);

        do {
            const auto& c_adj_list = adj_lists.at(c_id);
            auto c_adj = c_adj_list.GetAdj(level);
            for (int i = 0; i < c_adj.size(); i++) {
                const internal_id_t vid = c_adj[i];
                if (i < c_adj.size() - 1) Prefetch(c_adj[i + 1]);
                if (!visited.IsVisited(vid)) {
                    auto v_data = GetData<T, Dim>(vid);
                    auto v_dist = Distance(query, v_data, df);
                    Entry entry{v_dist, vid};
                    nearest_neighbors.insert(entry);
                    top_candidates.insert(entry);
                    visited.Mark(vid);
                }
            }

            while (nearest_neighbors.size() > ef) nearest_neighbors.pop();

            f_dist = nearest_neighbors.top()._distance;
            c_dist = top_candidates.top()._distance;
            c_id = top_candidates.top()._vid;
            top_candidates.pop();
        } while (!top_candidates.empty() && c_dist < f_dist);

        if (level == 0) visited.Advance();
        return nearest_neighbors;
    };

    internal_id_t HNSWGraph::Slide(internal_id_t enter_id, int level, std::span<const T, Dim> query) const {
        auto visited = VisitedTable::Global(node_capacity);
        auto c_id = enter_id;
        auto c_data = GetData<T, Dim>(enter_id);
        auto c_dist = Distance(query, c_data, df);
        bool updated;
        do {
            updated = false;
            const auto& c_adj_list = adj_lists.at(c_id);
            auto c_adj = c_adj_list.GetAdj(level);
            for (int i = 0; i < c_adj.size(); i++) {
                const internal_id_t vid = c_adj[i];
                if (i < c_adj.size() - 1) Prefetch(c_adj[i + 1]);
                if (!visited.IsVisited(vid)) {
                    auto v_data = GetData<T, Dim>(vid);
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

        if (level == 0) visited.Advance();
        return c_id;
    };
}
#endif //PICKLE_HNSW_HPP
