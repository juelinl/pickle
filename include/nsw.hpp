//
// Created by juelin on 7/1/24.
//

#ifndef PICKLE_NSW_HPP
#define PICKLE_NSW_HPP

#include "queue.hpp"
#include "graph.hpp"
#include "distance.hpp"
#include "util.hpp"

#include <iostream>
#include <set>
#include <atomic>
#include <cassert>
#include <omp.h>
#include <stack>

namespace pickle {

    inline std::vector<Entry> SelectNeighborsSimple(size_t top_k, MaxQueue queue) {
        while (queue.size() > top_k) queue.pop();
        
        size_t num_entry = queue.size();
        std::vector<Entry> ret(num_entry);
        for (size_t i = 0; i < num_entry; i++) {
            ret.at(num_entry - i - 1) = queue.top();
            queue.pop();
        }
        for (size_t i = 1; i < num_entry; i++) {
            assert(ret[i - 1] != ret[i]);
        }
        assert(queue.empty());
        return ret;
    }

    template<class T, size_t Dim>
    inline std::vector<Entry> SelectNeighborsHeuristic(DistFunc df, size_t dim, size_t top_k, MaxQueue queue, std::span<const T> all_data, const DynamicNSWGraphPtr& graph, bool keepPruned = true) {
        size_t num_entry = queue.size();
        MinQueue W;
        MinQueue Wd;
        std::vector<Entry> R;
        for (size_t i = 0; i < num_entry; i++) {
            W.emplace(queue.top());
            queue.pop();
            // TODO: allow extending the candidates W by the neighbors of the inserted ones
        }

        while (!W.empty() and R.size() < top_k) {
            auto v1 = W.top(); // v1 is the closest candidate to the query in the queue
            auto v1_q_dist = v1._distance;
            auto v1_id = v1._vid;
            std::span<const T, Dim> v1_data = GetDataForInternalID<T, Dim>(v1_id, dim, all_data, graph);
            bool insert_v1{true};
            for (auto& v2: R) {
                auto v2_id = v2._vid; // v2 is the nearest neighbors to return
                std::span<const T, Dim> v2_data = GetDataForInternalID<T, Dim>(v2_id, dim, all_data, graph);
                auto v1_v2_dist = Distance(v1_data, v2_data, df);
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
            } else if (keepPruned){
                Wd.push(v1);
            }
            W.pop();
        }

        if (keepPruned) {
            while(R.size() < top_k && !Wd.empty()) {
                R.push_back(Wd.top());
                Wd.pop();
            }
        }

        return R;
    }
    
    template<class T, std::size_t Dim>
    external_id_t SlideNSWLayer(DistFunc df, size_t dim,
                                internal_id_t entry_id,
                                std::span<const T, Dim> q_data, std::span<const T> all_data,
                                const DynamicNSWGraphPtr &graph) {

        std::span<const T, Dim> entry_data = GetDataForInternalID<T, Dim>(entry_id, dim, all_data, graph);
        distance_t c_dist = Distance(q_data, entry_data, df);
        auto c_id = entry_id;
        bool updated = true;
        while (updated) {
            updated = false;
            auto adjlist = graph->GetNeighborsID(c_id);
            for (internal_id_t vid: adjlist) {
                std::span<const T, Dim> v_data = GetDataForInternalID<T, Dim>(vid, dim, all_data, graph);
                auto v_dist = Distance(q_data, v_data, df);
                if (v_dist < c_dist) {
                    c_id = vid;
                    c_dist = v_dist;
                    updated = true;
                }
            }
        }
        return graph->GetExternalId(c_id);
    }

    template<class T, std::size_t Dim>
    external_id_t SlideNSWLayers(DistFunc df,
                                 size_t dim,
                                 size_t entry_layer,
                                 size_t stop_layer,
                                 std::span<const T, Dim> q_data,
                                 std::span<const T> all_data,
                                 const std::vector<DynamicNSWGraphPtr> &graphs) {

        assert(entry_layer < graphs.size());
        assert(entry_layer >= stop_layer);
        assert(stop_layer >= 1);

        external_id_t external_entry_id{empty_external_id};
        for (int cur_layer = (int) entry_layer; cur_layer >= stop_layer; cur_layer--) {
            const auto &cur_graph = graphs.at(cur_layer);
            internal_id_t entry_id = cur_graph->GetEntryInternalId(external_entry_id);
            external_entry_id = SlideNSWLayer(df, dim, entry_id, q_data, all_data, cur_graph);
        }
        return external_entry_id;
    }

    template<class T, std::size_t Dim>
    std::vector<Entry> SearchNSWLayerHeuristic(DistFunc df, size_t top_k, size_t ef,
                                               size_t dim, internal_id_t entry_id,
                                               std::span<const T, Dim> q_data, std::span<const T> all_data,
                                               const DynamicNSWGraphPtr &graph) {
        MinQueue top_candidates;    // min first data
        MaxQueue nearest_neighbors; // max first data

        std::span<const T, Dim> entry_data = GetDataForInternalID<T, Dim>(entry_id, dim, all_data, graph);
        auto entry_distance = Distance(q_data, entry_data, df);
        top_candidates.emplace(entry_distance, entry_id);
        nearest_neighbors.emplace(entry_distance, entry_id);

        std::vector<bool> visited(graph->GetNodeCapacity());
        visited.at(entry_id) = true;
        while (!top_candidates.empty()) {
            auto [c_dist, c_id] = top_candidates.top();
            auto f_dist = nearest_neighbors.top()._distance;
            top_candidates.pop();
            if (c_dist > f_dist)
                break;

            auto adjlist = graph->GetNeighborsID(c_id);
            for (internal_id_t vid: adjlist) {
                if (visited.at(vid)) continue;
                visited.at(vid) = true;
                std::span<const T, Dim> v_data = GetDataForInternalID<T, Dim>(vid, dim, all_data, graph);
                auto v_dist = Distance(q_data, v_data, df);
                if (nearest_neighbors.size() < ef ||
                    nearest_neighbors.top()._distance > v_dist) {
                    nearest_neighbors.emplace(v_dist, vid);
                    top_candidates.emplace(v_dist, vid);
                    if (nearest_neighbors.size() > ef) {
                        nearest_neighbors.pop();
                    }
                }
            }
        }
        return SelectNeighborsHeuristic<T, Dim>(df, dim, top_k, std::move(nearest_neighbors), all_data, graph);
    }

    template<class T, std::size_t Dim>
    std::vector<Entry> SearchNSWLayerSimple(DistFunc df, size_t top_k, size_t ef,
                                            size_t dim, internal_id_t entry_id,
                                            std::span<const T, Dim> q_data, std::span<const T> all_data,
                                            const DynamicNSWGraphPtr &graph) {
        MinQueue top_candidates;    // min first data
        MaxQueue nearest_neighbors; // max first data

        std::span<const T, Dim> entry_data = GetDataForInternalID<T, Dim>(entry_id, dim, all_data, graph);
        auto entry_distance = Distance(q_data, entry_data, df);
        top_candidates.emplace(entry_distance, entry_id);
        nearest_neighbors.emplace(entry_distance, entry_id);

        std::vector<bool> visited(graph->GetNodeCapacity());
        visited.at(entry_id) = true;
        while (!top_candidates.empty()) {
            auto [c_dist, c_id] = top_candidates.top();
            auto f_dist = nearest_neighbors.top()._distance;
            top_candidates.pop();
            if (c_dist > f_dist)
                break;

            auto adjlist = graph->GetNeighborsID(c_id);
            for (internal_id_t vid: adjlist) {
                if (visited.at(vid)) continue;
                visited.at(vid) = true;
                std::span<const T, Dim> v_data = GetDataForInternalID<T, Dim>(vid, dim, all_data, graph);
                auto v_dist = Distance(q_data, v_data, df);
                if (nearest_neighbors.size() < ef ||
                    nearest_neighbors.top()._distance > v_dist) {
                    nearest_neighbors.emplace(v_dist, vid);
                    top_candidates.emplace(v_dist, vid);
                    if (nearest_neighbors.size() > ef) {
                        nearest_neighbors.pop();
                    }
                }
            }
        }
        return SelectNeighborsSimple(top_k, std::move(nearest_neighbors));
    }

    template<class T, std::size_t Dim>
    std::vector<Entry> SearchBaseSimple(DistFunc df, size_t top_k, size_t ef,
                                        size_t dim, internal_id_t entry_id,
                                        std::span<const T, Dim> q_data, std::span<const T> all_data,
                                        const DynamicNSWGraphPtr &graph) {
        assert(graph->IsBase());
        MinQueue top_candidates;    // min first data
        MaxQueue nearest_neighbors; // max first data

        std::span<const T, Dim> entry_data = GetDataForInternalID<T, Dim>(entry_id, dim, all_data, graph);
        auto entry_distance = Distance(q_data, entry_data, df);
        top_candidates.emplace(entry_distance, entry_id);
        nearest_neighbors.emplace(entry_distance, entry_id);

        std::vector<bool> visited(graph->GetNodeCapacity());
        visited.at(entry_id) = true;
        while (!top_candidates.empty()) {
            auto [c_dist, c_id] = top_candidates.top();
            auto f_dist = nearest_neighbors.top()._distance;
            top_candidates.pop();
            if (c_dist > f_dist)
                break;

            auto adjlist = graph->GetNeighborsID(c_id);
            for (size_t i = 0; i < adjlist.size(); i++) {
                auto vid = adjlist[i];
                if (visited.at(vid)) continue;
                visited.at(vid) = true;

//                auto next_vid = adjlist[i + 1];
//                if (!visited.at(next_vid)) _mm_prefetch(GetDataForInternalID<T, Dim>(next_vid, dim, all_data, graph).data(), _MM_HINT_T2);
                std::span<const T, Dim> v_data = GetDataForInternalID<T, Dim>(vid, dim, all_data, graph);
                auto v_dist = Distance(q_data, v_data, df);
                if (nearest_neighbors.size() < ef ||
                    nearest_neighbors.top()._distance > v_dist) {
                    nearest_neighbors.emplace(v_dist, vid);
                    top_candidates.emplace(v_dist, vid);
                    if (nearest_neighbors.size() > ef) {
                        nearest_neighbors.pop();
                    }
                }
            }
        }
        return SelectNeighborsSimple(top_k, std::move(nearest_neighbors));
    }

    template<class T, std::size_t Dim = std::dynamic_extent>
    std::vector<Entry> SearchNSWLayersSimple(DistFunc df, size_t top_k, size_t ef,
                                             size_t dim, size_t entry_layer,
                                             std::span<const T, Dim> q_data, std::span<const T> all_data,
                                             const std::vector<DynamicNSWGraphPtr> &graphs) {
        assert(!graphs.empty());
        external_id_t external_entry_id{empty_external_id};
        if (entry_layer > 0) {
            size_t stop_layer{1};
            external_entry_id = SlideNSWLayers(df, dim, entry_layer, stop_layer, q_data, all_data, graphs);
        }
        internal_id_t base_entry_id = graphs.at(0)->GetEntryInternalId(external_entry_id);
        return SearchBaseSimple(df, top_k, ef, dim, base_entry_id, q_data, all_data, graphs.at(0));
    }

    template<class T, std::size_t Dim = std::dynamic_extent>
    external_id_t InsertNSWLayer(DistFunc df,
                                 size_t ef, size_t max_degree, size_t dim,
                                 internal_id_t entry_id,
                                 external_id_t q_id,
                                 std::span<const T, Dim> q_data,
                                 std::span<const T> all_data,
                                 DynamicNSWGraphPtr &graph) {
        assert(Dim == std::dynamic_extent || Dim == dim);
        assert(q_data.size() == dim);

        auto top_k = max_degree;
        auto vid = graph->GetNewInternalId(q_id);
        if (vid == 0) return empty_external_id; // handle edge case

        auto neighbors = SearchNSWLayerHeuristic(df, top_k, ef, dim, entry_id, q_data, all_data, graph);
        graph->AddNode(vid, neighbors);

        for (const auto &edge: neighbors) {
            graph->AddReverseEdge(vid, edge);
        }

        if (neighbors.size() >= 1) {
            assert(neighbors[0]._vid != vid);
            return graph->GetExternalId(neighbors[0]._vid);
        } else {
            return empty_external_id;
        }
    }

    template<class T, std::size_t Dim = std::dynamic_extent>
    void InsertNSWLayers(DistFunc df,
                         size_t ef, size_t max_degree, size_t dim,
                         size_t insert_level,
                         external_id_t external_entry_id,
                         external_id_t q_id,
                         std::span<const T, Dim> q_data,
                         std::span<const T> all_data,
                         const std::vector<DynamicNSWGraphPtr> &graphs) {

        assert(Dim == std::dynamic_extent || Dim == dim);
        assert(q_data.size() == dim);
        std::vector<internal_id_t> id_vec(insert_level + 1, empty_external_id);

        // create entries at different layer
        // must do this step first to avoid read before write condition
        // each entry by default has 0 edges so reading them is fine
        for (int i = 0; i <= insert_level; i++) {
            internal_id_t internal_id = graphs.at(i)->GetNewInternalId(q_id);
            id_vec.at(i) = internal_id;
            assert(internal_id >= 0);
        }

        // insert into hnsw graph
        for (int lc = (int) insert_level; lc >= 0; lc--) {
            auto graph = graphs.at(lc);
            auto entry_id = graph->GetEntryInternalId(external_entry_id);
            auto vid = id_vec.at(lc);
            auto top_k = lc == 0 ? max_degree * 2 : max_degree;
            auto neighbors = SearchNSWLayerHeuristic(df, top_k, ef, dim, entry_id, q_data, all_data, graph);
            if (vid == 0) {
                // handle edge case
                // keep vid 0's adj list uninitialized
                external_entry_id = empty_external_id;
            } else {
                assert(!neighbors.empty());
                internal_id_t internal_entry_id = neighbors.at(0)._vid;
                external_entry_id = graph->GetExternalId(internal_entry_id);
                graph->AddNode(vid, neighbors);
                for (const auto &edge: neighbors) {
                    graph->AddReverseEdge(vid, edge);
                }
            }
        }
    }

    template<class T, std::size_t Dim = std::dynamic_extent>
    DynamicNSWGraphPtr BuildNSWLayer(DistFunc df, size_t ef, size_t max_degree, size_t dim,
                                     const std::vector<external_id_t> &external_ids,
                                     std::span<const T> all_data) {
        assert(Dim == std::dynamic_extent || Dim == dim);
        auto graph = std::make_shared<DynamicNSWGraph>();
        graph->Init(max_degree, external_ids.size());

#pragma omp parallel for schedule(static, 128)
        for (external_id_t q_id: external_ids) {
            std::span<const T, Dim> q_data = GetDataForExternalID<T, Dim>(q_id, dim, all_data);
            internal_id_t entry_id = graph->GetEntryInternalId();
            InsertNSWLayer(df, ef, max_degree, dim, entry_id, q_id, q_data, all_data, graph);
        }
        return graph;
    }

    template<class T, std::size_t Dim = std::dynamic_extent>
    std::vector<DynamicNSWGraphPtr> BuildNSWLayers(DistFunc df, size_t ef, size_t max_degree, size_t dim,
                                                   const std::vector<external_id_t> &external_ids,
                                                   std::span<const T> all_data) {

        auto cumulative_probability = get_cumulative_probability(max_degree);
        auto start_layers = get_random_levels(external_ids.size(), cumulative_probability);
        int max_layer = *std::max_element(start_layers.begin(), start_layers.end());
        std::vector<size_t> num_nodes(max_layer + 1, 0);
        for (int l : start_layers) {
            for (size_t j = 0; j <= l; j ++) {
                num_nodes.at(j)++;
            }
        }

        assert(num_nodes[0] == external_ids.size());

        std::vector<DynamicNSWGraphPtr> graphs;
        for (size_t i = 0; i <= max_layer; i++) {
            bool is_base = i == 0;
            size_t M = i == 0 ? max_degree * 2 : max_degree;
            graphs.push_back(std::make_shared<DynamicNSWGraph>());
            graphs.at(i)->Init(M, num_nodes[i], is_base);
        }

        std::atomic<int> max_level{0};
#pragma omp parallel for schedule(static, 128)
        for (size_t i = 0; i < external_ids.size(); i++) {
            external_id_t q_id = external_ids.at(i);
            external_id_t external_entry_id{empty_external_id};

            std::span<const T, Dim> q_data = GetDataForExternalID<T, Dim>(q_id, dim, all_data);
            int l = start_layers.at(i); // layers to insert: 0, 1, ... l
            int L = max_level.load(); // current max layer number in the hnsw index: 0, 1, ... L

            if (L > l) {
                external_entry_id = SlideNSWLayers(df, dim, L, l + 1, q_data, all_data, graphs);
            }

            InsertNSWLayers(df, ef, max_degree, dim, l, external_entry_id, q_id, q_data, all_data, graphs);
            while (l > max_level) {
                if (max_level.compare_exchange_weak(L, l)) {
                    break;
                }
            }
        }
        return graphs;
    }
}
#endif //PICKLE_NSW_HPP
