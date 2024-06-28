#include "hnsw.hpp"
#include "distance.hpp"

#include <queue>
#include <cassert>

namespace pickle {
    typedef std::pair<distance_t, internal_id_t> Entry;

    struct MaxFirst {
        inline bool operator() (const Entry& lhs, const Entry& rhs) {
            return lhs.first < rhs.first;
        };
    };

    struct MinFirst {
        inline bool operator() (const Entry& lhs, const Entry& rhs) {
            return lhs.first > rhs.first;
        };
    };

    typedef std::priority_queue<Entry, std::vector<Entry>, MaxFirst> MaxQueue;
    typedef std::priority_queue<Entry, std::vector<Entry>, MinFirst> MinQueue;

    template<class T, std::size_t Dim>
    inline std::span<T, Dim> GetData(internal_id_t vid, size_t dim, std::span<T> all_data, const DynamicNSWGraphPtr& graph) {
        auto start = all_data.data() + graph->GetExternalId(vid) * dim;
        std::span<T, Dim> data{start, dim};
        return data;
    };

    template<class T, std::size_t Dim>
    MaxQueue search(DistanceFunction df,
                      size_t top_k,
                      size_t ef,
                      internal_id_t entry_id,
                      size_t dim,
                      std::span<T, Dim> q_data,
                      std::span<T> all_data,
                      const DynamicNSWGraphPtr &graph) {

        MinQueue top_candidates; // min first heap
        MaxQueue nearest_neighbors; // max first heap
        std::span<T, Dim> entry_data = GetData(entry_id, dim, all_data, graph);

        auto entry_distance = Distance(q_data, entry_data, df);
        top_candidates.emplace(entry_id, entry_distance);
        nearest_neighbors.emplace(entry_id, entry_distance);

        while (!top_candidates.empty()) {
            auto [c_dist, c_id] = top_candidates.top();
            auto f_dist = nearest_neighbors.top().first;
            top_candidates.pop();
            if (c_dist > f_dist) break;

            auto adjlist = graph->GetAdjlist(c_id);
            for (internal_id_t vid: adjlist) {
                std::span<T, Dim> v_data = GetData(vid, dim, all_data, graph);
                auto v_dist = Distance(q_data, v_data, df);
                if (nearest_neighbors.size() < ef || nearest_neighbors.top().first > v_dist) {
                    nearest_neighbors.emplace(v_dist, vid);
                    top_candidates.emplace(vid, v_dist);
                    if (nearest_neighbors.size() > ef) {
                        nearest_neighbors.pop();
                    }
                }
            }
        }
        while (nearest_neighbors.size() > top_k) nearest_neighbors.pop();
        return std::move(nearest_neighbors);
    };


    template<class T, std::size_t Dim>
    DynamicNSWGraphPtr BuildNSW(DistanceFunction df,
                                size_t ef,
                                size_t max_degree,
                                size_t dim,
                                const std::vector<external_id_t> &external_ids,
                                std::span<T> all_data) {
        assert(Dim == std::dynamic_extent || Dim == dim);
        auto top_k = max_degree;
        auto graph = std::make_shared<DynamicNSWGraph>();
        graph->Init(max_degree, external_ids.size());

        for (external_id_t q_id: external_ids) {
            auto vid = graph->GetInternalId(q_id);
            std::span<T, Dim> q_data = GetData(vid, dim, all_data, graph);
            auto entry_id = graph->GetEntryInternalId();
            auto search_result = search(df, top_k, ef, entry_id, dim, q_data, all_data, graph);
            size_t num_results = search_result.size();
            internal_id_t neighbors[num_results];
            distance_t distances[num_results];

            while (!search_result.empty()) {
                auto [dist, id] = search_result.top();
                search_result.pop();
                size_t remain_size = search_result.size();
                neighbors[remain_size] = id;
                distances[remain_size] = dist;
            };

            graph->AddNode(vid, {neighbors, num_results}, {distances, num_results});

            for (size_t j = 0; j < num_results; j++) {
                auto nid = neighbors[j];
                auto distance = distances[j];
                graph->UpdateNode(nid, vid, distance); // add reverse edge
            }
        }

        return graph;
    };

    template<>
    DynamicNSWGraphPtr BuildNSW<float, 128>(DistanceFunction df,
                                size_t ef,
                                size_t max_degree,
                                size_t dim,
                                const std::vector<external_id_t> &external_ids,
                                std::span<float> all_data);
};