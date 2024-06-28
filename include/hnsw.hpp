#pragma once
#include "graph.hpp"
#include "distance.hpp"

namespace pickle {
    template<class T, std::size_t Dim = std::dynamic_extent>
    DynamicNSWGraphPtr BuildNSW(DistanceFunction df,
                                size_t ef,
                                size_t max_degree,
                                size_t dim,
                                const std::vector<external_id_t> &external_ids,
                                std::span<T> all_data);
}