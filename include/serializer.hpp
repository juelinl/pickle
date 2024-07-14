//
// Created by juelin on 7/8/24.
//

#ifndef PICKLE_SERIALIZER_HPP
#define PICKLE_SERIALIZER_HPP
#include "graph.hpp"
#include "common.hpp"
#include <fstream>

namespace pickle
{
    inline void write_to(std::ofstream &file, auto data, long N) {
        file.write(reinterpret_cast<char*>(data), N);
    };

    inline void read_from(std::ifstream &file, auto data, long N) {
        file.read(reinterpret_cast<char*>(data), N);
    }

    class Serializer {
    public:
        static void to_disk(const std::string& filename, const std::vector<DynamicNSWGraphPtr> & graphs);
        static std::vector<DynamicNSWGraphPtr> from_disk(const std::string& filename);
    };

    void Serializer::to_disk(const std::string& filename, const std::vector<DynamicNSWGraphPtr> & graphs){
        std::ofstream file(filename, std::ios::binary | std::ios::trunc);
        ALWAYS_ASSERT(file.is_open());
        size_t num_graphs = graphs.size();
        write_to(file, &num_graphs, sizeof(num_graphs));
        for (const auto & graph : graphs) {
            size_t num_nodes = graph->m_max_node;
            size_t max_degree = graph->m_max_deg;
            size_t num_edges = num_nodes * max_degree;
            ALWAYS_ASSERT(graph->m_next_id == num_nodes);
            ALWAYS_ASSERT(graph->m_ext_list.size() == num_nodes);
            ALWAYS_ASSERT(graph->m_deg_list.size() == num_nodes);
            ALWAYS_ASSERT(graph->m_dist_list.size() == num_edges);
            ALWAYS_ASSERT(graph->m_adj_list.size() == num_edges);
            size_t graph_size = sizeof(graph_size) + sizeof(num_nodes) + sizeof(max_degree);
            graph_size += sizeof(external_id_t) * num_nodes;
            graph_size += sizeof(internal_id_t) * num_nodes;
            graph_size += sizeof(internal_id_t) * num_edges;
            graph_size += sizeof(internal_id_t) * num_edges;
            write_to(file, &graph_size, sizeof(graph_size));
            write_to(file, &num_nodes, sizeof(num_nodes));
            write_to(file, &max_degree, sizeof(max_degree));
            write_to(file, graph->m_ext_list.data(), sizeof(external_id_t) * num_nodes);
            write_to(file, graph->m_deg_list.data(), sizeof(internal_id_t) * num_nodes);
            write_to(file, graph->m_adj_list.data(), sizeof(internal_id_t) * num_edges);
            write_to(file, graph->m_dist_list.data(), sizeof(internal_id_t) * num_edges);
        }
        file.close();
    };

    std::vector<DynamicNSWGraphPtr> Serializer::from_disk(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        ALWAYS_ASSERT(file.is_open());

        std::vector<DynamicNSWGraphPtr> graphs;
        size_t num_graphs;
        read_from(file, &num_graphs, sizeof(num_graphs));
        for (size_t i = 0; i < num_graphs; i++) {
            DynamicNSWGraphPtr graph = std::make_shared<DynamicNSWGraph>();
            size_t graph_size, num_nodes, max_degree;
            read_from(file, &graph_size, sizeof(graph_size));
            read_from(file, &num_nodes, sizeof(num_nodes));
            read_from(file, &max_degree, sizeof(max_degree));
            size_t num_edges = num_nodes * max_degree;
            graph->Init(max_degree, num_nodes, i == 0);
            graph->m_next_id = num_nodes;
            read_from(file, graph->m_ext_list.data(), sizeof(external_id_t) * num_nodes);
            read_from(file, graph->m_deg_list.data(), sizeof(internal_id_t) * num_nodes);
            read_from(file, graph->m_adj_list.data(), sizeof(internal_id_t) * num_edges);
            read_from(file, graph->m_dist_list.data(), sizeof(internal_id_t) * num_edges);
            graph->CreateMap();
            graphs.push_back(graph);
        }
        return graphs;
    }
}
#endif //PICKLE_SERIALIZER_HPP
