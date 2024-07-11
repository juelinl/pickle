//
// Created by juelin on 7/10/24.
//

#ifndef PICKLE_VISITED_LIST_HPP
#define PICKLE_VISITED_LIST_HPP

#include "common.hpp"
#include <vector>
#include <cstring>

namespace pickle
{
    class VisitedTable
    {
    public:
        VisitedTable() = default;

        explicit VisitedTable(size_t num_nodes) {
            masks.resize(num_nodes, unvisited_mark);
        }

        static VisitedTable& Global(size_t num_nodes) {
            static thread_local size_t global_capacity{0};
            static thread_local VisitedTable table;
            if (global_capacity < num_nodes) {
                table = VisitedTable(num_nodes);
                global_capacity = num_nodes;
            } else {
                return table;
            }
        }

        void Advance() {
            visited_mark++;
            if (visited_mark == unvisited_mark) {
                // start over
                memset(masks.data(), masks.size(), unvisited_mark);
                visited_mark = 1;
            }
        }

        void Mark(internal_id_t id) {
            masks.at(id) = visited_mark;
        };

        bool IsVisited(internal_id_t id) {
            return masks.at(id) == visited_mark;
        }

    private:
        uint8_t visited_mark{1};
        uint8_t unvisited_mark{0};
        std::vector<uint8_t> masks;
    };
}
#endif //PICKLE_VISITED_LIST_HPP
