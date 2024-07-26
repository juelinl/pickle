//
// Created by juelin on 7/25/24.
//

#ifndef PICKLE_PROFILER_HPP
#define PICKLE_PROFILER_HPP

#include <atomic>
#include <memory>

namespace pickle
{
    class Profiler
    {
    private:
        std::atomic<size_t> num_hop;
        std::atomic<size_t> num_neighbor;
        std::atomic<size_t> num_dist;

    public:
        Profiler() = default;
        void Reset() {
            num_hop = 0;
            num_neighbor = 0;
            num_dist = 0;
        }

        static std::shared_ptr<Profiler> Global() {
            static auto profiler = std::make_shared<Profiler>();
            return profiler;
        };

        size_t GetHop() {
            return num_hop;
        }

        size_t GetDist() {
            return num_dist;
        }

        size_t GetNeighbor() {
            return num_neighbor;
        }

        void AddHop(size_t n) {
            num_hop += n;
        }

        void AddDist(size_t n) {
            num_dist += n;
        }

        void AddNeighbor(size_t n) {
            num_neighbor += n;
        }
    };
}
#endif //PICKLE_PROFILER_HPP
