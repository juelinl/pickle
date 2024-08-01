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
        static constexpr size_t max_level{16};
        std::array<std::atomic<int>, max_level> m_hop;
        std::array<std::atomic<int>, max_level> m_neighbor;
        std::array<std::atomic<int>, max_level> m_dist;

    public:
        Profiler() = default;
        void Reset() {
            for (int i = 0; i < max_level; i++){
                m_hop[i] = 0;
                m_neighbor[i] = 0;
                m_dist[i] = 0;
            }
        }

        static std::shared_ptr<Profiler> Global() {
            static auto profiler = std::make_shared<Profiler>();
            return profiler;
        };

        int GetHop() {
            return std::accumulate(m_hop.begin(), m_hop.end(), 0);
        }

        int GetHop(int level) {
            return m_hop[level];
        }

        int GetDist() {
            return std::accumulate(m_dist.begin(), m_dist.end(), 0);
        }

        int GetDist(int level) {
            return m_dist[level];
        }

        int GetNeighbor() {
            return std::accumulate(m_neighbor.begin(), m_neighbor.end(), 0);
        }

        int GetNeighbor(int level) {
            return m_neighbor[level];
        }

        void AddHop(int level, int num) {
            m_hop[level] += num;
        }
        void AddNeighbor(int level, int num) {
            m_neighbor[level] += num;
        }
        void AddDist(int level, int num) {
            m_dist[level] += num;
        }
    };
}
#endif //PICKLE_PROFILER_HPP
