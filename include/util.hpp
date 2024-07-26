//
// Created by juelin on 7/4/24.
//

#ifndef PICKLE_UTIL_HPP
#define PICKLE_UTIL_HPP
#include <vector>
#include <random>
#include <omp.h>

namespace pickle
{
    inline std::vector<double> get_cumulative_probability(size_t max_degree) {
        std::vector<double> ret;
        double r_logm = 1.0 / log(max_degree);
        double c_prob = 0.0; // cumulative probability at c_level
        while (true) {
            int c_level = ret.size();
            double delta = exp(-1.0 * c_level / r_logm) * (1 - exp(-1.0 / r_logm));
            if (delta < 1e-9) break;
            c_prob += delta;
            ret.push_back(c_prob);
        }
        return ret;
    }

    inline int get_random_level(const std::vector<double> &cumulative_probability) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double prob = distribution(rng);
        for (size_t i = 0; i < cumulative_probability.size(); i++) {
            if (prob < cumulative_probability.at(i)) return i;
        }
        return cumulative_probability.size() - 1;
    }

    inline std::vector<uint8_t> get_random_levels(size_t N, const std::vector<double> &cumulative_probability) {

        std::vector<uint8_t> res(N);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        std::mt19937 rng;
        
//#pragma omp parallel private(rng)
        rng.seed(std::random_device{}());

//#pragma parallel for schedule(static, 4096)
        for (size_t i = 0; i < N; i++) {
            bool is_set{false};
            for (int level = 0; level < cumulative_probability.size(); level++) {
                double prob = distribution(rng);
                if (prob < cumulative_probability.at(level)) {
                    res.at(i) = level;
                    is_set = true;
                    break;
                }
            }
            if (!is_set) {
                res.at(i) = int (cumulative_probability.size() - 1);
            }
        }

        return res;
    }

    template <class T> std::vector<T> GetRandIndices(size_t N) {
        std::vector<T> indices(N);
        // Fill the vector with 0, 1, ..., N-1
        for (size_t i = 0; i < N; ++i) {
            indices[i] = i;
        }

        // Obtain a time-based seed
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine rng(seed);

        // Shuffle the vector using the Fisher-Yates algorithm
        std::shuffle(indices.begin(), indices.end(), rng);
        return indices;
    };

    template <class T> std::vector<T> GetIndices(size_t N) {
        std::vector<T> indices(N);
        // Fill the vector with 0, 1, ..., N-1
        for (size_t i = 0; i < N; ++i) {
            indices[i] = i;
        }
        return indices;
    };
}
#endif //PICKLE_UTIL_HPP
