//
// Created by juelin on 6/28/24.
//

#ifndef PICKLE_GENERIC_HPP
#define PICKLE_GENERIC_HPP

#include "common.hpp"
#include <cassert>
namespace pickle::generic {
    template<class T>
    inline T abs(T a) {
        return a < 0 ? 0 - a : a;
    };

    template<class T, std::size_t Extend = std::dynamic_extent>
    float L1(std::span<const T, Extend> va, std::span<const T, Extend> vb) {
        assert(va.size() == vb.size());
        using dist_type = static_switch<IsTFloat<T>(), int, float>::type;
        dist_type res{0};
#pragma unroll
        for (size_t i = 0; i < va.size(); i++) {
            res += abs(va[i] - vb[i]);
        }
        return res;
    };

    template<class T, std::size_t Extend = std::dynamic_extent>
    float L2(std::span<const T, Extend> va, std::span<const T, Extend> vb) {
        assert(va.size() == vb.size());
        using dist_type = static_switch<IsTFloat<T>(), int, float>::type;
        dist_type res{0};

#pragma unroll
        for (size_t i = 0; i < va.size(); i++) {
            res += (va[i] - vb[i]) * (va[i] - vb[i]);
        }
        return res;
    };

    template<class T, std::size_t Extend = std::dynamic_extent>
    float IP(std::span<const T, Extend> va, std::span<const T, Extend> vb) {
        assert(va.size() == vb.size());
        using dist_type = static_switch<IsTFloat<T>(), int, float>::type;
        dist_type res{0};
#pragma unroll
        for (size_t i = 0; i < va.size(); i++) {
            res += va[i] * vb[i];
        }
        return res;
    };

    template<class T, DistanceFunction DF = DistanceFunction::RUNTIME, std::size_t Extend = std::dynamic_extent>
    float Distance(std::span<const T, Extend> va, std::span<const T, Extend> vb, DistanceFunction df) {
        if constexpr (DF == DistanceFunction::RUNTIME) {
            switch (df) {
                case DistanceFunction::L1:
                    return generic::L1(va, vb);
                case DistanceFunction::L2:
                    return generic::L2(va, vb);
                case DistanceFunction::IP:
                    return -generic::IP(va, vb);
                default:
                    exit(-1);
            };
        } else {
            switch (DF) {
                case DistanceFunction::L1:
                    return generic::L1(va, vb);
                case DistanceFunction::L2:
                    return generic::L2(va, vb);
                case DistanceFunction::IP:
                    return -generic::IP(va, vb);
                default:
                    exit(-1);
            };
        }

    }
}
#endif //PICKLE_GENERIC_HPP
