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

    template<class T, std::size_t Extent = std::dynamic_extent>
    float L1(std::span<const T, Extent> va, std::span<const T, Extent> vb) {
        assert(va.size() == vb.size());
        using dist_type = static_switch<IsTFloat<T>(), int, float>::type;
        dist_type res{0};
#pragma unroll
        for (size_t i = 0; i < va.size(); i++) {
            res += abs(va[i] - vb[i]);
        }
        return res;
    };

    template<class T, std::size_t Extent = std::dynamic_extent>
    float L2(std::span<const T, Extent> va, std::span<const T, Extent> vb) {
        assert(va.size() == vb.size());
        using dist_type = static_switch<IsTFloat<T>(), int, float>::type;
        dist_type res{0};

#pragma unroll
        for (size_t i = 0; i < va.size(); i++) {
            res += (va[i] - vb[i]) * (va[i] - vb[i]);
        }
        return res;
    };

    template<class T, std::size_t Extent = std::dynamic_extent>
    float IP(std::span<const T, Extent> va, std::span<const T, Extent> vb) {
        assert(va.size() == vb.size());
        using dist_type = static_switch<IsTFloat<T>(), int, float>::type;
        dist_type res{0};
#pragma unroll
        for (size_t i = 0; i < va.size(); i++) {
            res += va[i] * vb[i];
        }
        return res;
    };

    template<class T, std::size_t Extent = std::dynamic_extent>
    float Distance(std::span<const T, Extent> va, std::span<const T, Extent> vb, DistFunc df) {
        switch (df) {
//            case DistFunc::L1:
//                return generic::L1<T, Extent>(va, vb);
            case DistFunc::L2:
                return generic::L2<T, Extent>(va, vb);
            case DistFunc::IP:
                return -generic::IP<T, Extent>(va, vb);
            default:
                exit(-1);
        };
    }
}
#endif //PICKLE_GENERIC_HPP
