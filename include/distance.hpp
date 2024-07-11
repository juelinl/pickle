#pragma once

#include "distance/generic.hpp"
// #include "distance/avx512.hpp"
namespace pickle {
    template<class T, DistFunc DF = DistFunc::RUNTIME, std::size_t Extent = std::dynamic_extent>
    float Distance(std::span<const T, Extent> va, std::span<const T, Extent> vb, DistFunc df) {
        return generic::Distance<T, DF, Extent>(va, vb, df);
    };
}