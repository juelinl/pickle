#pragma once

#include "distance/generic.hpp"

#ifdef __AVX512F__
#include "distance/avx512.hpp"
#endif

namespace pickle {
    template<class T, std::size_t Extent = std::dynamic_extent>
    float Distance(std::span<const T, Extent> va, std::span<const T, Extent> vb, DistFunc df) {
#ifdef __AVX512F__
        return avx512::Distance<T, Extent>(va, vb, df);
#else
        return generic::Distance<T, Extent>(va, vb, df);
#endif

//        return generic::Distance<T, Extent>(va, vb, df);

    };
}