#pragma once

#include "distance/generic.hpp"

namespace pickle {
    template<class T, DistanceFunction DF = DistanceFunction::RUNTIME, std::size_t Extend = std::dynamic_extent>
    float Distance(std::span<const T, Extend> va, std::span<const T, Extend> vb, DistanceFunction df) {
        return generic::Distance<T, DF, Extend>(va, vb, df);
    };
}