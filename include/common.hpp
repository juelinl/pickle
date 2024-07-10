//
// Created by juelin on 6/28/24.
//

#ifndef PICKLE_COMMON_HPP
#define PICKLE_COMMON_HPP

#include <span>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include "marco.hpp"

namespace pickle {    
    typedef int32_t external_id_t;
    typedef int32_t internal_id_t;
    typedef float distance_t;
    constexpr external_id_t empty_external_id = -1;
    constexpr internal_id_t empty_internal_id = -1;

    enum class DataType {
        Uint8 = 0,
        Int8 = 1,
        Float16 = 3,
        Float32 = 2,
    };

    enum class DistanceFunction {
        L1,
        L2,
        IP,
        RUNTIME
    };

    template<bool flag, typename T, typename U>
    struct static_switch {
    };

    template<typename T, typename U>
    struct static_switch<false, T, U> {
        typedef T type;
    };

    template<typename T, typename U>
    struct static_switch<true, T, U> {
        typedef U type;
    };

    template<typename T>
    consteval bool IsTFloat() {
        return std::is_same_v<T, float> || std::is_same_v<T, float16_t>;
    }

    template <size_t Scale, size_t Extent> constexpr size_t get_main(size_t dim) {
        if constexpr (Extent == std::dynamic_extent) {
            return dim - dim % Scale;
        } else {
            return Extent - Extent % Scale;
        }
    };

    template <size_t Scale, size_t Extent> constexpr size_t get_residual(size_t dim) {
        if constexpr (Extent == std::dynamic_extent) {
            return dim % Scale;
        } else {
            return Extent % Scale;
        }
    };

    template <size_t Scale, size_t Extent> constexpr size_t get_all(size_t dim) {
        if constexpr (Extent == std::dynamic_extent) {
            return dim;
        } else {
            return Extent;
        }
    };

}
#endif //PICKLE_COMMON_HPP
