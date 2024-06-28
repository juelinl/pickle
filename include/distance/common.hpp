//
// Created by juelin on 6/28/24.
//

#ifndef PICKLE_COMMON_HPP
#define PICKLE_COMMON_HPP
#include <span>
#include <cstdint>
#include <numeric>
#include <cmath>
namespace pickle
{
#ifndef float16_t
#ifdef __clang__
#define float16_t __fp16
#elif __GNUC__
#define float16_t _Float16
#endif
#endif

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

    enum class DistanceFunction {
        L1,
        L2,
        IP,
        RUNTIME
    };

}
#endif //PICKLE_COMMON_HPP
