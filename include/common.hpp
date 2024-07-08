//
// Created by juelin on 6/28/24.
//

#ifndef PICKLE_COMMON_HPP
#define PICKLE_COMMON_HPP

#include <span>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace pickle {
    // TODO: swtich to <stdfloat>
#ifndef float16_t
#ifdef __clang__
#define float16_t __fp16
#elif __GNUC__
#define float16_t _Float16
#endif
#endif

// Always assert macro
#define ALWAYS_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "Assertion failed: " << #expr << " in " << __FILE__ \
                      << " at line " << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (false)

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


    typedef int32_t external_id_t;
    typedef int32_t internal_id_t;
    typedef float distance_t;
    constexpr external_id_t empty_external_id = -1;
    constexpr internal_id_t empty_internal_id = -1;
}
#endif //PICKLE_COMMON_HPP
