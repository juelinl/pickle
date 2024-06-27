#include "distance.hpp"
#include "marco.hpp"
#include <cassert>
#include <cstdint>

namespace pickle::distance
{
    template<class T, std::size_t Extend = std::dynamic_extent>
    float Ip(const std::span<T, Extend> va, const std::span<T, Extend> vb){
        assert(va.size() == vb.size());
        using dist_type = static_switch<IsTFloat<T>(), int, float>::type;
        dist_type res{0};
        #pragma unroll
        for(size_t i = 0; i < va.size(); i++) {
            res += va[i] * vb[i];
        }
        return res;
    };

    DISTANCE_TEMPLATE_EXPAND(float, Ip);
    DISTANCE_TEMPLATE_EXPAND(float16_t, Ip);
    DISTANCE_TEMPLATE_EXPAND(uint8_t, Ip);
    DISTANCE_TEMPLATE_EXPAND(int8_t, Ip);
}