#include "distance.hpp"
#include "marco.hpp"
#include <cassert>
#include <cstdint>
#include <cmath>

namespace pickle::distance
{
    template<class T, std::size_t Extend = std::dynamic_extent>
    float L1(const std::span<T, Extend> va, const std::span<T, Extend> vb){
        assert(va.size() == vb.size());
        using dist_type = static_switch<IsTFloat<T>(), int, float>::type;
        dist_type res{0};
        #pragma unroll
        for(size_t i = 0; i < va.size(); i++) {
            res += std::abs(va[i] - vb[i]);
        }
        return res;
    };

    DISTANCE_TEMPLATE_EXPAND(float, L1);
    DISTANCE_TEMPLATE_EXPAND(float16_t, L1);
    DISTANCE_TEMPLATE_EXPAND(uint8_t, L1);
    DISTANCE_TEMPLATE_EXPAND(int8_t, L1);
}