//
// Created by juelin on 6/28/24.
//

#include "distance.hpp"

namespace pickle {
#define DISTANCE_TEMPLATE_EXPAND(Fn, T, DF)                                    \
  template float Fn<T, DF>(const std::span<const T>, const std::span<const T>, DistanceFunction);                \
  template float Fn<T, DF>(const std::span<const T, 1>, const std::span<const T, 1>, DistanceFunction);          \
  template float Fn<T, DF>(const std::span<const T, 2>, const std::span<const T, 2>, DistanceFunction);          \
  template float Fn<T, DF>(const std::span<const T, 4>, const std::span<const T, 4>, DistanceFunction);          \
  template float Fn<T, DF>(const std::span<const T, 8>, const std::span<const T, 8>, DistanceFunction);          \
  template float Fn<T, DF>(const std::span<const T, 16>, const std::span<const T, 16>, DistanceFunction);        \
  template float Fn<T, DF>(const std::span<const T, 32>, const std::span<const T, 32>, DistanceFunction);        \
  template float Fn<T, DF>(const std::span<const T, 64>, const std::span<const T, 64>, DistanceFunction);        \
  template float Fn<T, DF>(const std::span<const T, 128>, const std::span<const T, 128>, DistanceFunction);      \
  template float Fn<T, DF>(const std::span<const T, 256>, const std::span<const T, 256>, DistanceFunction);      \
  template float Fn<T, DF>(const std::span<const T, 512>, const std::span<const T, 512>, DistanceFunction);

    DISTANCE_TEMPLATE_EXPAND(Distance, uint8_t, DistanceFunction::L2);
    DISTANCE_TEMPLATE_EXPAND(Distance, int8_t, DistanceFunction::L2);
    DISTANCE_TEMPLATE_EXPAND(Distance, float16_t, DistanceFunction::L2);
    DISTANCE_TEMPLATE_EXPAND(Distance, float, DistanceFunction::L2);

    DISTANCE_TEMPLATE_EXPAND(Distance, uint8_t, DistanceFunction::L1);
    DISTANCE_TEMPLATE_EXPAND(Distance, int8_t, DistanceFunction::L1);
    DISTANCE_TEMPLATE_EXPAND(Distance, float16_t, DistanceFunction::L1);
    DISTANCE_TEMPLATE_EXPAND(Distance, float, DistanceFunction::L1);

    DISTANCE_TEMPLATE_EXPAND(Distance, uint8_t, DistanceFunction::IP);
    DISTANCE_TEMPLATE_EXPAND(Distance, int8_t, DistanceFunction::IP);
    DISTANCE_TEMPLATE_EXPAND(Distance, float16_t, DistanceFunction::IP);
    DISTANCE_TEMPLATE_EXPAND(Distance, float, DistanceFunction::IP);
}