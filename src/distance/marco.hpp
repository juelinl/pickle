#pragma once

#define DISTANCE_TEMPLATE_EXPAND(T, Fn)                                        \
  template <> float Fn(const std::span<T>, const std::span<T>);                \
  template <> float Fn(const std::span<T, 1>, const std::span<T, 1>);          \
  template <> float Fn(const std::span<T, 2>, const std::span<T, 2>);          \
  template <> float Fn(const std::span<T, 4>, const std::span<T, 4>);          \
  template <> float Fn(const std::span<T, 8>, const std::span<T, 8>);          \
  template <> float Fn(const std::span<T, 16>, const std::span<T, 16>);        \
  template <> float Fn(const std::span<T, 32>, const std::span<T, 32>);        \
  template <> float Fn(const std::span<T, 64>, const std::span<T, 64>);        \
  template <> float Fn(const std::span<T, 128>, const std::span<T, 128>);      \
  template <> float Fn(const std::span<T, 256>, const std::span<T, 256>);      \
  template <> float Fn(const std::span<T, 512>, const std::span<T, 512>);

#ifndef float16_t
#ifdef __clang__
#define float16_t __fp16
#elif __GNUC__
#define float16_t _Float16
#endif
#endif

 template<bool flag, typename T, typename U>
 struct static_switch {};

 template<typename T, typename U>
 struct static_switch<false, T, U>{typedef T type;};

 template<typename T, typename U>
 struct static_switch<true, T, U>{typedef U type;};

 template<typename T>
 consteval bool IsTFloat() {
    return std::is_same_v<T, float> || std::is_same_v<T, float16_t>;
 } 