//
// Created by juelin on 7/9/24.
//

#ifndef PICKLE_MARCO_HPP
#define PICKLE_MARCO_HPP

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
#define ALWAYS_ASSERT(expr)                                                    \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << "Assertion failed: " << #expr << " in " << __FILE__         \
                << " at line " << __LINE__ << std::endl;                       \
      std::abort();                                                            \
    }                                                                          \
  } while (false)

#define ATEN_DSTR_SWITCH(val, DType, ...)                                      \
  do {                                                                         \
    if (val == "uint8") {                                                      \
      typedef uint8_t DType;                                                   \
      { __VA_ARGS__ }                                                          \
    } else if (val == "int8") {                                                \
      typedef int8_t DType;                                                    \
      { __VA_ARGS__ }                                                          \
    } else if (val == "float16") {                                             \
      typedef float16_t DType;                                                 \
      { __VA_ARGS__ }                                                          \
    } else if (val == "float32") {                                             \
      typedef float DType;                                                     \
      { __VA_ARGS__ }                                                          \
    } else {                                                                   \
      std::cerr << "DType can only be uint8, int8, float16, or float32\n";     \
      std::cerr << "Unsupported dtype: " << val;                               \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define ATEN_DTYPE_SWITCH(val, DType, ...)                                     \
  do {                                                                         \
    if ((val) == DataType::Uint8) {                                            \
      typedef uint8_t DType;                                                   \
      { __VA_ARGS__ }                                                          \
    } else if ((val) == DataType::Int8) {                                      \
      typedef int8_t DType;                                                    \
      { __VA_ARGS__ }                                                          \
    } else if ((val) == DataType::Float32) {                                   \
      typedef float DType;                                                     \
      { __VA_ARGS__ }                                                          \
    } else if ((val) == DataType::Float16) {                                   \
      typedef float16_t DType;                                                 \
      { __VA_ARGS__ }                                                          \
    } else {                                                                   \
      std::cerr << "DType can only be uint8_t, int8_t, float16_t, or float";   \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define ATEN_DIM_SWITCH(val, DIM, ...)                                         \
  do {                                                                         \
    if (val == 16) {                                                           \
      constexpr size_t DIM = 16;                                               \
      { __VA_ARGS__ }                                                          \
    } else if (val == 32) {                                                    \
      constexpr size_t DIM = 32;                                               \
      { __VA_ARGS__ }                                                          \
    } else if (val == 64) {                                                    \
      constexpr size_t DIM = 64;                                               \
      { __VA_ARGS__ }                                                          \
    } else if (val == 96) {                                                    \
      constexpr size_t DIM = 96;                                               \
      { __VA_ARGS__ }                                                          \
    } else if (val == 100) {                                                   \
      constexpr size_t DIM = 100;                                              \
      { __VA_ARGS__ }                                                          \
    } else if (val == 128) {                                                   \
      constexpr size_t DIM = 128;                                              \
      { __VA_ARGS__ }                                                          \
    } else if (val == 256) {                                                   \
      constexpr size_t DIM = 256;                                              \
      { __VA_ARGS__ }                                                          \
    } else if (val == 384) {                                                   \
      constexpr size_t DIM = 384;                                              \
      { __VA_ARGS__ }                                                          \
    } else if (val == 512) {                                                   \
      constexpr size_t DIM = 512;                                              \
      { __VA_ARGS__ }                                                          \
    } else if (val == 768) {                                                   \
      constexpr size_t DIM = 768;                                              \
      { __VA_ARGS__ }                                                          \
    } else {                                                                   \
      const size_t DIM = std::dynamic_extent;                                  \
      { __VA_ARGS__ }                                                          \
    }                                                                          \
  } while (0)

#define ATEN_DISTANCE_SWITCH(val, DF, ...)                                     \
  do {                                                                         \
    if (val == DistanceFunction::IP) {                                             \
      constexpr DistanceFunction DF = DistanceFunction::IP;                            \
      { __VA_ARGS__ }                                                          \
    } else if (val == DistanceFunction::L2) {                                      \
      constexpr DistanceFunction DF = DistanceFunction::L2;                            \
      { __VA_ARGS__ }                                                          \
    } else if (val == DistanceFunction::L1) {                                      \
      constexpr DistanceFunction DF = DistanceFunction::L1;                            \
      { __VA_ARGS__ }                                                          \
    } else {                                                                   \
      std::cerr << "DF can only be Inner Product (IP), L2, or L1";             \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define Sum8(arr)                                                              \
  arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]

#define Sum16(arr)                                                             \
  arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7] +      \
      arr[8] + arr[9] + arr[10] + arr[11] + arr[12] + arr[13] + arr[14] +      \
      arr[15]

} // namespace pickle
#endif // PICKLE_MARCO_HPP
