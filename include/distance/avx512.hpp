//
// Created by juelin on 7/9/24.
//
#ifndef PICKLE_AVX512_HPP
#define PICKLE_AVX512_HPP

#include "common.hpp"
#include <cassert>
#include <immintrin.h>

namespace pickle::avx512 {
#define Main get_main<Scale,Extent>(va.size())
#define All get_all<Scale, Extent>(va.size())
    template<size_t Extent = std::dynamic_extent>
    float L2(const std::span<const uint8_t, Extent> va,
             const std::span<const uint8_t, Extent> vb) {
        __m512i temp = _mm512_set1_epi32(0);
        constexpr size_t Scale = sizeof(temp) / sizeof(int16_t);
        int __attribute__((aligned(sizeof(temp)))) TmpRes[Scale];

#pragma unroll
        for (size_t i = 0; i < Main; i += Scale) {
            const auto a_casted =
                    _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i_u *) (va.data() + i)));
            const auto b_casted =
                    _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i_u *) (vb.data() + i)));
            const auto diff = _mm512_sub_epi16(a_casted, b_casted);
            const auto diff_sq = _mm512_mullo_epi16(diff, diff);
            temp = _mm512_add_epi32(
                    temp, _mm512_cvtepu16_epi32(_mm512_castsi512_si256(diff_sq)));
            temp = _mm512_add_epi32(
                    temp, _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(diff_sq, 1)));
        }
        _mm512_store_si512((__m512i *) TmpRes, temp);

        int remain{0};
#pragma unroll
        for (size_t j = Main; j < va.size(); j++) {
            remain += (va[j] - vb[j]) * (va[j] - vb[j]);
        }
        return Sum16(TmpRes) + remain;
    };

    template<size_t Extent = std::dynamic_extent>
    float IP(const std::span<const uint8_t, Extent> va,
             const std::span<const uint8_t, Extent> vb) {
        __m512i temp = _mm512_set1_epi32(0);
        constexpr size_t Scale = sizeof(temp) / sizeof(int16_t);
        int __attribute__((aligned(sizeof(temp)))) TmpRes[Scale];

#pragma unroll
        for (size_t i = 0; i < Main; i += Scale) {
            const auto a_casted =
                    _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i_u *) (va.data() + i)));
            const auto b_casted =
                    _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i_u *) (vb.data() + i)));
            const auto diff_sq = _mm512_mullo_epi16(a_casted, b_casted);
            temp = _mm512_add_epi32(
                    temp, _mm512_cvtepu16_epi32(_mm512_castsi512_si256(diff_sq)));
            temp = _mm512_add_epi32(
                    temp, _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(diff_sq, 1)));
        }
        _mm512_store_si512((__m512i *) TmpRes, temp);

        int remain{0};
#pragma unroll
        for (size_t j = Main; j < va.size(); j++) {
            remain += va[j] * vb[j];
        }
        return Sum16(TmpRes) + remain;
    };

    template<size_t Extent = std::dynamic_extent>
    float L2(std::span<const int8_t, Extent> va,
             std::span<const int8_t, Extent> vb) {
        __m512i temp = _mm512_set1_epi32(0);
        constexpr size_t Scale = sizeof(temp) / sizeof(int16_t);
        int __attribute__((aligned(sizeof(temp)))) TmpRes[Scale];

#pragma unroll
        for (size_t i = 0; i < Main; i += Scale) {
            const auto a_casted =
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i_u *) (va.data() + i)));
            const auto b_casted =
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i_u *) (vb.data() + i)));
            const auto diff = _mm512_sub_epi16(a_casted, b_casted);
            const auto diff_sq = _mm512_mullo_epi16(diff, diff);
            temp = _mm512_add_epi32(
                    temp, _mm512_cvtepu16_epi32(_mm512_castsi512_si256(diff_sq)));
            temp = _mm512_add_epi32(
                    temp, _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(diff_sq, 1)));
        }
        _mm512_store_si512((__m512i *) TmpRes, temp);

        int remain{0};
#pragma unroll
        for (size_t j = Main; j < va.size(); j++) {
            remain += (va[j] - vb[j]) * (va[j] - vb[j]);
        }
        return Sum16(TmpRes) + remain;
    };

    template<size_t Extent = std::dynamic_extent>
    float IP(const std::span<const int8_t, Extent> va,
             const std::span<const int8_t, Extent> vb) {
        __m512i temp = _mm512_set1_epi32(0);
        constexpr size_t Scale = sizeof(temp) / sizeof(int16_t);
        int __attribute__((aligned(sizeof(temp)))) TmpRes[Scale];

#pragma unroll
        for (size_t i = 0; i < Main; i += Scale) {
            const auto a_casted =
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i_u *) (va.data() + i)));
            const auto b_casted =
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i_u *) (vb.data() + i)));
            const auto diff_sq = _mm512_mullo_epi16(a_casted, b_casted);
            temp = _mm512_add_epi32(
                    temp, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(diff_sq)));
            temp = _mm512_add_epi32(
                    temp, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(diff_sq, 1)));
        }
        _mm512_store_si512((__m512i *) TmpRes, temp);
        int remain{0};
#pragma unroll
        for (size_t j = Main; j < va.size(); j++) {
            remain += va[j] * vb[j];
        }
        return Sum16(TmpRes) + remain;
    };

    template<size_t Extent = std::dynamic_extent>
    float L2(std::span<const float16_t, Extent> va,
             std::span<const float16_t, Extent> vb) {
        __m512 temp = _mm512_set1_ps(0);
        constexpr size_t Scale = sizeof(temp) / sizeof(float);
        float __attribute__((aligned(sizeof(temp)))) TmpRes[Scale];

#pragma unroll
        for (int i = 0; i < Main; i += Scale) {
            const __m512 diff = _mm512_sub_ps(
                    _mm512_cvtph_ps(_mm256_loadu_si256(
                            (const __m256i_u *) (reinterpret_cast<const uint16_t *>(va.data()) +
                                                 i))),
                    _mm512_cvtph_ps(_mm256_loadu_si256(
                            (const __m256i_u *) (reinterpret_cast<const uint16_t *>(vb.data()) +
                                                 i))));
            temp = _mm512_add_ps(temp, _mm512_mul_ps(diff, diff));
        }
        _mm512_store_ps(TmpRes, temp);
        
        float remain{0};
#pragma unroll
        for (size_t j = Main; j < va.size(); j++) {
            remain += (va[j] - vb[j]) * (va[j] - vb[j]);
        }
        return Sum16(TmpRes) + remain;
    };

    template<size_t Extent = std::dynamic_extent>
    float IP(std::span<const float16_t, Extent> va,
             std::span<const float16_t, Extent> vb) {
        __m512 temp = _mm512_set1_ps(0);
        constexpr size_t Scale = sizeof(temp) / sizeof(float);
        float __attribute__((aligned(sizeof(temp)))) TmpRes[Scale];

#pragma unroll
        for (int i = 0; i < Main; i += Scale) {
            const __m512 diff = _mm512_mul_ps(
                    _mm512_cvtph_ps(_mm256_loadu_si256(
                            (const __m256i_u *) (reinterpret_cast<const uint16_t *>(va.data()) +
                                                 i))),
                    _mm512_cvtph_ps(_mm256_loadu_si256(
                            (const __m256i_u *) (reinterpret_cast<const uint16_t *>(vb.data()) +
                                                 i))));
            temp = _mm512_add_ps(temp, diff);
        }
        _mm512_store_ps(TmpRes, temp);

        float remain{0};
#pragma unroll
        for (size_t j = Main; j < va.size(); j++) {
            remain += va[j] * vb[j];
        }
        return Sum16(TmpRes) + remain;
    };

    template<size_t Extent = std::dynamic_extent>
    float L2(std::span<const float, Extent> va, std::span<const float, Extent> vb) {
        __m512 temp = _mm512_set1_ps(0);
        constexpr size_t Scale = sizeof(temp) / sizeof(float);
        float __attribute__((aligned(sizeof(temp)))) TmpRes[Scale];

#pragma unroll
        for (int i = 0; i < Main; i += Scale) {
            const __m512 diff =
                    _mm512_sub_ps(_mm512_loadu_ps(va.data() + i), _mm512_loadu_ps(vb.data() + i));
            temp = _mm512_add_ps(temp, _mm512_mul_ps(diff, diff));
        }

        _mm512_store_ps(TmpRes, temp);

        float remain{0};
#pragma unroll
        for (size_t j = Main; j < va.size(); j++) {
            remain += (va[j] - vb[j]) * (va[j] - vb[j]);
        }
        return Sum16(TmpRes) + remain;
    };

    template<size_t Extent = std::dynamic_extent>
    float IP(std::span<const float, Extent> va, std::span<const float, Extent> vb) {
        __m512 temp = _mm512_set1_ps(0);
        constexpr size_t Scale = sizeof(temp) / sizeof(float);
        float __attribute__((aligned(sizeof(temp)))) TmpRes[Scale];

#pragma unroll
        for (int i = 0; i < Main; i += Scale) {
            const __m512 diff =
                    _mm512_mul_ps(_mm512_loadu_ps(va.data() + i), _mm512_loadu_ps(vb.data() + i));
            temp = _mm512_add_ps(temp, diff);
        }

        _mm512_store_ps(TmpRes, temp);

        float remain{0};
#pragma unroll
        for (size_t j = Main; j < va.size(); j++) {
            remain += va[j] * vb[j];
        }
        return Sum16(TmpRes) + remain;
    };

    template<class T, std::size_t Extent = std::dynamic_extent>
    float Distance(std::span<const T, Extent> va, std::span<const T, Extent> vb, DistFunc df) {
        switch (df) {
            case DistFunc::L2:
                return avx512::L2<Extent>(va, vb);
            case DistFunc::IP:
                return -avx512::IP<Extent>(va, vb);
            default:
                exit(-1);
        };
    }
} // namespace pickle::avx512
#endif // PICKLE_AVX512_HPP
