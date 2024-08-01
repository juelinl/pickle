//
// Created by juelin on 7/10/24.
//

#ifndef PICKLE_NDARRAY_HPP
#define PICKLE_NDARRAY_HPP

#include "common.hpp"
#include <cassert>
#include <memory>
#include <utility>
#include <vector>
#include <xmmintrin.h>

namespace pickle
{
    struct NDArray {
        std::shared_ptr<std::vector<char>> m_data{nullptr};
        std::vector<size_t> m_shape;
        size_t m_byte{0};
        size_t m_stride{0};
        DataType m_dtype{DataType::Float32};

        NDArray() = default;
        NDArray(DataType dtype, const std::vector<size_t >& shape) {
            assert(shape.size() == 2);
            m_dtype = dtype;
            m_shape = shape;
            if (m_dtype == DataType::Float32) {
                m_byte = 4;
            } else if (m_dtype == DataType::Float16) {
                m_byte = 2;
            } else if (m_dtype == DataType::Int8 || m_dtype == DataType::Uint8){
                m_byte = 1;
            }
            m_stride = m_byte * m_shape[1];
            m_data = std::make_shared<std::vector<char>>(m_stride * m_shape[0]);
        }

        NDArray(const std::shared_ptr<std::vector<char>>& data, DataType dtype, const std::vector<size_t >& shape) {
            assert(shape.size() == 2);
            m_dtype = dtype;
            m_shape = shape;
            if (m_dtype == DataType::Float32) {
                m_byte = 4;
            } else if (m_dtype == DataType::Float16) {
                m_byte = 2;
            } else if (m_dtype == DataType::Int8 || m_dtype == DataType::Uint8){
                m_byte = 1;
            }
            m_stride = m_byte * m_shape[1];
            m_data = data;
        }

        void *get_raw(size_t row_id) {
            return static_cast<char *>(m_data->data()) + row_id * m_stride;
        };

        [[nodiscard]] const void *get_raw(size_t row_id) const {
            return static_cast<const char *>(m_data->data()) + row_id * m_stride;
        };

        template <typename T> T *get_ptr(size_t row_id) {
            return static_cast<T *>(get_raw(row_id));
        }

        template <typename T> const T *get_ptr(size_t row_id) const {
            return static_cast<const T *>(get_raw(row_id));
        }

        template <typename T> std::span<T> get_span(size_t row_id) {
            return std::span{get_ptr<T>(row_id), m_shape[1]};
        }

        template <typename T> std::span<const T> get_span(size_t row_id) const {
            return std::span{get_ptr<T>(row_id), m_shape[1]};
        }

        template <typename T, size_t Dim> std::span<T, Dim> get_span(size_t row_id) {
            assert(Dim == m_shape[1] || Dim == std::dynamic_extent);
            return std::span<T, Dim>{get_ptr<T>(row_id), m_shape[1]};
        }

        template <typename T, size_t Dim> std::span<const T, Dim> get_span(size_t row_id) const {
            assert(Dim == m_shape[1] || Dim == std::dynamic_extent);
            return std::span<const T, Dim>{get_ptr<T>(row_id), m_shape[1]};
        }

        template <typename T> T *data() { return static_cast<T *>(m_data); }

        template <typename T> const T *data() const { return static_cast<T *>(m_data); }

        template <typename T> std::span<T> span() {
            return {data<T>(), m_shape[0] * m_shape[1]};
        };


        void Prefetch(size_t row_id) const {
            _mm_prefetch(get_raw(row_id), _MM_HINT_T0);
        }
    };
}
#endif //PICKLE_NDARRAY_HPP
