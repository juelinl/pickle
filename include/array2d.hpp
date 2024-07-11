//
// Created by juelin on 7/10/24.
//

#ifndef PICKLE_ARRAY2D_HPP
#define PICKLE_ARRAY2D_HPP

#include "common.hpp"
#include <cassert>
#include <memory>

namespace pickle
{
    struct Array2D {
        void *_data{nullptr};
        bool _managed{false};
        size_t _shape[2]{0, 0};
        size_t _element_size{0};
        DataType _data_type{DataType::Float32};

        Array2D() = default;
        explicit Array2D(bool managed): _managed{managed} {};

        ~Array2D() {
            _shape[0] = 0;
            _shape[1] = 0;
            _element_size = 0;
            if (_managed && _data)
                free(_data);
        }

        void *get_raw(size_t row_id) {
            return static_cast<char *>(_data) + 1ull * row_id * _shape[1] * _element_size;
        };

        const void *get_raw(size_t row_id) const {
            return static_cast<char *>(_data) + 1ull * row_id * _shape[1] * _element_size;
        };

        template <typename T> T *get_ptr(size_t row_id) {
            return static_cast<T *>(get_raw(row_id));
        }

        template <typename T> const T *get_ptr(size_t row_id) const {
            return static_cast<T *>(get_raw(row_id));
        }

        template <typename T> std::span<T> get_span(size_t row_id) {
            return std::span{get_ptr<T>(row_id), _shape[1]};
        }

        template <typename T> std::span<const T> get_span(size_t row_id) const {
            return std::span{get_ptr<T>(row_id), _shape[1]};
        }

        template <typename T, size_t Dim> std::span<T, Dim> get_span(size_t row_id) {
            assert(Dim == _shape[1]);
            return std::span<T, Dim>{get_ptr<T>(row_id), _shape[1]};
        }

        template <typename T, size_t Dim> std::span<const T, Dim> get_span(size_t row_id) const {
            assert(Dim == _shape[1]);
            return std::span<const T, Dim>{get_ptr<T>(row_id), _shape[1]};
        }

        template <typename T> T *data() { return static_cast<T *>(_data); }

        template <typename T> const T *data() const { return static_cast<T *>(_data); }

        template <typename T> std::span<T> span() {
            return {data<T>(), _shape[0] * _shape[1]};
        };
    };
    using Array2DPtr = std::shared_ptr<Array2D>;
}
#endif //PICKLE_ARRAY2D_HPP
