//
// Created by juelin on 7/1/24.
//

#ifndef PICKLE_QUEUE_HPP
#define PICKLE_QUEUE_HPP

#include "common.hpp"
#include "mempool.hpp"

#include <queue>
#include <numeric>
#include <functional>
#include <cstring>

namespace pickle {
    struct Entry {
        distance_t _distance{std::numeric_limits<distance_t>::max()};
        internal_id_t _vid{empty_internal_id};

        Entry(distance_t distance, internal_id_t vid) : _distance{distance}, _vid{vid} {};

        Entry() = default;

        bool inline operator==(const Entry &other) const {
            return _distance == other._distance && _vid == other._vid;
        }
    };

    struct MaxFirst {
        inline bool operator()(const Entry &lhs, const Entry &rhs) const {
            return lhs._distance < rhs._distance;
        };
    };

    struct MinFirst {
        inline bool operator()(const Entry &lhs, const Entry &rhs) const {
            return lhs._distance > rhs._distance;
        };
    };

    // TODO: accelerate pop operation using SIMD instructions
    // TODO: customized container to reduce memory allocation overhead
    typedef std::priority_queue<Entry, std::vector<Entry>, MaxFirst> MaxQueue;
    typedef std::priority_queue<Entry, std::vector<Entry>, MinFirst> MinQueue;

    class EntryVector
    {
    private:
        Entry* _data{nullptr};
        int _len{0};
        int _capacity{0};
    public:
        EntryVector() = default;
        explicit EntryVector(size_t capacity = 512) {
            _capacity = capacity;
            _data = WorkMemoryPool::Global().Alloc<Entry>(_capacity * sizeof(Entry));
        }

        ~EntryVector() {
            if (_data) {
                WorkMemoryPool::Global().Free(_data);
                _capacity = 0;
                _len = 0;
            }
        }
        void set_size(int size){_len = size;};
        Entry* begin() {return _data;};
        const Entry* begin() const {return _data;};
        Entry* end() {return _data + _len;};
        const Entry* end() const {return _data + _len;};
        bool empty() const {return _len == 0;};

        inline Entry &operator[](size_t i) {
            assert(i < _len);
            return _data[i];
        };

        inline Entry operator[](size_t i) const {
            assert(i < _len);
            return _data[i];
        };

        void swap(EntryVector &other) noexcept {
            std::swap(_data, other._data);
            std::swap(_len, other._len);
            std::swap(_capacity, other._capacity);
        };

        EntryVector(const EntryVector& other) = delete;
        EntryVector &operator=(const EntryVector& other) = delete;
        EntryVector(EntryVector && other) noexcept {swap(other);};
        EntryVector &operator=(EntryVector && other) noexcept {
            swap(other);
            return *this;
        }
    };

    template<class Comp = MinFirst>
    class EntryHeap {
    private:
        Entry * _data{nullptr};
        int _capacity{0};
        int _len{0};
        Comp _comp;

        int parentIdx(int index){
            return (index - 1) / 2;
        }

        int leftIdx (int index) {
            return 2 * index + 1;
        }

        int rightIdx (int index) {
            return 2 * index + 2;
        }

        void heapifyUp() {
            int index = _len - 1;
            while (index > 0 && _comp(_data[parentIdx(index)], _data[index])){
                std::swap(_data[parentIdx(index)], _data[index]);
                index = parentIdx(index);
            }
        }

        void heapifyDown() {
            assert(!empty());
            int largest{0};
            int index{0};

            while (true) {
                int left = leftIdx(index);
                int right = rightIdx(index);
                if (left < _len && _comp(_data[largest], _data[left])) {
                    largest = left;
                }

                if (right < _len && _comp(_data[largest], _data[right])) {
                    largest = right;
                }

                if (largest != index) {
                    std::swap(_data[index], _data[largest]);
                    index = largest;
                } else {
                    return;
                }
            }
        }
    public:
        explicit EntryHeap(size_t capacity = 512): _comp() {
            _capacity = capacity;
            _data = WorkMemoryPool::Global().Alloc<Entry>(_capacity * sizeof(Entry));
        }

        ~EntryHeap() {
            if (_data) {
                WorkMemoryPool::Global().Free(_data);
                _capacity = 0;
                _len = 0;
            }
        }

        bool empty() const {return _len == 0;};

        void insert(Entry entry) {
            if (_len >= _capacity) {
                auto old = _data;
                _capacity = 4 * _capacity;
                _data = WorkMemoryPool::Global().Alloc<Entry>(_capacity * sizeof(Entry));
                std::memcpy(_data, old, sizeof(Entry) * _len);
                WorkMemoryPool::Global().Free(old);
            }
            _data[_len++] = entry;
            heapifyUp();
        }

        void insert(distance_t distance, internal_id_t vid) {
            if (_len >= _capacity) {
                auto old = _data;
                _capacity = 4 * _capacity;
                _data = WorkMemoryPool::Global().Alloc<Entry>(_capacity * sizeof(Entry));
                std::memcpy(_data, old, sizeof(Entry) * _len);
                WorkMemoryPool::Global().Free(old);
            }
            _data[_len++] = {distance, vid};
            heapifyUp();
        }
        Entry back() {
            assert(!empty());
            return _data[_len - 1];
        }

        Entry top() {
            assert(!empty());
            return _data[0];
        }

        void pop() {
            assert(!empty());
            _data[0] = back();
            _len -= 1;
            if (_len > 0) heapifyDown();
        }

        Entry min() {
            assert(!empty());
            size_t min_entry_idx = 0;
            for (size_t i = 1; i < _len; i++) {
                if (_data[i]._distance < _data[min_entry_idx]._distance) {
                    min_entry_idx = i;
                }
            }
            return _data[min_entry_idx];
        }
        Entry extractTop() {
            Entry ret = top();
            pop();
            return ret;
        }

        EntryVector extractTopK(int k) {
            assert(k <= _len);
            EntryVector vec(k);
            vec.set_size(k);
            for (int i = 0; i < k; i++) {
                vec[i] = extractTop();
            }
            return vec;
        }

        EntryVector extractLastK(int k) {
            assert(k <= _len);
            EntryVector vec(k);
            vec.set_size(k);
            while(_len > k) pop();
            for (int i = 0; i < k; i++) {
                vec[i] = extractTop();
            }
            return vec;
        }

        size_t size() const {
            return _len;
        }
    };
    typedef EntryHeap<MinFirst> MinHeap;
    typedef EntryHeap<MaxFirst> MaxHeap;

}
#endif //PICKLE_QUEUE_HPP
