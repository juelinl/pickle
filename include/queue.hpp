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
        distance_t m_dist{std::numeric_limits<distance_t>::max()};
        internal_id_t m_vid{empty_internal_id};

        Entry(distance_t distance, internal_id_t vid) : m_dist{distance}, m_vid{vid} {};

        Entry() = default;

        bool inline operator==(const Entry &other) const {
            return m_dist == other.m_dist && m_vid == other.m_vid;
        }
    };

    struct MaxFirst {
        inline bool operator()(const Entry &lhs, const Entry &rhs) const {
            return lhs.m_dist < rhs.m_dist;
        };
    };

    struct MinFirst {
        inline bool operator()(const Entry &lhs, const Entry &rhs) const {
            return lhs.m_dist > rhs.m_dist;
        };
    };

    // TODO: accelerate pop operation using SIMD instructions
    // TODO: customized container to reduce memory allocation overhead
    typedef std::priority_queue<Entry, std::vector<Entry>, MaxFirst> MaxQueue;
    typedef std::priority_queue<Entry, std::vector<Entry>, MinFirst> MinQueue;

    class EntryVector
    {
    private:
        int m_len{0};
        int m_capacity{0};
        Entry* m_data{nullptr};
    public:
        EntryVector() = default;
        explicit EntryVector(size_t capacity) {
            m_capacity = capacity;
            m_data = WorkMemoryPool::Global().Alloc<Entry>(m_capacity * sizeof(Entry));
        }

        ~EntryVector() {
            if (m_data) {
                WorkMemoryPool::Global().Free(m_data);
                m_capacity = 0;
                m_len = 0;
            }
        }
        void set_size(int size){ m_len = size;};
        Entry* begin() {return m_data;};
        const Entry* begin() const {return m_data;};
        Entry* end() {return m_data + m_len;};
        const Entry* end() const {return m_data + m_len;};
        bool empty() const {return m_len == 0;};

        inline Entry &operator[](size_t i) {
            assert(i < m_len);
            return m_data[i];
        };

        inline Entry operator[](size_t i) const {
            assert(i < m_len);
            return m_data[i];
        };

        void swap(EntryVector &other) noexcept {
            std::swap(m_data, other.m_data);
            std::swap(m_len, other.m_len);
            std::swap(m_capacity, other.m_capacity);
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
        Comp m_comp;
        int m_capacity{0};
        int m_len{0};
        Entry * m_data{nullptr};

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
            int index = m_len - 1;
            while (index > 0 && m_comp(m_data[parentIdx(index)], m_data[index])){
                std::swap(m_data[parentIdx(index)], m_data[index]);
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
                if (left < m_len && m_comp(m_data[largest], m_data[left])) {
                    largest = left;
                }

                if (right < m_len && m_comp(m_data[largest], m_data[right])) {
                    largest = right;
                }

                if (largest != index) {
                    std::swap(m_data[index], m_data[largest]);
                    index = largest;
                } else {
                    return;
                }
            }
        }
    public:
        EntryHeap(): m_comp(), m_capacity{512}, m_len{0} {
            m_data = WorkMemoryPool::Global().Alloc<Entry>(m_capacity * sizeof(Entry));
//            memset(m_data, 0, capacity * sizeof(Entry));
        }

        EntryHeap(const EntryHeap& other) = delete;
        EntryHeap &operator=(const EntryHeap& other) = delete;
        EntryHeap(EntryHeap && other) noexcept {swap(other);};
        EntryHeap &operator=(EntryHeap && other) noexcept {
            swap(other);
            return *this;
        }

        void swap(EntryHeap &other) noexcept {
            std::swap(m_data, other.m_data);
            std::swap(m_len, other.m_len);
            std::swap(m_capacity, other.m_capacity);
        };


        Entry* begin() {return m_data;}
        [[nodiscard]] const Entry* begin() const {return static_cast<const Entry*>(m_data);}
        Entry* end() {return begin() + m_len;}
        [[nodiscard]] const Entry* end() const {return begin() + m_len;}

        ~EntryHeap() {
            if (m_data) {
                WorkMemoryPool::Global().Free(m_data);
                m_capacity = 0;
                m_len = 0;
            }
        }

        [[nodiscard]] bool empty() const {return size() == 0;};

        [[nodiscard]] size_t size() const {
            return m_len;
        }

        void insert(Entry entry) {
            if (m_len >= m_capacity) {
                auto old = begin();
                m_capacity = 8 * m_capacity;
                m_data = WorkMemoryPool::Global().Alloc<Entry>(m_capacity * sizeof(Entry));
                std::memcpy(begin(), old, sizeof(Entry) * m_len);
                WorkMemoryPool::Global().Free(old);
            }
            begin()[m_len++] = entry;
//            std::push_heap(begin(), end(), m_comp);
            heapifyUp();
        }

        void insert(distance_t distance, internal_id_t vid) {
            if (m_len >= m_capacity) {
                auto old = begin();
                m_capacity = 8 * m_capacity;
                m_data = WorkMemoryPool::Global().Alloc<Entry>(m_capacity * sizeof(Entry));
                std::memcpy(begin(), old, sizeof(Entry) * m_len);
                WorkMemoryPool::Global().Free(old);
            }
            begin()[m_len].m_dist = distance;
            begin()[m_len].m_vid = vid;
            m_len++;
            std::push_heap(begin(), end(), m_comp);
        }

        [[nodiscard]] Entry back() const {
            assert(!empty());
            return begin()[m_len - 1];
        }

        [[nodiscard]] Entry top() const {
            assert(!empty());
            Entry ret = begin()[0];
            assert(ret.m_dist >= 0);
            return ret;
        }

        void pop() {
            assert(!empty());
            begin()[0] = back();
            m_len -= 1;
//            if (!empty()) std::pop_heap(begin(), end(), m_comp);
            if (!empty()) heapifyDown();
        }

        [[nodiscard]] Entry min() const {
            assert(!empty());
            size_t min_entry_idx = 0;
            for (size_t i = 1; i < m_len; i++) {
                if (m_data[i].m_dist < m_data[min_entry_idx].m_dist) {
                    min_entry_idx = i;
                }
            }
            return begin()[min_entry_idx];
        }

        Entry extractTop() {
            Entry ret = top();
            pop();
            return ret;
        }

        EntryVector extractTopK(int k) {
            k = std::min(k, m_len);
            EntryVector vec(k);
            vec.set_size(k);
            for (int i = 0; i < k; i++) {
                vec[i] = extractTop();
            }
            return vec;
        }

        EntryVector extractMinTopK(int k) {
            k = std::min(k, m_len);
            EntryVector vec(k);
            vec.set_size(k);

            if constexpr(std::is_same_v<Comp, MaxFirst>){
                while (m_len > k) pop();
                for (int i = 0; i < k; i++) {
                    vec[k - i - 1] = extractTop();
                }
            } else {
                for (int i = 0; i < k; i++) {
                    vec[i] = extractTop();
                }
            }

            std::span<Entry> sp(vec);
            for (int i = 1; i < k; i++) {
                assert(sp[i].m_dist >= sp[i - 1].m_dist);
            }
            return vec;
        }
    };
    typedef EntryHeap<MinFirst> MinHeap;
    typedef EntryHeap<MaxFirst> MaxHeap;

}
#endif //PICKLE_QUEUE_HPP
