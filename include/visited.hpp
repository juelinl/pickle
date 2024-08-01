//
// Created by juelin on 7/10/24.
//

#ifndef PICKLE_VISITED_LIST_HPP
#define PICKLE_VISITED_LIST_HPP

#include "common.hpp"
#include "bloom_filter.hpp"

#include <unordered_set>
#include <vector>
#include <cstring>

namespace pickle {
    // Adopted from NGT: https://github.com/yahoojapan/NGT/blob/main/lib/NGT/HashBasedBooleanSet.h
    class HashBitmap {
    private:
        internal_id_t *m_table{nullptr};
        internal_id_t m_table_size{0};
        internal_id_t m_table_mask{0};
        std::unordered_set<uint32_t> m_set;

        [[nodiscard]] internal_id_t hash(internal_id_t id) const {
            return id & m_table_mask;
        }

    public:
        HashBitmap() = default;

        HashBitmap(const HashBitmap &other) = delete;

        HashBitmap &operator=(const HashBitmap &other) = delete;

        explicit HashBitmap(size_t capacity) {
            size_t bitSize = 0;
            size_t bit = capacity;
            while (bit != 0) {
                bitSize++;
                bit >>= 1;
            }
            m_table_size = 0x1 << ((bitSize + 4) / 2 + 3);
            m_table_mask = m_table_size - 1;
            m_table = WorkMemoryPool::ThreadLocal().Alloc<internal_id_t>(m_table_size * sizeof(internal_id_t));
            memset(m_table, 0, sizeof(uint32_t) * m_table_size); // Initialize all elements to 0
        }

        ~HashBitmap() {
            if (m_table) {
                WorkMemoryPool::ThreadLocal().Free(m_table);
                m_table_size = 0;
                m_table_mask = 0;
                m_set.clear();
            }
        }

        void set(internal_id_t id) {
            internal_id_t pos = hash(id);
            if (m_table[pos] == 0) {
                m_table[pos] = id;
            } else {
                if (m_table[pos] != id) {
                    m_set.insert(id);
                }
            }
        }

        [[nodiscard]] bool test(internal_id_t id) const {
            internal_id_t pos = hash(id);
            auto flag = m_table[pos];
            if (flag == 0) {
                return false;
            } else if (flag == id) {
                return true;
            } else {
                return m_set.contains(id);
            }
        }

        static HashBitmap &ThreadLocal(size_t max_node) {
            static thread_local HashBitmap table{max_node};
            return table;
        }

        void Advance() {
            memset(m_table, 0, sizeof(uint32_t) * m_table_size); // Initialize all elements to 0
            m_set.clear();
        };

        void Mark(internal_id_t id) {
            set(id);
        };

        [[nodiscard]] bool IsVisited(internal_id_t id) const {
            return test(id);
        };

        void Prefetch(internal_id_t id) const {
            _mm_prefetch(m_table + hash(id), _MM_HINT_T0);
        }
    };

    class BloomFilter {
    public:
        BloomFilter() = default;

        explicit BloomFilter(size_t num_insert) {
            bloom_parameters parameters;
            parameters.projected_element_count = num_insert;
            parameters.false_positive_probability = 0.0001;
            parameters.compute_optimal_parameters();
            filter = bloom_filter(parameters);
        }

        static BloomFilter &ThreadLocal(size_t num_insert) {
            static thread_local size_t global_capacity{0};
            static thread_local BloomFilter table;
            if (global_capacity < num_insert) {
                table = BloomFilter(num_insert);
                global_capacity = num_insert;
            }
            return table;
        }

        void Advance() {
            filter.clear();
        }

        void Mark(internal_id_t id) {
            filter.insert(id);
        };

        [[nodiscard]] bool IsVisited(internal_id_t id) const {
            return filter.contains(id);
        };
    private:
        bloom_filter filter;
    };

    namespace impl {
        class Bitmap {
        private:
            uint32_t *m_data{nullptr};  // Using vector of uint32_t for storage
            uint32_t m_capacity{0};
            uint32_t m_size{0};
            int m_max_index{0};

        public:
            Bitmap() = default;

            Bitmap(const Bitmap &other) = delete;

            Bitmap &operator=(const Bitmap &other) = delete;

            Bitmap(size_t capacity, size_t size) {
                // Calculate number of uint32_t elements needed to store 'size' bits
                m_size = (size + 31) / 32;
                m_capacity = (capacity + 31) / 32;
                m_capacity = std::max(m_size, m_capacity);
                m_data = WorkMemoryPool::ThreadLocal().Alloc<uint32_t>(m_capacity * sizeof(uint32_t));
                memset(m_data, 0, sizeof(uint32_t) * m_capacity); // Initialize all elements to 0
            }

            ~Bitmap() {
                if (m_data) {
                    m_size = 0;
                    m_capacity = 0;
                    WorkMemoryPool::ThreadLocal().Free(m_data);
                }
            }

            void set(int pos) {
                int index = pos / 32;
                int bit = pos % 32;
                m_data[index] |= (1 << bit);
                m_max_index = std::max(index, m_max_index);
            }

            void clear(int pos) {
                int index = pos / 32;
                int bit = pos % 32;
                m_data[index] &= ~(1 << bit);
            }

            [[nodiscard]] bool test(int pos) const {
                int index = pos / 32;
                int bit = pos % 32;
                return (m_data[index] & (1 << bit)) != 0;
            }

            void reset() {
//            memset(m_data, 0, m_size * sizeof(uint32_t ));
                memset(m_data, 0, (m_max_index + 1) * sizeof(uint32_t));
                m_max_index = 0;
            }

            void prefetch(internal_id_t id) {
                int idx = (id / 512) * (512 / 32);
                _mm_prefetch(m_data + idx, _MM_HINT_T0);
            }

            void set_size(size_t new_size) {
                m_size = (new_size + 31) / 32;
                assert(m_capacity >= m_size);
                // reset();
            }
        };
    }

    class BitMap {
    public:
        BitMap() = default;

        BitMap(size_t max_node, size_t num_nodes) : m_bitmap(max_node, num_nodes) {}

        BitMap(const BitMap &other) = delete;

        BitMap &operator=(const BitMap &other) = delete;

        static BitMap &ThreadLocal(size_t max_node, size_t num_nodes) {
            static thread_local size_t global_capacity{max_node};
            static thread_local BitMap table{max_node, num_nodes};
            assert(max_node >= num_nodes);
            table.m_bitmap.set_size(num_nodes);
            return table;
        }

        void Advance() {
            m_bitmap.reset();
        }

        void Prefetch(internal_id_t id) {
            m_bitmap.prefetch(id);
        }

        void Mark(internal_id_t id) {
            m_bitmap.set(id);
        };

        [[nodiscard]] bool IsVisited(internal_id_t id) const {
            return m_bitmap.test(id);
        };

    private:
        impl::Bitmap m_bitmap{};
    };

    class ByteMap {
    public:
        using MapType = uint8_t;

        ByteMap() = default;

        ByteMap(const ByteMap &other) = delete;

        ByteMap &operator=(const ByteMap &other) = delete;

        void Init(size_t num_nodes) {
             if (masks.size() != num_nodes) {
                 visited_mark = 1;
                 masks.clear();
                 masks.resize(num_nodes, unvisited_mark);
             }
        }

        static ByteMap &ThreadLocal(int level, int max_level, size_t num_nodes) {
            static thread_local std::vector<ByteMap> tables(max_level);
            tables.at(level).Init(num_nodes);
            return tables.at(level);
        }

        void Advance() {
            visited_mark++;
            if (visited_mark == unvisited_mark) {
                // start over
                memset(masks.data(), unvisited_mark, masks.size() * sizeof(MapType));
                visited_mark = 1;
            }
        }

        void Prefetch(internal_id_t id) {
            _mm_prefetch(masks.data() + id, _MM_HINT_T0);
        }

        void Mark(internal_id_t id) {
            masks.at(id) = visited_mark;
        };

        bool IsVisited(internal_id_t id) {
            return masks.at(id) == visited_mark;
        }

    private:
        MapType visited_mark{1};
        const MapType unvisited_mark{0};
        std::vector<MapType> masks;
    };
}
#endif //PICKLE_VISITED_LIST_HPP
