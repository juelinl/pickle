//
// Created by juelin on 7/9/24.
//

#ifndef PICKLE_MEMPOOL_HPP
#define PICKLE_MEMPOOL_HPP
#include <mutex>
#include <vector>
#include <set>
#include <stack>
#include <cassert>

namespace pickle {
    // Need to manually free it
    struct DataPage
    {
        char * _ptr{nullptr};
        size_t _size{0};
        DataPage() = default;
        explicit DataPage(size_t size, size_t alignment) {
            _ptr = static_cast<char*>(std::aligned_alloc(alignment, size));
            _size = size;
        }

        void * Alloc(size_t N) {
            assert(_size >= N);
            char* old = _ptr;
            _ptr += N;
            _size -= N;
            return old;
        }

        bool operator<(const DataPage& other) const {
            return _size > other._size;
        }
    };

    class WorkMemoryPool {
    private:
        std::vector<void *> alloc_small_page; // 4KiB per page
        std::stack<void *> free_small_page; // 4KiB per page

        std::vector<void *> alloc_mid_page; // 32KiB per page
        std::stack<void *> free_mid_page; // 32KiB per page

        std::vector<void*> alloc_large; // allocated large pages

        template<class T>
        T* AllocSmall(size_t N) {
            void * ptr{nullptr};
            if (free_small_page.empty()) {
                ptr = std::aligned_alloc(16, 4096);
                alloc_small_page.push_back(ptr);
            } else {
                ptr = free_small_page.top();
                free_small_page.pop();
            }
            return static_cast<T*>(ptr);
        };

        template<class T>
        T* AllocMid(size_t N) {
            void * ptr{nullptr};
            if (free_mid_page.empty()) {
                ptr = std::aligned_alloc(16, 32768);
                alloc_mid_page.push_back(ptr);
            } else {
                ptr = free_mid_page.top();
                free_mid_page.pop();
            }
            return static_cast<T*>(ptr);
        };

        template<class T>
        T* AllocLarge(size_t N) {
            auto ptr = std::aligned_alloc(16, N);
            alloc_large.push_back(ptr);
            return static_cast<T*>(ptr);
        };

        template<class T>
        inline bool contains(const std::vector<T>& input, T item) {
            for (const auto& x: input) {
                if (x == item) return true;
            }
            return false;
        }
    public:
        WorkMemoryPool() = default;
        ~WorkMemoryPool() {
            for (auto ptr: alloc_small_page) {
                free(ptr);
            }

            for (auto ptr: alloc_mid_page) {
                free(ptr);
            }

            for (auto ptr: alloc_large) {
                free(ptr);
            }
        }
        template<class T = void>
        T* Alloc(size_t N) {
            if (N <= 4096) {
               return AllocSmall<T>(N);
            } else if (N <= 32768) {
                return AllocMid<T>(N);
            } else {
                return AllocLarge<T>(N);
            }
        };

        void Free(void *ptr) {
            if (contains(alloc_small_page, ptr)) {
                free_small_page.push(ptr);
            } else if (contains(alloc_mid_page, ptr)) {
                free_mid_page.push(ptr);
            } else {
                assert(contains(alloc_large, ptr));
            }
        }

        static WorkMemoryPool& Global() {
            thread_local WorkMemoryPool pool;
            return pool;
        }
    };


    /**
     * memory allocated will not be freed until the termination of the program
     */
    class DataMemoryPool
    {
    private:
        std::mutex _mutex;
        std::vector<DataPage> _allocated_pages;
        DataPage _free_page;
        const size_t _default_size{1048576}; // 1 MiB
        const size_t _alignment{16};
    public:
        DataMemoryPool() = default;

        ~DataMemoryPool() {
            for (auto page: _allocated_pages) {
                free(page._ptr);
            }
        }

        template<class T>
        T* Alloc(size_t size) {
            std::unique_lock<std::mutex> lock(_mutex);
            if (_free_page._size < _default_size) {
                size = std::max(size, _default_size);
                _free_page = DataPage(size, _alignment);
                _allocated_pages.push_back(_free_page);
            }
            return static_cast<T*>(_free_page.Alloc(size));
        }

        size_t total_bytes() {
            size_t size{0};
            for (auto page: _allocated_pages) {
                size += page._size;
            }
            return size;
        }

        static DataMemoryPool& Global() {
            static DataMemoryPool pool;
            return pool;
        }
    };
}
#endif //PICKLE_MEMPOOL_HPP
