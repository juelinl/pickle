//
// Created by juelin on 7/9/24.
//

#ifndef PICKLE_MEMPOOL_HPP
#define PICKLE_MEMPOOL_HPP
#include <mutex>
#include <vector>
#include <cassert>
#include <algorithm>

namespace pickle {
    // Need to manually free it
    struct DataPage
    {
        char * _ptr{nullptr};
        int64_t _size{0};
        DataPage() = default;
        explicit DataPage(size_t size, size_t alignment) {
            _ptr = static_cast<char*>(std::aligned_alloc(alignment, size));
            _size = size;
        }

        void * Alloc(int64_t N) {
            assert(_size >= N);
            assert(N % 4 == 0);
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
        size_t small_page_size{4096}; // 4KiB per page
        size_t mid_page_size{32768}; // 32KiB per page
        size_t large_page_size{1048576}; // 1MiB per page

        std::vector<void *> alloc_small_page;
        std::vector<void *> free_small_page;

        std::vector<void *> alloc_mid_page;
        std::vector<void *> free_mid_page;

        std::vector<void*> alloc_large_page; // at least 1MiB per page
        std::vector<void *> free_large_page;

        template<class T>
        T* AllocSmall(size_t N) {
            assert(N <= small_page_size);
            void * ptr{nullptr};
            if (free_small_page.empty()) {
                ptr = std::aligned_alloc(64, small_page_size);
                alloc_small_page.push_back(ptr);
            } else {
                ptr = free_small_page.back();
                free_small_page.pop_back();
            }
//            memset(ptr, 0, small_page_size);
            return static_cast<T*>(ptr);
        };

        template<class T>
        T* AllocMid(size_t N) {
            assert(N <= mid_page_size);
            void * ptr{nullptr};
            if (free_mid_page.empty()) {
                ptr = std::aligned_alloc(64, mid_page_size);
                alloc_mid_page.push_back(ptr);
            } else {
                ptr = free_mid_page.back();
                free_mid_page.pop_back();
            }
//            memset(ptr, 0, mid_page_size);
            return static_cast<T*>(ptr);
        };

        template<class T>
        T* AllocLarge(size_t N) {
            assert(N <= large_page_size);
            void * ptr{nullptr};
            if (free_large_page.empty()) {
                ptr = std::aligned_alloc(64, large_page_size);
                alloc_large_page.push_back(ptr);
            } else {
                ptr = free_large_page.back();
                free_large_page.pop_back();
            }
//            memset(ptr, 0, large_page_size);
            return static_cast<T*>(ptr);
        };

        static inline bool contains(const std::vector<void*>& input, void* ptr) {
            return std::find(input.begin(), input.end(), ptr) != input.end();
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

            for (auto ptr: alloc_large_page) {
                free(ptr);
            }
        }
        template<class T = void>
        T* Alloc(size_t N) {
            if (N <= small_page_size) {
               return AllocSmall<T>(N);
            } else if (N <= mid_page_size) {
                return AllocMid<T>(N);
            } else if (N <= large_page_size){
                return AllocLarge<T>(N);
            } else {
                return static_cast<T*>(std::aligned_alloc(64, N));
            }
        };

        void Free(void *ptr) {
            if (contains(alloc_small_page, ptr)) {
                assert(!contains(free_small_page, ptr));
                free_small_page.push_back(ptr);
            } else if (contains(alloc_mid_page, ptr)) {
                assert(!contains(free_mid_page, ptr));
                free_mid_page.push_back(ptr);
            } else if (contains(alloc_large_page, ptr)){
                assert(!contains(free_large_page, ptr));
                free_large_page.push_back(ptr);
            } else {
                free(ptr);
            }
        }

        static WorkMemoryPool& ThreadLocal() {
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
        const size_t _default_size{16 * 1048576}; // 16 MiB
        const size_t _alignment{64};
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
            if (_free_page._size < size) {
                size_t page_size = std::max(size, _default_size);
                _free_page = DataPage(page_size, _alignment);
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

    class LayerMemoryPool {
    private:
        std::array<DataMemoryPool, 10> pool;
    public:
        static LayerMemoryPool& Global() {
            static LayerMemoryPool pool;
            return pool;
        }

        template<class T>
        T* Alloc(size_t size, size_t level) {
            assert(level < 10);
            return pool[level].Alloc<T>(size);
        }
    };
}
#endif //PICKLE_MEMPOOL_HPP
