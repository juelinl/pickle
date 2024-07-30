cmake_minimum_required(VERSION 3.22)
set(MKL_DIR /data/opt/intel/mkl/2024.2/lib/cmake/mkl/)
set(FAISS_ENABLE_GPU OFF)
set(FAISS_ENABLE_PYTHON OFF)
set(FAISS_ENABLE_RAFT OFF)

project(pickle)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-inline")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funsafe-math-optimizations")

# Official HNSW Repo
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/hnswlib)

# Faiss
find_package(MKL REQUIRED)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/faiss)

# Read Numpy File in C++ Program
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/cnpy)
include_directories(${CMAKE_SOURCE_DIR}/third_party/cnpy/include)

# Logging
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/spdlog)
include_directories(${CMAKE_SOURCE_DIR}/third_party/spdlog/include)

# OpenMP for Multithreading
find_package(OpenMP REQUIRED)
link_libraries(OpenMP::OpenMP_CXX)

include_directories(${CMAKE_SOURCE_DIR}/include)
add_subdirectory(benchmark)

#add_subdirectory(pybind)
#add_subdirectory(program)
#add_subdirectory(test)