cmake_minimum_required(VERSION 3.22)

project(pickle)

set(CMAKE_CXX_STANDARD 23)
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-inline")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -funsafe-math-optimizations")

include_directories("${CMAKE_SOURCE_DIR}/include")
find_package(OpenMP REQUIRED)
link_libraries(OpenMP::OpenMP_CXX)
add_subdirectory(pybind)