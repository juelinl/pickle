cmake_minimum_required(VERSION 3.22)

project(pickle)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -funsafe-math-optimizations")

include_directories("${CMAKE_SOURCE_DIR}/include")

add_subdirectory(src)