add_library(distance distance/generic.cpp)
add_library(hnsw hnsw.cpp)
target_link_libraries(hnsw PRIVATE distance)