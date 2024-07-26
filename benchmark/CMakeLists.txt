add_executable(hnsw_official hnsw_official.cpp)
target_link_libraries(hnsw_official PRIVATE hnswlib::hnswlib spdlog::spdlog cnpy)

add_executable(hnsw_faiss hnsw_faiss.cpp)
target_link_libraries(hnsw_faiss PRIVATE faiss spdlog::spdlog cnpy)

add_executable(hnsw_pickle hnsw_pickle.cpp)
target_link_libraries(hnsw_pickle PRIVATE spdlog::spdlog cnpy)